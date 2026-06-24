"""
Simple Recursive Language Model (RLM) with REPL environment.
"""

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Any

from rlm import RLM, RLMResult
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 max_iterations: int = 20,
                 depth: int = 0,
                 enable_logging: bool = False,
                 custom_tools: Optional[dict[str, Any]] = None,
                 final_answer_validator: Optional[Callable[[Any], Any]] = None,
                 request_timeout: Optional[float] = 300,
                 ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.custom_tools = custom_tools
        self.final_answer_validator = final_answer_validator
        self.llm = OpenAIClient(api_key, model) # Replace with other client
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        self.depth = depth # Unused in this version.
        self._max_iterations = max_iterations
        self.request_timeout = request_timeout
        
        # Initialize colorful logger
        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)
        
        self.messages = [] # Initialize messages list
        self.query = None
        self.reset_trajectory()

    def _message_chars(self, messages: List[Dict[str, str]]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    def _root_completion(self, messages: List[Dict[str, str]]) -> str:
        kwargs = {}
        if self.request_timeout is not None:
            kwargs["timeout"] = self.request_timeout
        response = self.llm.completion(messages, **kwargs)
        return "" if response is None else str(response)

    def _final_answer_from_code_blocks(self, iteration_record: Dict[str, Any]) -> str | None:
        for code_block in iteration_record["code_blocks"]:
            if code_block.get("final_answer") is not None:
                return code_block["final_answer"]
        return None

    def reset_trajectory(self):
        """Reset structured execution trajectory for the next completion call."""
        self.trajectory = {
            "status": "initialized",
            "query": None,
            "root_model": self.model,
            "recursive_model": self.recursive_model,
            "max_iterations": self._max_iterations,
            "context": None,
            "initial_messages": [],
            "iterations": [],
            "final_answer": None,
            "fallback": None,
        }

    def get_trajectory(self) -> Dict[str, Any]:
        """Return a copy of the latest structured execution trajectory."""
        return deepcopy(self.trajectory)

    def _build_result(self, response: str) -> RLMResult:
        """Build a completion result using the current trajectory as metadata."""
        return RLMResult(response=response, metadata=self.get_trajectory())

    def _final_answer_rejection(self, response: Any) -> str:
        if self.final_answer_validator is None:
            return ""
        try:
            verdict = self.final_answer_validator(response)
        except Exception as exc:
            return f"final answer validator failed: {type(exc).__name__}: {exc}"
        if verdict is None or verdict is True or verdict == "":
            return ""
        if verdict is False:
            return "final answer rejected by validator"
        if isinstance(verdict, tuple) and verdict:
            if bool(verdict[0]):
                return ""
            return str(verdict[1] if len(verdict) > 1 else "final answer rejected")
        return str(verdict)

    def _summarize_context(self, context: List[str] | str | List[Dict[str, str]]) -> Dict[str, Any]:
        """Keep context metadata small; the full context already lives in the REPL."""
        if isinstance(context, str):
            return {"type": "str", "chars": len(context)}
        if isinstance(context, list):
            return {"type": "list", "items": len(context)}
        if isinstance(context, dict):
            return {"type": "dict", "keys": list(context.keys())[:20]}
        return {"type": type(context).__name__}
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.reset_trajectory()
        self.trajectory.update(
            {
                "status": "running",
                "query": query,
                "context": self._summarize_context(context),
            }
        )
        self.logger.log_query_start(query)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt(custom_tools=self.custom_tools)
        self.trajectory["initial_messages"] = deepcopy(self.messages)
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
            custom_tools=self.custom_tools,
            final_answer_validator=self.final_answer_validator,
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> RLMResult:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        """
        self.messages = self.setup_context(context, query)
        
        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):
            
            # Query root LM to interact with REPL environment
            action_prompt = next_action_prompt(query, iteration)
            current_prompt = self.messages + [action_prompt]
            response = self._root_completion(current_prompt)
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            iteration_record = {
                "iteration": iteration,
                "prompt": deepcopy(current_prompt),
                "prompt_message_count": len(current_prompt),
                "prompt_chars": self._message_chars(current_prompt),
                "response": response,
                "code_blocks": [],
                "final_answer": None,
            }
            self.trajectory["iterations"].append(iteration_record)
            self.logger.log_model_response(response, has_tool_calls=bool(code_blocks))
            
            # Process code execution or add assistant message
            if code_blocks:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env, 
                    self.repl_env_logger, self.logger,
                    trajectory_iteration=iteration_record,
                )
            else:
                # Add assistant message when there are no code blocks
                assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                self.messages.append(assistant_message)
            
            # Match the full RLM termination protocol: the REPL signals
            # completion by setting answer["ready"] = True during execution.
            final_answer = self._final_answer_from_code_blocks(iteration_record)
            iteration_record["final_answer"] = final_answer

            # In practice, you may need some guardrails here.
            if final_answer is not None:
                self.trajectory["status"] = "completed"
                self.trajectory["final_answer"] = final_answer
                self.logger.log_final_response(final_answer)
                return self._build_result(final_answer)

            
        # If we reach here, no final answer was found in any iteration
        print("No final answer found in any iteration")
        fallback_prompt = next_action_prompt(query, self._max_iterations, final_answer=True)
        self.messages.append(fallback_prompt)
        fallback_messages = deepcopy(self.messages)
        fallback_response = self._root_completion(self.messages)
        fallback_record = {
            "prompt": fallback_messages,
            "prompt_message_count": len(fallback_messages),
            "prompt_chars": self._message_chars(fallback_messages),
            "response": fallback_response,
            "code_blocks": [],
            "final_answer": None,
        }
        fallback_code_blocks = utils.find_code_blocks(fallback_response)
        if fallback_code_blocks:
            self.messages = utils.process_code_execution(
                fallback_response,
                self.messages,
                self.repl_env,
                self.repl_env_logger,
                self.logger,
                trajectory_iteration=fallback_record,
            )
        final_answer = self._final_answer_from_code_blocks(fallback_record)
        fallback_record["final_answer"] = final_answer
        if final_answer is None:
            final_answer = fallback_response
        rejection = self._final_answer_rejection(final_answer)
        self.trajectory["status"] = (
            "fallback_completed"
            if fallback_record["final_answer"] is not None
            else "max_iterations_exceeded"
        )
        self.trajectory["fallback"] = fallback_record
        if rejection:
            self.trajectory["status"] = "final_answer_rejected"
            self.trajectory["fallback"]["rejection"] = rejection
            self.trajectory["final_answer"] = None
            self.logger.log_final_response(f"Final answer rejected: {rejection}")
            return self._build_result(f"Final answer rejected: {rejection}")
        self.trajectory["final_answer"] = final_answer
        self.logger.log_final_response(final_answer)

        return self._build_result(final_answer)
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv(
            custom_tools=self.custom_tools,
            final_answer_validator=self.final_answer_validator,
        )
        self.messages = []
        self.query = None
        self.reset_trajectory()


if __name__ == "__main__":
    pass
