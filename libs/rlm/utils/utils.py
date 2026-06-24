"""
Utility functions for the RLM REPL Client.
"""

import re
from typing import List, Dict, Optional, Any

def find_code_blocks(text: str | None) -> List[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    if text is None:
        return []
    text = str(text)
    pattern = r'```repl\s*\n(.*?)\n```'
    results = []
    
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)
    
    return results

def add_execution_result_to_messages(messages: List[Dict[str, str]], 
                                   code: str, 
                                   result: str,
                                   max_character_length: int = 100000,
                                   ) -> List[Dict[str, str]]:
    """
    Add code execution result to the conversation messages.
    
    Args:
        messages: Current conversation messages
        code: The code that was executed
        result: Result from code execution
        max_character_length: Maximum character length of the result
        disable_recursive: When recursion is disabled, we need to use the entire stdout
        
    Returns:
        Updated messages list
    """
    # Truncate result if it exceeds 100k characters
    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."
    
    # Add the code execution result
    execution_message = {
        "role": "user",
        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}"
    }
    messages.append(execution_message)
    return messages

def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: Dict[str, Any],
    final_answer: Optional[str] = None,
    truncate_length: int = 100
) -> str:
    """
    Format the execution result as a string for display.
    
    Args:
        stdout: Standard output from execution
        stderr: Standard error from execution
        locals_dict: Local variables after execution
        truncate_length: Maximum length of the string to display per var
    """
    result_parts = []
    
    if stdout:
        result_parts.append(f"\n{stdout}")
    
    if stderr:
        result_parts.append(f"\n{stderr}")

    if final_answer is not None:
        result_parts.append(f"Final answer signaled: {final_answer}")
    
    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in locals_dict.items():
        if not key.startswith('_') and not key in ['__builtins__', '__name__', '__doc__']:
            try:
                # Only show simple types or short representations
                if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                    if isinstance(value, str) and len(value) > truncate_length:
                        important_vars[key] = f"'{value[:truncate_length]}...'"
                    else:
                        important_vars[key] = repr(value)
            except:
                important_vars[key] = f"<{type(value).__name__}>"
    
    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")
    
    return "\n\n".join(result_parts) if result_parts else "No output"


def truncate_for_trajectory(text: str, max_character_length: int = 10000) -> str:
    """Keep trajectory records bounded when REPL output is large."""
    if len(text) <= max_character_length:
        return text
    return text[:max_character_length] + "..."


def summarize_locals_for_trajectory(locals_dict: Dict[str, Any]) -> Dict[str, str]:
    """Record variable names and types without copying large values into trajectory."""
    return {
        key: type(value).__name__
        for key, value in locals_dict.items()
        if not key.startswith("_") and key not in ["__builtins__", "__name__", "__doc__"]
    }


def execute_code(
    repl_env,
    code: str,
    repl_env_logger,
    logger,
    trajectory_entry: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Execute code in the REPL environment and return formatted result.
    
    Args:
        repl_env: The REPL environment
        code: Python code to execute
        repl_env_logger: Logger for execution environment
        logger: Main logger
        state_logger: Optional state logger
        
    Returns:
        Formatted execution result
    """
    try:
        result = repl_env.code_execution(code)
        
        formatted_result = format_execution_result(
            result.stdout, result.stderr, result.locals, result.final_answer
        )

        if trajectory_entry is not None:
            trajectory_entry.update(
                {
                    "stdout": truncate_for_trajectory(result.stdout),
                    "stderr": truncate_for_trajectory(result.stderr),
                    "locals": summarize_locals_for_trajectory(result.locals),
                    "execution_time": result.execution_time,
                    "final_answer": result.final_answer,
                    "formatted_result": truncate_for_trajectory(formatted_result),
                }
            )

        repl_env_logger.log_execution(code, result.stdout, result.stderr, result.execution_time)
        repl_env_logger.display_last()

        # Print out tool execution to root
        logger.log_tool_execution("CODE_EXECUTION", formatted_result)
        
        return formatted_result
        
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        if trajectory_entry is not None:
            trajectory_entry.update({"error": error_msg})
        return error_msg

def process_code_execution(
    response: str,
    messages: List[Dict[str, str]],
    repl_env,
    repl_env_logger,
    logger,
    trajectory_iteration: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Process code execution from the model response. If recursive is disabled, we should
    return the entire stdout block.
    
    Args:
        response: The model response containing code
        messages: Current conversation messages
        repl_env: The REPL environment
        repl_env_logger: Logger for execution environment
        logger: Main logger
        
    Returns:
        Updated messages list
    """
    # Extract code blocks from response
    code_blocks = find_code_blocks(response)
    
    if code_blocks:
        # Execute each code block
        for code in code_blocks:
            trajectory_entry = {"code": code}
            execution_result = execute_code(
                repl_env,
                code,
                repl_env_logger,
                logger,
                trajectory_entry=trajectory_entry,
            )

            if trajectory_iteration is not None:
                trajectory_iteration.setdefault("code_blocks", []).append(trajectory_entry)
            
            # Add execution result to conversation
            messages = add_execution_result_to_messages(
                messages, code, execution_result, 
            )
    
    return messages

def convert_context_for_repl(context):
    """
    Convert REPL context to either some 
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None
    
    return context_data, context_str
