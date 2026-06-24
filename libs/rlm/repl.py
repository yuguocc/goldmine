import io
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rlm import RLM
from rlm.utils.custom_tools import extract_tool_value, validate_custom_tools


_PROCESS_STATE_LOCK = threading.RLock()


# Simple sub LM for REPL environment. Note: This could also be just the RLM itself!
class Sub_RLM(RLM):
    """Recursive LLM client for REPL environment with fixed configuration."""

    def __init__(self, model: str = "gpt-5"):
        # Configuration - model can be specified
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model = model

        # Initialize OpenAI client
        from rlm.utils.llm import OpenAIClient

        self.client = OpenAIClient(api_key=self.api_key, model=model)

    def completion(self, prompt) -> str:
        """
        Simple LM query for sub-LM call.
        """
        try:
            # Handle both string and dictionary/list inputs
            response = self.client.completion(messages=prompt, timeout=300)

            return response

        except Exception as e:
            error_msg = f"Error making LLM query: {str(e)}"
            return error_msg

    def cost_summary(self) -> dict[str, float]:
        raise NotImplementedError("Cost tracking is not implemented for the Sub-RLM.")

    def reset(self):
        raise NotImplementedError("Reset is not implemented for the Sub-RLM.")


class _AnswerDict(dict):
    """REPL-visible dict where answer["ready"] = True signals completion."""

    def __init__(
        self,
        on_ready=None,
        *,
        validator: Callable[[Any], Any] | None = None,
        on_reject=None,
    ):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready
        self._validator = validator
        self._on_reject = on_reject

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            content = self.get("content", "")
            rejection = self._rejection_reason(content)
            if rejection:
                super().__setitem__("ready", False)
                super().__setitem__("rejected", True)
                super().__setitem__("rejection_reason", rejection)
                if self._on_reject is not None:
                    try:
                        self._on_reject(rejection)
                    except Exception:
                        pass
                return
            try:
                self._on_ready(content)
            except Exception:
                pass

    def _rejection_reason(self, content: Any) -> str:
        if self._validator is None:
            return ""
        try:
            verdict = self._validator(content)
        except Exception as exc:
            return f"final answer validator failed: {type(exc).__name__}: {exc}"
        if verdict is None or verdict is True or verdict == "":
            return ""
        if verdict is False:
            return "final answer rejected by validator"
        if isinstance(verdict, tuple) and verdict:
            ok = bool(verdict[0])
            if ok:
                return ""
            return str(verdict[1] if len(verdict) > 1 else "final answer rejected")
        return str(verdict)


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    final_answer: str | None = None

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict,
        execution_time: float = None,
        final_answer: str | None = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.final_answer = final_answer

    def __str__(self):
        return (
            f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, "
            f"locals={self.locals}, execution_time={self.execution_time}, "
            f"final_answer={self.final_answer})"
        )


class REPLEnv:
    def __init__(
        self,
        recursive_model: str = "gpt-5-mini",
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
        setup_code: str = None,
        custom_tools: Optional[dict[str, Any]] = None,
        final_answer_validator: Callable[[Any], Any] | None = None,
    ):
        # Store the original working directory
        self.original_cwd = os.getcwd()

        # Each REPL owns a temp working directory so relative files created by
        # code blocks do not collide across parallel RLM runs.
        self.temp_dir = self._make_temp_dir()

        self.custom_tools = custom_tools or {}
        validate_custom_tools(self.custom_tools)

        # Initialize minimal RLM / LM client. Change this to support more depths.
        self.sub_rlm: RLM = Sub_RLM(model=recursive_model)

        # Create safe globals with only string-safe built-ins
        self.globals = {
            "__builtins__": {
                # Safe built-ins for string manipulation
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "chr": chr,
                "ord": ord,
                "hex": hex,
                "bin": bin,
                "oct": oct,
                "repr": repr,
                "ascii": ascii,
                "format": format,
                "__import__": __import__,  # Allow imports
                "open": open,  # Allow file access
                # Add commonly used built-ins that were missing
                "any": any,
                "all": all,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "delattr": delattr,
                "dir": dir,
                "vars": vars,
                "range": range,  # Add range function
                "reversed": reversed,  # Add reversed function
                "slice": slice,  # Add slice function
                "iter": iter,  # Add iter function
                "next": next,  # Add next function
                "pow": pow,  # Add pow function
                "divmod": divmod,  # Add divmod function
                "complex": complex,  # Add complex function
                "bytes": bytes,  # Add bytes function
                "bytearray": bytearray,  # Add bytearray function
                "memoryview": memoryview,  # Add memoryview function
                "hash": hash,  # Add hash function
                "id": id,  # Add id function
                "callable": callable,  # Add callable function
                "issubclass": issubclass,  # Add issubclass function
                "super": super,  # Add super function
                "property": property,  # Add property function
                "staticmethod": staticmethod,  # Add staticmethod function
                "classmethod": classmethod,  # Add classmethod function
                "object": object,  # Add object class
                "BaseException": BaseException,  # Add BaseException class
                "ArithmeticError": ArithmeticError,  # Add ArithmeticError class
                "LookupError": LookupError,  # Add LookupError class
                "EnvironmentError": EnvironmentError,  # Add EnvironmentError class
                "AssertionError": AssertionError,  # Add AssertionError class
                "NotImplementedError": NotImplementedError,  # Add NotImplementedError class
                "UnicodeError": UnicodeError,  # Add UnicodeError class
                "Warning": Warning,  # Add Warning class
                "UserWarning": UserWarning,  # Add UserWarning class
                "DeprecationWarning": DeprecationWarning,  # Add DeprecationWarning class
                "PendingDeprecationWarning": PendingDeprecationWarning,  # Add PendingDeprecationWarning class
                "SyntaxWarning": SyntaxWarning,  # Add SyntaxWarning class
                "RuntimeWarning": RuntimeWarning,  # Add RuntimeWarning class
                "FutureWarning": FutureWarning,  # Add FutureWarning class
                "ImportWarning": ImportWarning,  # Add ImportWarning class
                "UnicodeWarning": UnicodeWarning,  # Add UnicodeWarning class
                "BytesWarning": BytesWarning,  # Add BytesWarning class
                "ResourceWarning": ResourceWarning,  # Add ResourceWarning class
                # Add exception classes
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "FileNotFoundError": FileNotFoundError,
                "OSError": OSError,
                "IOError": IOError,
                "RuntimeError": RuntimeError,
                "NameError": NameError,
                "ImportError": ImportError,
                "StopIteration": StopIteration,
                "GeneratorExit": GeneratorExit,
                "SystemExit": SystemExit,
                "KeyboardInterrupt": KeyboardInterrupt,
                # Disallow the following built-ins
                "input": None,  # Block input
                "eval": None,  # Block eval
                "exec": None,  # Block exec
                "compile": None,  # Block compile
                "globals": None,  # Block globals access
                "locals": None,  # Block locals access
            }
        }
        self.locals = {}
        self._last_final_answer: str | None = None
        self._last_answer_rejection: str = ""
        self._final_answer_validator = final_answer_validator
        self.locals["answer"] = self._new_answer_dict()
        self._lock = threading.Lock()
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()

        def llm_query(prompt: str) -> str:
            """Query the LLM with the given prompt."""
            return self.sub_rlm.completion(prompt)

        self._llm_query_fn = llm_query
        # Add (R)LM query function to globals
        self.globals["llm_query"] = llm_query
        self.globals["SHOW_VARS"] = self._show_vars

        self._inject_custom_tools()
        self.load_context(context_json, context_str)

        # Finally, run any setup code if provided
        if setup_code:
            self.code_execution(setup_code)

    def _make_temp_dir(self) -> str:
        root = os.getenv("RLM_REPL_TEMP_DIR") or None
        try:
            temp_dir = tempfile.mkdtemp(prefix="repl_env_", dir=root)
            probe_path = os.path.join(temp_dir, ".write_test")
            with open(probe_path, "w") as f:
                f.write("ok")
            os.remove(probe_path)
        except Exception:
            try:
                if "temp_dir" in locals():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
            root_msg = root or tempfile.gettempdir()
            raise RuntimeError(
                "REPL temporary directory is not writable. "
                f"Checked root: {root_msg!r}. Set RLM_REPL_TEMP_DIR to a writable directory."
            )
        return temp_dir

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def _reject_answer(self, reason: Any) -> None:
        self._last_answer_rejection = str(reason)

    def _new_answer_dict(self) -> _AnswerDict:
        return _AnswerDict(
            on_ready=self._capture_answer,
            validator=self._final_answer_validator,
            on_reject=self._reject_answer,
        )

    def _show_vars(self) -> str:
        available = {
            key: type(value).__name__
            for key, value in self.locals.items()
            if not key.startswith("_") and key != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _restore_scaffold(self) -> None:
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query_fn
        if "context_0" in self.locals:
            self.locals["context"] = self.locals["context_0"]

        current = self.locals.get("answer")
        if isinstance(current, _AnswerDict):
            return

        replacement = self._new_answer_dict()
        if isinstance(current, dict):
            for key, value in current.items():
                if key != "ready":
                    dict.__setitem__(replacement, key, value)
            if current.get("ready"):
                replacement["ready"] = True
            else:
                dict.__setitem__(replacement, "ready", current.get("ready", False))
        self.locals["answer"] = replacement

    def _inject_custom_tools(self):
        for name, entry in self.custom_tools.items():
            value = extract_tool_value(entry)
            if callable(value):
                self.globals[name] = value
            else:
                self.locals[name] = value

    def load_context(
        self,
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
    ):
        if context_json is not None:
            self.locals["context_0"] = context_json
            self.locals["context"] = context_json

        if context_str is not None:
            self.locals["context_0"] = context_str
            self.locals["context"] = context_str

    def __del__(self):
        """Clean up the per-environment temporary directory."""
        try:
            temp_dir = getattr(self, "temp_dir", None)
            if temp_dir and os.path.basename(os.path.normpath(temp_dir)).startswith(
                "repl_env_"
            ):
                shutil.rmtree(temp_dir)
        except:
            pass

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr"""
        with _PROCESS_STATE_LOCK, self._lock:
            # Store original streams
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            # Create new buffers for this execution
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                # Redirect streams
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                # Restore original streams
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @contextmanager
    def _temp_working_directory(self):
        """Temporarily run REPL code from this environment's temp directory."""
        with _PROCESS_STATE_LOCK:
            old_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
                yield
            finally:
                os.chdir(old_cwd)

    def code_execution(self, code) -> REPLResult:
        """
        Simple code execution "notebook-style" in a REPL environment.
        """
        start_time = time.time()
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    combined_namespace = {**self.globals, **self.locals}
                    exec(code, combined_namespace, combined_namespace)

                    # Update locals with any new variables created.
                    for key, value in combined_namespace.items():
                        if key not in self.globals:
                            self.locals[key] = value
                    self._restore_scaffold()

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()
                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()

        end_time = time.time()
        execution_time = end_time - start_time

        # Store output in locals for access
        if self._last_answer_rejection:
            stderr_content = (
                stderr_content
                + ("\n" if stderr_content else "")
                + f"Final answer rejected: {self._last_answer_rejection}"
            )
            self._last_answer_rejection = ""
        self.locals["_stdout"] = stdout_content
        self.locals["_stderr"] = stderr_content
        final_answer = self._last_final_answer
        self._last_final_answer = None

        return REPLResult(
            stdout_content,
            stderr_content,
            self.locals.copy(),
            execution_time,
            final_answer=final_answer,
        )

    def get_cost_summary(self):
        raise NotImplementedError(
            "Cost tracking is not implemented for the REPL Environment."
        )
