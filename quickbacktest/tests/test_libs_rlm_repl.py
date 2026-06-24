from __future__ import annotations

import sys
import uuid
from pathlib import Path


LIBS_ROOT = Path(__file__).resolve().parents[2] / "libs"
if str(LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBS_ROOT))


def _writable_temp_root(name: str) -> Path:
    path = (
        Path("runs")
        / "test_libs_rlm_repl"
        / f"{name}_{uuid.uuid4().hex}"
    ).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_rlm_messages_are_not_truncated_at_trajectory_limit(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    import rlm.repl as repl_module
    import rlm.rlm_repl as rlm_repl

    monkeypatch.setattr(
        repl_module.REPLEnv,
        "_make_temp_dir",
        lambda self: str(_writable_temp_root("long_messages")),
    )

    prompts = []

    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass

        def completion(self, messages, **kwargs):
            prompts.append(messages)
            if len(prompts) == 1:
                return "```repl\nprint('START' + 'x' * 12050 + 'END')\n```"
            return (
                "```repl\n"
                "answer['content'] = 'done'\n"
                "answer['ready'] = True\n"
                "```"
            )

    monkeypatch.setattr(rlm_repl, "OpenAIClient", FakeOpenAIClient)

    rlm = rlm_repl.RLM_REPL(api_key="test-key", max_iterations=2)
    result = rlm.completion(context="context", query="query")

    second_prompt_text = "\n".join(item["content"] for item in prompts[1])
    assert result.response == "done"
    assert result.metadata["status"] == "completed"
    assert len(second_prompt_text) > 10_000
    assert "START" in second_prompt_text
    assert "END" in second_prompt_text
    assert result.metadata["iterations"][1]["prompt_chars"] > 10_000


def test_rlm_fallback_executes_repl_final_answer(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    import rlm.repl as repl_module
    import rlm.rlm_repl as rlm_repl

    monkeypatch.setattr(
        repl_module.REPLEnv,
        "_make_temp_dir",
        lambda self: str(_writable_temp_root("fallback")),
    )

    prompts = []

    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass

        def completion(self, messages, **kwargs):
            prompts.append(messages)
            if len(prompts) == 1:
                return "I have not finished yet."
            return (
                "```repl\n"
                "answer['content'] = 'fallback done'\n"
                "answer['ready'] = True\n"
                "```"
            )

    monkeypatch.setattr(rlm_repl, "OpenAIClient", FakeOpenAIClient)

    rlm = rlm_repl.RLM_REPL(api_key="test-key", max_iterations=1)
    result = rlm.completion(context="context", query="query")

    assert result.response == "fallback done"
    assert result.metadata["status"] == "fallback_completed"
    assert result.metadata["fallback"]["final_answer"] == "fallback done"


def test_rlm_fallback_validator_rejection_returns_bounded_result(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    import rlm.repl as repl_module
    import rlm.rlm_repl as rlm_repl

    monkeypatch.setattr(
        repl_module.REPLEnv,
        "_make_temp_dir",
        lambda self: str(_writable_temp_root("fallback_rejected")),
    )

    prompts = []

    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass

        def completion(self, messages, **kwargs):
            prompts.append(messages)
            if len(prompts) == 1:
                return "I have not finished yet."
            return (
                "```repl\n"
                "answer['content'] = 'fallback rejected'\n"
                "answer['ready'] = True\n"
                "```"
            )

    monkeypatch.setattr(rlm_repl, "OpenAIClient", FakeOpenAIClient)

    rlm = rlm_repl.RLM_REPL(
        api_key="test-key",
        max_iterations=1,
        final_answer_validator=lambda content: (False, "validator says no"),
    )
    result = rlm.completion(context="context", query="query")

    assert result.response == "Final answer rejected: validator says no"
    assert result.metadata["status"] == "final_answer_rejected"
    assert result.metadata["final_answer"] is None
    assert result.metadata["fallback"]["rejection"] == "validator says no"
    assert len(prompts) == 2


def test_rlm_none_model_response_falls_back_without_type_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    import rlm.repl as repl_module
    import rlm.rlm_repl as rlm_repl
    from rlm.utils.utils import find_code_blocks

    assert find_code_blocks(None) == []

    monkeypatch.setattr(
        repl_module.REPLEnv,
        "_make_temp_dir",
        lambda self: str(_writable_temp_root("none_response")),
    )

    prompts = []

    class FakeOpenAIClient:
        def __init__(self, *args, **kwargs):
            pass

        def completion(self, messages, **kwargs):
            prompts.append(messages)
            if len(prompts) == 1:
                return None
            return (
                "```repl\n"
                "answer['content'] = 'fallback after none'\n"
                "answer['ready'] = True\n"
                "```"
            )

    monkeypatch.setattr(rlm_repl, "OpenAIClient", FakeOpenAIClient)

    rlm = rlm_repl.RLM_REPL(api_key="test-key", max_iterations=1)
    result = rlm.completion(context="context", query="query")

    assert result.response == "fallback after none"
    assert result.metadata["status"] == "fallback_completed"
    assert result.metadata["iterations"][0]["response"] == ""
