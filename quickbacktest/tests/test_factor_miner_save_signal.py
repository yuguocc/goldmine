import json
from pathlib import Path
import sys
import uuid

import factor_miner
from factor_miner import (
    FactorMinerCaseConfig,
    build_signal_code_from_compute,
    build_factor_miner_custom_tools,
    build_factor_miner_final_answer_validator,
    compute_examples_prompt_text,
    initialize_qlib_for_factor_miner,
    save_signal_code,
)

sys.path.insert(0, str(factor_miner.LIBS_ROOT))
from rlm.repl import _AnswerDict


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_factor_miner_save_signal" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_save_signal_code_writes_without_factor_validation():
    workspace = _workspace_tmp("save")
    code = """from quickbacktest.base_types import BaseSignal


class BrokenAtRuntime(BaseSignal):
    name = "BrokenAtRuntime"

    def compute(self, **kwargs):
        return missing_name
"""

    result = save_signal_code(workspace, "BrokenAtRuntime", code)

    assert result["ok"] is True
    signal_path = Path(result["signal_path"])
    assert signal_path.exists()
    assert "missing_name" in signal_path.read_text(encoding="utf-8")


def test_save_signal_code_overwrites_by_default():
    workspace = _workspace_tmp("overwrite")
    first = "class A:\n    pass\n"
    second = "class A:\n    value = 2\n"

    first_result = save_signal_code(workspace, "OverwriteFactor", first)
    second_result = save_signal_code(workspace, "OverwriteFactor", second)

    assert first_result["ok"] is True
    assert second_result["ok"] is True
    assert "value = 2" in Path(second_result["signal_path"]).read_text(encoding="utf-8")


def test_factor_miner_custom_tools_expose_submit_compute_by_default():
    tools = build_factor_miner_custom_tools(_workspace_tmp("tools"))

    assert set(tools) == {"submit_compute"}
    assert "submit_compute" in tools["submit_compute"]["description"]
    assert "ok=True" in tools["submit_compute"]["description"]


def test_factor_miner_custom_tools_can_expose_legacy_save_and_run():
    tools = build_factor_miner_custom_tools(
        _workspace_tmp("legacy_tools"),
        expose_legacy_tools=True,
    )

    assert set(tools) == {
        "read_signal_template",
        "submit_compute",
        "submit_signal",
        "save_signal",
        "run_signal",
    }
    assert "Prefer the compute examples in query" in tools["read_signal_template"]["description"]
    assert "Prefer submit_compute" in tools["save_signal"]["description"]
    assert "Prefer submit_compute" in tools["run_signal"]["description"]


def test_compute_examples_are_in_query_text_not_context():
    context = factor_miner._factor_generation_context(FactorMinerCaseConfig())
    prompt_text = compute_examples_prompt_text()

    assert "compute_template" not in context
    assert "compute_examples" not in context
    assert "Example: momentum" in prompt_text
    assert "Example: mean_reversion" in prompt_text
    assert "Preserve the original DataFrame index/columns" in prompt_text
    assert "pd.Series(values, index=x.index)" in prompt_text
    assert "volatility_normalized_momentum" not in prompt_text
    assert "return " not in prompt_text


def test_initialize_qlib_for_factor_miner_uses_configured_provider(monkeypatch):
    provider = _workspace_tmp("provider")
    seen = {}

    class FakeQlibAdapter:
        def init_qlib_once(self, provider_uri=None):
            seen["provider_uri"] = provider_uri
            return str(provider_uri)

    monkeypatch.setattr(
        factor_miner,
        "_import_qlib_adapter",
        lambda: FakeQlibAdapter(),
    )

    result = initialize_qlib_for_factor_miner(
        FactorMinerCaseConfig(provider_uri=provider)
    )

    assert result == str(provider.resolve())
    assert seen["provider_uri"] == str(provider.resolve())


def test_build_signal_code_from_compute_wraps_body():
    source = build_signal_code_from_compute(
        "WrappedFactor",
        "signal = self.close.pct_change(5, fill_method=None)",
    )

    assert "class WrappedFactor(BaseSignal):" in source
    assert 'name = "WrappedFactor"' in source
    assert "def compute(self, **kwargs) -> pd.DataFrame:" in source
    assert source.count("return signal") == 1
    assert source.rstrip().endswith("return signal")


def test_build_signal_code_from_compute_rejects_full_class():
    try:
        build_signal_code_from_compute(
            "WrappedFactor",
            "class Bad:\n    pass\n",
        )
    except ValueError as exc:
        assert "class definitions" in str(exc)
    else:
        raise AssertionError("expected full class compute code to be rejected")


def test_build_signal_code_from_compute_extracts_def_body():
    source = build_signal_code_from_compute(
        "WrappedFactor",
        "def compute(self, **kwargs):\n    signal = self.close\n    return signal\n",
    )

    assert "        signal = self.close" in source
    assert source.count("def compute") == 1
    assert source.count("return signal") == 1


def test_build_signal_code_from_compute_requires_signal_assignment():
    try:
        build_signal_code_from_compute(
            "WrappedFactor",
            "factor = self.close.pct_change(5, fill_method=None)",
        )
    except ValueError as exc:
        assert "assign the final DataFrame to signal" in str(exc)
    else:
        raise AssertionError("expected missing signal assignment to be rejected")


def test_build_signal_code_from_compute_rejects_non_signal_return():
    try:
        build_signal_code_from_compute(
            "WrappedFactor",
            "factor = self.close.pct_change(5, fill_method=None)\nreturn factor",
        )
    except ValueError as exc:
        assert "do not write return statements" in str(exc)
    else:
        raise AssertionError("expected non-signal return to be rejected")


def test_submit_compute_saves_wrapped_signal_then_runs(monkeypatch):
    workspace = _workspace_tmp("submit_compute")
    seen = {}

    def fake_run_saved_signal(config, module_name=None):
        seen["workspace"] = config.workspace
        seen["module_name"] = module_name
        return {
            "ok": True,
            "module_name": module_name,
            "rank_ic": 0.03,
            "rank_ic_name": "daily_rank_ic_mean",
            "rows": 5,
            "run_signal_json": str(workspace / "run.json"),
        }

    monkeypatch.setattr(factor_miner, "run_saved_signal", fake_run_saved_signal)
    tools = build_factor_miner_custom_tools(workspace)
    compute_code = "signal = self.close.pct_change(5, fill_method=None)\n"

    result = tools["submit_compute"]["tool"](compute_code)

    assert result == {"ok": True}
    assert seen["workspace"] == workspace
    assert seen["module_name"] == factor_miner.GENERATED_SIGNAL_MODULE
    signal_path = workspace / "signals" / f"{factor_miner.GENERATED_SIGNAL_MODULE}.py"
    assert signal_path.exists()
    assert f"class {factor_miner.GENERATED_SIGNAL_MODULE}(BaseSignal):" in (
        signal_path.read_text(encoding="utf-8")
    )


def test_submit_compute_validation_failure_updates_confirmation_status():
    workspace = _workspace_tmp("submit_compute_failure")
    tools = build_factor_miner_custom_tools(workspace)

    result = tools["submit_compute"]["tool"]("")
    status = json.loads((workspace / "run_signal_status.json").read_text())

    assert result["ok"] is False
    assert result["error_type"] == "ComputeCodeError"
    assert "compute_code must be a non-empty string" in result["error"]
    assert status["ok"] is False
    assert status["error_type"] == "ComputeCodeError"


def test_submit_compute_run_failure_returns_only_ok_and_error(monkeypatch):
    workspace = _workspace_tmp("submit_compute_run_failure")

    def fake_run_saved_signal(config, module_name=None):
        return {
            "ok": False,
            "module_name": module_name,
            "error_type": "RuntimeError",
            "error": "boom",
            "rank_ic": -0.1,
            "rows": 10,
            "run_signal_json": str(workspace / "run.json"),
        }

    monkeypatch.setattr(factor_miner, "run_saved_signal", fake_run_saved_signal)
    tools = build_factor_miner_custom_tools(workspace)

    result = tools["submit_compute"]["tool"]("signal = self.close\n")

    assert result == {
        "ok": False,
        "error_type": "RuntimeError",
        "error": "boom",
    }


def test_answer_ready_can_be_rejected_by_validator():
    captured = []
    rejected = []
    answer = _AnswerDict(
        on_ready=captured.append,
        validator=lambda _content: "missing runnable factor",
        on_reject=rejected.append,
    )

    answer["content"] = "final"
    answer["ready"] = True

    assert captured == []
    assert rejected == ["missing runnable factor"]
    assert answer["ready"] is False
    assert answer["rejected"] is True
    assert answer["rejection_reason"] == "missing runnable factor"


def test_answer_ready_without_validator_keeps_legacy_behavior():
    captured = []
    answer = _AnswerDict(on_ready=captured.append)

    answer["content"] = "final"
    answer["ready"] = True

    assert captured == ["final"]
    assert answer["ready"] is True
    assert "rejected" not in answer


def test_factor_miner_final_answer_validator_requires_saved_ok_signal():
    workspace = _workspace_tmp("validator")
    module_name = factor_miner.GENERATED_SIGNAL_MODULE
    validate = build_factor_miner_final_answer_validator(workspace, module_name)

    assert "missing signal file" in validate("")

    signal_dir = workspace / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    (signal_dir / f"{module_name}.py").write_text("class Placeholder:\n    pass\n")
    (workspace / "run_signal_status.json").write_text(
        json.dumps(
            {
                "ok": False,
                "module_name": module_name,
                "error_type": "SyntaxError",
                "error": "invalid syntax",
            }
        ),
        encoding="utf-8",
    )
    assert "SyntaxError" in validate("")

    (workspace / "run_signal_status.json").write_text(
        json.dumps({"ok": True, "module_name": module_name}),
        encoding="utf-8",
    )
    assert validate("") == ""
