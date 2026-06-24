from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from factor_mining.hypothesis_ledger import HypothesisLedger, make_hypothesis_id
from factor_mining.quickbacktest_tools import QuickBacktestToolbox
from rlm_tools import RLMToolRuntime, summarize_value
from rlm_tools.alphaagent import (
    RLMNativeRoundTools,
    evaluate_similarity,
    factor_score_series,
    spearman_corr,
    truth_label_from_metrics,
)
from rlm_alphaagent_factor_research_glm5 import (
    FACTOR_DESIGN_POLICY,
    build_native_context,
    native_root_prompt,
    parse_args,
    validate_native_round_result,
)


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_alphaagent_factor_research" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _factor_frame(values):
    return pd.DataFrame(
        {
            "trade_time": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "code": ["A", "B", "A", "B"],
            "score": values,
        }
    )


def test_hypothesis_ledger_writes_run_and_global_records():
    base = _workspace_tmp("ledger")
    ledger = HypothesisLedger(base / "run", base / "global.jsonl")
    hypothesis_id = make_hypothesis_id("smoke", 1, "C01")
    record = {
        "run_id": "smoke",
        "round": 1,
        "candidate_id": "C01",
        "hypothesis_id": hypothesis_id,
        "hypothesis_text": "momentum persists",
        "factor_id": "RlmAlphaR01C01",
        "status": "evaluated",
        "truth_label": "confirmed",
    }

    stored = ledger.append(record)

    assert stored["truth_label"] == "confirmed"
    assert len(ledger.run_records()) == 1
    assert len(ledger.global_records()) == 1
    assert ledger.read(hypothesis_id)["factor_id"] == "RlmAlphaR01C01"


def test_data_corr_similarity_rejects_identical_scores():
    left = factor_score_series(_factor_frame([1.0, 2.0, 3.0, 4.0]))
    right = factor_score_series(_factor_frame([1.0, 2.0, 3.0, 4.0]))

    assert spearman_corr(left, right) == 1.0
    matrix, violations = evaluate_similarity(
        candidate_series={"C01": left},
        reference_series={"factor_pool:Existing": right},
        max_factor_corr=0.5,
    )

    assert matrix["C01"]["factor_pool:Existing"] == 1.0
    assert violations["C01"]["max_abs_corr"] == 1.0


def test_data_corr_similarity_allows_low_corr_scores():
    left = factor_score_series(_factor_frame([1.0, 2.0, 3.0, 4.0]))
    right = factor_score_series(_factor_frame([1.0, 4.0, 2.0, 3.0]))

    _, violations = evaluate_similarity(
        candidate_series={"C01": left},
        reference_series={"factor_pool:Different": right},
        max_factor_corr=0.5,
    )

    assert violations == {}


def test_truth_label_rules():
    assert (
        truth_label_from_metrics(
            {"daily_rank_ic_mean": 0.03, "icir": 0.2, "coverage": 0.8}
        )[0]
        == "confirmed"
    )
    assert (
        truth_label_from_metrics(
            {"daily_rank_ic_mean": -0.01, "icir": 0.2, "coverage": 0.8}
        )[0]
        == "rejected"
    )
    assert (
        truth_label_from_metrics(
            {"daily_rank_ic_mean": 0.01, "icir": 0.05, "coverage": 0.8}
        )[0]
        == "inconclusive"
    )


def test_hypothesis_tools_are_read_only():
    tools = QuickBacktestToolbox(_workspace_tmp("tools")).to_rlm_tools()

    assert "summary" in tools
    assert "list_hypotheses" in tools
    assert "read_hypothesis" in tools
    assert "save_hypothesis" not in tools
    assert "write_hypothesis" not in tools


def test_summary_compacts_common_python_objects():
    payload = {
        "alpha": [1, 2, 3, 4],
        "beta": {"x": 1, "y": 2},
        "gamma": "long-text",
    }
    overview = summarize_value(payload, max_items=2, max_depth=2)

    assert overview["type"] == "dict"
    assert overview["length"] == 3
    assert overview["keys_count"] == 3
    assert overview["keys"] == ["'alpha'", "'beta'"]
    assert overview["sample"]["alpha"]["type"] == "list"
    assert overview["sample"]["alpha"]["length"] == 4

    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame_overview = summarize_value(frame, max_items=1)
    assert frame_overview["type"] == "DataFrame"
    assert frame_overview["shape"] == [2, 2]
    assert frame_overview["columns"] == ["a"]


def test_summary_tool_manifest_is_formatting_helper():
    toolbox = QuickBacktestToolbox(_workspace_tmp("summary_tool"))
    runtime = RLMToolRuntime.from_quickbacktest_toolbox(toolbox, profile="researcher")
    manifest = runtime.to_manifest()

    assert "summary" in runtime.to_custom_tools()
    assert manifest["summary"]["category"] == "formatting"
    assert manifest["summary"]["permission"] == "format"


def test_prompt_context_allows_free_factor_design():
    base = _workspace_tmp("policy").resolve()
    ledger = HypothesisLedger(base, base / "global_hypotheses.jsonl")
    context = build_native_context(
        original_input={"initial_hypothesis": "find alpha"},
        round_number=1,
        current_base_hypothesis="find alpha",
        previous_round_result=None,
        ledger=ledger,
        factor_pool_full=[],
        tool_manifest={},
        module_prefix="RlmAlpha",
        candidates_per_round=3,
        instruments="all",
        start="2020-01-02",
        end="2020-06-30",
        max_factor_corr=0.5,
        round_dir=base / "round_01",
    )
    prompt = native_root_prompt(1, 3, "all")

    assert context["factor_design_policy"] == FACTOR_DESIGN_POLICY
    assert context["universe"]["instruments"] == "all"
    assert context["universe"]["instrument_kind"] == "all_market"
    assert "ML/DL" in prompt
    assert "nonlinear" in prompt
    assert "current universe" in prompt
    assert "full A-share market" in prompt
    assert "summary(value) as a preview" in prompt
    assert "then use print(...)" in prompt
    assert any("summary(value) to preview" in item for item in context["required_internal_protocol"])
    assert "Do not generate multiple candidates by default" in prompt
    assert "BaseSignal" in context["factor_design_policy"]["required_output_contract"]


def test_default_candidate_count_is_single_iterative_flow(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rlm_alphaagent_factor_research_glm5.py"])
    args = parse_args()

    assert args.candidates_per_round == 1


def test_native_round_tools_keep_artifacts_in_round_workspace():
    base = _workspace_tmp("native").resolve()
    pool_dir = base / "factor_pool"
    ledger = HypothesisLedger(base, base / "global_hypotheses.jsonl")
    tools = RLMNativeRoundTools(
        base_dir=base,
        round_number=1,
        run_id="smoke",
        ledger=ledger,
        instruments=["SH600000"],
        start="2020-01-02",
        end="2020-01-10",
        max_factor_corr=0.5,
        factor_pool_dir=pool_dir,
    )
    code = """from quickbacktest.base_types import BaseSignal

class RlmAlphaR01C01(BaseSignal):
    name = "RlmAlphaR01C01"

    def compute(self, **kwargs):
        return self.close.pct_change().shift(1)
"""

    materialized = tools.materialize_candidate_signal("RlmAlphaR01C01", code, "round_01")
    assert materialized["ok"] is True
    assert Path(materialized["signal_path"]).is_relative_to(base / "round_01")
    assert Path(materialized["model_dir"]).is_relative_to(base / "round_01")

    recorded = tools.record_hypothesis_result(
        {
            "candidate_id": "C01",
            "hypothesis_text": "lagged momentum works",
            "factor_id": "RlmAlphaR01C01",
            "status": "evaluated",
            "truth_label": "confirmed",
        }
    )
    assert recorded["ok"] is True
    assert len(ledger.run_records()) == 1

    report = tools.write_round_report("<!doctype html><html><body>ok</body></html>")
    assert report["ok"] is True
    assert Path(report["report_path"]).is_relative_to(base / "round_01")

    saved = tools.save_selected_factor(
        "RlmAlphaR01C01",
        {"hypothesis": "lagged momentum works"},
        {"truth_label": "confirmed", "metrics": {"daily_rank_ic_mean": 0.03}},
    )
    assert saved["ok"] is True
    factor_dir = pool_dir / "RlmAlphaR01C01"
    assert (factor_dir / "factor.py").exists()
    assert (factor_dir / "meta.json").exists()
    assert (factor_dir / "latest_result.json").exists()
    assert not (factor_dir / "latest_report.html").exists()


def test_native_host_insurance_accepts_tool_completed_payload():
    base = _workspace_tmp("insurance").resolve()
    pool_dir = base / "factor_pool"
    ledger = HypothesisLedger(base, base / "global_hypotheses.jsonl")
    tools = RLMNativeRoundTools(
        base_dir=base,
        round_number=1,
        run_id="smoke",
        ledger=ledger,
        instruments=["SH600000"],
        start="2020-01-02",
        end="2020-01-10",
        max_factor_corr=0.5,
        factor_pool_dir=pool_dir,
    )
    code = """from quickbacktest.base_types import BaseSignal

class RlmAlphaR01C01(BaseSignal):
    name = "RlmAlphaR01C01"

    def compute(self, **kwargs):
        return self.close.pct_change().shift(1)
"""
    assert tools.materialize_candidate_signal("RlmAlphaR01C01", code)["ok"] is True
    ledger_result = tools.record_hypothesis_result(
        {
            "candidate_id": "C01",
            "hypothesis_text": "lagged momentum works",
            "factor_id": "RlmAlphaR01C01",
            "status": "evaluated",
            "truth_label": "confirmed",
        }
    )
    assert ledger_result["ok"] is True
    assert tools.write_round_report("<!doctype html><html><body>ok</body></html>")["ok"] is True
    assert tools.save_selected_factor(
        "RlmAlphaR01C01",
        {"hypothesis": "lagged momentum works"},
        {"truth_label": "confirmed"},
    )["ok"] is True

    errors = validate_native_round_result(
        payload={
            "status": "completed",
            "round": 1,
            "candidates": [
                {
                    "candidate_id": "C01",
                    "module_name": "RlmAlphaR01C01",
                    "hypothesis": "lagged momentum works",
                    "truth_label": "confirmed",
                    "factor_resume": {},
                    "metrics": {},
                    "corr_check": {},
                    "ledger_id": ledger_result["ledger_id"],
                    "failure_reason": None,
                }
            ],
            "selected_module_name": "RlmAlphaR01C01",
            "selected_hypothesis_id": ledger_result["ledger_id"],
            "next_hypothesis": "try conditional momentum",
            "report_path": "round_01/report.html",
        },
        native_tools=tools,
        candidates_per_round=1,
    )

    assert errors == []
