from __future__ import annotations

import argparse
from pathlib import Path
import uuid

from factor_pool import pool as factor_pool
from rlm_report_factor_iteration_glm5 import (
    build_custom_tools_and_manifest,
    load_factor_pool_memory,
    run_report_factor_iteration,
)
from factor_mining.quickbacktest_tools import QuickBacktestToolbox


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_report_factor_iteration" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _signal_code(module_name: str) -> str:
    return f'''from quickbacktest.base_types import BaseSignal


class {module_name}(BaseSignal):
    name = "{module_name}"

    def compute(self, **kwargs):
        return self.close.pct_change(20).shift(1)
'''


def _args(tmp_path: Path, report_file: Path) -> argparse.Namespace:
    return argparse.Namespace(
        report_file=str(report_file),
        initial_hypothesis="Extract one lagged momentum factor from the report.",
        module_prefix="RlmReportFactor",
        base_dir=str(tmp_path / "run"),
        rounds=1,
        pdf_max_pages=80,
        instruments="all",
        start="2020-01-02",
        end="2020-06-30",
        benchmark="SH000300",
        topk=50,
        n_drop=5,
        crypto_symbol="BTCUSDT",
        factor_pool_dir=str(tmp_path / "factor_pool"),
        output=None,
        final_report_output=None,
        backend="fake",
        model="fake-model",
        max_depth=2,
        max_iterations=40,
        max_timeout=900.0,
        verbose=False,
    )


def test_factor_pool_memory_omits_html_body():
    tmp_path = _workspace_tmp("memory")
    pool_dir = tmp_path / "factor_pool"
    code = _signal_code("ExistingFactor")
    factor_pool.save_factor("ExistingFactor", code, {"hypothesis": "old"}, pool_dir=pool_dir)
    factor_pool.save_factor_result(
        "ExistingFactor",
        {"metrics": {"daily_rank_ic_mean": 0.03}},
        report_html="<html><body>old report</body></html>",
        pool_dir=pool_dir,
    )

    memory = load_factor_pool_memory(pool_dir)

    assert len(memory) == 1
    assert memory[0]["factor_id"] == "ExistingFactor"
    assert "latest_report_html" not in memory[0]
    assert memory[0]["has_latest_report_html"] is True


def test_report_iteration_tool_set_uses_factor_mining_tools_only():
    tmp_path = _workspace_tmp("tools")
    toolbox = QuickBacktestToolbox(tmp_path, factor_pool_dir=tmp_path / "factor_pool")

    custom_tools, manifest = build_custom_tools_and_manifest(toolbox)

    assert set(manifest) == {"shared", "crypto", "qlib"}
    assert "summary" in custom_tools
    assert "materialize_signal" in custom_tools
    assert "analyze_qlib_factors" in custom_tools
    assert "simulate_qlib_portfolio" in custom_tools
    assert "query_market_summary" in custom_tools
    assert "run_strategy_backtest" in custom_tools
    assert "save_factor" in custom_tools
    assert "save_factor_result" in custom_tools
    assert "train_qlib_alpha158_augmented_model" not in custom_tools
    assert "materialize_signal" in manifest["shared"]
    assert "analyze_qlib_factors" in manifest["qlib"]
    assert "query_market_summary" in manifest["crypto"]
    assert "run_strategy_backtest" in manifest["crypto"]


def test_fake_rlm_report_iteration_saves_round_artifacts():
    tmp_path = _workspace_tmp("fake")
    report = tmp_path / "report.txt"
    report.write_text("The report proposes delayed price momentum in A-shares.", encoding="utf-8")
    args = _args(tmp_path, report)

    class FakeCompletion:
        response = ""
        metadata = {"fake": True}
        usage_summary = None

    class FakeRLM:
        def __init__(self, custom_tools):
            self.custom_tools = custom_tools

        def completion(self, context, *, root_prompt):
            module_name = context["module_name"]
            code = _signal_code(module_name)
            assert "alphaagent" not in root_prompt.lower()
            assert context["workflow_goal"] == "report_based_factor_iteration"
            assert context["universe"]["instruments"] == "all"
            assert context["data_backends"]["qlib"]["tools_manifest_key"] == "qlib"
            assert context["data_backends"]["crypto"]["symbol"] == "BTCUSDT"
            assert set(context["tool_manifest"]) == {"shared", "crypto", "qlib"}
            assert "factor_pool_memory" in context
            assert "latest_report_html" not in str(context["factor_pool_memory"])
            assert "materialize_signal" in self.custom_tools
            assert "save_factor" in self.custom_tools

            self.custom_tools["materialize_signal"]["tool"](module_name, code)
            self.custom_tools["save_factor"]["tool"](
                module_name,
                code,
                {
                    "factor_id": module_name,
                    "hypothesis": "lagged price momentum",
                    "source": "research_report",
                    "markets": ["ashare"],
                },
            )
            self.custom_tools["save_factor_result"]["tool"](
                module_name,
                {
                    "backend": "qlib",
                    "mode": "fake",
                    "metrics": {"daily_rank_ic_mean": 0.03, "icir": 0.2, "coverage": 0.9},
                },
                report_html=None,
            )
            html = "<!doctype html><html><body><h1>Round report</h1></body></html>"
            completion = FakeCompletion()
            completion.response = (
                "{"
                '"status":"completed",'
                f'"module_name":"{module_name}",'
                '"hypothesis":"lagged price momentum",'
                '"truth_label":"confirmed",'
                '"evidence_summary":{"daily_rank_ic_mean":0.03},'
                '"next_hypothesis":"test momentum with volume confirmation",'
                f'"report_html":{html!r}'
                "}"
            ).replace("'", '"')
            return completion

    def fake_factory(_args, custom_tools):
        return FakeRLM(custom_tools)

    state = run_report_factor_iteration(args, rlm_factory=fake_factory)

    base_dir = Path(args.base_dir).resolve()
    module_name = "RlmReportFactorR01"
    factor_dir = Path(args.factor_pool_dir).resolve() / module_name
    round_dir = base_dir / "round_01"

    assert state["status"] == "completed"
    assert (round_dir / "signals" / f"{module_name}.py").exists()
    assert (round_dir / "round_report.html").exists()
    assert (round_dir / "result.json").exists()
    assert (base_dir / "iteration_state.json").exists()
    assert (base_dir / "final_report.html").exists()
    assert (factor_dir / "factor.py").exists()
    assert (factor_dir / "meta.json").exists()
    assert (factor_dir / "latest_result.json").exists()
    assert not (factor_dir / "latest_report.html").exists()
