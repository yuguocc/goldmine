from __future__ import annotations

import argparse
from pathlib import Path
import uuid

import pytest

from factor_mining.quickbacktest_tools import QuickBacktestToolbox
from factor_mining.report_loader import load_research_report
from factor_pool import pool as factor_pool
from rlm_pdf_factor_backtest_report_glm5 import (
    build_context,
    build_custom_tools_and_manifest,
    load_factor_pool_memory,
    run_pdf_factor_report,
)
from rlm_tools.universe import build_qlib_universe_context


SIGNAL_CODE = """from quickbacktest.base_types import BaseSignal

class RlmPdfFactor(BaseSignal):
    name = "RlmPdfFactor"

    def compute(self, **kwargs):
        return self.close.pct_change().shift(1)
"""


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_pdf_factor_report" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _args(tmp_path: Path, report_file: Path) -> argparse.Namespace:
    return argparse.Namespace(
        report_file=str(report_file),
        module_name="RlmPdfFactor",
        base_dir=str(tmp_path / "run"),
        output=None,
        report_output=None,
        report_title="PDF factor report",
        pdf_max_pages=80,
        instruments="all",
        start="2020-01-02",
        end="2020-06-30",
        benchmark="SH000300",
        topk=50,
        n_drop=5,
        factor_pool_dir=str(tmp_path / "factor_pool"),
        backend="fake",
        model="fake-model",
        max_depth=2,
        max_iterations=40,
        max_timeout=900.0,
        verbose=False,
    )


def test_report_loader_loads_text_file():
    tmp_path = _workspace_tmp("loader")
    report = tmp_path / "report.txt"
    report.write_text("Momentum improves after volume confirmation.", encoding="utf-8")

    content, source = load_research_report(report)

    assert "Momentum improves" in content
    assert source["type"] == "text_file"
    assert source["path"] == str(report.resolve())


def test_materialize_signal_and_factor_research_tools():
    tmp_path = _workspace_tmp("materialize")
    toolbox = QuickBacktestToolbox(tmp_path)
    result = toolbox.materialize_signal("RlmPdfFactor", SIGNAL_CODE)

    assert result["ok"] is True
    assert (tmp_path / "signals" / "RlmPdfFactor.py").exists()

    tools = toolbox.to_factor_research_tools()
    assert "materialize_signal" in tools
    assert "save_factor" in tools
    assert "save_factor_result" in tools
    assert "analyze_qlib_factors" in tools
    assert "simulate_qlib_portfolio" in tools
    assert "materialize_strategy" not in tools
    assert "materialize_evaluator" not in tools

    bad_code = SIGNAL_CODE.replace("class RlmPdfFactor", "class OtherFactor")
    with pytest.raises(ValueError, match="class RlmPdfFactor"):
        toolbox.materialize_signal("RlmPdfFactor", bad_code)


def test_factor_pool_memory_omits_report_html():
    tmp_path = _workspace_tmp("memory")
    pool_dir = tmp_path / "factor_pool"
    factor_pool.save_factor(
        "ExistingFactor",
        SIGNAL_CODE.replace("RlmPdfFactor", "ExistingFactor"),
        {"hypothesis": "old"},
        pool_dir=pool_dir,
    )
    factor_pool.save_factor_result(
        "ExistingFactor",
        {"metrics": {"rank_ic": 0.1}},
        report_html="<html>old report</html>",
        pool_dir=pool_dir,
    )

    memory = load_factor_pool_memory(pool_dir=pool_dir)

    assert memory[0]["factor_id"] == "ExistingFactor"
    assert "latest_report_html" not in memory[0]
    assert memory[0]["has_latest_report_html"] is True


def test_pdf_factor_context_contains_memory_without_html():
    universe = build_qlib_universe_context(
        instruments="all",
        provider_uri=".qlib/qlib_data/cn_data",
        start="2020-01-02",
        end="2020-06-30",
    )
    context = build_context(
        module_name="RlmPdfFactor",
        report_title="Report",
        report_content="raw report",
        report_source={"type": "text_file"},
        factor_pool_memory=[{"factor_id": "Existing", "has_latest_report_html": True}],
        universe=universe,
        tool_manifest={"materialize_signal": {"description": "x"}},
        benchmark="SH000300",
        topk=50,
        n_drop=5,
    )

    assert context["workflow_goal"] == "pdf_to_factor_backtest_revised_report"
    assert context["factor_pool_memory"][0]["factor_id"] == "Existing"
    assert "latest_report_html" not in context["factor_pool_memory"][0]
    assert context["answer_contract"] == {"type": "html_only", "output": "html"}


def test_fake_rlm_pdf_factor_report_saves_artifacts():
    tmp_path = _workspace_tmp("fake")
    report = tmp_path / "report.txt"
    report.write_text("A lagged momentum factor should be tested.", encoding="utf-8")
    args = _args(tmp_path, report)

    class FakeCompletion:
        response = "<!doctype html><html><body><h1>Revised report</h1></body></html>"
        metadata = {"fake": True}
        usage_summary = None

    class FakeRLM:
        def __init__(self, custom_tools):
            self.custom_tools = custom_tools

        def completion(self, context, *, root_prompt):
            assert "rlm_tools.alphaagent" not in root_prompt
            assert context["workflow_goal"] == "pdf_to_factor_backtest_revised_report"
            assert "factor_pool_memory" in context
            assert all("latest_report_html" not in item for item in context["factor_pool_memory"])
            assert "materialize_signal" in self.custom_tools
            assert "train_qlib_alpha158_augmented_model" not in self.custom_tools
            self.custom_tools["materialize_signal"]["tool"]("RlmPdfFactor", SIGNAL_CODE)
            self.custom_tools["save_factor"]["tool"](
                "RlmPdfFactor",
                SIGNAL_CODE,
                {"hypothesis": "lagged momentum", "source": "pdf"},
            )
            self.custom_tools["save_factor_result"]["tool"](
                "RlmPdfFactor",
                {"backend": "qlib", "metrics": {"daily_rank_ic_mean": 0.03}},
                report_html=None,
            )
            return FakeCompletion()

    def fake_factory(_args, custom_tools):
        return FakeRLM(custom_tools)

    result = run_pdf_factor_report(args, rlm_factory=fake_factory)

    signal_path = Path(result["signal_path"])
    report_path = Path(result["report_path"])
    output_path = Path(result["result_path"])
    factor_dir = Path(args.factor_pool_dir) / "RlmPdfFactor"

    assert signal_path.exists()
    assert report_path.exists()
    assert output_path.exists()
    assert (factor_dir / "factor.py").exists()
    assert (factor_dir / "meta.json").exists()
    assert (factor_dir / "latest_result.json").exists()
    assert not (factor_dir / "latest_report.html").exists()
    assert Path(result["factor_pool_paths"]["factor"]) == (factor_dir / "factor.py").resolve()


def test_pdf_factor_tools_are_limited_to_factor_mining_set():
    tmp_path = _workspace_tmp("tools")
    toolbox = QuickBacktestToolbox(tmp_path, factor_pool_dir=tmp_path / "factor_pool")
    custom_tools, manifest = build_custom_tools_and_manifest(toolbox=toolbox)

    assert set(custom_tools) == set(manifest)
    assert "materialize_signal" in custom_tools
    assert "save_factor" in custom_tools
    assert "save_factor_result" in custom_tools
    assert "train_qlib_alpha158_augmented_model" not in custom_tools
    assert "materialize_strategy" not in custom_tools
    assert "materialize_evaluator" not in custom_tools
    assert "run_strategy_backtest" not in custom_tools
