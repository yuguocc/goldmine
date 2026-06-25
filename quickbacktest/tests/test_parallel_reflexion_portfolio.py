from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import uuid

import pandas as pd

import src.factor_miner_parallel_reflexion.portfolio as portfolio_module
from quickbacktest import FactorLibrary
from src.factor_miner_parallel_reflexion.models import ParallelReflexionConfig
from src.factor_miner_parallel_reflexion.portfolio import (
    FactorLibraryPortfolioService,
    PortfolioHistoryRecorder,
)


def _workspace_tmp(name: str) -> Path:
    path = (
        Path("runs")
        / "test_parallel_reflexion_portfolio"
        / f"{name}_{uuid.uuid4().hex}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_accepted_factor(
    library: FactorLibrary,
    *,
    name: str,
    rank_ic: float,
    scores: list[float],
) -> None:
    library.save_factor(
        name=name,
        signal_code="class Signal: pass\n",
        metrics={
            "daily_rank_ic_mean": rank_ic,
            "coverage": 0.95,
            "missing_rate": 0.01,
            "rank_icir": 0.5,
            "daily_rank_ic_count": 2,
            "layered_ic": {"layer_type": "decile", "deciles": []},
        },
        description="accepted test factor",
        signal_class=name,
        status="accepted",
    )
    data = pd.DataFrame(
        {
            "trade_time": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "code": ["A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 19.0],
            "score": scores,
        }
    )
    data.to_csv(library.root / name / "factor_data.csv", index=False, encoding="utf-8")


def test_composite_factor_analysis_saves_rank_ic(monkeypatch):
    root = _workspace_tmp("library")
    output_dir = _workspace_tmp("portfolio")
    library = FactorLibrary(root)
    _save_accepted_factor(library, name="factor-a", rank_ic=0.04, scores=[1, 2, 2, 3])
    _save_accepted_factor(library, name="factor-b", rank_ic=0.03, scores=[2, 1, 4, 3])

    service = FactorLibraryPortfolioService()
    composite = service.build_composite_factor(
        library_root=root,
        output_dir=output_dir,
    )

    assert composite["status"] == "built"
    assert Path(composite["composite_analysis_factor_csv"]).exists()
    analysis_input = pd.read_csv(composite["composite_analysis_factor_csv"])
    assert {"trade_time", "code", "score", "close"}.issubset(analysis_input.columns)

    class FakeQlibAdapter:
        @staticmethod
        def suppress_qlib_console():
            return nullcontext()

        @staticmethod
        def analyze_qlib_factors(**kwargs):
            factor_df = kwargs["factor_df"]
            assert {"score", "close"}.issubset(factor_df.columns)
            return {
                "metrics": {
                    "score": {
                        "daily_rank_ic_mean": 0.123,
                        "rank_ic_distribution": {"mean": 0.123, "count": 2},
                    }
                }
            }

    monkeypatch.setattr(
        portfolio_module,
        "_import_qlib_adapter",
        lambda: FakeQlibAdapter,
    )
    analysis = service.analyze_composite_factor(
        composite=composite,
        config=ParallelReflexionConfig(provider_uri=root),
        output_dir=output_dir,
    )

    assert analysis["status"] == "completed"
    assert analysis["rank_ic"] == 0.123
    assert analysis["rank_ic_name"] == "daily_rank_ic_mean"
    saved = json.loads(Path(analysis["analysis_json"]).read_text(encoding="utf-8"))
    assert saved["rank_ic"] == 0.123


def test_portfolio_history_records_composite_rank_ic():
    output_dir = _workspace_tmp("history")
    recorder = PortfolioHistoryRecorder()

    result = recorder.record_round(
        output_dir=output_dir,
        round_summary={
            "round": 1,
            "round_ic": {
                "best_ic": 0.08,
                "best_module": "RlmGeneratedFactorR001C001",
                "improved": True,
            },
            "factor_library_admission": {"status": "accepted"},
            "factor_library_portfolio": {
                "status": "portfolio_completed",
                "component_count": 2,
                "score_rows": 4,
                "composite_rank_ic": 0.123,
                "composite_rank_ic_name": "daily_rank_ic_mean",
                "composite_analysis_json": "analysis.json",
                "composite_analysis": {"status": "completed"},
                "portfolio": {"cumulative_return_after_cost": 0.05},
            },
        },
    )

    history = json.loads(
        Path(result["portfolio_history_json"]).read_text(encoding="utf-8")
    )
    row = history["rounds"][0]
    assert row["composite_rank_ic"] == 0.123
    assert row["composite_rank_ic_name"] == "daily_rank_ic_mean"
    assert row["round_best_rank_ic"] == 0.08
    assert row["round_best_module"] == "RlmGeneratedFactorR001C001"
    assert row["round_best_improved"] is True
    assert row["composite_analysis_status"] == "completed"

    csv_text = Path(result["portfolio_history_csv"]).read_text(encoding="utf-8")
    assert "composite_rank_ic" in csv_text
    assert "round_best_rank_ic" in csv_text
    assert "0.123" in csv_text


def test_oos_period_uses_one_year_after_in_sample_end():
    period = FactorLibraryPortfolioService.oos_period(
        ParallelReflexionConfig(
            end="2024-12-31",
            oos_start=None,
            oos_end=None,
        )
    )

    assert period == {
        "in_sample_end": "2024-12-31",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "warmup_start": "2024-01-01",
        "warmup_days": 366,
        "source": {
            "start": "default",
            "end": "default",
            "warmup_start": "default",
        },
    }


def test_oos_period_uses_config_default_dates():
    period = FactorLibraryPortfolioService.oos_period(ParallelReflexionConfig())

    assert period == {
        "in_sample_end": "2024-12-31",
        "start": "2025-01-01",
        "end": "2026-01-31",
        "warmup_start": "2024-01-01",
        "warmup_days": 366,
        "source": {
            "start": "configured",
            "end": "configured",
            "warmup_start": "default",
        },
    }


def test_oos_period_uses_configured_dates():
    period = FactorLibraryPortfolioService.oos_period(
        ParallelReflexionConfig(
            end="2024-12-31",
            oos_start="2025-07-01",
            oos_end="2026-06-30",
            oos_warmup_start="2025-01-01",
        )
    )

    assert period == {
        "in_sample_end": "2024-12-31",
        "start": "2025-07-01",
        "end": "2026-06-30",
        "warmup_start": "2025-01-01",
        "warmup_days": 181,
        "source": {
            "start": "configured",
            "end": "configured",
            "warmup_start": "configured",
        },
    }


def test_final_oos_recomputes_components_and_runs_portfolio(monkeypatch):
    root = _workspace_tmp("oos_library")
    output_dir = _workspace_tmp("oos_output")
    library = FactorLibrary(root)
    library.save_factor(
        name="factor-a",
        signal_code="class FactorA:\n    pass\n",
        metrics={
            "daily_rank_ic_mean": 0.04,
            "coverage": 0.95,
            "missing_rate": 0.01,
        },
        description="accepted test factor",
        signal_class="FactorA",
        status="accepted",
    )
    library.save_factor(
        name="factor-b",
        signal_code="class FactorB:\n    pass\n",
        metrics={
            "daily_rank_ic_mean": 0.03,
            "coverage": 0.95,
            "missing_rate": 0.01,
        },
        description="accepted test factor",
        signal_class="FactorB",
        status="accepted",
    )

    compute_calls = []
    analyze_calls = []
    portfolio_calls = []

    class FakeQlibAdapter:
        @staticmethod
        def suppress_qlib_console():
            return nullcontext()

        @staticmethod
        def compute_qlib_factor_dataframe(**kwargs):
            compute_calls.append(kwargs)
            module = kwargs["signal_modules"][0]
            base_scores = [1.0, 2.0, 3.0, 4.0]
            if module == "FactorB":
                base_scores = [4.0, 3.0, 2.0, 1.0]
            return pd.DataFrame(
                {
                    "trade_time": [
                        "2024-12-31",
                        "2025-01-01",
                        "2025-01-01",
                        "2026-01-31",
                    ],
                    "code": ["A", "A", "B", "A"],
                    "close": [10.0, 11.0, 20.0, 12.0],
                    module: base_scores,
                    "score": base_scores,
                }
            )

        @staticmethod
        def analyze_qlib_factors(**kwargs):
            analyze_calls.append(kwargs)
            factor_df = kwargs["factor_df"]
            assert factor_df["trade_time"].min() == "2025-01-01"
            assert factor_df["trade_time"].max() == "2026-01-31"
            return {
                "metrics": {
                    "score": {
                        "daily_rank_ic_mean": 0.234,
                        "rank_ic_distribution": {"mean": 0.234, "count": 2},
                    }
                }
            }

        @staticmethod
        def factor_df_to_qlib_signal(factor_df, score_column="score"):
            return factor_df.set_index(["trade_time", "code"])[score_column]

        @staticmethod
        def simulate_qlib_portfolio(**kwargs):
            portfolio_calls.append(kwargs)
            return {
                "cumulative_return_after_cost": 0.12,
                "prediction_rows": int(len(kwargs["pred"])),
            }

    monkeypatch.setattr(
        portfolio_module,
        "_import_qlib_adapter",
        lambda: FakeQlibAdapter,
    )

    result = FactorLibraryPortfolioService().run_final_oos(
        config=ParallelReflexionConfig(
            provider_uri=root,
            factor_library_path=root,
            end="2024-12-31",
        ),
        output_dir=output_dir,
    )

    assert result["status"] == "oos_portfolio_completed"
    assert result["oos_period"]["start"] == "2025-01-01"
    assert result["oos_period"]["end"] == "2026-01-31"
    assert result["composite_rank_ic"] == 0.234
    assert len(compute_calls) == 2
    assert {call["signal_modules"][0] for call in compute_calls} == {
        "FactorA",
        "FactorB",
    }
    assert {call["start"] for call in compute_calls} == {"2024-01-01"}
    assert {call["end"] for call in compute_calls} == {"2026-01-31"}
    assert len(analyze_calls) == 1
    assert len(portfolio_calls) == 1
    assert Path(result["composite_factor_csv"]).exists()
    assert Path(result["composite_analysis_json"]).exists()
    composite_df = pd.read_csv(result["composite_factor_csv"])
    assert composite_df["trade_time"].min() == "2025-01-01"
