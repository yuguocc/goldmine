from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd

from scripts.plot_portfolio_history import (
    load_portfolio_history,
    load_round_return_curves,
    plot_portfolio_history,
)


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_plot_portfolio_history" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_round(root: Path, round_number: int, offset: float) -> dict[str, object]:
    round_dir = root / f"round_{round_number:03d}" / "factor_library_portfolio"
    portfolio_dir = round_dir / "portfolio"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    round_summary_path = root / f"round_{round_number:03d}" / "round_summary.json"
    round_summary_path.write_text(
        json.dumps(
            {
                "round": round_number,
                "round_ic": {
                    "best_ic": 0.04 + offset,
                    "best_module": f"RlmGeneratedFactorR{round_number:03d}C001",
                    "improved": round_number == 1,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    curve_path = portfolio_dir / "qlib_return_curve.csv"
    pd.DataFrame(
        {
            "datetime": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "strategy_return": [0.0, 0.01 + offset, 0.02 + offset],
            "strategy_return_after_cost": [0.0, 0.008 + offset, 0.018 + offset],
            "benchmark_return": [0.0, 0.005, 0.01],
            "excess_return_after_cost": [0.0, 0.003 + offset, 0.008 + offset],
        }
    ).to_csv(curve_path, index=False, encoding="utf-8")
    portfolio_json = round_dir / "library_portfolio.json"
    portfolio_json.write_text(
        json.dumps(
            {
                "status": "portfolio_completed",
                "portfolio": {
                    "artifacts": {
                        "return_curve_csv": str(curve_path),
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "round": round_number,
        "status": "portfolio_completed",
        "component_count": round_number,
        "composite_rank_ic": 0.02 + offset,
        "admission_status": "accepted",
        "admitted_factors": f"factor-{round_number}",
        "portfolio_json": str(portfolio_json),
        "cumulative_return_after_cost": 0.018 + offset,
        "cumulative_benchmark_return": 0.01,
        "cumulative_excess_return_after_cost": 0.008 + offset,
    }


def test_load_round_return_curves_and_plot_outputs():
    root = _workspace_tmp("curves")
    history_path = root / "portfolio_history.json"
    history_path.write_text(
        json.dumps(
            {
                "rounds": [
                    _write_round(root, 1, 0.0),
                    _write_round(root, 2, 0.01),
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    curves = load_round_return_curves(history_path)
    assert sorted(curves["round"].unique()) == [1, 2]
    assert len(curves) == 6
    assert {"strategy_return_after_cost", "benchmark_return"}.issubset(curves.columns)

    history = load_portfolio_history(history_path)
    assert history["round_best_rank_ic"].tolist() == [0.04, 0.05]
    assert history["round_best_module"].tolist() == [
        "RlmGeneratedFactorR001C001",
        "RlmGeneratedFactorR002C001",
    ]

    result = plot_portfolio_history(history_path)
    plots = result["plots"]
    assert result["round_curve_count"] == 2
    assert Path(plots["rank_ic"]).exists()
    assert Path(plots["round_return_curves"]).exists()
    assert Path(plots["round_excess_curves"]).exists()
