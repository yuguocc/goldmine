from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd

from quickbacktest import FactorLibrary
from src.factor_miner_parallel_reflexion.library import FactorLibraryAdmissionService
from src.factor_miner_parallel_reflexion.models import (
    CandidateResult,
    ParallelReflexionConfig,
)
from src.factor_miner_parallel_reflexion.portfolio import FactorLibraryPortfolioService


def _workspace_tmp(name: str) -> Path:
    path = (
        Path("runs")
        / "test_parallel_reflexion_marginal_gate"
        / f"{name}_{uuid.uuid4().hex}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metrics(rank_ic: float) -> dict:
    return {
        "coverage": 0.95,
        "missing_rate": 0.01,
        "daily_rank_ic_mean": rank_ic,
        "rank_icir": 0.8,
        "daily_rank_ic_count": 3,
        "rank_ic_distribution": {"mean": rank_ic, "count": 3},
        "layered_ic": {
            "layer_type": "decile",
            "deciles": {"D1": {"rows": 10}, "D10": {"rows": 10}},
        },
    }


def _factor_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_time": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "code": ["A", "B", "A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0],
            "score": scores,
        }
    )


def _write_factor_csv(path: Path, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _factor_frame(scores).to_csv(path, index=False, encoding="utf-8")


def _save_accepted_factor(
    library: FactorLibrary,
    *,
    name: str,
    rank_ic: float,
    scores: list[float],
) -> None:
    library.save_factor(
        name=name,
        signal_code="class AcceptedFactor: pass\n",
        metrics=_metrics(rank_ic),
        description="accepted test factor",
        signal_class="AcceptedFactor",
        status="accepted",
    )
    _write_factor_csv(library.root / name / "factor_data.csv", scores)


def test_marginal_contribution_gate_rejects_negative_delta(monkeypatch):
    library_root = _workspace_tmp("library")
    output_dir = _workspace_tmp("gate")
    candidate_csv = output_dir / "candidate_factor_data.csv"
    library = FactorLibrary(library_root)
    _save_accepted_factor(
        library,
        name="accepted-factor",
        rank_ic=0.04,
        scores=[1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
    )
    _write_factor_csv(candidate_csv, [6.0, 1.0, 5.0, 2.0, 4.0, 3.0])

    def fake_analyze_composite_factor(*, composite, config, output_dir):
        rank_ic = 0.05 if Path(output_dir).name == "baseline" else 0.045
        return {
            "status": "completed",
            "rank_ic": rank_ic,
            "rank_ic_name": "daily_rank_ic_mean",
            "analysis_json": str(Path(output_dir) / "analysis.json"),
        }

    monkeypatch.setattr(
        FactorLibraryPortfolioService,
        "analyze_composite_factor",
        staticmethod(fake_analyze_composite_factor),
    )

    result = FactorLibraryPortfolioService().marginal_contribution_check(
        config=ParallelReflexionConfig(
            provider_uri=library_root,
            factor_library_path=library_root,
        ),
        library_root=library_root,
        candidate_name="candidate-factor",
        candidate_metrics=_metrics(0.06),
        candidate_factor_data_csv=candidate_csv,
        output_dir=output_dir / "marginal_contribution",
    )

    assert result["passed"] is False
    assert result["verdict"] == "rejected"
    assert result["baseline_rank_ic"] == 0.05
    assert result["with_candidate_rank_ic"] == 0.045
    assert result["delta_rank_ic"] < 0.0
    assert Path(result["baseline"]["composite_factor_csv"]).exists()
    saved = json.loads(
        (output_dir / "marginal_contribution" / "marginal_contribution_gate.json")
        .read_text(encoding="utf-8")
    )
    assert saved["verdict"] == "rejected"


def test_admission_rejects_before_saving_when_marginal_gate_fails(monkeypatch):
    library_root = _workspace_tmp("admission_library")
    round_dir = _workspace_tmp("round")
    workspace = _workspace_tmp("candidate")
    library = FactorLibrary(library_root)
    _save_accepted_factor(
        library,
        name="accepted-factor",
        rank_ic=0.04,
        scores=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    signal_path = workspace / "signals" / "RlmGeneratedFactorR001C001.py"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text("class RlmGeneratedFactorR001C001: pass\n", encoding="utf-8")
    candidate_csv = workspace / "factor_data.csv"
    _write_factor_csv(candidate_csv, [6.0, 1.0, 5.0, 2.0, 4.0, 3.0])

    def fake_marginal_contribution_check(self, **kwargs):
        return {
            "available": True,
            "passed": False,
            "verdict": "rejected",
            "reason": "candidate reduces composite rank IC",
            "baseline_rank_ic": 0.05,
            "with_candidate_rank_ic": 0.045,
            "delta_rank_ic": -0.005,
            "min_delta": 0.0,
        }

    monkeypatch.setattr(
        FactorLibraryPortfolioService,
        "marginal_contribution_check",
        fake_marginal_contribution_check,
    )

    candidate = CandidateResult(
        round_number=1,
        candidate_index=1,
        module_name="RlmGeneratedFactorR001C001",
        workspace=workspace,
        label="best",
        ok=True,
        ic=0.06,
        ic_name="daily_rank_ic_mean",
        signal_path=signal_path,
        factor_data_csv=candidate_csv,
        metrics=_metrics(0.06),
    )
    admission = FactorLibraryAdmissionService().admit_candidates(
        config=ParallelReflexionConfig(
            provider_uri=library_root,
            factor_library_path=library_root,
        ),
        round_dir=round_dir,
        candidates=[candidate],
    )

    assert admission["status"] == "rejected"
    assert admission["attempts"][0]["reason"] == "marginal_contribution_gate_failed"
    assert admission["attempts"][0]["marginal_contribution_verdict"] == "rejected"
    assert not (library_root / "rlm-generated-factor-r001-c001").exists()
