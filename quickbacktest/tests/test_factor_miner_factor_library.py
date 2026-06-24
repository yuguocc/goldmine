from __future__ import annotations

from pathlib import Path
import uuid

from factor_miner import FactorMinerCaseConfig, save_reviewed_factor


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_factor_miner_factor_library" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_factor_miner_saves_successful_run_to_factor_library():
    workspace = _workspace_tmp("workspace")
    factor_root = _workspace_tmp("factors")
    signal_path = workspace / "signals" / "RlmGeneratedFactor.py"
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text("from quickbacktest.base_types import BaseSignal\n", encoding="utf-8")
    config = FactorMinerCaseConfig(
        workspace=workspace,
        run_portfolio=False,
        train_alpha158=False,
    )
    analysis = {
        "metrics": {
            "RlmGeneratedFactor": {
                "coverage": 0.95,
                "missing_rate": 0.05,
                "daily_rank_ic_mean": 0.03,
                "rank_icir": 0.4,
                "daily_rank_ic_count": 20,
                "ic_distribution": {"count": 20},
                "rank_ic_distribution": {"count": 20},
                "layered_ic": {
                    "layer_type": "decile",
                    "deciles": {"Decile1": {"rows": 100}},
                },
            }
        }
    }

    result = save_reviewed_factor(
        config=config,
        module_name="RlmGeneratedFactor",
        signal_path=signal_path,
        analysis=analysis,
        rlm_summary="done",
        library_root=factor_root,
    )

    assert result["factor_name"] == "rlm-generated-factor"
    assert result["status"] == "accepted"
    assert Path(result["factor_card"]).exists()
    assert Path(result["factor_metrics"]).exists()
    assert Path(result["factor_review"]).exists()
