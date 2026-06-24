from __future__ import annotations

from pathlib import Path
import uuid

from quickbacktest import FactorLibrary, build_rlm_factor_tools, review_factor_metrics


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_factor_library" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metrics(rank_ic: float = 0.03, rank_icir: float = 0.4) -> dict:
    return {
        "coverage": 0.95,
        "missing_rate": 0.05,
        "daily_rank_ic_mean": rank_ic,
        "rank_icir": rank_icir,
        "daily_rank_ic_count": 20,
        "ic_distribution": {"count": 20, "mean": 0.02},
        "rank_ic_distribution": {"count": 20, "mean": rank_ic},
        "layered_ic": {
            "layer_type": "decile",
            "deciles": {"Decile1": {"rows": 100}},
        },
    }


def test_factor_library_save_read_review_and_status():
    root = _workspace_tmp("save")
    library = FactorLibrary(root)

    record = library.save_factor(
        name="rlm-test-factor",
        signal_code="class RlmTestFactor: pass\n",
        metrics=_metrics(),
        description="RLM test factor.",
        rlm_summary="Generated from a momentum hypothesis.",
        signal_class="RlmTestFactor",
        universe="csi500",
        horizon=1,
        factor_shift=1,
        tags=["rlm", "factor"],
    )

    assert record.name == "rlm-test-factor"
    assert (root / "rlm-test-factor" / "FACTOR.md").exists()
    assert (root / "rlm-test-factor" / "signal.py").exists()
    assert (root / "rlm-test-factor" / "metrics.json").exists()
    factor_card = (root / "rlm-test-factor" / "FACTOR.md").read_text(encoding="utf-8")
    assert "daily_rank_ic_mean" not in factor_card
    assert "rank_icir" not in factor_card
    assert "`metrics.json`" in factor_card

    review = review_factor_metrics(_metrics())
    assert review["verdict"] == "accepted"
    library.save_review("rlm-test-factor", review)
    library.update_status("rlm-test-factor", "accepted")

    saved = library.read_factor("rlm-test-factor")
    assert saved["metadata"]["status"] == "accepted"
    assert saved["metrics"]["daily_rank_ic_mean"] == 0.03
    assert saved["review"]["verdict"] == "accepted"
    assert library.list_factors()[0]["name"] == "rlm-test-factor"


def test_review_factor_metrics_rejects_weak_factor():
    review = review_factor_metrics(_metrics(rank_ic=-0.01, rank_icir=-0.1))

    assert review["verdict"] == "rejected"
    assert review["checks"]["rank_ic_ok"] is False
    assert review["checks"]["icir_ok"] is False


def test_factor_library_exposes_read_only_rlm_tools():
    root = _workspace_tmp("tools")
    library = FactorLibrary(root)
    library.save_factor(
        name="rlm-test-factor",
        signal_code="class RlmTestFactor: pass\n",
        metrics=_metrics(),
        description="RLM test factor.",
        signal_class="RlmTestFactor",
        universe="csi500",
    )

    tools = build_rlm_factor_tools(root)

    assert set(tools) == {"list_factors", "read_factor"}
    assert tools["list_factors"]["tool"]()[0]["name"] == "rlm-test-factor"
    assert tools["read_factor"]["tool"]("rlm-test-factor")["metrics"]["coverage"] == 0.95
