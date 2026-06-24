from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factor_miner import LIBS_ROOT, _clean_json_value, _write_json

from .constants import FAST_SCREEN_RANK_IC_THRESHOLD, MAX_SIGNAL_SOURCE_CHARS
from .memory import RlmFactorMemoryManager
from .models import CandidateResult

def _ensure_rlm_import_path() -> None:
    libs_root = str(LIBS_ROOT)
    if libs_root not in sys.path:
        sys.path.insert(0, libs_root)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_memory(memory: Any) -> dict[str, Any]:
    return RlmFactorMemoryManager.from_payload(memory).to_dict()


def _memory_text(memory: dict[str, Any]) -> str:
    return RlmFactorMemoryManager.from_payload(memory).prompt_text()


def _memory_for_rlm(memory: dict[str, Any]) -> dict[str, Any]:
    """Return only non-debug memory that should be visible to RLM."""
    return RlmFactorMemoryManager.from_payload(memory).visible_for_rlm()


def _append_memory(
    memory: dict[str, Any],
    *,
    round_number: int,
    reflexions: list[dict[str, Any]],
    memory_size: int,
    round_ic: dict[str, Any],
    best_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    manager = RlmFactorMemoryManager.from_payload(memory, max_entries=memory_size)
    admission = {"status": "accepted" if best_signal is not None else "skipped"}
    if best_signal is not None:
        admission.update(
            {
                "factor_name": best_signal.get("factor_name"),
                "factor_dir": best_signal.get("factor_dir"),
                "candidate_module": best_signal.get("module"),
                "candidate_ic": best_signal.get("ic"),
            }
        )
    manager.update(
        round_number=round_number,
        reflexions=reflexions,
        round_ic=round_ic,
        best_signal=best_signal,
        results=[],
        admission=admission,
    )
    return manager.to_dict()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _previous_global_best_ic(memory: dict[str, Any]) -> tuple[float | None, int | None]:
    if isinstance(memory.get("state"), dict):
        return RlmFactorMemoryManager.from_payload(memory).previous_global_best_ic()
    best_ic = _finite_float(memory.get("global_best_ic"))
    best_round = memory.get("global_best_round")
    if best_ic is not None:
        try:
            return best_ic, int(best_round) if best_round is not None else None
        except (TypeError, ValueError):
            return best_ic, None

    history = memory.get("ic_history", [])
    if not isinstance(history, list):
        return None, None
    best_record: dict[str, Any] | None = None
    best_value: float | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        value = _finite_float(item.get("best_ic"))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_record = item
    if best_value is None:
        return None, None
    try:
        round_number = int(best_record.get("round")) if best_record else None
    except (TypeError, ValueError):
        round_number = None
    return best_value, round_number


def _round_ic_record(
    memory: dict[str, Any],
    *,
    round_number: int,
    results: list[CandidateResult],
) -> dict[str, Any]:
    return RoundEvaluator(ParallelReflexionConfig()).round_ic_record(
        memory,
        round_number=round_number,
        results=results,
    )


def _best_signal_history(memory: dict[str, Any]) -> list[dict[str, Any]]:
    state = memory.get("state") if isinstance(memory, dict) else None
    best_signals = (
        state.get("best_signals", [])
        if isinstance(state, dict)
        else memory.get("best_signals", [])
    )
    if not isinstance(best_signals, list):
        return []
    return [
        item
        for item in best_signals
        if isinstance(item, dict) and str(item.get("signal_source", "") or "").strip()
        and item.get("admission_status") == "accepted"
    ]


def _global_best_signal_from_history(
    best_signals: list[dict[str, Any]],
) -> dict[str, Any] | None:
    valid = [
        item
        for item in best_signals
        if isinstance(item, dict)
        and item.get("admission_status") == "accepted"
        and _finite_float(item.get("ic")) is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: _finite_float(item.get("ic")) or float("-inf"))


def _latest_accepted_memory_signal_source(memory: dict[str, Any]) -> str:
    if isinstance(memory.get("state"), dict):
        return RlmFactorMemoryManager.from_payload(memory).latest_accepted_signal_source()
    history = _best_signal_history(memory)
    if not history:
        return ""
    best = _global_best_signal_from_history(history) or history[-1]
    return str(best.get("signal_source", "") or "").strip()


def _trim_signal_source(source: str) -> str:
    if len(source) <= MAX_SIGNAL_SOURCE_CHARS:
        return source
    return (
        source[:MAX_SIGNAL_SOURCE_CHARS]
        + "\n\n# ... truncated: signal source exceeded context budget ..."
    )


def _round_best_signal_record(
    *,
    round_number: int,
    best: CandidateResult | None,
    admission: dict[str, Any],
) -> dict[str, Any] | None:
    if admission.get("status") != "accepted":
        return None
    if best is None or best.signal_path is None or not best.signal_path.exists():
        return None
    return {
        "source": "accepted_library_candidate",
        "admission_status": "accepted",
        "round": round_number,
        "module": best.module_name,
        "research_branch": best.research_branch,
        "factor_name": admission.get("factor_name"),
        "factor_dir": admission.get("factor_dir"),
        "ic": best.ic,
        "ic_name": best.ic_name,
        "signal_path": str(best.signal_path),
        "signal_source": _trim_signal_source(
            best.signal_path.read_text(encoding="utf-8")
        ),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

def _extract_ic_with_name(metrics: dict[str, Any] | None) -> tuple[float | None, str]:
    if not isinstance(metrics, dict):
        return None, ""
    ic = _finite_float(metrics.get("daily_rank_ic_mean"))
    if ic is not None:
        return ic, "daily_rank_ic_mean"
    ic_distribution = metrics.get("rank_ic_distribution")
    if isinstance(ic_distribution, dict):
        ic = _finite_float(ic_distribution.get("mean"))
        if ic is not None:
            return ic, "rank_ic_distribution.mean"
    ic = _finite_float(metrics.get("rank_ic"))
    if ic is not None:
        return ic, "rank_ic"
    ic = _finite_float(metrics.get("daily_ic_mean"))
    if ic is not None:
        return ic, "daily_ic_mean"
    ic_distribution = metrics.get("ic_distribution")
    if isinstance(ic_distribution, dict):
        ic = _finite_float(ic_distribution.get("mean"))
        if ic is not None:
            return ic, "ic_distribution.mean"
    return None, ""


def _candidate_ic(
    *,
    metrics: dict[str, Any] | None,
) -> tuple[float | None, str]:
    ic, ic_name = _extract_ic_with_name(metrics)
    if ic is not None:
        return ic, ic_name
    return None, ""


def _is_positive_ic_result(result: CandidateResult) -> bool:
    """A candidate is usable only after finite rank IC passes fast screen."""
    ic = _finite_float(result.ic)
    return bool(
        result.ok
        and ic is not None
        and ic > FAST_SCREEN_RANK_IC_THRESHOLD
    )


def _has_factor_data(result: CandidateResult) -> bool:
    """Correlation checks require the materialized factor scores on disk."""
    return result.factor_data_csv is not None and result.factor_data_csv.exists()


def _select_best_result(results: list[CandidateResult]) -> CandidateResult | None:
    return next((result for result in results if result.label == "best"), None)


def _result_payload(result: CandidateResult) -> dict[str, Any]:
    return _clean_json_value(asdict(result))


def _write_candidate_result(result: CandidateResult) -> None:
    path = result.workspace / "candidate_result.json"
    result.candidate_result_json = path
    _write_json(path, _result_payload(result))


def _factor_library_name(module_name: str) -> str:
    import re

    words = re.sub(r"(?<!^)(?=[A-Z])", "-", module_name).replace("_", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", words).strip("-").lower()


def _factor_metric_ic(metrics: dict[str, Any] | None) -> float | None:
    ic, _name = _extract_ic_with_name(metrics)
    return ic

def _series_from_factor_csv(path: Path) -> Any:
    import pandas as pd

    df = pd.read_csv(path)
    required = {"trade_time", "code"}
    if not required.issubset(df.columns):
        return None
    if "score" in df.columns:
        value_col = "score"
    else:
        excluded = {
            "trade_time",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "amount",
        }
        numeric_cols = [
            col
            for col in df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not numeric_cols:
            return None
        value_col = numeric_cols[-1]
    frame = df[["trade_time", "code", value_col]].copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"])
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    return frame.set_index(["trade_time", "code"])[value_col].sort_index()


def _spearman_corr_from_csv(left_path: Path, right_path: Path) -> float | None:
    import pandas as pd

    left = _series_from_factor_csv(left_path)
    right = _series_from_factor_csv(right_path)
    if left is None or right is None:
        return None
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(joined) < 2:
        return None
    corr = joined["left"].corr(joined["right"], method="spearman")
    return _finite_float(corr)


def _validate_config(config: Any) -> None:
    if config.candidates < 1:
        raise ValueError("candidates must be >= 1")
    if config.rounds < 1:
        raise ValueError("rounds must be >= 1")
    if config.max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if config.memory_size < 1:
        raise ValueError("memory_size must be >= 1")
