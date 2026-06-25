from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SEARCH_METADATA_LIST_KEYS = {
    "keywords",
    "operators",
    "data_fields",
    "failure_types",
    "tools_or_functions",
    "data_contracts",
}

_REFLEXION_HEADING_TITLES = {
    "search metadata",
    "overall guidance",
    "best hypothesis",
    "weak hypotheses",
    "hypothesis improvements",
    "failed function calls",
    "error frequency",
    "error patterns",
    "recommended directions (p_succ)",
    "forbidden directions (p_fail)",
    "robust patterns",
    "next round fixes",
}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return _split_csvish(value)
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _as_metadata_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    metadata: dict[str, Any] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()
        if not clean_key:
            continue
        if clean_key in _SEARCH_METADATA_LIST_KEYS:
            metadata[clean_key] = _as_str_list(raw)
        elif isinstance(raw, list):
            metadata[clean_key] = _as_str_list(raw)
        elif raw is None:
            metadata[clean_key] = ""
        elif isinstance(raw, (str, int, float, bool)):
            metadata[clean_key] = raw
        else:
            metadata[clean_key] = str(raw)
    return metadata


def _split_csvish(value: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[,;\n]+", str(value or "")):
        text = raw.strip(" -`'\t\"")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _safe_token(value: str, *, fallback: str = "pattern") -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "").strip().lower())
    token = token.strip("_")
    if not token:
        return fallback
    return token[:64].rstrip("_") or fallback


def _parse_search_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    in_section = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        heading = stripped.strip("# \t").lower()
        if stripped.startswith("## "):
            if heading == "search metadata":
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        if not stripped or ":" not in stripped:
            continue
        entry = stripped.lstrip("-* ").strip()
        if ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        clean_key = key.strip().lower().replace(" ", "_")
        if not clean_key:
            continue
        clean_value = value.strip()
        if clean_key in _SEARCH_METADATA_LIST_KEYS:
            metadata[clean_key] = _split_csvish(clean_value)
        else:
            metadata[clean_key] = clean_value
    return metadata


def _parse_reflexion_metadata(text: str) -> dict[str, Any]:
    metadata = _parse_search_metadata(text)
    metadata = _merge_metadata(metadata, _parse_economic_direction_sections(text))
    error_frequency = _parse_error_frequency(text)
    next_hypotheses = _parse_next_round_hypotheses(text)
    if next_hypotheses:
        metadata["next_round_hypotheses"] = next_hypotheses
    if error_frequency:
        frequency_types = [item["error_type"] for item in error_frequency]
        existing_types = [
            item
            for item in _as_str_list(metadata.get("failure_types"))
            if _useful_failure_type(item)
        ]
        metadata["failure_types"] = _merge_unique(frequency_types, existing_types)
        metadata["error_frequency"] = [
            _format_error_frequency_record(item) for item in error_frequency
        ]
    return metadata


def _parse_economic_direction_sections(text: str) -> dict[str, Any]:
    sections = {
        "recommended directions (p_succ)": "recommended",
        "forbidden directions (p_fail)": "forbidden",
    }
    records: dict[str, list[dict[str, str]]] = {"recommended": [], "forbidden": []}
    section: str | None = None
    current: dict[str, str] | None = None

    def clean_key(value: str) -> str:
        return value.strip().lower().replace(" ", "_").replace("-", "_")

    structured_keys = {
        "hypothesis",
        "selection_reason",
        "hypothesis_construction",
        "construction_rule",
        "mutation_guidance",
        "duplicate_avoidance",
        "avoid_duplicate_note",
        "example",
        "rejection_reason",
        "reason",
        "forbidden_construction",
        "allowed_exception",
    }

    def store_current() -> None:
        nonlocal current
        if section and current and any(str(v).strip() for v in current.values()):
            records[section].append(dict(current))
        current = None

    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            store_current()
            heading = stripped.strip("# \t").lower()
            section = sections.get(heading)
            continue
        if section is None or not stripped:
            continue
        is_bullet = stripped.startswith(("- ", "* "))
        entry = stripped[2:].strip() if is_bullet else stripped
        if not entry:
            continue
        if ":" in entry:
            key, value = entry.split(":", 1)
            key = clean_key(key)
            value = value.strip()
            if is_bullet and key not in structured_keys:
                store_current()
                current = {"hypothesis": entry}
                continue
            if key == "hypothesis":
                store_current()
                current = {"hypothesis": value}
            else:
                if current is None:
                    current = {}
                current[key] = value
            continue
        if is_bullet:
            store_current()
            current = {"hypothesis": entry}
        elif current is not None:
            current["notes"] = " ".join(
                part for part in [current.get("notes", ""), entry] if part
            )

    store_current()

    metadata: dict[str, Any] = {}
    recommended = records["recommended"]
    forbidden = records["forbidden"]
    if recommended:
        metadata["recommended_hypotheses"] = _record_values(
            recommended,
            ["hypothesis"],
        )
        metadata["recommended_selection"] = _record_values(
            recommended,
            ["selection_reason"],
        )
        metadata["recommended_construction"] = _record_values(
            recommended,
            ["hypothesis_construction", "construction_rule"],
        )
        metadata["mutation_guidance"] = _record_values(
            recommended,
            ["mutation_guidance"],
        )
        metadata["duplicate_avoidance"] = _record_values(
            recommended,
            ["duplicate_avoidance", "avoid_duplicate_note"],
        )
        metadata["recommended_examples"] = _record_values(recommended, ["example"])
    if forbidden:
        metadata["forbidden_hypotheses"] = _record_values(forbidden, ["hypothesis"])
        metadata["forbidden_reasons"] = _record_values(
            forbidden,
            ["rejection_reason", "reason"],
        )
        metadata["forbidden_construction"] = _record_values(
            forbidden,
            ["forbidden_construction"],
        )
        metadata["allowed_exceptions"] = _record_values(
            forbidden,
            ["allowed_exception"],
        )
        metadata["forbidden_examples"] = _record_values(forbidden, ["example"])
    return metadata


def _record_values(records: list[dict[str, str]], keys: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in keys:
            value = str(record.get(key, "") or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            values.append(value)
            break
    return values


def _parse_next_round_hypotheses(text: str) -> list[str]:
    hypotheses: list[str] = []
    seen: set[str] = set()
    in_section = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        heading = stripped.strip("# \t").lower()
        if stripped.startswith("## "):
            if heading == "next round hypotheses":
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        entry = stripped.lstrip("-* ").strip()
        if not entry:
            continue
        entry = re.sub(r"^(?:H|hypothesis)\s*\d+\s*[:.)-]\s*", "", entry, flags=re.I)
        entry = re.sub(r"^\d+\s*[:.)-]\s*", "", entry).strip()
        if not entry:
            continue
        key = entry.lower()
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(entry)
    return hypotheses


def _parse_error_frequency(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    in_section = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        heading = stripped.strip("# \t").lower()
        if stripped.startswith("## "):
            if heading == "error frequency":
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        entry = stripped.lstrip("-* ").strip()
        if not entry:
            continue
        record = _parse_error_frequency_line(entry)
        if record is not None:
            records.append(record)
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("error_type", "") or "").lower()
        if key not in deduped:
            deduped[key] = record
            continue
        deduped[key]["count"] = int(deduped[key].get("count", 0) or 0) + int(
            record.get("count", 0) or 0
        )
        if not deduped[key].get("cause") and record.get("cause"):
            deduped[key]["cause"] = record["cause"]
        if not deduped[key].get("evidence") and record.get("evidence"):
            deduped[key]["evidence"] = record["evidence"]
        if not deduped[key].get("fix") and record.get("fix"):
            deduped[key]["fix"] = record["fix"]
    return sorted(
        deduped.values(),
        key=lambda item: (-int(item.get("count", 0) or 0), str(item.get("error_type"))),
    )


def _parse_error_frequency_line(entry: str) -> dict[str, Any] | None:
    error_type = ""
    rest = entry
    explicit = re.search(r"\berror_type\s*[=:]\s*([^,;]+)", entry, flags=re.I)
    if explicit:
        error_type = explicit.group(1).strip()
    else:
        prefix = re.match(r"([^:]+):\s*(.*)$", entry)
        if prefix:
            error_type = prefix.group(1).strip()
            rest = prefix.group(2).strip()
    if not _useful_failure_type(error_type):
        return None
    count_match = re.search(r"\bcount\s*[=:]\s*(\d+)", entry, flags=re.I)
    frequency_match = re.search(r"\bfrequency\s*[=:]\s*([0-9]*\.?[0-9]+)", entry, flags=re.I)
    cause_match = re.search(
        r"\bcause\s*[=:]\s*([^;]+?)(?=\s*;\s*(?:evidence|fix)\s*[=:]|$)",
        entry,
        flags=re.I,
    )
    evidence_match = re.search(
        r"\bevidence\s*[=:]\s*([^;]+?)(?=\s*;\s*(?:fix|cause)\s*[=:]|$)",
        entry,
        flags=re.I,
    )
    fix_match = re.search(
        r"\bfix\s*[=:]\s*([^;]+?)(?=\s*;\s*(?:cause|evidence)\s*[=:]|$)",
        entry,
        flags=re.I,
    )
    count = int(count_match.group(1)) if count_match else 1
    frequency = float(frequency_match.group(1)) if frequency_match else None
    cause = cause_match.group(1).strip(" ,;") if cause_match else ""
    evidence = evidence_match.group(1).strip(" ,;") if evidence_match else ""
    fix = fix_match.group(1).strip(" ,;") if fix_match else ""
    return {
        "error_type": error_type,
        "count": count,
        "frequency": frequency,
        "cause": cause or rest,
        "evidence": evidence,
        "fix": fix,
    }


def _useful_failure_type(value: str) -> bool:
    text = str(value or "").strip().strip("<>")
    if not text:
        return False
    return text.lower() not in {
        "error_type",
        "repeated failure classes",
        "none",
        "n/a",
        "na",
        "unknown",
    }


def _format_error_frequency_record(record: dict[str, Any]) -> str:
    error_type = str(record.get("error_type", "") or "").strip()
    count = record.get("count")
    frequency = record.get("frequency")
    cause = str(record.get("cause", "") or "").strip()
    evidence = str(record.get("evidence", "") or "").strip()
    fix = str(record.get("fix", "") or "").strip()
    parts = [f"{error_type}: count={count}"]
    if frequency is not None:
        parts.append(f"frequency={frequency:g}")
    if cause:
        parts.append(f"cause={cause}")
    if evidence:
        parts.append(f"evidence={evidence}")
    if fix:
        parts.append(f"fix={fix}")
    return ", ".join(parts)


def _first_metadata_value(metadata: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            text = ", ".join(str(item) for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
        if text:
            return text
    return ""


def _merge_unique(existing: list[str], new: list[str]) -> list[str]:
    values = list(existing)
    seen = {item.lower() for item in values}
    for item in new:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


def _merge_metadata(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = _as_metadata_dict(existing)
    for key, value in _as_metadata_dict(new).items():
        if isinstance(value, list):
            current = merged.get(key)
            current_values = current if isinstance(current, list) else _as_str_list(current)
            merged[key] = _merge_unique(current_values, value)
        elif value:
            merged[key] = value
    return merged


def _compact_text(text: str, *, limit: int = 1200) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 36].rstrip() + " ... [truncated]"


def _first_meaningful_line(text: str, *, fallback: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip(" -#\t")
        if stripped and stripped.lower() not in _REFLEXION_HEADING_TITLES:
            return stripped
    return fallback


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error_type_key(error_type: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(error_type or "").strip().lower()).strip("_")


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    values: list[int] = []
    seen: set[int] = set()
    for item in value:
        parsed = _optional_int(item)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        values.append(parsed)
    return sorted(values)


def _as_failure_type_stats(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    stats: dict[str, dict[str, Any]] = {}
    for key, raw in value.items():
        payload = raw if isinstance(raw, dict) else {}
        error_type = str(payload.get("error_type", "") or key).strip()
        if not _useful_failure_type(error_type):
            continue
        stat_key = _error_type_key(error_type)
        if not stat_key:
            continue
        count = _optional_int(payload.get("count"))
        stats[stat_key] = {
            "error_type": error_type,
            "count": max(0, count or 0),
            "rounds": _as_int_list(payload.get("rounds")),
            "latest_round": _optional_int(payload.get("latest_round")),
            "latest_cause": str(payload.get("latest_cause", "") or ""),
            "latest_evidence": str(payload.get("latest_evidence", "") or ""),
            "latest_fix": str(payload.get("latest_fix", "") or ""),
        }
    return _top_failure_type_stats(stats, limit=None)


def _top_failure_type_stats(
    stats: dict[str, dict[str, Any]],
    *,
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        stats.items(),
        key=lambda item: (
            -int(item[1].get("count", 0) or 0),
            -int(item[1].get("latest_round", 0) or 0),
            str(item[1].get("error_type", "")),
        ),
    )
    if limit is not None:
        ordered = ordered[: max(0, int(limit))]
    return {key: value for key, value in ordered}


@dataclass
class RlmMiningState:
    """State S: compact loop state and debug-only round history."""

    latest_round: int = 0
    library_size: int = 0
    recent_admissions: list[dict[str, Any]] = field(default_factory=list)
    recent_rejections: list[dict[str, Any]] = field(default_factory=list)
    domain_saturation: dict[str, float] = field(default_factory=dict)
    failure_type_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    admission_log: list[dict[str, Any]] = field(default_factory=list)
    ic_history: list[dict[str, Any]] = field(default_factory=list)
    best_signals: list[dict[str, Any]] = field(default_factory=list)
    latest_best_ic: float | None = None
    global_best_ic: float | None = None
    global_best_round: int | None = None
    latest_round_improved: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RlmMiningState":
        return cls(
            latest_round=int(payload.get("latest_round", payload.get("round", 0)) or 0),
            library_size=int(payload.get("library_size", 0) or 0),
            recent_admissions=_as_dict_list(payload.get("recent_admissions")),
            recent_rejections=_as_dict_list(payload.get("recent_rejections")),
            domain_saturation=dict(payload.get("domain_saturation") or {}),
            failure_type_stats=_as_failure_type_stats(
                payload.get("failure_type_stats")
            ),
            admission_log=_as_dict_list(payload.get("admission_log")),
            ic_history=_as_dict_list(payload.get("ic_history")),
            best_signals=_as_dict_list(payload.get("best_signals")),
            latest_best_ic=_finite_float(payload.get("latest_best_ic")),
            global_best_ic=_finite_float(payload.get("global_best_ic")),
            global_best_round=_optional_int(payload.get("global_best_round")),
            latest_round_improved=bool(payload.get("latest_round_improved", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RlmSuccessPattern:
    """P_succ: reusable economic hypothesis or admitted-signal pattern."""

    name: str
    description: str
    template: str = ""
    success_rate: str = "Medium"
    example_factors: list[str] = field(default_factory=list)
    occurrence_count: int = 1
    confidence: float = 0.5
    source_round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RlmSuccessPattern":
        return cls(
            name=str(payload.get("name", "") or "unnamed_success_pattern"),
            description=str(payload.get("description", "") or ""),
            template=str(payload.get("template", "") or ""),
            success_rate=str(payload.get("success_rate", "Medium") or "Medium"),
            example_factors=[str(item) for item in payload.get("example_factors", [])],
            occurrence_count=int(payload.get("occurrence_count", 1) or 1),
            confidence=float(payload.get("confidence", 0.5) or 0.5),
            source_round=int(payload.get("source_round", 0) or 0),
            metadata=_as_metadata_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RlmForbiddenDirection:
    """P_fail: repeated implementation failure or redundant factor direction."""

    name: str
    description: str
    reason: str = ""
    examples: list[str] = field(default_factory=list)
    correlated_factors: list[str] = field(default_factory=list)
    typical_correlation: float = 0.0
    occurrence_count: int = 1
    source_round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RlmForbiddenDirection":
        return cls(
            name=str(payload.get("name", "") or "unnamed_forbidden_direction"),
            description=str(payload.get("description", "") or ""),
            reason=str(payload.get("reason", "") or ""),
            examples=[str(item) for item in payload.get("examples", [])],
            correlated_factors=[
                str(item) for item in payload.get("correlated_factors", [])
            ],
            typical_correlation=float(payload.get("typical_correlation", 0.0) or 0.0),
            occurrence_count=int(payload.get("occurrence_count", 1) or 1),
            source_round=int(payload.get("source_round", 0) or 0),
            metadata=_as_metadata_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RlmStrategicInsight:
    """I: high-level lesson used as concise future generation guidance."""

    insight: str
    evidence: str = ""
    batch_source: int = 0
    phase: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RlmStrategicInsight":
        return cls(
            insight=str(payload.get("insight", "") or ""),
            evidence=str(payload.get("evidence", "") or ""),
            batch_source=int(payload.get("batch_source", 0) or 0),
            phase=str(payload.get("phase", "") or ""),
            metadata=_as_metadata_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RlmExperienceMemory:
    """Complete RLM factor memory M = {S, P_succ, P_fail, I}."""

    state: RlmMiningState = field(default_factory=RlmMiningState)
    success_patterns: list[RlmSuccessPattern] = field(default_factory=list)
    forbidden_directions: list[RlmForbiddenDirection] = field(default_factory=list)
    insights: list[RlmStrategicInsight] = field(default_factory=list)
    version: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RlmExperienceMemory":
        if "state" in payload:
            return cls(
                state=RlmMiningState.from_dict(dict(payload.get("state") or {})),
                success_patterns=[
                    RlmSuccessPattern.from_dict(item)
                    for item in _as_dict_list(payload.get("success_patterns"))
                ],
                forbidden_directions=[
                    RlmForbiddenDirection.from_dict(item)
                    for item in _as_dict_list(payload.get("forbidden_directions"))
                ],
                insights=[
                    RlmStrategicInsight.from_dict(item)
                    for item in _as_dict_list(payload.get("insights"))
                ],
                version=int(payload.get("version", 4) or 4),
            )
        return cls.from_legacy_dict(payload)

    @classmethod
    def from_legacy_dict(cls, payload: dict[str, Any]) -> "RlmExperienceMemory":
        state = RlmMiningState.from_dict(
            {
                "latest_round": payload.get("latest_round", payload.get("round", 0)),
                "ic_history": payload.get("ic_history", []),
                "best_signals": payload.get("best_signals", []),
                "latest_best_ic": payload.get("latest_best_ic"),
                "global_best_ic": payload.get("global_best_ic"),
                "global_best_round": payload.get("global_best_round"),
                "latest_round_improved": payload.get("latest_round_improved", False),
            }
        )
        hypothesis_entries = _as_dict_list(payload.get("hypothesis_memory"))
        implementation_entries = _as_dict_list(payload.get("implementation_memory"))
        legacy_buffer = _as_dict_list(payload.get("buffer"))
        if legacy_buffer and not (hypothesis_entries or implementation_entries):
            for item in legacy_buffer:
                if str(item.get("phase", "")) == "implementation_errors":
                    implementation_entries.append(item)
                else:
                    hypothesis_entries.append(item)

        legacy_reflexion = str(payload.get("reflexion", "") or "").strip()
        if legacy_reflexion and not hypothesis_entries:
            hypothesis_entries.append(
                {
                    "round": state.latest_round,
                    "phase": "economic_hypothesis",
                    "reflexion": legacy_reflexion,
                }
            )

        success_patterns = [
            RlmSuccessPattern(
                name=f"economic_hypothesis_round_{int(item.get('round', 0) or 0)}",
                description=_compact_text(str(item.get("reflexion", "") or "")),
                template="legacy_reflexion",
                success_rate="Medium",
                example_factors=[],
                occurrence_count=1,
                confidence=0.5,
                source_round=int(item.get("round", 0) or 0),
            )
            for item in hypothesis_entries
            if str(item.get("reflexion", "") or "").strip()
        ]
        forbidden = [
            RlmForbiddenDirection(
                name=f"implementation_errors_round_{int(item.get('round', 0) or 0)}",
                description=_compact_text(str(item.get("reflexion", "") or "")),
                reason="legacy_implementation_reflexion",
                examples=[],
                occurrence_count=1,
                source_round=int(item.get("round", 0) or 0),
            )
            for item in implementation_entries
            if str(item.get("reflexion", "") or "").strip()
        ]
        insights = [
            RlmStrategicInsight(
                insight=_first_meaningful_line(
                    str(item.get("reflexion", "") or ""),
                    fallback="Legacy economic reflexion",
                ),
                evidence=_compact_text(str(item.get("reflexion", "") or "")),
                batch_source=int(item.get("round", 0) or 0),
                phase=str(item.get("phase", "economic_hypothesis") or ""),
            )
            for item in hypothesis_entries + implementation_entries
            if str(item.get("reflexion", "") or "").strip()
        ]
        return cls(
            state=state,
            success_patterns=success_patterns,
            forbidden_directions=forbidden,
            insights=insights,
            version=4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "state": self.state.to_dict(),
            "success_patterns": [item.to_dict() for item in self.success_patterns],
            "forbidden_directions": [
                item.to_dict() for item in self.forbidden_directions
            ],
            "insights": [item.to_dict() for item in self.insights],
        }


class RlmFactorMemoryManager:
    """OOP facade for retrieval, formation, evolution, and persistence."""

    def __init__(self, memory: RlmExperienceMemory | None = None, *, max_entries: int = 5):
        self.memory = memory or RlmExperienceMemory()
        self.max_entries = max(1, int(max_entries))

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        max_entries: int = 5,
    ) -> "RlmFactorMemoryManager":
        if isinstance(payload, dict):
            memory = RlmExperienceMemory.from_dict(payload)
        else:
            memory = RlmExperienceMemory()
        return cls(memory, max_entries=max_entries)

    @classmethod
    def load(cls, path: Path, *, max_entries: int = 5) -> "RlmFactorMemoryManager":
        if not path.exists():
            return cls(max_entries=max_entries)
        return cls.from_payload(
            json.loads(path.read_text(encoding="utf-8")),
            max_entries=max_entries,
        )

    @property
    def latest_round(self) -> int:
        return self.memory.state.latest_round

    def to_dict(self) -> dict[str, Any]:
        payload = self.memory.to_dict()
        payload["memory_policy"] = self.schema()
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def schema(self) -> dict[str, Any]:
        return {
            "policy": "rlm_factor_memory",
            "versioning": "monotonic_round",
            "state_schema": {
                "latest_round": "int",
                "library_size": "int",
                "recent_admissions": "list[dict]",
                "recent_rejections": "list[dict]",
                "domain_saturation": "dict[str,float]",
                "failure_type_stats": "dict[str,dict] implementation_error_counter",
                "admission_log": "list[dict]",
                "ic_history": "list[dict] debug_only",
                "best_signals": "list[dict] accepted_only",
            },
            "memory_formula": "M={S,P_succ,P_fail,I}",
            "retrieval_signal": {
                "recommended_directions": "selected P_succ entries",
                "forbidden_directions": "selected P_fail entries",
                "insights": "selected I entries",
                "library_state": "compact state diagnostics",
                "prompt_text": "Factorminer-style prompt section",
            },
            "search_metadata": {
                "source": "economic reflection direction sections and optional ## Search Metadata sections",
                "stored_on": ["success_patterns", "forbidden_directions", "insights"],
                "fields": [
                    "keywords",
                    "signal_family",
                    "operators",
                    "data_fields",
                    "recommended_hypotheses",
                    "recommended_selection",
                    "recommended_construction",
                    "mutation_guidance",
                    "duplicate_avoidance",
                    "forbidden_hypotheses",
                    "forbidden_reasons",
                    "forbidden_construction",
                    "allowed_exceptions",
                    "failure_types",
                    "tools_or_functions",
                    "data_contracts",
                    "next_action",
                ],
            },
            "limits": {"max_entries": self.max_entries},
        }

    def visible_for_rlm(self) -> dict[str, Any]:
        return self.retrieve_memory_signal()

    def retrieve_memory_signal(self) -> dict[str, Any]:
        """Factorminer-style R(M, L): select compact priors for generation."""
        state = self.memory.state
        library_state = self._library_state_for_retrieval()
        recommended = self._select_recommended_directions()
        forbidden = self._select_forbidden_directions()
        insights = self._select_insights()
        return {
            "version": self.memory.version,
            "structure": "M={state, success_patterns, forbidden_directions, insights}",
            "state": {
                "latest_round": state.latest_round,
                "library_size": state.library_size,
                "recent_admissions": [
                    self._public_state_record(item)
                    for item in state.recent_admissions[-self.max_entries :]
                ],
                "recent_rejections": [
                    self._public_state_record(item)
                    for item in state.recent_rejections[-self.max_entries :]
                ],
                "domain_saturation": state.domain_saturation,
                "failure_type_stats": list(
                    _top_failure_type_stats(
                        state.failure_type_stats,
                        limit=self.max_entries,
                    ).values()
                ),
            },
            "recommended_directions": recommended,
            "forbidden_directions": forbidden,
            "insights": insights,
            "library_state": library_state,
            "prompt_text": self._format_retrieval_prompt(
                library_state=library_state,
                recommended=recommended,
                forbidden=forbidden,
                insights=insights,
            ),
        }

    def _library_state_for_retrieval(self) -> dict[str, Any]:
        state = self.memory.state
        recent_logs = state.admission_log[-self.max_entries :]
        admission_rate = 0.0
        if recent_logs:
            admission_rate = sum(
                float(item.get("admission_rate", 0.0) or 0.0)
                for item in recent_logs
            ) / len(recent_logs)
        saturated = {
            domain: value
            for domain, value in state.domain_saturation.items()
            if float(value or 0.0) >= 0.5
        }
        return {
            "library_size": state.library_size,
            "latest_round": state.latest_round,
            "recent_admission_rate": round(admission_rate, 3),
            "saturated_domains": saturated,
            "recent_admissions_count": len(state.recent_admissions),
            "recent_rejections_count": len(state.recent_rejections),
        }

    def _select_recommended_directions(self) -> list[dict[str, Any]]:
        scored = [
            (item, self._success_relevance_score(item))
            for item in self.memory.success_patterns
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item.to_dict() for item, _ in scored[: self.max_entries]]

    def _select_forbidden_directions(self) -> list[dict[str, Any]]:
        scored = [
            (item, self._forbidden_relevance_score(item))
            for item in self.memory.forbidden_directions
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item.to_dict() for item, _ in scored[: self.max_entries]]

    def _select_insights(self) -> list[dict[str, Any]]:
        insights = sorted(
            self.memory.insights,
            key=lambda item: (item.batch_source, item.insight),
            reverse=True,
        )
        return [item.to_dict() for item in insights[: self.max_entries]]

    def _success_relevance_score(self, pattern: RlmSuccessPattern) -> float:
        score = {"High": 2.0, "Medium": 1.0, "Low": 0.5}.get(
            pattern.success_rate,
            1.0,
        )
        if pattern.occurrence_count > 0:
            score *= 1.0 + math.log1p(pattern.occurrence_count)
        signal_family = str(pattern.metadata.get("signal_family", "") or pattern.name)
        saturation = float(self.memory.state.domain_saturation.get(signal_family, 0.0) or 0.0)
        if saturation >= 0.7:
            score *= 0.2
        elif saturation >= 0.5:
            score *= 0.6
        return score

    def _forbidden_relevance_score(self, direction: RlmForbiddenDirection) -> float:
        score = 1.0 + float(direction.typical_correlation or 0.0)
        if direction.occurrence_count > 0:
            score *= 1.0 + math.log1p(direction.occurrence_count)
        text = " ".join(
            [
                direction.name,
                direction.description,
                direction.reason,
                " ".join(direction.examples),
            ]
        ).lower()
        for rejection in self.memory.state.recent_rejections[-self.max_entries :]:
            reason = str(rejection.get("reason", "") or "").lower()
            error_type = str(rejection.get("error_type", "") or "").lower()
            if reason and any(word in text for word in reason.split() if len(word) > 4):
                score *= 1.5
                break
            if error_type and error_type in text:
                score *= 1.5
                break
        return score

    def _format_retrieval_prompt(
        self,
        *,
        library_state: dict[str, Any],
        recommended: list[dict[str, Any]],
        forbidden: list[dict[str, Any]],
        insights: list[dict[str, Any]],
    ) -> str:
        sections: list[str] = ["## Factorminer Memory Priors"]

        sections.append(
            "\n".join(
                [
                    "### Current Library State",
                    f"- library_size: {library_state.get('library_size', 0)}",
                    "- recent_admission_rate: "
                    f"{float(library_state.get('recent_admission_rate', 0.0) or 0.0):.1%}",
                    "- recent_admissions_count: "
                    f"{library_state.get('recent_admissions_count', 0)}",
                    "- recent_rejections_count: "
                    f"{library_state.get('recent_rejections_count', 0)}",
                ]
            )
        )

        saturated = library_state.get("saturated_domains")
        if isinstance(saturated, dict) and saturated:
            lines = ["### Saturated Domains To Avoid"]
            for domain, value in saturated.items():
                lines.append(f"- {domain}: {float(value or 0.0):.0%}")
            sections.append("\n".join(lines))

        sections.append(
            self._recommended_direction_section(recommended)
            or "### Recommended Directions (P_succ)\n- None yet; start from simple, economically grounded price/volume factors."
        )
        sections.append(
            self._forbidden_direction_section(forbidden)
            or "### Forbidden Directions (P_fail)\n- None yet; still avoid duplicate formulas, weak IC, alignment bugs, NaN/inf output, and excessive complexity."
        )
        insight_section = self._strategic_insight_section(insights)
        if insight_section:
            sections.append(insight_section)
        return "\n\n".join(sections)

    def _recommended_direction_section(self, items: list[dict[str, Any]]) -> str:
        lines = ["### Recommended Directions (P_succ)"]
        for index, item in enumerate(items, start=1):
            metadata = _as_metadata_dict(item.get("metadata"))
            parts = [
                self._compact_pattern_value(item.get("name"), limit=80),
                self._compact_pattern_value(item.get("description")),
                self._compact_pattern_value(item.get("template"), limit=120),
                self._compact_pattern_value(metadata.get("recommended_hypotheses")),
                self._compact_pattern_value(metadata.get("recommended_selection")),
                self._compact_pattern_value(metadata.get("recommended_construction")),
                self._compact_pattern_value(metadata.get("mutation_guidance")),
                self._compact_pattern_value(metadata.get("duplicate_avoidance")),
                self._compact_pattern_value(metadata.get("recommended_examples")),
                self._compact_pattern_value(metadata.get("positive_pattern")),
                self._compact_pattern_value(metadata.get("next_action")),
            ]
            text = " | ".join(part for part in parts if part)
            if text:
                lines.append(f"- P_succ_{index}: {text}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _forbidden_direction_section(self, items: list[dict[str, Any]]) -> str:
        lines = ["### Forbidden Directions (P_fail)"]
        for index, item in enumerate(items, start=1):
            metadata = _as_metadata_dict(item.get("metadata"))
            parts = [
                self._compact_pattern_value(item.get("name"), limit=80),
                self._compact_pattern_value(item.get("reason")),
                self._compact_pattern_value(item.get("description")),
                self._compact_pattern_value(metadata.get("forbidden_hypotheses")),
                self._compact_pattern_value(metadata.get("forbidden_reasons")),
                self._compact_pattern_value(metadata.get("forbidden_construction")),
                self._compact_pattern_value(metadata.get("allowed_exceptions")),
                self._compact_pattern_value(metadata.get("forbidden_examples")),
                self._compact_pattern_value(metadata.get("weak_pattern")),
                self._compact_pattern_value(metadata.get("avoid_pattern")),
                self._compact_pattern_value(metadata.get("next_action")),
            ]
            text = " | ".join(part for part in parts if part)
            if text:
                lines.append(f"- P_fail_{index}: {text}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _strategic_insight_section(self, items: list[dict[str, Any]]) -> str:
        lines = ["### Strategic Insights (I)"]
        for index, item in enumerate(items, start=1):
            parts = [
                self._compact_pattern_value(item.get("insight")),
                self._compact_pattern_value(item.get("evidence"), limit=260),
            ]
            text = " | ".join(part for part in parts if part)
            if text:
                lines.append(f"- I_{index}: {text}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def visible_for_reflexion(self) -> dict[str, Any]:
        """Return only previous-round memory for reflection agents."""
        state = self.memory.state
        previous_round = int(state.latest_round or 0)
        visible = self.visible_for_rlm()
        return {
            "version": self.memory.version,
            "structure": "M={state, success_patterns, forbidden_directions, insights}",
            "reflection_scope": "previous_round_only",
            "previous_round": previous_round if previous_round > 0 else None,
            "state": {
                "latest_round": previous_round,
                "library_size": state.library_size,
                "latest_round_improved": state.latest_round_improved,
                "failure_type_stats": list(
                    _top_failure_type_stats(
                        state.failure_type_stats,
                        limit=self.max_entries,
                    ).values()
                ),
            },
            "previous_round_admissions": [
                self._public_state_record(item)
                for item in state.recent_admissions
                if _optional_int(item.get("round")) == previous_round
            ],
            "previous_round_rejections": [
                self._public_state_record(item)
                for item in state.recent_rejections
                if _optional_int(item.get("round")) == previous_round
            ],
            "recommended_directions": [
                item
                for item in visible["recommended_directions"]
                if _optional_int(item.get("source_round")) == previous_round
            ],
            "forbidden_directions": [
                item
                for item in visible["forbidden_directions"]
                if _optional_int(item.get("source_round")) == previous_round
            ],
            "insights": [
                item
                for item in visible["insights"]
                if _optional_int(item.get("batch_source")) == previous_round
            ],
        }

    def prompt_text(self) -> str:
        return str(self.retrieve_memory_signal().get("prompt_text", ""))

    def seed_success_patterns(self, hypotheses: list[str], *, source_round: int = 0) -> None:
        """Seed P_succ from externally supplied or bootstrapped hypotheses."""
        patterns: list[RlmSuccessPattern] = []
        seen: set[str] = set()
        for index, item in enumerate(hypotheses, start=1):
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            patterns.append(
                RlmSuccessPattern(
                    name=f"seed_hypothesis_{index}",
                    description=text,
                    template=text,
                    success_rate="Medium",
                    example_factors=[],
                    occurrence_count=0,
                    confidence=0.5,
                    source_round=source_round,
                    metadata={
                        "signal_family": "seed_hypothesis",
                        "positive_pattern": text,
                        "next_action": "test this hypothesis with a runnable price/volume compute body",
                    },
                )
            )
        if patterns:
            self.memory.success_patterns = self._merge_success_patterns(
                self.memory.success_patterns,
                patterns,
            )[-self.max_entries :]

    def reflexion_prompt_text(self) -> str:
        return self.pattern_text(previous_round_only=True)

    def pattern_text(self, *, previous_round_only: bool) -> str:
        visible = self.visible_for_reflexion() if previous_round_only else self.visible_for_rlm()
        title = (
            "## Previous Round Reflection Patterns"
            if previous_round_only
            else "## Historical Reflection Patterns"
        )
        sections = [title]
        sections.extend(
            self._success_pattern_sections(visible.get("recommended_directions", []))
        )
        sections.extend(
            self._failure_pattern_sections(visible.get("forbidden_directions", []))
        )
        sections.extend(self._reflection_sections(visible.get("insights", [])))
        sections.extend(self._insight_pattern_sections(visible.get("insights", [])))
        if len(sections) == 1:
            return f"{title}\n- No prior reflection patterns."
        return "\n\n".join(sections)

    @staticmethod
    def _compact_pattern_value(value: Any, *, limit: int = 220) -> str:
        values = value if isinstance(value, list) else [value]
        cleaned = [str(item or "").strip() for item in values if str(item or "").strip()]
        return _compact_text("; ".join(cleaned), limit=limit)

    def _append_compact_bullet(
        self,
        lines: list[str],
        *,
        prefix: str,
        value: Any,
        limit: int = 220,
    ) -> None:
        text = self._compact_pattern_value(value, limit=limit)
        if text:
            lines.append(f"- {prefix}: {text}")

    def _success_pattern_sections(self, items: list[dict[str, Any]]) -> list[str]:
        lines = ["### Success Patterns"]
        for index, item in enumerate(items, start=1):
            metadata = _as_metadata_dict(item.get("metadata"))
            parts = [
                self._compact_pattern_value(item.get("example_factors"), limit=80),
                self._compact_pattern_value(metadata.get("signal_family"), limit=80),
                self._compact_pattern_value(metadata.get("positive_pattern")),
                self._compact_pattern_value(metadata.get("next_action")),
            ]
            text = " | ".join(part for part in parts if part)
            if text:
                lines.append(f"- success_{index}: {text}")
        return ["\n".join(lines)] if len(lines) > 1 else []

    def _failure_pattern_sections(self, items: list[dict[str, Any]]) -> list[str]:
        lines = ["### Failed Patterns"]
        for index, item in enumerate(items, start=1):
            metadata = _as_metadata_dict(item.get("metadata"))
            parts = [
                self._compact_pattern_value(metadata.get("failure_types"), limit=100),
                self._compact_pattern_value(metadata.get("avoid_pattern")),
                self._compact_pattern_value(metadata.get("next_action")),
                self._compact_pattern_value(item.get("examples"), limit=100),
            ]
            text = " | ".join(part for part in parts if part)
            if text:
                lines.append(f"- failed_{index}: {text}")
        return ["\n".join(lines)] if len(lines) > 1 else []

    def _failure_type_stats_sections(self, items: list[dict[str, Any]]) -> list[str]:
        if not items:
            return []
        lines = ["### high_frequency_failure_types"]
        for item in items:
            error_type = str(item.get("error_type", "") or "").strip()
            if not error_type:
                continue
            parts = [f"{error_type}: count={int(item.get('count', 0) or 0)}"]
            rounds = item.get("rounds")
            if isinstance(rounds, list) and rounds:
                parts.append(
                    "rounds=" + ",".join(str(round_number) for round_number in rounds)
                )
            latest_round = _optional_int(item.get("latest_round"))
            if latest_round is not None:
                parts.append(f"latest_round={latest_round}")
            latest_cause = str(item.get("latest_cause", "") or "").strip()
            if latest_cause:
                parts.append(f"cause={latest_cause}")
            latest_fix = str(item.get("latest_fix", "") or "").strip()
            if latest_fix:
                parts.append(f"fix={latest_fix}")
            lines.append("- " + "; ".join(parts))
        return ["\n".join(lines)] if len(lines) > 1 else []

    def _insight_pattern_sections(self, items: list[dict[str, Any]]) -> list[str]:
        lines = ["### Insights"]
        for index, item in enumerate(items, start=1):
            metadata = _as_metadata_dict(item.get("metadata"))
            parts = [
                self._compact_pattern_value(item.get("insight")),
                self._compact_pattern_value(metadata.get("next_action")),
            ]
            text = " | ".join(part for part in parts if part)
            if text:
                lines.append(f"- insight_{index}: {text}")
        return ["\n".join(lines)] if len(lines) > 1 else []

    def _reflection_sections(self, items: list[dict[str, Any]]) -> list[str]:
        lines = ["### Reflections"]
        for index, item in enumerate(items, start=1):
            evidence = self._compact_pattern_value(item.get("evidence"), limit=300)
            if not evidence:
                continue
            phase = str(item.get("phase", "") or "reflection")
            source = item.get("batch_source")
            label = f"{phase}"
            if source:
                label += f"_round_{source}"
            lines.append(f"- reflection_{index}: {label}: {evidence}")
        return ["\n".join(lines)] if len(lines) > 1 else []

    @staticmethod
    def _public_state_record(record: dict[str, Any]) -> dict[str, Any]:
        hidden = {"ic", "candidate_ic", "best_ic", "previous_best_ic"}
        return {key: value for key, value in record.items() if key not in hidden}

    def previous_global_best_ic(self) -> tuple[float | None, int | None]:
        state = self.memory.state
        if state.global_best_ic is not None:
            return state.global_best_ic, state.global_best_round
        best_record: dict[str, Any] | None = None
        best_value: float | None = None
        for item in state.ic_history:
            value = _finite_float(item.get("best_ic"))
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_record = item
        if best_value is None:
            return None, None
        try:
            return best_value, int(best_record.get("round")) if best_record else None
        except (TypeError, ValueError):
            return best_value, None

    def latest_accepted_signal_source(self) -> str:
        valid = [
            item
            for item in self.memory.state.best_signals
            if item.get("admission_status") == "accepted"
            and str(item.get("signal_source", "") or "").strip()
        ]
        if not valid:
            return ""
        best = max(
            valid,
            key=lambda item: _finite_float(item.get("ic")) or float("-inf"),
        )
        return str(best.get("signal_source", "") or "").strip()

    def update(
        self,
        *,
        round_number: int,
        reflexions: list[dict[str, Any]],
        round_ic: dict[str, Any],
        best_signal: dict[str, Any] | None,
        results: list[Any],
        admission: dict[str, Any],
        ) -> None:
        self.memory.version = 4
        formed = self._form(
            round_number=round_number,
            reflexions=reflexions,
            results=results,
            admission=admission,
        )
        self._evolve(formed)
        state = self.memory.state
        state.latest_round = round_number
        state.latest_best_ic = _finite_float(round_ic.get("best_ic"))
        state.latest_round_improved = bool(round_ic.get("improved"))
        state.ic_history.append(dict(round_ic))
        if best_signal is not None:
            state.best_signals.append(dict(best_signal))
        state.best_signals = state.best_signals[-self.max_entries :]

        accepted_best = self._global_best_signal()
        if accepted_best is not None:
            state.global_best_ic = _finite_float(accepted_best.get("ic"))
            try:
                state.global_best_round = int(accepted_best.get("round"))
            except (TypeError, ValueError):
                state.global_best_round = None
        else:
            state.global_best_ic = None
            state.global_best_round = None

    def _form(
        self,
        *,
        round_number: int,
        reflexions: list[dict[str, Any]],
        results: list[Any],
        admission: dict[str, Any],
    ) -> RlmExperienceMemory:
        accepted = admission.get("status") == "accepted"
        candidate_count = len(results)
        accepted_candidates = _as_dict_list(admission.get("accepted_candidates"))
        accepted_count = _optional_int(admission.get("accepted_count"))
        if accepted_count is None:
            if accepted_candidates:
                accepted_count = len(accepted_candidates)
            else:
                accepted_count = 1 if accepted else 0
        if not accepted:
            accepted_count = 0
        archived_count = sum(
            1
            for item in _as_dict_list(admission.get("archived_replacements"))
            if item.get("status") == "archived"
        )
        library_delta = accepted_count - archived_count
        formed_state = RlmMiningState(
            latest_round=round_number,
            library_size=max(0, self.memory.state.library_size + library_delta),
            recent_admissions=self._admission_records(round_number, admission),
            recent_rejections=self._rejection_records(round_number, results, admission),
            domain_saturation=dict(self.memory.state.domain_saturation),
            failure_type_stats=self._round_failure_type_stats(
                round_number,
                reflexions,
            ),
            admission_log=[
                {
                    "round": round_number,
                    "admitted": accepted_count,
                    "rejected": max(0, candidate_count - accepted_count),
                    "admission_rate": (
                        accepted_count / candidate_count
                        if candidate_count
                        else 0.0
                    ),
                }
            ],
        )
        return RlmExperienceMemory(
            state=formed_state,
            success_patterns=(
                self._success_patterns(round_number, reflexions, admission)
                if reflexions
                else []
            ),
            forbidden_directions=self._forbidden_directions(
                round_number, reflexions, results, admission
            )
            if reflexions
            else [],
            insights=(
                self._insights(round_number, reflexions, admission)
                if reflexions
                else []
            ),
            version=4,
        )

    def _evolve(self, formed: RlmExperienceMemory) -> None:
        state = self.memory.state
        state.library_size = formed.state.library_size
        state.recent_admissions = (
            state.recent_admissions + formed.state.recent_admissions
        )[-self.max_entries :]
        state.recent_rejections = (
            state.recent_rejections + formed.state.recent_rejections
        )[-self.max_entries :]
        state.admission_log = (
            state.admission_log + formed.state.admission_log
        )[-self.max_entries :]
        state.failure_type_stats = self._merge_failure_type_stats(
            state.failure_type_stats,
            formed.state.failure_type_stats,
        )

        self.memory.success_patterns = self._merge_success_patterns(
            self.memory.success_patterns,
            formed.success_patterns,
        )[-self.max_entries :]
        self.memory.forbidden_directions = self._merge_forbidden_directions(
            self.memory.forbidden_directions,
            formed.forbidden_directions,
        )[-self.max_entries :]
        self.memory.insights = self._merge_insights(
            self.memory.insights,
            formed.insights,
        )[-self.max_entries :]
        self.memory.version = 4

    def _round_failure_type_stats(
        self,
        round_number: int,
        reflexions: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        impl_text = self._reflexion_text(reflexions, "implementation_errors")
        stats: dict[str, dict[str, Any]] = {}
        for record in _parse_error_frequency(impl_text):
            error_type = str(record.get("error_type", "") or "").strip()
            if not _useful_failure_type(error_type):
                continue
            key = _error_type_key(error_type)
            count = max(1, int(record.get("count", 1) or 1))
            stats[key] = {
                "error_type": error_type,
                "count": count,
                "rounds": [round_number],
                "latest_round": round_number,
                "latest_cause": str(record.get("cause", "") or ""),
                "latest_evidence": str(record.get("evidence", "") or ""),
                "latest_fix": str(record.get("fix", "") or ""),
            }
        return stats

    def _merge_failure_type_stats(
        self,
        existing: dict[str, dict[str, Any]],
        new: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        merged = _as_failure_type_stats(existing)
        for key, item in _as_failure_type_stats(new).items():
            if key not in merged:
                merged[key] = item
                continue
            merged[key]["count"] = int(merged[key].get("count", 0) or 0) + int(
                item.get("count", 0) or 0
            )
            merged[key]["rounds"] = _as_int_list(
                list(merged[key].get("rounds", [])) + list(item.get("rounds", []))
            )
            old_latest = _optional_int(merged[key].get("latest_round")) or 0
            new_latest = _optional_int(item.get("latest_round")) or 0
            if new_latest >= old_latest:
                merged[key]["latest_round"] = new_latest
                merged[key]["latest_cause"] = item.get("latest_cause", "")
                merged[key]["latest_evidence"] = item.get("latest_evidence", "")
                merged[key]["latest_fix"] = item.get("latest_fix", "")
                merged[key]["error_type"] = item.get("error_type") or merged[key].get(
                    "error_type",
                    key,
                )
        return _top_failure_type_stats(merged, limit=self.max_entries)

    def _global_best_signal(self) -> dict[str, Any] | None:
        valid = [
            item
            for item in self.memory.state.best_signals
            if item.get("admission_status") == "accepted"
            and _finite_float(item.get("ic")) is not None
        ]
        if not valid:
            return None
        return max(valid, key=lambda item: _finite_float(item.get("ic")) or float("-inf"))

    def _admission_records(
        self,
        round_number: int,
        admission: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if admission.get("status") != "accepted":
            return []
        accepted_candidates = _as_dict_list(admission.get("accepted_candidates"))
        if accepted_candidates:
            return [
                {
                    "round": round_number,
                    "factor_name": item.get("factor_name"),
                    "candidate_module": item.get("candidate_module"),
                    "research_branch": item.get("research_branch"),
                    "candidate_mode": item.get("candidate_mode"),
                    "mutation_axis": item.get("mutation_axis"),
                    "mutation_parent": item.get("mutation_parent"),
                    "candidate_ic": item.get("candidate_ic"),
                    "factor_dir": item.get("factor_dir"),
                }
                for item in accepted_candidates
            ]
        return [
            {
                "round": round_number,
                "factor_name": admission.get("factor_name"),
                "candidate_module": admission.get("candidate_module"),
                "research_branch": admission.get("research_branch"),
                "candidate_mode": admission.get("candidate_mode"),
                "mutation_axis": admission.get("mutation_axis"),
                "mutation_parent": admission.get("mutation_parent"),
                "candidate_ic": admission.get("candidate_ic"),
                "factor_dir": admission.get("factor_dir"),
            }
        ]

    def _rejection_records(
        self,
        round_number: int,
        results: list[Any],
        admission: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for result in results:
            label = str(getattr(result, "label", "") or "")
            if label not in {"failed", "duplicate"}:
                continue
            records.append(
                {
                    "round": round_number,
                    "module": getattr(result, "module_name", ""),
                    "research_branch": getattr(result, "research_branch", ""),
                    "candidate_mode": getattr(result, "candidate_mode", ""),
                    "mutation_axis": getattr(result, "mutation_axis", ""),
                    "mutation_parent": getattr(result, "mutation_parent", None),
                    "label": label,
                    "reason": getattr(result, "error", ""),
                    "error_type": getattr(result, "error_type", ""),
                    "ic": getattr(result, "ic", None),
                }
            )
        if admission.get("status") == "rejected":
            records.append(
                {
                    "round": round_number,
                    "module": admission.get("candidate_module"),
                    "research_branch": admission.get("research_branch"),
                    "candidate_mode": admission.get("candidate_mode"),
                    "mutation_axis": admission.get("mutation_axis"),
                    "mutation_parent": admission.get("mutation_parent"),
                    "label": "library_rejected",
                    "reason": admission.get("reason", ""),
                    "error_type": "FactorLibraryRejected",
                    "ic": admission.get("candidate_ic"),
                }
            )
        return records[-self.max_entries :]

    def _success_patterns(
        self,
        round_number: int,
        reflexions: list[dict[str, Any]],
        admission: dict[str, Any],
    ) -> list[RlmSuccessPattern]:
        if admission.get("status") != "accepted":
            return []
        economic_text = self._reflexion_text(reflexions, "economic_hypothesis")
        metadata = _parse_reflexion_metadata(economic_text)
        description = _first_metadata_value(
            metadata,
            [
                "recommended_hypotheses",
                "recommended_selection",
                "recommended_construction",
                "mutation_guidance",
                "positive_pattern",
                "next_action",
                "weak_pattern",
            ],
        ) or _compact_text(economic_text)
        accepted_candidates = _as_dict_list(admission.get("accepted_candidates"))
        if not accepted_candidates:
            accepted_candidates = [admission]

        patterns: list[RlmSuccessPattern] = []
        for item in accepted_candidates:
            name = str(
                item.get("factor_name", "") or f"round_{round_number}_accepted"
            )
            branch = str(item.get("research_branch", "") or "").strip()
            family = _safe_token(
                str(metadata.get("signal_family", "") or branch),
                fallback="",
            )
            pattern_name = f"accepted_{family}_{name}" if family else f"accepted_{name}"
            pattern_metadata = _merge_metadata(
                metadata,
                {"research_branch": branch} if branch else {},
            )
            patterns.append(
                RlmSuccessPattern(
                    name=pattern_name,
                    description=description,
                    template=str(item.get("candidate_module", "") or name),
                    success_rate="High" if item.get("candidate_ic") else "Medium",
                    example_factors=[name],
                    occurrence_count=1,
                    confidence=0.7,
                    source_round=round_number,
                    metadata=pattern_metadata,
                )
            )
        return patterns

    def _forbidden_directions(
        self,
        round_number: int,
        reflexions: list[dict[str, Any]],
        results: list[Any],
        admission: dict[str, Any],
    ) -> list[RlmForbiddenDirection]:
        impl_text = self._reflexion_text(reflexions, "implementation_errors")
        economic_text = self._reflexion_text(reflexions, "economic_hypothesis")
        failed = [
            str(getattr(item, "module_name", "") or "")
            for item in results
            if str(getattr(item, "label", "") or "") in {"failed", "duplicate"}
        ]
        if admission.get("status") == "rejected":
            failed.append(str(admission.get("candidate_module", "") or ""))
        if not impl_text and not economic_text and not failed:
            return []
        metadata = _parse_reflexion_metadata(impl_text or economic_text)
        failure_types = metadata.get("failure_types")
        failure_name = ""
        if isinstance(failure_types, list) and failure_types:
            failure_name = _safe_token(failure_types[0], fallback="")
        branch = self._dominant_failed_branch(results, admission)
        signal_family = _safe_token(
            str(metadata.get("signal_family", "") or branch),
            fallback="",
        )
        name = (
            f"implementation_{failure_name}_round_{round_number}"
            if failure_name
            else f"weak_{signal_family}_round_{round_number}"
            if signal_family
            else f"weak_or_rejected_patterns_round_{round_number}"
        )
        description = _first_metadata_value(
            metadata,
            [
                "forbidden_hypotheses",
                "forbidden_reasons",
                "forbidden_construction",
                "allowed_exceptions",
                "weak_pattern",
                "avoid_pattern",
                "next_action",
            ],
        ) or _compact_text(impl_text or economic_text or "Avoid repeated failed candidates.")
        reason = _first_metadata_value(
            metadata,
            [
                "forbidden_reasons",
                "forbidden_hypotheses",
                "forbidden_construction",
                "weak_pattern",
                "avoid_pattern",
                "next_action",
            ],
        ) or "weak_factor_or_candidate_rejection"
        return [
            RlmForbiddenDirection(
                name=name,
                description=description,
                reason=reason,
                examples=_merge_unique([], [item for item in failed if item]),
                occurrence_count=max(1, len(failed)),
                source_round=round_number,
                metadata=_merge_metadata(
                    metadata,
                    {"research_branch": branch} if branch else {},
                ),
            )
        ]

    @staticmethod
    def _dominant_failed_branch(results: list[Any], admission: dict[str, Any]) -> str:
        branches: list[str] = []
        for result in results:
            if str(getattr(result, "label", "") or "") not in {"failed", "duplicate"}:
                continue
            branch = str(getattr(result, "research_branch", "") or "").strip()
            if branch:
                branches.append(branch)
        if admission.get("status") == "rejected":
            branch = str(admission.get("research_branch", "") or "").strip()
            if branch:
                branches.append(branch)
        if not branches:
            return ""
        return max(set(branches), key=branches.count)

    def _insights(
        self,
        round_number: int,
        reflexions: list[dict[str, Any]],
        admission: dict[str, Any],
    ) -> list[RlmStrategicInsight]:
        insights: list[RlmStrategicInsight] = []
        for item in reflexions:
            text = str(item.get("reflexion", "") or "").strip()
            if not text:
                continue
            phase = str(item.get("phase", "") or "")
            metadata = _parse_reflexion_metadata(text)
            insight = _first_metadata_value(
                metadata,
                [
                    "next_action",
                    "recommended_hypotheses",
                    "recommended_selection",
                    "recommended_construction",
                    "forbidden_hypotheses",
                    "forbidden_reasons",
                    "forbidden_construction",
                    "positive_pattern",
                    "robust_pattern",
                    "avoid_pattern",
                    "weak_pattern",
                ],
            ) or _first_meaningful_line(text, fallback=f"{phase} reflexion")
            insights.append(
                RlmStrategicInsight(
                    insight=insight,
                    evidence=_compact_text(text),
                    batch_source=round_number,
                    phase=phase,
                    metadata=metadata,
                )
            )
        if admission.get("status") == "rejected":
            insights.append(
                RlmStrategicInsight(
                    insight="Round best failed factor library admission",
                    evidence=str(admission.get("reason", "") or ""),
                    batch_source=round_number,
                    phase="library_admission",
                )
            )
        return insights

    def _merge_success_patterns(
        self,
        existing: list[RlmSuccessPattern],
        new: list[RlmSuccessPattern],
    ) -> list[RlmSuccessPattern]:
        merged = {item.name: item for item in existing}
        for item in new:
            if item.name in merged:
                merged[item.name].occurrence_count += item.occurrence_count
                merged[item.name].description = item.description or merged[item.name].description
                merged[item.name].confidence = max(merged[item.name].confidence, item.confidence)
                merged[item.name].metadata = _merge_metadata(
                    merged[item.name].metadata,
                    item.metadata,
                )
                for factor in item.example_factors:
                    if factor not in merged[item.name].example_factors:
                        merged[item.name].example_factors.append(factor)
            else:
                merged[item.name] = item
        return list(merged.values())

    def _merge_forbidden_directions(
        self,
        existing: list[RlmForbiddenDirection],
        new: list[RlmForbiddenDirection],
    ) -> list[RlmForbiddenDirection]:
        merged = {item.name: item for item in existing}
        for item in new:
            if item.name in merged:
                merged[item.name].occurrence_count += item.occurrence_count
                merged[item.name].description = item.description or merged[item.name].description
                merged[item.name].reason = item.reason or merged[item.name].reason
                merged[item.name].metadata = _merge_metadata(
                    merged[item.name].metadata,
                    item.metadata,
                )
                for example in item.examples:
                    if example not in merged[item.name].examples:
                        merged[item.name].examples.append(example)
            else:
                merged[item.name] = item
        return list(merged.values())

    def _merge_insights(
        self,
        existing: list[RlmStrategicInsight],
        new: list[RlmStrategicInsight],
    ) -> list[RlmStrategicInsight]:
        merged = list(existing)
        seen = {(item.phase, item.insight): item for item in merged}
        for item in new:
            key = (item.phase, item.insight)
            if key not in seen:
                merged.append(item)
                seen[key] = item
            else:
                seen[key].metadata = _merge_metadata(seen[key].metadata, item.metadata)
        return merged

    @staticmethod
    def _reflexion_text(reflexions: list[dict[str, Any]], phase: str) -> str:
        for item in reflexions:
            if str(item.get("phase", "") or "") == phase:
                return str(item.get("reflexion", "") or "").strip()
        return ""
