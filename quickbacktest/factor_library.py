"""Filesystem-backed factor library for quickbacktest factor-miner runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_FACTOR_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VALID_STATUSES = {"candidate", "accepted", "rejected", "archived"}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("factor name must be a string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("factor name must be non-empty")
    if not _FACTOR_NAME_RE.match(normalized):
        raise ValueError(
            "factor name must be a directory-safe name: letters, numbers, dot, "
            "underscore, or hyphen; no spaces or slashes"
        )
    return normalized


def _quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_inside_root(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
        raise ValueError("resolved factor path escapes library root")
    return resolved_path


def _status_or_raise(status: str) -> str:
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid factor status: {status}")
    return status


def _metric(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def _format_frontmatter(record: "FactorRecord") -> str:
    lines = [
        "---",
        f"name: {_quote(record.name)}",
        f"description: {_quote(record.description)}",
        f"status: {_quote(record.status)}",
        f"source: {_quote(record.source)}",
        f"signal_class: {_quote(record.signal_class)}",
        f"universe: {_quote(record.universe)}",
        f"horizon: {_quote(record.horizon)}",
        f"factor_shift: {_quote(record.factor_shift)}",
    ]
    if record.tags:
        lines.append("tags:")
        for tag in record.tags:
            lines.append(f"  - {_quote(tag)}")
    lines.append("---")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_factor_markdown(record: "FactorRecord") -> str:
    body = [
        _format_frontmatter(record),
        "",
        "## RLM Summary",
        record.rlm_summary.strip() or "No RLM summary provided.",
        "",
        "## Review",
        "- status: see frontmatter `status`",
        "- review details: see `review.json`",
        "- numeric metrics: see `metrics.json`",
        "",
        "## Notes",
        "The signal was saved and run successfully before this factor card was created.",
        "Factor shift is applied by quickbacktest after signal computation.",
        "",
    ]
    return "\n".join(body)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("FACTOR.md must start with frontmatter")
    end = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = idx
            break
    if end is None:
        raise ValueError("FACTOR.md frontmatter is missing closing ---")

    meta: dict[str, Any] = {}
    section: str | None = None
    for raw in lines[1:end]:
        if raw.startswith("  - ") and section == "tags":
            meta.setdefault("tags", []).append(json.loads(raw[4:]))
            continue
        key, sep, value = raw.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key == "tags":
            meta["tags"] = []
            section = "tags"
        else:
            section = None
            try:
                meta[key] = json.loads(value)
            except json.JSONDecodeError:
                meta[key] = value.strip("'\"")
    return meta


@dataclass
class FactorRecord:
    name: str
    description: str
    signal_class: str
    signal_code: str
    metrics: dict[str, Any]
    rlm_summary: str = ""
    status: str = "candidate"
    source: str = "rlm"
    universe: str = ""
    horizon: int = 1
    factor_shift: int = 1
    tags: list[str] = field(default_factory=list)
    review: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.name = _normalize_name(self.name)
        self.status = _status_or_raise(self.status)
        if not self.description.strip():
            raise ValueError("factor description must be non-empty")
        if not self.signal_class.strip():
            raise ValueError("signal_class must be non-empty")
        if not self.signal_code.strip():
            raise ValueError("signal_code must be non-empty")
        self.tags = [str(tag) for tag in self.tags]

    @property
    def markdown(self) -> str:
        return _build_factor_markdown(self)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "source": self.source,
            "signal_class": self.signal_class,
            "universe": self.universe,
            "horizon": self.horizon,
            "factor_shift": self.factor_shift,
            "tags": list(self.tags),
        }


class FactorLibrary:
    """Read and write factor assets under ``root/<factor-name>/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _factor_dir(self, name: str) -> Path:
        normalized = _normalize_name(name)
        return _assert_inside_root(self.root / normalized, self.root)

    def save_factor(
        self,
        *,
        name: str,
        signal_code: str,
        metrics: dict[str, Any],
        description: str,
        rlm_summary: str = "",
        signal_class: str,
        universe: str = "",
        horizon: int = 1,
        factor_shift: int = 1,
        status: str = "candidate",
        source: str = "rlm",
        tags: list[str] | None = None,
        overwrite: bool = True,
    ) -> FactorRecord:
        record = FactorRecord(
            name=name,
            description=description,
            signal_class=signal_class,
            signal_code=signal_code,
            metrics=dict(metrics),
            rlm_summary=rlm_summary,
            status=status,
            source=source,
            universe=universe,
            horizon=int(horizon),
            factor_shift=int(factor_shift),
            tags=tags or [],
        )
        factor_dir = self._factor_dir(record.name)
        if factor_dir.exists() and not overwrite:
            raise FileExistsError(f"factor already exists: {record.name}")
        factor_dir.mkdir(parents=True, exist_ok=True)
        (factor_dir / "FACTOR.md").write_text(record.markdown, encoding="utf-8")
        (factor_dir / "signal.py").write_text(record.signal_code, encoding="utf-8")
        _write_json(factor_dir / "metrics.json", record.metrics)
        return record

    def list_factors(self) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []
        if not self.root.exists():
            return factors
        for path in sorted(self.root.glob("*/FACTOR.md")):
            try:
                factors.append(_parse_frontmatter(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return factors

    def read_factor(self, name: str) -> dict[str, Any]:
        factor_dir = self._factor_dir(name)
        card_path = factor_dir / "FACTOR.md"
        if not card_path.exists():
            raise KeyError(f"factor not found: {_normalize_name(name)}")
        review_path = factor_dir / "review.json"
        return {
            "metadata": _parse_frontmatter(card_path.read_text(encoding="utf-8")),
            "card": card_path.read_text(encoding="utf-8"),
            "signal_code": (factor_dir / "signal.py").read_text(encoding="utf-8"),
            "metrics": _read_json(factor_dir / "metrics.json"),
            "review": _read_json(review_path) if review_path.exists() else None,
        }

    def save_review(self, name: str, review: dict[str, Any]) -> dict[str, Any]:
        factor_dir = self._factor_dir(name)
        if not (factor_dir / "FACTOR.md").exists():
            raise KeyError(f"factor not found: {_normalize_name(name)}")
        _write_json(factor_dir / "review.json", review)
        return review

    def update_status(self, name: str, status: str) -> None:
        status = _status_or_raise(status)
        factor_dir = self._factor_dir(name)
        card_path = factor_dir / "FACTOR.md"
        if not card_path.exists():
            raise KeyError(f"factor not found: {_normalize_name(name)}")
        text = card_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        replaced = False
        for idx, line in enumerate(lines):
            if line.startswith("status:"):
                lines[idx] = f"status: {_quote(status)}"
                replaced = True
                break
        if not replaced:
            raise ValueError("FACTOR.md frontmatter is missing status")
        card_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def to_rlm_tools(self) -> dict[str, dict[str, Any]]:
        """Expose read-only factor library tools for RLM context."""

        def list_factors() -> list[dict[str, Any]]:
            """List saved factors with metadata and status."""
            return self.list_factors()

        def read_factor(name: str) -> dict[str, Any]:
            """Read one saved factor card, signal source, metrics, and review."""
            return self.read_factor(name)

        return {
            "list_factors": {
                "tool": list_factors,
                "description": "list_factors() -> list saved factor cards and statuses",
            },
            "read_factor": {
                "tool": read_factor,
                "description": (
                    "read_factor(name) -> read one saved factor's FACTOR.md, "
                    "signal.py, metrics.json, and review.json if present"
                ),
            },
        }


def review_factor_metrics(
    metrics: dict[str, Any],
    *,
    runs_without_error: bool = True,
) -> dict[str, Any]:
    """Deterministic first-pass factor review from quickbacktest metrics."""
    coverage = _metric(metrics, "coverage")
    missing_rate = _metric(metrics, "missing_rate")
    rank_ic = _metric(metrics, "daily_rank_ic_mean")
    rank_icir = _metric(metrics, "rank_icir")
    daily_count = int(_metric(metrics, "daily_rank_ic_count", 0))
    layered_ic = metrics.get("layered_ic") if isinstance(metrics, dict) else {}
    checks = {
        "runs_without_error": bool(runs_without_error),
        "coverage_ok": coverage >= 0.8,
        "missing_rate_ok": missing_rate <= 0.2,
        "rank_ic_ok": rank_ic > 0.0,
        "icir_ok": rank_icir > 0.0,
        "daily_rank_ic_count_ok": daily_count > 0,
        "decile_layer_ok": (
            isinstance(layered_ic, dict)
            and layered_ic.get("layer_type") == "decile"
            and bool(layered_ic.get("deciles"))
        ),
    }
    accepted = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "verdict": "accepted" if accepted else "rejected",
        "summary": (
            "Runs successfully with acceptable coverage and positive rank IC."
            if accepted
            else "Rejected by deterministic review: " + ", ".join(failed)
        ),
        "checks": checks,
        "evidence": {
            "coverage": coverage,
            "missing_rate": missing_rate,
            "daily_rank_ic_mean": rank_ic,
            "rank_icir": rank_icir,
            "daily_rank_ic_count": daily_count,
            "ic_distribution": metrics.get("ic_distribution"),
            "rank_ic_distribution": metrics.get("rank_ic_distribution"),
            "layered_ic": layered_ic,
        },
        "risks": [] if accepted else ["Review thresholds failed."],
        "required_fixes": [] if accepted else ["Improve failed review checks."],
        "next_steps": ["Evaluate on a longer period.", "Check correlation with accepted factors."],
    }


def build_rlm_factor_tools(root: str | Path) -> dict[str, dict[str, Any]]:
    """Build read-only RLM tools for an on-disk factor library."""
    return FactorLibrary(root).to_rlm_tools()
