from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import FAST_SCREEN_RANK_IC_THRESHOLD, LIBRARY_CORRELATION_THRESHOLD
from .models import CandidateResult, ParallelReflexionConfig
from .utils import (
    _factor_metric_ic,
    _finite_float,
    _latest_accepted_memory_signal_source,
    _spearman_corr_from_csv,
    _trim_signal_source,
    _write_json,
)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _factor_library_name(module_name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "-", module_name).replace("_", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", words).strip("-").lower()

class FactorLibraryAdmissionService:
    """Encapsulate factor-library lookup, correlation review, and admission."""

    def best_signal_record(self, library_root: Path) -> dict[str, Any] | None:
        from quickbacktest import FactorLibrary

        library = FactorLibrary(library_root)
        candidates: list[dict[str, Any]] = []
        for meta in library.list_factors():
            if meta.get("status") != "accepted":
                continue
            name = str(meta.get("name", "") or "").strip()
            if not name:
                continue
            try:
                factor = library.read_factor(name)
            except Exception:
                continue
            ic = _factor_metric_ic(factor.get("metrics"))
            if ic is None:
                continue
            signal_code = str(factor.get("signal_code", "") or "").strip()
            if not signal_code:
                continue
            factor_dir = self.factor_dir(library_root, name)
            candidates.append(
                {
                    "source": "factor_library",
                    "admission_status": "accepted",
                    "factor_name": name,
                    "factor_dir": str(factor_dir),
                    "signal_path": str(factor_dir / "signal.py"),
                    "ic": ic,
                    "signal_source": _trim_signal_source(signal_code),
                }
            )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: _finite_float(item.get("ic")) or float("-inf"),
        )

    def best_signal_source(
        self,
        *,
        config: ParallelReflexionConfig,
        memory: dict[str, Any],
    ) -> str:
        library_best = self.best_signal_record(config.factor_library_path)
        if library_best is not None:
            return str(library_best.get("signal_source", "") or "").strip()
        return _latest_accepted_memory_signal_source(memory)

    def admit_best(
        self,
        *,
        config: ParallelReflexionConfig,
        round_dir: Path,
        best: CandidateResult | None,
    ) -> dict[str, Any]:
        return self.admit_candidates(
            config=config,
            round_dir=round_dir,
            candidates=[best] if best is not None else [],
        )

    def admit_candidates(
        self,
        *,
        config: ParallelReflexionConfig,
        round_dir: Path,
        candidates: list[CandidateResult],
    ) -> dict[str, Any]:
        ordered = sorted(
            [candidate for candidate in candidates if candidate is not None],
            key=lambda item: _finite_float(item.ic) or float("-inf"),
            reverse=True,
        )
        if not ordered:
            return self._write_admission(
                round_dir,
                {"status": "skipped", "reason": "no admissible candidates"},
            )

        attempts: list[dict[str, Any]] = []
        accepted_attempts: list[dict[str, Any]] = []
        for rank, candidate in enumerate(ordered, start=1):
            attempt = self._admit_candidate(
                config=config,
                candidate=candidate,
                attempt_rank=rank,
            )
            attempts.append(self._attempt_summary(attempt))
            if attempt.get("status") != "accepted":
                continue
            accepted_attempts.append(attempt)

        if accepted_attempts:
            primary = dict(accepted_attempts[0])
            primary["selection_policy"] = (
                "all_candidates_that_pass_fast_screen_and_library_review"
            )
            primary["accepted_count"] = len(accepted_attempts)
            primary["accepted_candidates"] = [
                self._accepted_candidate_record(attempt)
                for attempt in accepted_attempts
            ]
            primary["archived_replacements"] = [
                item
                for attempt in accepted_attempts
                for item in _as_list(attempt.get("archived_replacements"))
                if isinstance(item, dict)
            ]
            primary["attempts"] = attempts
            primary["candidate_pool"] = [
                self._candidate_pool_record(item) for item in ordered
            ]
            return self._write_admission(round_dir, primary)

        return self._write_admission(
            round_dir,
            {
                "status": "rejected",
                "reason": "no_candidate_passed_rank_ic_threshold_or_library_review",
                "selection_policy": "try_all_positive_nonduplicate_candidates_by_ic",
                "candidate_pool": [self._candidate_pool_record(item) for item in ordered],
                "attempts": attempts,
            },
        )

    def _admit_candidate(
        self,
        *,
        config: ParallelReflexionConfig,
        candidate: CandidateResult,
        attempt_rank: int,
    ) -> dict[str, Any]:
        from quickbacktest import FactorLibrary, review_factor_metrics

        if candidate.signal_path is None or not candidate.signal_path.exists():
            return {
                "status": "skipped",
                "reason": "candidate signal source missing",
                "candidate_module": candidate.module_name,
                "research_branch": candidate.research_branch,
                "attempt_rank": attempt_rank,
            }
        if not isinstance(candidate.metrics, dict):
            return {
                "status": "skipped",
                "reason": "candidate metrics missing",
                "candidate_module": candidate.module_name,
                "research_branch": candidate.research_branch,
                "attempt_rank": attempt_rank,
            }

        factor_name = _factor_library_name(candidate.module_name)
        library = FactorLibrary(config.factor_library_path)
        candidate_ic = _factor_metric_ic(candidate.metrics)
        if candidate_ic is None or candidate_ic <= FAST_SCREEN_RANK_IC_THRESHOLD:
            return {
                "status": "rejected",
                "reason": "rank_ic_fast_screen_failed",
                "factor_name": factor_name,
                "candidate_module": candidate.module_name,
                "research_branch": candidate.research_branch,
                "candidate_signal": str(candidate.signal_path),
                "candidate_ic": candidate_ic,
                "rank_ic_threshold": FAST_SCREEN_RANK_IC_THRESHOLD,
                "attempt_rank": attempt_rank,
            }

        deterministic_review = review_factor_metrics(
            candidate.metrics,
            runs_without_error=True,
        )
        library_corr = self.correlation_check(
            candidate,
            library_root=config.factor_library_path,
        )
        replacement_matches, blocking_matches = self.split_correlation_matches(
            library_corr.get("high_correlation", []),
            candidate_ic=candidate_ic,
        )
        correlation_verdict = self.correlation_verdict(
            replacement_matches=replacement_matches,
            blocking_matches=blocking_matches,
        )

        deterministic_accepted = deterministic_review.get("verdict") == "accepted"
        status = (
            "accepted"
            if deterministic_accepted and correlation_verdict != "rejected"
            else "rejected"
        )
        combined_review = {
            **deterministic_review,
            "verdict": status,
            "library_correlation_check": library_corr,
            "replacement_check": {
                "verdict": correlation_verdict,
                "replacement_candidates": replacement_matches,
                "blocking_matches": blocking_matches,
            },
        }
        if blocking_matches:
            combined_review["summary"] = (
                deterministic_review.get("summary", "")
                + " Rejected by library correlation check."
            ).strip()

        if status != "accepted":
            return {
                "status": "rejected",
                "reason": "deterministic_review_or_library_correlation_failed",
                "factor_name": factor_name,
                "candidate_module": candidate.module_name,
                "research_branch": candidate.research_branch,
                "candidate_signal": str(candidate.signal_path),
                "candidate_ic": candidate_ic,
                "attempt_rank": attempt_rank,
                "deterministic_review": deterministic_review,
                "library_correlation_check": library_corr,
                "replacement_check": combined_review["replacement_check"],
                "review": combined_review,
            }

        record = library.save_factor(
            name=factor_name,
            signal_code=candidate.signal_path.read_text(encoding="utf-8"),
            metrics=candidate.metrics,
            description="RLM parallel reflexion generated factor signal.",
            rlm_summary=candidate.rlm_summary,
            signal_class=candidate.module_name,
            universe=config.instruments,
            horizon=config.horizon,
            factor_shift=config.factor_shift,
            status="accepted",
            source="rlm_parallel_reflexion",
            tags=["rlm", "factor", "parallel-reflexion"],
            overwrite=True,
        )
        library.save_review(record.name, combined_review)
        factor_dir = self.factor_dir(config.factor_library_path, record.name)
        if candidate.factor_data_csv is not None and candidate.factor_data_csv.exists():
            import shutil

            shutil.copyfile(candidate.factor_data_csv, factor_dir / "factor_data.csv")
        archived_replacements = self.archive_replaced_factors(
            library,
            replacement_matches=replacement_matches,
            new_factor_name=record.name,
            candidate_ic=candidate_ic,
        )

        return {
            "status": "accepted",
            "factor_name": record.name,
            "factor_dir": str(factor_dir),
            "factor_card": str(factor_dir / "FACTOR.md"),
            "factor_signal": str(factor_dir / "signal.py"),
            "factor_metrics": str(factor_dir / "metrics.json"),
            "factor_review": str(factor_dir / "review.json"),
            "factor_data": str(factor_dir / "factor_data.csv"),
            "candidate_module": candidate.module_name,
            "research_branch": candidate.research_branch,
            "candidate_ic": candidate_ic,
            "attempt_rank": attempt_rank,
            "deterministic_review": deterministic_review,
            "library_correlation_check": library_corr,
            "replacement_check": combined_review["replacement_check"],
            "archived_replacements": archived_replacements,
        }

    def correlation_check(
        self,
        candidate: CandidateResult,
        *,
        library_root: Path,
    ) -> dict[str, Any]:
        from quickbacktest import FactorLibrary

        if candidate.factor_data_csv is None or not candidate.factor_data_csv.exists():
            return {"available": False, "reason": "candidate factor_data.csv missing"}

        library = FactorLibrary(library_root)
        matches: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for meta in library.list_factors():
            if meta.get("status") != "accepted":
                continue
            name = str(meta.get("name", ""))
            factor_data_path = self.library_factor_data_path(library_root, name)
            if not factor_data_path.exists():
                skipped.append({"factor": name, "reason": "factor_data.csv missing"})
                continue
            corr = _spearman_corr_from_csv(candidate.factor_data_csv, factor_data_path)
            if corr is None:
                skipped.append({"factor": name, "reason": "correlation unavailable"})
                continue
            factor = library.read_factor(name)
            existing_ic = _factor_metric_ic(factor.get("metrics"))
            matches.append(
                {
                    "factor": name,
                    "spearman": corr,
                    "abs_spearman": abs(corr),
                    "existing_ic": existing_ic,
                }
            )

        matches = sorted(matches, key=lambda item: item["abs_spearman"], reverse=True)
        nearest = matches[0] if matches else None
        high_corr = [
            item
            for item in matches
            if item["abs_spearman"] >= LIBRARY_CORRELATION_THRESHOLD
        ]
        return {
            "available": True,
            "threshold": LIBRARY_CORRELATION_THRESHOLD,
            "nearest": nearest,
            "high_correlation": high_corr,
            "skipped": skipped,
        }

    def split_correlation_matches(
        self,
        high_corr: Any,
        *,
        candidate_ic: float | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        blocking_matches: list[dict[str, Any]] = []
        replacement_matches: list[dict[str, Any]] = []
        for match in high_corr if isinstance(high_corr, list) else []:
            existing_ic = _finite_float(match.get("existing_ic"))
            improves_existing = candidate_ic is not None and (
                existing_ic is None or candidate_ic > existing_ic
            )
            if improves_existing:
                replacement_matches.append(match)
            else:
                blocking_matches.append(match)
        return replacement_matches, blocking_matches

    @staticmethod
    def correlation_verdict(
        *,
        replacement_matches: list[dict[str, Any]],
        blocking_matches: list[dict[str, Any]],
    ) -> str:
        if blocking_matches:
            return "rejected"
        if replacement_matches:
            return "replacement_candidate"
        return "accepted"

    def archive_replaced_factors(
        self,
        library: Any,
        *,
        replacement_matches: list[dict[str, Any]],
        new_factor_name: str,
        candidate_ic: float | None,
    ) -> list[dict[str, Any]]:
        archived: list[dict[str, Any]] = []
        for match in replacement_matches:
            old_name = str(match.get("factor", "") or "").strip()
            if not old_name or old_name == new_factor_name:
                continue
            archive_event = {
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "archived_by": new_factor_name,
                "reason": "replaced_by_higher_ic_correlated_factor",
                "replacement_ic": candidate_ic,
                "replaced_factor_ic": match.get("existing_ic"),
                "spearman": match.get("spearman"),
                "threshold": LIBRARY_CORRELATION_THRESHOLD,
            }
            try:
                existing = library.read_factor(old_name)
                review = existing.get("review")
                review_payload = dict(review) if isinstance(review, dict) else {}
                history = review_payload.get("archive_history", [])
                if not isinstance(history, list):
                    history = []
                history.append(archive_event)
                review_payload["archive_history"] = history
                review_payload["latest_archive_event"] = archive_event
                library.save_review(old_name, review_payload)
                library.update_status(old_name, "archived")
                archived.append(
                    {"factor": old_name, "status": "archived", **archive_event}
                )
            except Exception as exc:
                archived.append(
                    {
                        "factor": old_name,
                        "status": "archive_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        **archive_event,
                    }
                )
        return archived

    @staticmethod
    def factor_dir(root: Path, factor_name: str) -> Path:
        return root / factor_name

    def library_factor_data_path(self, root: Path, factor_name: str) -> Path:
        return self.factor_dir(root, factor_name) / "factor_data.csv"

    @staticmethod
    def _candidate_pool_record(candidate: CandidateResult) -> dict[str, Any]:
        return {
            "module": candidate.module_name,
            "research_branch": candidate.research_branch,
            "label": candidate.label,
            "ic": candidate.ic,
            "ic_name": candidate.ic_name,
            "rank_ic_threshold": FAST_SCREEN_RANK_IC_THRESHOLD,
        }

    @staticmethod
    def _accepted_candidate_record(attempt: dict[str, Any]) -> dict[str, Any]:
        return {
            "factor_name": attempt.get("factor_name"),
            "factor_dir": attempt.get("factor_dir"),
            "factor_card": attempt.get("factor_card"),
            "factor_signal": attempt.get("factor_signal"),
            "factor_metrics": attempt.get("factor_metrics"),
            "factor_review": attempt.get("factor_review"),
            "factor_data": attempt.get("factor_data"),
            "candidate_module": attempt.get("candidate_module"),
            "research_branch": attempt.get("research_branch"),
            "candidate_ic": attempt.get("candidate_ic"),
            "attempt_rank": attempt.get("attempt_rank"),
            "archived_replacements": attempt.get("archived_replacements", []),
        }

    @staticmethod
    def _attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
        review = attempt.get("review")
        review_payload = review if isinstance(review, dict) else {}
        replacement_check = attempt.get("replacement_check")
        replacement_payload = replacement_check if isinstance(replacement_check, dict) else {}
        return {
            "attempt_rank": attempt.get("attempt_rank"),
            "status": attempt.get("status"),
            "reason": attempt.get("reason", ""),
            "candidate_module": attempt.get("candidate_module"),
            "research_branch": attempt.get("research_branch"),
            "candidate_ic": attempt.get("candidate_ic"),
            "factor_name": attempt.get("factor_name"),
            "review_verdict": review_payload.get("verdict"),
            "replacement_verdict": replacement_payload.get("verdict"),
        }

    @staticmethod
    def _write_admission(round_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
        _write_json(round_dir / "factor_library_admission.json", result)
        return result

def _best_factor_library_signal_record(library_root: Path) -> dict[str, Any] | None:
    return FactorLibraryAdmissionService().best_signal_record(library_root)


def _best_signal_source_for_generation(
    *,
    config: ParallelReflexionConfig,
    memory: dict[str, Any],
) -> str:
    return FactorLibraryAdmissionService().best_signal_source(
        config=config,
        memory=memory,
    )

def _factor_library_dir(root: Path, factor_name: str) -> Path:
    return FactorLibraryAdmissionService.factor_dir(root, factor_name)


def _library_factor_data_path(root: Path, factor_name: str) -> Path:
    return FactorLibraryAdmissionService().library_factor_data_path(root, factor_name)


def _library_correlation_check(
    candidate: CandidateResult,
    *,
    library_root: Path,
) -> dict[str, Any]:
    return FactorLibraryAdmissionService().correlation_check(
        candidate,
        library_root=library_root,
    )


def _split_library_correlation_matches(
    high_corr: Any,
    *,
    candidate_ic: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return replacement candidates and blocking matches from library hits."""
    return FactorLibraryAdmissionService().split_correlation_matches(
        high_corr,
        candidate_ic=candidate_ic,
    )


def _correlation_verdict(
    *,
    replacement_matches: list[dict[str, Any]],
    blocking_matches: list[dict[str, Any]],
) -> str:
    return FactorLibraryAdmissionService.correlation_verdict(
        replacement_matches=replacement_matches,
        blocking_matches=blocking_matches,
    )
