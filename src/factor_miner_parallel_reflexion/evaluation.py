from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import BATCH_DUPLICATE_CORRELATION_THRESHOLD
from .library import FactorLibraryAdmissionService
from .models import CandidateResult, ParallelReflexionConfig, RoundEvaluation
from .portfolio import FactorLibraryPortfolioService
from .utils import (
    _finite_float,
    _has_factor_data,
    _is_positive_ic_result,
    _previous_global_best_ic,
    _round_best_signal_record,
    _select_best_result,
    _spearman_corr_from_csv,
    _write_candidate_result,
)

class RoundEvaluator:
    """Apply Ralph-style candidate screening and factor-library admission."""

    def __init__(
        self,
        config: ParallelReflexionConfig,
        *,
        admission_service: FactorLibraryAdmissionService | None = None,
        portfolio_service: FactorLibraryPortfolioService | None = None,
    ) -> None:
        self.config = config
        self.admission_service = admission_service or FactorLibraryAdmissionService()
        self.portfolio_service = portfolio_service or FactorLibraryPortfolioService()

    def evaluate(
        self,
        *,
        round_number: int,
        round_dir: Path,
        memory: dict[str, Any],
        results: list[CandidateResult],
    ) -> RoundEvaluation:
        self.label_results(results)
        batch_dedup = self.deduplicate_batch_results(results)
        best_result = _select_best_result(results)
        round_ic = self.round_ic_record(
            memory,
            round_number=round_number,
            results=results,
        )
        admission_candidates = self.admission_candidates(results)
        factor_library_admission = self.admission_service.admit_candidates(
            config=self.config,
            round_dir=round_dir,
            candidates=admission_candidates,
        )
        admitted_results = self.admitted_results(results, factor_library_admission)
        admitted_result = admitted_results[0] if admitted_results else None
        best_signal = self.round_best_signal_record(
            round_number=round_number,
            best=admitted_result,
            admission=factor_library_admission,
        )
        factor_library_portfolio = self.portfolio_service.run(
            config=self.config,
            round_dir=round_dir,
        )
        return RoundEvaluation(
            batch_dedup=batch_dedup,
            round_ic=round_ic,
            best_signal=best_signal,
            best_result=best_result,
            admitted_result=admitted_result,
            admitted_results=admitted_results,
            factor_library_admission=factor_library_admission,
            factor_library_portfolio=factor_library_portfolio,
        )

    def label_results(self, results: list[CandidateResult]) -> None:
        successful = [result for result in results if _is_positive_ic_result(result)]
        best: CandidateResult | None = (
            max(successful, key=lambda item: item.ic) if successful else None
        )
        for result in results:
            if not _is_positive_ic_result(result):
                result.label = "failed"
            elif best is not None and result.candidate_index == best.candidate_index:
                result.label = "best"
            else:
                result.label = "average"
            _write_candidate_result(result)

    def deduplicate_batch_results(
        self,
        results: list[CandidateResult],
        *,
        threshold: float = BATCH_DUPLICATE_CORRELATION_THRESHOLD,
    ) -> dict[str, Any]:
        candidates = sorted(
            [
                result
                for result in results
                if _is_positive_ic_result(result) and _has_factor_data(result)
            ],
            key=lambda item: float(item.ic or float("-inf")),
            reverse=True,
        )
        kept: list[CandidateResult] = []
        duplicates: list[dict[str, Any]] = []
        for result in candidates:
            duplicate_of: CandidateResult | None = None
            duplicate_corr: float | None = None
            for prior in kept:
                if result.factor_data_csv is None or prior.factor_data_csv is None:
                    continue
                corr = _spearman_corr_from_csv(
                    result.factor_data_csv,
                    prior.factor_data_csv,
                )
                if corr is None:
                    continue
                if abs(corr) >= threshold:
                    duplicate_of = prior
                    duplicate_corr = corr
                    break
            if duplicate_of is None:
                kept.append(result)
                continue
            self._mark_duplicate(
                result,
                duplicate_of=duplicate_of,
                duplicate_corr=duplicate_corr,
            )
            duplicates.append(
                {
                    "module": result.module_name,
                    "duplicate_of": duplicate_of.module_name,
                    "spearman": duplicate_corr,
                    "threshold": threshold,
                }
            )

        for index, result in enumerate(kept):
            result.label = "best" if index == 0 else "average"

        kept_ids = {id(result) for result in kept}
        for result in results:
            if id(result) not in kept_ids and result.label != "duplicate":
                if not _is_positive_ic_result(result):
                    result.label = "failed"
            _write_candidate_result(result)

        return {
            "threshold": threshold,
            "kept": [result.module_name for result in kept],
            "duplicates": duplicates,
        }

    def round_ic_record(
        self,
        memory: dict[str, Any],
        *,
        round_number: int,
        results: list[CandidateResult],
    ) -> dict[str, Any]:
        best = next((result for result in results if result.label == "best"), None)
        best_ic = _finite_float(best.ic if best is not None else None)
        previous_best_ic, previous_best_round = _previous_global_best_ic(memory)
        improved = best_ic is not None and (
            previous_best_ic is None or best_ic > previous_best_ic
        )
        return {
            "round": round_number,
            "best_ic": best_ic,
            "best_module": best.module_name if best is not None else None,
            "previous_best_ic": previous_best_ic,
            "previous_best_round": previous_best_round,
            "improved": improved,
            "delta_vs_previous_best": (
                None
                if best_ic is None or previous_best_ic is None
                else best_ic - previous_best_ic
            ),
        }

    def round_best_signal_record(
        self,
        *,
        round_number: int,
        best: CandidateResult | None,
        admission: dict[str, Any],
    ) -> dict[str, Any] | None:
        return _round_best_signal_record(
            round_number=round_number,
            best=best,
            admission=admission,
        )

    @staticmethod
    def admission_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
        return sorted(
            [
                result
                for result in results
                if result.label in {"best", "average"}
                and _is_positive_ic_result(result)
                and _has_factor_data(result)
            ],
            key=lambda item: float(item.ic or float("-inf")),
            reverse=True,
        )

    @staticmethod
    def admitted_result(
        results: list[CandidateResult],
        admission: dict[str, Any],
    ) -> CandidateResult | None:
        admitted = RoundEvaluator.admitted_results(results, admission)
        return admitted[0] if admitted else None

    @staticmethod
    def admitted_results(
        results: list[CandidateResult],
        admission: dict[str, Any],
    ) -> list[CandidateResult]:
        if admission.get("status") != "accepted":
            return []
        modules = [
            str(item.get("candidate_module", "") or "")
            for item in admission.get("accepted_candidates", [])
            if isinstance(item, dict)
        ]
        if not modules:
            modules = [str(admission.get("candidate_module", "") or "")]
        by_module = {result.module_name: result for result in results}
        return [by_module[module] for module in modules if module in by_module]

    @staticmethod
    def _mark_duplicate(
        result: CandidateResult,
        *,
        duplicate_of: CandidateResult,
        duplicate_corr: float | None,
    ) -> None:
        result.label = "duplicate"
        result.dedup_of = duplicate_of.module_name
        result.dedup_correlation = duplicate_corr
        result.error_type = "IntraBatchDuplicate"
        result.error = (
            f"Signal is highly correlated with {duplicate_of.module_name}; "
            f"spearman={duplicate_corr:.6g}"
            if duplicate_corr is not None
            else f"Signal is highly correlated with {duplicate_of.module_name}."
        )

def _deduplicate_batch_results(
    results: list[CandidateResult],
    *,
    threshold: float = BATCH_DUPLICATE_CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    return RoundEvaluator(ParallelReflexionConfig()).deduplicate_batch_results(
        results,
        threshold=threshold,
    )

def _label_results(results: list[CandidateResult]) -> None:
    RoundEvaluator(ParallelReflexionConfig()).label_results(results)

def _evaluate_round_results(
    *,
    config: ParallelReflexionConfig,
    round_number: int,
    round_dir: Path,
    memory: dict[str, Any],
    results: list[CandidateResult],
) -> RoundEvaluation:
    """Apply fast IC screen, intra-batch dedup, and factor-library admission."""
    return RoundEvaluator(config).evaluate(
        round_number=round_number,
        round_dir=round_dir,
        memory=memory,
        results=results,
    )
