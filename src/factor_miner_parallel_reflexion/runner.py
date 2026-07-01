from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.factor_miner import _clean_json_value, _write_json

from .candidate import CandidateBatchExecutor, CandidateJobFactory
from .constants import DEFAULT_FACTOR_LIBRARY_PATH, DEFAULT_OUTPUT_DIR, MEMORY_FILENAME
from .evaluation import RoundEvaluator
from .library import FactorLibraryAdmissionService
from .memory import RlmFactorMemoryManager
from .models import ParallelReflexionConfig
from .portfolio import FactorLibraryPortfolioService, PortfolioHistoryRecorder
from .reflexion import RoundArtifactBuilder, RoundReflexionAgent
from .utils import _result_payload, _validate_config

class ParallelReflexionRunner:
    """Stateful runner for the RLM Ralph-style factor mining loop."""

    def __init__(self, config: ParallelReflexionConfig) -> None:
        self.config = self._with_run_local_factor_library(config)
        _validate_config(self.config)
        self.output_dir = self.config.output_dir.resolve()
        self.memory_path = self.output_dir / MEMORY_FILENAME
        self.memory_manager: RlmFactorMemoryManager | None = None
        self.admission_service = FactorLibraryAdmissionService()
        self.job_factory = CandidateJobFactory(self.config)
        self.batch_executor = CandidateBatchExecutor(max_workers=self.config.max_workers)
        self.round_evaluator = RoundEvaluator(
            self.config,
            admission_service=self.admission_service,
        )
        self.portfolio_service = FactorLibraryPortfolioService()
        self.portfolio_history_recorder = PortfolioHistoryRecorder()
        self.artifact_builder = RoundArtifactBuilder()
        self.reflexion_agent = RoundReflexionAgent(self.config)

    @staticmethod
    def _with_run_local_factor_library(
        config: ParallelReflexionConfig,
    ) -> ParallelReflexionConfig:
        output_dir = config.output_dir.resolve()
        configured_library = Path(config.factor_library_path)
        default_library_paths = {
            DEFAULT_FACTOR_LIBRARY_PATH.resolve(),
            (DEFAULT_OUTPUT_DIR / "factor_library").resolve(),
        }
        library_path = configured_library.resolve()
        if library_path in default_library_paths:
            library_path = output_dir / "factor_library"
        return replace(
            config,
            output_dir=output_dir,
            factor_library_path=library_path,
        )

    def run(self) -> dict[str, Any]:
        run_started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_manager = RlmFactorMemoryManager.load(
            self.memory_path,
            max_entries=self.config.memory_size,
        )
        round_summaries: list[dict[str, Any]] = []
        start_round = self.memory_manager.latest_round + 1
        end_round = start_round + self.config.rounds
        for round_number in range(start_round, end_round):
            round_summaries.append(
                self._run_round(
                    round_number,
                    run_reflexion=self._should_run_reflexion(
                        round_number,
                        end_round,
                    ),
                )
            )

        final_oos = self.portfolio_service.run_final_oos(
            config=self.config,
            output_dir=self.output_dir / "final_oos",
        )
        portfolio_history_plots = self._plot_portfolio_history()
        total_elapsed_seconds = time.perf_counter() - run_started
        final_summary = {
            "output_dir": self.output_dir,
            "factor_library_path": self.config.factor_library_path,
            "memory_json": self.memory_path,
            "portfolio_history_json": self.output_dir
            / self.portfolio_history_recorder.json_name,
            "portfolio_history_csv": self.output_dir
            / self.portfolio_history_recorder.csv_name,
            "timing": {
                "total_elapsed_seconds": total_elapsed_seconds,
                "round_count": len(round_summaries),
            },
            "total_elapsed_seconds": total_elapsed_seconds,
            "final_oos": final_oos,
            "portfolio_history_plots": portfolio_history_plots,
            "memory": self.memory_manager.to_dict(),
            "rounds": round_summaries,
        }
        _write_json(self.output_dir / "summary.json", final_summary)
        return _clean_json_value(final_summary)

    def _plot_portfolio_history(self) -> dict[str, Any]:
        history_path = self.output_dir / self.portfolio_history_recorder.json_name
        if not history_path.exists():
            return {
                "status": "skipped",
                "reason": "portfolio_history.json missing",
                "history_json": str(history_path),
            }
        try:
            from scripts.plot_portfolio_history import plot_portfolio_history

            result = plot_portfolio_history(
                history_path=history_path,
                output_dir=self.output_dir,
            )
            return {"status": "completed", **result}
        except Exception as exc:
            return {
                "status": "failed",
                "history_json": str(history_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def _run_round(
        self,
        round_number: int,
        *,
        run_reflexion: bool = True,
    ) -> dict[str, Any]:
        round_started = time.perf_counter()
        memory_manager = self._memory_manager()
        round_dir = self.output_dir / f"round_{round_number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        memory = memory_manager.to_dict()

        jobs = self.job_factory.make_jobs(
            round_number=round_number,
            round_dir=round_dir,
            memory_text=memory_manager.prompt_text(),
            memory=memory,
        )
        results = self.batch_executor.run(jobs)
        evaluation = self.round_evaluator.evaluate(
            round_number=round_number,
            round_dir=round_dir,
            memory=memory,
            results=results,
        )
        reflexion_inputs = None
        reflexions: list[dict[str, Any]] = []
        if run_reflexion:
            reflexion_inputs = self.artifact_builder.write_inputs(
                round_dir=round_dir,
                results=results,
                factor_library_path=self.config.factor_library_path,
            )
            reflexions = self.reflexion_agent.run_round(
                round_number=round_number,
                round_dir=round_dir,
                inputs=reflexion_inputs,
            )
        memory_manager.update(
            round_number=round_number,
            reflexions=reflexions,
            round_ic=evaluation.round_ic,
            best_signal=evaluation.best_signal,
            results=results,
            admission=evaluation.factor_library_admission,
        )
        memory_manager.save(self.memory_path)

        summary = {
            "round": round_number,
            "round_dir": round_dir,
            "memory_json": self.memory_path,
            "round_trajectories_json": (
                reflexion_inputs.trajectory_path
                if reflexion_inputs is not None
                else None
            ),
            "round_economic_context_md": (
                reflexion_inputs.final_answer_path
                if reflexion_inputs is not None
                else None
            ),
            "reflexion_skipped": not run_reflexion,
            "reflexion_skip_reason": "final_round" if not run_reflexion else "",
            "round_ic": evaluation.round_ic,
            "batch_dedup": evaluation.batch_dedup,
            "best_signal": evaluation.best_signal,
            "factor_library_admission": evaluation.factor_library_admission,
            "factor_library_portfolio": evaluation.factor_library_portfolio,
            "reflexions": reflexions,
            "best": (
                _result_payload(evaluation.best_result)
                if evaluation.best_result is not None
                else None
            ),
            "admitted": (
                _result_payload(evaluation.admitted_result)
                if evaluation.admitted_result is not None
                else None
            ),
            "admitted_factors": [
                _result_payload(result) for result in evaluation.admitted_results
            ],
            "candidates": [_result_payload(result) for result in results],
        }
        summary["portfolio_history"] = self.portfolio_history_recorder.record_round(
            output_dir=self.output_dir,
            round_summary=summary,
        )
        round_elapsed_seconds = time.perf_counter() - round_started
        summary["timing"] = {"round_elapsed_seconds": round_elapsed_seconds}
        summary["round_elapsed_seconds"] = round_elapsed_seconds
        _write_json(round_dir / "round_summary.json", summary)
        return _clean_json_value(summary)

    @staticmethod
    def _should_run_reflexion(round_number: int, end_round: int) -> bool:
        return round_number < end_round - 1

    def _memory_manager(self) -> RlmFactorMemoryManager:
        if self.memory_manager is None:
            raise RuntimeError("memory manager is not initialized; call run() first")
        return self.memory_manager

def run_parallel_reflexion(config: ParallelReflexionConfig) -> dict[str, Any]:
    return ParallelReflexionRunner(config).run()
