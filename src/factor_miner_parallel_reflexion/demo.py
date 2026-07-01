from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.factor_miner import PROJECT_ROOT, _clean_json_value, _write_json

from .candidate import CandidateJobFactory
from .constants import FAST_SCREEN_RANK_IC_THRESHOLD, MEMORY_FILENAME
from .evaluation import RoundEvaluator
from .library import FactorLibraryAdmissionService
from .memory import RlmFactorMemoryManager
from .models import CandidateJob, CandidateResult, ParallelReflexionConfig
from .portfolio import PortfolioHistoryRecorder
from .reflexion import RoundArtifactBuilder
from .utils import _candidate_ic, _result_payload


DEFAULT_DEMO_ROOT = PROJECT_ROOT / "runs" / "factor_miner_parallel_reflexion_demo"


@dataclass(frozen=True)
class NoLlmDemoConfig:
    output_dir: Path
    rounds: int = 2
    candidates: int = 6
    memory_size: int = 5
    instruments: str = "mock_csi500"
    factor_shift: int = 1

    def to_parallel_config(self) -> ParallelReflexionConfig:
        return ParallelReflexionConfig(
            output_dir=self.output_dir,
            factor_library_path=self.output_dir / "factor_library",
            instruments=self.instruments,
            start="2024-01-02",
            end="2024-01-12",
            candidates=self.candidates,
            rounds=self.rounds,
            max_workers=1,
            memory_size=self.memory_size,
            factor_shift=self.factor_shift,
            run_library_portfolio=False,
            marginal_contribution_gate=False,
            enable_rlm_logging=False,
        )


class NoLlmParallelReflexionDemo:
    """Run the parallel-reflexion pipeline with deterministic mock candidates."""

    def __init__(self, config: NoLlmDemoConfig) -> None:
        self.demo_config = config
        self.config = config.to_parallel_config()
        self.output_dir = self.config.output_dir.resolve()
        self.memory_path = self.output_dir / MEMORY_FILENAME
        self.memory_manager = RlmFactorMemoryManager(max_entries=config.memory_size)
        self.admission_service = FactorLibraryAdmissionService()
        self.job_factory = CandidateJobFactory(self.config)
        self.round_evaluator = RoundEvaluator(
            self.config,
            admission_service=self.admission_service,
        )
        self.portfolio_history_recorder = PortfolioHistoryRecorder()
        self.artifact_builder = RoundArtifactBuilder()

    def run(self) -> dict[str, Any]:
        run_started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        round_summaries: list[dict[str, Any]] = []
        for round_number in range(1, self.demo_config.rounds + 1):
            round_summaries.append(self._run_round(round_number))

        total_elapsed_seconds = time.perf_counter() - run_started
        summary = {
            "mode": "no_llm_demo",
            "output_dir": self.output_dir,
            "memory_json": self.memory_path,
            "factor_library_path": self.config.factor_library_path,
            "portfolio_history_json": self.output_dir
            / self.portfolio_history_recorder.json_name,
            "portfolio_history_csv": self.output_dir
            / self.portfolio_history_recorder.csv_name,
            "timing": {
                "total_elapsed_seconds": total_elapsed_seconds,
                "round_count": len(round_summaries),
            },
            "total_elapsed_seconds": total_elapsed_seconds,
            "memory": self.memory_manager.to_dict(),
            "rounds": round_summaries,
        }
        _write_json(self.output_dir / "summary.json", summary)
        return _clean_json_value(summary)

    def _run_round(self, round_number: int) -> dict[str, Any]:
        round_started = time.perf_counter()
        round_dir = self.output_dir / f"round_{round_number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        memory = self.memory_manager.to_dict()
        jobs = self.job_factory.make_jobs(
            round_number=round_number,
            round_dir=round_dir,
            memory_text=self.memory_manager.prompt_text(),
            memory=memory,
        )
        results = [
            self._mock_candidate_result(job, self._candidate_spec(round_number, index))
            for index, job in enumerate(jobs, start=1)
        ]
        evaluation = self.round_evaluator.evaluate(
            round_number=round_number,
            round_dir=round_dir,
            memory=memory,
            results=results,
        )
        reflexion_inputs = self.artifact_builder.write_inputs(
            round_dir=round_dir,
            results=results,
            factor_library_path=self.config.factor_library_path,
        )
        reflexions = self._mock_reflexions(round_number, evaluation)
        self.memory_manager.update(
            round_number=round_number,
            reflexions=reflexions,
            round_ic=evaluation.round_ic,
            best_signal=evaluation.best_signal,
            results=results,
            admission=evaluation.factor_library_admission,
        )
        self.memory_manager.save(self.memory_path)

        summary = {
            "round": round_number,
            "round_dir": round_dir,
            "round_trajectories_json": reflexion_inputs.trajectory_path,
            "round_economic_context_md": reflexion_inputs.final_answer_path,
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

    def _candidate_spec(self, round_number: int, candidate_index: int) -> dict[str, Any]:
        specs = {
            1: [
                {
                    "kind": "base",
                    "ic": 0.035,
                    "rank_ic": 0.041,
                    "rank_icir": 1.25,
                    "summary": "Mock momentum-liquidity interaction with positive rank IC.",
                    "ok": True,
                },
                {
                    "kind": "duplicate",
                    "ic": 0.025,
                    "rank_ic": 0.030,
                    "rank_icir": 0.95,
                    "summary": "Mock near-duplicate of the first signal.",
                    "ok": True,
                },
                {
                    "kind": "failed",
                    "ic": -0.012,
                    "rank_ic": -0.010,
                    "rank_icir": -0.30,
                    "summary": "Mock candidate with wrong direction.",
                    "ok": False,
                },
            ],
            2: [
                {
                    "kind": "replacement",
                    "ic": 0.061,
                    "rank_ic": 0.068,
                    "rank_icir": 1.80,
                    "summary": "Mock evolved version of prior best with stronger IC.",
                    "ok": True,
                },
                {
                    "kind": "orthogonal",
                    "ic": 0.033,
                    "rank_ic": 0.036,
                    "rank_icir": 1.05,
                    "summary": "Mock orthogonal value-volume signal.",
                    "ok": True,
                },
                {
                    "kind": "failed",
                    "ic": 0.0,
                    "rank_ic": 0.0,
                    "rank_icir": 0.0,
                    "summary": "Mock flat signal with no useful IC.",
                    "ok": False,
                },
            ],
        }
        round_specs = specs.get(round_number, specs[2])
        return round_specs[(candidate_index - 1) % len(round_specs)]

    def _mock_candidate_result(
        self,
        job: CandidateJob,
        spec: dict[str, Any],
    ) -> CandidateResult:
        workspace = job.workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        result = CandidateResult(
            round_number=job.round_number,
            candidate_index=job.candidate_index,
            module_name=job.module_name,
            workspace=workspace,
            rlm_summary=self._rlm_summary(job, spec),
            research_branch=job.research_branch.name,
            research_branch_goal=job.research_branch.goal,
            candidate_mode=job.candidate_mode,
            mutation_parent=job.mutation_parent,
            mutation_axis=job.mutation_axis,
        )
        trajectory_path = workspace / "rlm_trajectory.json"
        _write_json(
            trajectory_path,
            {
                "response": result.rlm_summary,
                "metadata": {
                    "status": "mocked_without_llm",
                    "iterations": [],
                    "final_answer": result.rlm_summary,
                },
                "note": "This demo never calls RLM_REPL or any LLM provider.",
            },
        )
        result.trajectory_path = trajectory_path

        if spec.get("ok", True):
            signal_path = workspace / "signals" / f"{job.module_name}.py"
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.write_text(self._signal_source(job, spec), encoding="utf-8")
            result.signal_path = signal_path
            factor_csv = workspace / "factor_data.csv"
            factor_csv.write_text(self._factor_csv(spec), encoding="utf-8")
            result.factor_data_csv = factor_csv
            result.metrics = self._metrics(spec)
            analysis_path = workspace / "analysis.json"
            _write_json(
                analysis_path,
                {
                    "mode": "mock_no_llm",
                    "metrics": {job.module_name: result.metrics},
                },
            )
            result.analysis_json = analysis_path
            result.ic, result.ic_name = _candidate_ic(metrics=result.metrics)
            result.ok = (
                result.ic is not None
                and result.ic > FAST_SCREEN_RANK_IC_THRESHOLD
            )
        else:
            result.metrics = self._metrics(spec)
            result.ic, result.ic_name = _candidate_ic(metrics=result.metrics)
            result.ok = False
            result.error_type = "MockRejectedSignal"
            result.error = "Mock candidate intentionally failed IC or direction checks."
        return result

    def _rlm_summary(self, job: CandidateJob, spec: dict[str, Any]) -> str:
        return "\n".join(
            [
                "## Hypothesis",
                str(spec["summary"]),
                "## Explanation",
                f"{job.module_name} is generated under the {job.research_branch.name} branch in {job.candidate_mode} mode.",
                f"mutation_axis: {job.mutation_axis or '<none>'}",
                "## Implementation",
                "Signal artifacts, metrics, trajectory, and factor data are mocked.",
            ]
        )

    @staticmethod
    def _signal_source(job: CandidateJob, spec: dict[str, Any]) -> str:
        return "\n".join(
            [
                "from __future__ import annotations",
                "",
                f"class {job.module_name}:",
                f"    \"\"\"Mock no-LLM signal: {spec['kind']}.\"\"\"",
                "",
                "    def run_signal(self, data):",
                "        raise RuntimeError('This mock signal is not meant to be executed.')",
                "",
            ]
        )

    def _factor_csv(self, spec: dict[str, Any]) -> str:
        rows = ["trade_time,code,score"]
        for day in range(1, 7):
            for code_index in range(1, 11):
                base = day * 0.1 + code_index
                kind = str(spec["kind"])
                if kind in {"base", "duplicate", "replacement"}:
                    score = base * (1.0 if kind != "duplicate" else 1.001)
                elif kind == "orthogonal":
                    score = ((-1) ** code_index) * code_index + day * 0.03
                else:
                    score = 0.0
                rows.append(f"2024-01-{day + 1:02d},S{code_index:03d},{score:.8f}")
        return "\n".join(rows) + "\n"

    @staticmethod
    def _metrics(spec: dict[str, Any]) -> dict[str, Any]:
        ic = float(spec["ic"])
        rank_ic = float(spec["rank_ic"])
        rank_icir = float(spec["rank_icir"])
        return {
            "coverage": 1.0,
            "missing_rate": 0.0,
            "daily_ic_mean": ic,
            "daily_ic_std": 0.02,
            "daily_rank_ic_mean": rank_ic,
            "daily_rank_ic_std": 0.02,
            "icir": ic / 0.02,
            "rank_icir": rank_icir,
            "daily_ic_count": 6,
            "daily_rank_ic_count": 6,
            "ic_distribution": {
                "count": 6,
                "mean": ic,
                "std": 0.02,
                "min": ic - 0.015,
                "p05": ic - 0.012,
                "p25": ic - 0.006,
                "median": ic,
                "p75": ic + 0.006,
                "p95": ic + 0.012,
                "max": ic + 0.015,
                "positive_rate": 1.0 if ic > 0 else 0.0,
                "negative_rate": 0.0 if ic > 0 else 1.0,
                "zero_rate": 1.0 if ic == 0 else 0.0,
            },
            "rank_ic_distribution": {
                "count": 6,
                "mean": rank_ic,
                "std": 0.02,
                "min": rank_ic - 0.015,
                "p05": rank_ic - 0.012,
                "p25": rank_ic - 0.006,
                "median": rank_ic,
                "p75": rank_ic + 0.006,
                "p95": rank_ic + 0.012,
                "max": rank_ic + 0.015,
                "positive_rate": 1.0 if rank_ic > 0 else 0.0,
                "negative_rate": 0.0 if rank_ic > 0 else 1.0,
                "zero_rate": 1.0 if rank_ic == 0 else 0.0,
            },
            "layered_ic": {
                "layer_type": "decile",
                "deciles": {
                    f"D{i}": {
                        "rows": 6,
                        "daily_ic_mean": ic + (i - 5) * 0.001,
                        "daily_rank_ic_mean": rank_ic + (i - 5) * 0.001,
                        "rank_icir": rank_icir,
                        "daily_rank_ic_count": 6,
                    }
                    for i in range(1, 11)
                },
                "diagnostics": {"joined_rows": 60, "layered_rows": 60},
            },
        }

    @staticmethod
    def _mock_reflexions(
        round_number: int,
        evaluation: Any,
    ) -> list[dict[str, Any]]:
        best_module = (
            evaluation.best_result.module_name
            if evaluation.best_result is not None
            else "none"
        )
        return [
            {
                "phase": "economic_hypothesis",
                "reflexion": (
                    "## Recommended Directions (P_succ)\n"
                    f"- Continue hypotheses like {best_module}: participation-supported continuation is worth selecting when the story explains persistence rather than a one-day shock; next round should mutate the confirmation concept, not implementation details.\n"
                    "## Forbidden Directions (P_fail)\n"
                    "- Avoid activity-alone hypotheses: high activity by itself is too broad and should only be revisited when tied to persistence, crowding unwind, or informed participation."
                ),
                "trajectory_json": None,
            },
        ]


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_DEMO_ROOT / stamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a no-LLM demo of the parallel reflexion factor miner.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--memory-size", type=int, default=5)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> NoLlmDemoConfig:
    return NoLlmDemoConfig(
        output_dir=Path(args.output_dir) if args.output_dir else default_output_dir(),
        rounds=args.rounds,
        candidates=args.candidates,
        memory_size=args.memory_size,
    )


def run_demo(config: NoLlmDemoConfig | None = None) -> dict[str, Any]:
    return NoLlmParallelReflexionDemo(config or NoLlmDemoConfig(default_output_dir())).run()


def main(argv: list[str] | None = None) -> None:
    summary = run_demo(config_from_args(parse_args(argv)))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
