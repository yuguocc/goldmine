from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import ECONOMIC_REFLEXION_PROMPT, FAST_SCREEN_RANK_IC_THRESHOLD
from .models import CandidateResult, ParallelReflexionConfig, RoundReflexionInputs
from .utils import _ensure_rlm_import_path, _write_json

class RoundArtifactBuilder:
    """Build and persist round-level inputs for reflexion agents."""

    def write_inputs(
        self,
        *,
        round_dir: Path,
        results: list[CandidateResult],
        factor_library_path: Path,
    ) -> RoundReflexionInputs:
        trajectory_context = self.trajectory_context(results=results)
        trajectory_path = round_dir / "round_trajectories.json"
        _write_json(trajectory_path, trajectory_context)

        final_answer_context = self.economic_context(
            results=results,
            factor_library_path=factor_library_path,
        )
        final_answer_path = round_dir / "round_economic_context.md"
        final_answer_path.write_text(final_answer_context, encoding="utf-8")

        return RoundReflexionInputs(
            trajectory_context=trajectory_context,
            trajectory_path=trajectory_path,
            final_answer_context=final_answer_context,
            final_answer_path=final_answer_path,
        )

    def trajectory_context(self, *, results: list[CandidateResult]) -> dict[str, Any]:
        return {
            "type": "round_labeled_trajectory_code_blocks",
            "schema_version": 2,
            "candidates": [self.trajectory_candidate_payload(result) for result in results],
        }

    def economic_context(
        self,
        *,
        results: list[CandidateResult],
        factor_library_path: Path,
    ) -> str:
        return "\n\n".join(
            [
                "# Current Factor Library Hypotheses",
                self.format_factor_library_hypotheses(
                    self.factor_library_hypotheses(factor_library_path)
                ),
                "# Current Round Labeled Final Answers",
                self.format_labeled_final_answers(
                    [
                        self.labeled_final_answer_payload(result)
                        for result in results
                    ]
                ),
                "",
            ]
        )

    @staticmethod
    def format_factor_library_hypotheses(
        hypotheses: list[dict[str, Any]],
    ) -> str:
        if not hypotheses:
            return "- No admitted factor-library hypotheses yet."
        blocks: list[str] = []
        for index, item in enumerate(hypotheses, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"## Library Factor {index}: {item.get('name', '')}",
                        f"- status: {item.get('status', '')}",
                        f"- signal_class: {item.get('signal_class', '')}",
                        "- hypothesis:",
                        str(item.get("hypothesis", "") or "<missing>").strip(),
                        "- rlm_summary:",
                        str(item.get("rlm_summary", "") or "<missing>").strip(),
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def format_labeled_final_answers(
        final_answers: list[dict[str, Any]],
    ) -> str:
        if not final_answers:
            return "- No candidate final answers."
        blocks: list[str] = []
        for item in final_answers:
            blocks.append(
                "\n".join(
                    [
                        f"## Candidate {item.get('candidate_index')}",
                        f"- label: {item.get('label')}",
                        f"- module: {item.get('module')}",
                        f"- research_branch: {item.get('research_branch')}",
                        f"- ok: {item.get('ok')}",
                        f"- ic: {item.get('ic')}",
                        f"- failure_reason: {item.get('failure_reason')}",
                        f"- failure_detail: {item.get('failure_detail')}",
                        "- final_answer:",
                        str(item.get("final_answer", "") or "<empty>").strip(),
                    ]
                )
            )
        return "\n\n".join(blocks)

    def trajectory_candidate_payload(self, result: CandidateResult) -> dict[str, Any]:
        trajectory, parse_error = self.load_candidate_trajectory(result)
        return {
            "candidate_index": result.candidate_index,
            "label": result.label,
            "module": result.module_name,
            "research_branch": result.research_branch,
            "candidate_mode": result.candidate_mode,
            "mutation_parent": result.mutation_parent,
            "mutation_axis": result.mutation_axis,
            "ok": result.ok,
            "error_type": result.error_type,
            "error": result.error,
            "code_blocks": self.extract_iteration_code_blocks(trajectory),
            "trajectory_parse_error": parse_error,
        }

    def labeled_final_answer_payload(self, result: CandidateResult) -> dict[str, Any]:
        return {
            "candidate_index": result.candidate_index,
            "label": result.label,
            "module": result.module_name,
            "research_branch": result.research_branch,
            "research_branch_goal": result.research_branch_goal,
            "candidate_mode": result.candidate_mode,
            "mutation_parent": result.mutation_parent,
            "mutation_axis": result.mutation_axis,
            "ok": result.ok,
            "ic": result.ic,
            "failure_reason": self.failure_reason(result),
            "failure_detail": self.failure_detail(result),
            "final_answer": result.rlm_summary.strip() or "<empty>",
        }

    def factor_library_hypotheses(self, library_path: Path) -> list[dict[str, Any]]:
        from quickbacktest import FactorLibrary

        library = FactorLibrary(library_path)
        hypotheses: list[dict[str, Any]] = []
        for meta in library.list_factors():
            name = str(meta.get("name", "") or "").strip()
            if not name:
                continue
            try:
                factor = library.read_factor(name)
            except Exception:
                continue
            card = str(factor.get("card", "") or "")
            summary = self.extract_rlm_summary(card)
            hypotheses.append(
                {
                    "name": name,
                    "status": str(meta.get("status", "") or ""),
                    "signal_class": str(meta.get("signal_class", "") or ""),
                    "hypothesis": self.extract_hypothesis(summary),
                    "rlm_summary": self.compact_inline(summary, limit=1200),
                }
            )
        return hypotheses

    @staticmethod
    def extract_rlm_summary(card: str) -> str:
        lines = str(card or "").splitlines()
        in_summary = False
        summary: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == "## RLM Summary":
                in_summary = True
                continue
            if in_summary and stripped.startswith("## "):
                break
            if in_summary:
                summary.append(line)
        return "\n".join(summary).strip()

    @staticmethod
    def extract_hypothesis(summary: str) -> str:
        lines = str(summary or "").splitlines()
        captured: list[str] = []
        in_hypothesis = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("hypothesis:"):
                in_hypothesis = True
                captured.append(stripped)
                continue
            if not in_hypothesis:
                continue
            if stripped and not line.startswith((" ", "\t")) and ":" in stripped:
                break
            captured.append(line.rstrip())
        return "\n".join(captured).strip()

    @classmethod
    def extract_iteration_code_blocks(cls, trajectory: Any) -> list[dict[str, Any]]:
        if not isinstance(trajectory, dict):
            return []
        metadata = trajectory.get("metadata")
        source = metadata if isinstance(metadata, dict) else trajectory
        iterations = source.get("iterations") if isinstance(source, dict) else None
        if not isinstance(iterations, list):
            return []

        blocks: list[dict[str, Any]] = []
        for iteration_index, iteration in enumerate(iterations):
            if not isinstance(iteration, dict):
                continue
            code_blocks = iteration.get("code_blocks")
            if not isinstance(code_blocks, list):
                continue
            for block_index, block in enumerate(code_blocks):
                if not isinstance(block, dict):
                    continue
                blocks.append(
                    {
                        "iteration": iteration.get("iteration", iteration_index),
                        "block_index": block_index,
                        "code": cls.compact_multiline(block.get("code"), limit=4000),
                        "stdout": cls.compact_multiline(block.get("stdout"), limit=1200),
                        "stderr": cls.compact_multiline(block.get("stderr"), limit=1200),
                        "error": cls.compact_inline(block.get("error"), limit=300),
                        "final_answer": cls.compact_inline(
                            block.get("final_answer"),
                            limit=500,
                        ),
                    }
                )
        return blocks

    @staticmethod
    def load_candidate_trajectory(result: CandidateResult) -> tuple[Any, str]:
        if result.trajectory_path is None or not result.trajectory_path.exists():
            return None, ""
        try:
            return json.loads(result.trajectory_path.read_text(encoding="utf-8")), ""
        except (OSError, json.JSONDecodeError) as exc:
            return None, str(exc)

    @staticmethod
    def format_final_answer(result: CandidateResult) -> str:
        return "\n".join(
            [
                f"# Candidate {result.candidate_index}",
                f"label: {result.label}",
                f"module: {result.module_name}",
                f"research_branch: {result.research_branch}",
                f"ok: {result.ok}",
                f"ic: {RoundArtifactBuilder.format_ic(result)}",
                f"failure_reason: {RoundArtifactBuilder.failure_reason(result)}",
                f"failure_detail: {RoundArtifactBuilder.failure_detail(result)}",
                "final_answer:",
                result.rlm_summary.strip() or "<empty>",
            ]
        )

    @staticmethod
    def format_ic(result: CandidateResult) -> str:
        if result.ic is None:
            return "None"
        return f"{result.ic:.6g}"

    @staticmethod
    def failure_reason(result: CandidateResult) -> str:
        label = str(result.label or "")
        error_type = str(result.error_type or "")
        if label == "best":
            return "none"
        if label == "duplicate":
            return "duplicate"
        if label == "average":
            return "factor_not_good"
        if label == "failed":
            if error_type in {
                "MissingIC",
                "NonPositiveIC",
                "NegativeRankICRequiresReverse",
                "LowRankIC",
                "MockRejectedSignal",
            }:
                return "factor_not_good"
            if result.error or error_type or result.traceback:
                return "error"
            if (
                result.ic is None
                or result.ic <= FAST_SCREEN_RANK_IC_THRESHOLD
            ):
                return "factor_not_good"
            return "unknown_failure"
        return "none" if result.ok else "unknown_failure"

    @staticmethod
    def failure_detail(result: CandidateResult) -> str:
        label = str(result.label or "")
        if label == "best":
            return "round best candidate"
        if label == "average":
            return "valid factor, but weaker than the round best"
        if label == "duplicate":
            detail = f"duplicate of {result.dedup_of or '<unknown>'}"
            if result.dedup_correlation is not None:
                detail += f"; correlation={result.dedup_correlation:.6g}"
            return detail
        if result.error_type or result.error:
            return RoundArtifactBuilder.compact_inline(
                f"{result.error_type}: {result.error}".strip(": ")
            )
        if result.ic is None:
            return "no finite rank IC"
        if result.ic <= FAST_SCREEN_RANK_IC_THRESHOLD:
            return f"rank IC <= {FAST_SCREEN_RANK_IC_THRESHOLD:g}"
        return "not applicable"

    @staticmethod
    def compact_multiline(text: Any, *, limit: int) -> str:
        value = "" if text is None else str(text)
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 18)].rstrip() + "\n... [truncated]"

    @staticmethod
    def compact_inline(text: Any, *, limit: int = 240) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 15].rstrip() + " ... [truncated]"

def _load_candidate_trajectory(result: CandidateResult) -> tuple[Any, str]:
    return RoundArtifactBuilder.load_candidate_trajectory(result)


def _trajectory_candidate_payload(result: CandidateResult) -> dict[str, Any]:
    return RoundArtifactBuilder().trajectory_candidate_payload(result)


def _format_final_answer_for_reflexion(result: CandidateResult) -> str:
    return RoundArtifactBuilder.format_final_answer(result)

class RoundReflexionAgent:
    """Run enabled reflexion phases for a completed round."""

    def __init__(self, config: ParallelReflexionConfig) -> None:
        self.config = config

    def run_round(
        self,
        *,
        round_number: int,
        round_dir: Path,
        inputs: RoundReflexionInputs,
    ) -> list[dict[str, Any]]:
        reflexion_specs = [
            (
                "economic_hypothesis",
                self.economic_prompt(),
                inputs.final_answer_context,
            ),
        ]
        reflexions: list[dict[str, Any]] = []
        for phase, prompt, context in reflexion_specs:
            reflexion, trajectory_json = self.run_phase(
                phase=phase,
                prompt=prompt,
                round_number=round_number,
                round_dir=round_dir,
                context=context,
            )
            reflexions.append(
                {
                    "phase": phase,
                    "reflexion": reflexion,
                    "trajectory_json": trajectory_json,
                }
            )
        return reflexions

    def run_phase(
        self,
        *,
        phase: str,
        prompt: str,
        round_number: int,
        round_dir: Path,
        context: Any,
    ) -> tuple[str, str]:
        _ensure_rlm_import_path()
        from rlm.rlm_repl import RLM_REPL

        rlm = RLM_REPL(
            model=self.config.model,
            recursive_model=self.config.recursive_model,
            max_iterations=max(3, min(self.config.max_iterations, 8)),
            enable_logging=self.config.enable_rlm_logging,
        )
        query = self.query(prompt)
        result = rlm.completion(context=context, query=query)
        trajectory_path = round_dir / f"{phase}_reflexion_trajectory.json"
        _write_json(
            trajectory_path,
            {
                "response": result.response,
                "metadata": result.metadata,
                "phase": phase,
                "round": round_number,
                "prompt": prompt,
                "query": query,
            },
        )
        return str(result.response).strip(), str(trajectory_path)

    @staticmethod
    def query(prompt: str) -> str:
        return prompt

    def economic_prompt(self) -> str:
        return (
            f"{ECONOMIC_REFLEXION_PROMPT}\n"
            f"candidate_count_observed: {self.config.candidates}\n"
            "Do not allocate one hypothesis per candidate. Extract reusable "
            "P_succ/P_fail memory priors for retrieval-driven exploration."
        )


def _reflexion_query(prompt: str) -> str:
    return RoundReflexionAgent.query(prompt)


def _run_reflexion(
    *,
    phase: str,
    prompt: str,
    round_number: int,
    round_dir: Path,
    context: Any,
    config: ParallelReflexionConfig,
) -> tuple[str, str]:
    return RoundReflexionAgent(config).run_phase(
        phase=phase,
        prompt=prompt,
        round_number=round_number,
        round_dir=round_dir,
        context=context,
    )


def _run_round_reflexions(
    *,
    round_number: int,
    round_dir: Path,
    inputs: RoundReflexionInputs,
    config: ParallelReflexionConfig,
) -> list[dict[str, Any]]:
    """Run the enabled reflexion phases for one mining round."""
    return RoundReflexionAgent(config).run_round(
        round_number=round_number,
        round_dir=round_dir,
        inputs=inputs,
    )
