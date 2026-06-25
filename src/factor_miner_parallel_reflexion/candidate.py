from __future__ import annotations

import json
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from factor_miner import (
    FactorMinerCaseConfig,
    _factor_generation_context,
    _import_qlib_adapter,
    build_factor_miner_custom_tools,
    build_factor_miner_final_answer_validator,
    compute_examples_prompt_text,
    compute_signal_analysis,
    initialize_qlib_for_factor_miner,
)

from .branches import branch_for_candidate
from .constants import (
    DEFAULT_SKILL_PATH,
    FAST_SCREEN_RANK_IC_THRESHOLD,
    RLM_SUMMARY_SPEC,
)
from .models import (
    CandidateJob,
    CandidateResult,
    ParallelReflexionConfig,
    ResearchBranch,
)
from .scheduler import CandidateAssignment, ResearchDirectionScheduler
from .utils import (
    _candidate_ic,
    _finite_float,
    _ensure_rlm_import_path,
    _result_payload,
    _write_candidate_result,
    _write_json,
)

NOVELTY_CANDIDATE_SLOTS = 3
MUTATION_CANDIDATE_SLOTS = 3
MUTATION_AXES: tuple[str, ...] = (
    "replace_gate",
    "change_normalization",
    "add_interaction",
    "change_horizon_family",
)


class MissingRunSignalConfirmation(RuntimeError):
    """Raised when RLM finishes without submitting a runnable signal."""


class RunSignalConfirmationFailed(RuntimeError):
    """Raised when RLM submits a signal but leaves it failed or invalid."""


class CandidatePromptBuilder:
    """Build all RLM-facing prompt and context payloads for one candidate."""

    max_existing_factors = 20
    max_existing_factor_summary_chars = 600

    def query(
        self,
        module_name: str,
        memory_text: str = "",
        *,
        candidate_index: int = 1,
        candidate_count: int = 1,
        research_branch: ResearchBranch | None = None,
        candidate_mode: str = "novelty",
        mutation_parent: dict[str, Any] | None = None,
        mutation_axis: str = "",
    ) -> str:
        # Compatibility shim: memory is now supplied through context["memory_priors"].
        _ = memory_text
        branch_text = self.research_assignment_text(research_branch)
        mode_text = self.candidate_mode_text(
            candidate_mode=candidate_mode,
            mutation_parent=mutation_parent,
            mutation_axis=mutation_axis,
        )
        return (
            "Task: write only the Python body of BaseSignal.compute(**kwargs). "
            "Do not write imports, class definitions, def compute, markdown fences, "
            "signal names, or return statements. Assign the final wide DataFrame "
            "to variable signal; the wrapper appends return signal. Target positive "
            "daily rank IC using the template/examples below and the Factorminer "
            "memory priors.\n\n"
            f"Candidate class name: {module_name}\n"
            f"Candidate index: {candidate_index} of {candidate_count}\n\n"
            f"{branch_text}\n\n"
            f"{mode_text}\n\n"
            f"{compute_examples_prompt_text()}\n\n"
            "Flow:\n"
            "1. Read context.available_data_fields and context.available_libraries.\n"
            "2. Read context.candidate_mode. If it is mutation, read "
            "context.mutation_parent and context.mutation_axis first; preserve "
            "the parent's broad economic family but change the mechanism only "
            "on the assigned mutation axis.\n"
            "3. Read context.existing_factors as already admitted factor-library "
            "ideas. Avoid duplicating their hypotheses, operators, horizons, "
            "conditioning gates, field combinations, or simple sign-flipped variants.\n"
            "4. Before writing compute_code, silently run a novelty check against "
            "all Existing Factors: identify the closest existing idea, reject the "
            "candidate if it is mainly the same hypothesis, fields, operator stack, "
            "horizon, gate, sign flip, or parameter-only variant, then choose a "
            "materially different mechanism inside the assigned branch.\n"
            "5. Treat Research Assignment as the primary style constraint. "
            "Read context.memory_priors as secondary guidance: prefer Recommended Directions (P_succ), "
            "avoid Forbidden Directions (P_fail), and use Strategic Insights (I). "
            "If Memory priors conflict with the assigned branch, stay inside the "
            "branch but choose a safer mechanism.\n"
            "6. Put only compute-body code into compute_code; the last effective "
            "result must be assigned to signal.\n"
            "7. Call submit_compute(compute_code); it wraps, saves, and runs the real Qlib path.\n"
            "8. If submit_compute returns ok=False, fix execution/contract errors "
            "and submit again.\n"
            "9. If error_type is NegativeRankICRequiresReverse, reverse the final "
            "output by multiplying signal by -1, then submit_compute(compute_code) "
            "again.\n"
            "10. Final answer is allowed only after submit_compute returns ok=True. "
            "A high-IC candidate is still invalid if it duplicates an Existing Factor.\n\n"
            "Final answer must follow:\n"
            f"{RLM_SUMMARY_SPEC}\n"
            "Novelty evidence required inside the same Markdown fields: mention "
            "the closest Existing Factor considered, why this candidate is not a "
            "duplicate, the key new field interaction/operator/horizon/regime, "
            "and the assigned branch. For mutation mode, also mention the parent "
            "factor, mutation axis, preserved hypothesis, changed mechanism, and "
            "why it should be less duplicate/correlated than the parent. Do not "
            "add numeric metrics."
        )

    @staticmethod
    def research_assignment_text(branch: ResearchBranch | None) -> str:
        if branch is None:
            return (
                "Research Assignment:\n"
                "- branch: General price-volume factor\n"
                "- goal: Generate one runnable factor with positive rank IC."
            )
        lines = [
            "Research Assignment:",
            f"- branch: {branch.name}",
            f"- goal: {branch.goal}",
        ]
        if branch.must_use:
            lines.append("- must_use: " + "; ".join(branch.must_use))
        if branch.must_avoid:
            lines.append("- must_avoid: " + "; ".join(branch.must_avoid))
        if branch.examples:
            lines.append("- example_mechanisms: " + "; ".join(branch.examples))
        lines.append(
            "- final_summary: include this branch name, closest Existing Factor considered, and the material non-duplicate difference."
        )
        return "\n".join(lines)

    @staticmethod
    def candidate_mode_text(
        *,
        candidate_mode: str,
        mutation_parent: dict[str, Any] | None,
        mutation_axis: str,
    ) -> str:
        mode = str(candidate_mode or "novelty").strip() or "novelty"
        if mode != "mutation":
            return (
                "Candidate Mode:\n"
                "- mode: novelty\n"
                "- contract: generate a materially new factor inside the assigned branch."
            )
        parent_name = str((mutation_parent or {}).get("name", "") or "<missing>")
        parent_summary = str((mutation_parent or {}).get("summary", "") or "")
        parent_space = str(
            (mutation_parent or {}).get("occupied_idea_space", "") or parent_summary
        )
        return "\n".join(
            [
                "Candidate Mode:",
                "- mode: mutation",
                f"- parent_factor: {parent_name}",
                f"- mutation_axis: {mutation_axis or '<missing>'}",
                "- contract: preserve the parent's broad economic family, but change the signal mechanism on the assigned mutation axis.",
                "- forbidden: copying the parent formula, simple sign flip, parameter-only window/threshold change, or cosmetic rewrite.",
                "- parent_occupied_idea_space: "
                + CandidatePromptBuilder.compact_text(parent_space, limit=500),
            ]
        )

    def context(self, job: CandidateJob) -> dict[str, Any]:
        context = _factor_generation_context(job.case_config)
        context["parallel_generation"] = (
            f"factor_class_name: {job.module_name}; "
            f"candidate_index: {job.candidate_index}; "
            f"candidate_count: {job.candidate_count}; "
            f"candidate_mode: {job.candidate_mode}; "
            f"research_branch: {job.research_branch.name}; "
            "choose a materially novel structural variant inside the assigned "
            "branch; do not switch to a common high-score pattern outside this "
            "branch; do not use sign flips or parameter-only changes as novelty"
        )
        context["memory_priors"] = job.memory_text
        context["candidate_mode"] = job.candidate_mode
        context["mutation_parent"] = job.mutation_parent
        context["mutation_axis"] = job.mutation_axis
        context["mutation_policy"] = (
            "Mutation candidates must preserve the parent factor's broad "
            "economic family while changing exactly one structural axis. Valid "
            "axes are replace_gate, change_normalization, add_interaction, and "
            "change_horizon_family. Mutation must not copy the parent formula, "
            "flip the sign, or only tweak windows/thresholds."
        )
        context["existing_factors"] = self.existing_factors_context(
            job.factor_library_path
        )
        context["duplicate_avoidance_policy"] = (
            "Treat context.existing_factors as Existing Factors already admitted "
            "to the factor library. Do not regenerate accepted ideas. Before "
            "coding, compare the candidate idea with every Existing Factor and "
            "discard it if it mainly repeats the same economic hypothesis, "
            "input-field combination, operator or transform stack, lookback "
            "horizon, volume/volatility gate, normalization, field interaction, "
            "simple sign flip, or trivial window/threshold-only variant. If the "
            "assigned branch overlaps an existing factor, make a materially "
            "different structural change such as a new field interaction, "
            "conditioning regime, time scale with a clear reason, or "
            "normalization/comparison logic. Novelty is required before "
            "performance; a high-IC duplicate is invalid. Explain the closest "
            "Existing Factor and the non-duplicate mechanism in the final summary."
        )
        return context

    def existing_factors_context(self, library_path: Path) -> dict[str, Any]:
        from quickbacktest import FactorLibrary

        library = FactorLibrary(library_path)
        items: list[dict[str, Any]] = []
        for meta in library.list_factors():
            if meta.get("status") != "accepted":
                continue
            name = str(meta.get("name", "") or "").strip()
            if not name:
                continue
            try:
                factor = library.read_factor(name)
            except Exception:
                factor = {}
            metrics = factor.get("metrics") if isinstance(factor, dict) else {}
            rank_ic, rank_ic_name = _candidate_ic(
                metrics=metrics if isinstance(metrics, dict) else None
            )
            card = str(factor.get("card", "") if isinstance(factor, dict) else "")
            summary = self.extract_rlm_summary(card)
            if not summary:
                summary = str(meta.get("description", "") or "")
            description = self.compact_text(meta.get("description", ""))
            compact_summary = self.compact_text(
                summary,
                limit=self.max_existing_factor_summary_chars,
            )
            occupied_parts = []
            if description:
                occupied_parts.append(f"description={description}")
            if compact_summary:
                occupied_parts.append(f"summary={compact_summary}")
            items.append(
                {
                    "name": name,
                    "status": "accepted",
                    "signal_class": str(meta.get("signal_class", "") or ""),
                    "description": description,
                    "rank_ic": rank_ic,
                    "rank_ic_name": rank_ic_name,
                    "summary": compact_summary,
                    "occupied_idea_space": self.compact_text(
                        "; ".join(occupied_parts),
                        limit=self.max_existing_factor_summary_chars,
                    ),
                    "do_not_duplicate": (
                        "Do not recreate this hypothesis, its sign flip, or a "
                        "trivial parameter/window/threshold variant."
                    ),
                }
            )

        items.sort(
            key=lambda item: (
                item["rank_ic"] is not None,
                float(item["rank_ic"] or float("-inf")),
            ),
            reverse=True,
        )
        total_count = len(items)
        truncated = total_count > self.max_existing_factors
        items = items[: self.max_existing_factors]
        return {
            "label": "Existing Factors",
            "instruction": (
                "These accepted factor-library entries already exist. Avoid "
                "duplicating their hypotheses, operators, horizons, gates, and "
                "simple sign-flipped variants. Treat each occupied_idea_space as "
                "claimed territory; choose a materially different mechanism before "
                "writing compute_code."
            ),
            "count": total_count,
            "returned_count": len(items),
            "truncated": truncated,
            "items": items,
        }

    @staticmethod
    def extract_rlm_summary(card: str) -> str:
        in_summary = False
        lines: list[str] = []
        for line in str(card or "").splitlines():
            stripped = line.strip()
            if stripped == "## RLM Summary":
                in_summary = True
                continue
            if in_summary and stripped.startswith("## "):
                break
            if in_summary:
                lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def compact_text(value: Any, *, limit: int = 240) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def skills_manifest(self) -> list[dict[str, str]]:
        from quickbacktest import SkillsLibrary

        return [
            {
                "name": str(skill.get("name", "")),
                "description": str(skill.get("description", "")),
            }
            for skill in SkillsLibrary(DEFAULT_SKILL_PATH).manifest()
            if str(skill.get("name", "")).strip()
        ]


class CandidateJobFactory:
    """Create isolated ProcessPool candidate jobs for a round."""

    def __init__(self, config: ParallelReflexionConfig) -> None:
        self.config = config

    def make_jobs(
        self,
        *,
        round_number: int,
        round_dir: Path,
        memory_text: str,
        memory: dict[str, Any] | None = None,
    ) -> list[CandidateJob]:
        jobs: list[CandidateJob] = []
        mutation_parents = self.mutation_parent_pool()
        assignments = ResearchDirectionScheduler().allocate(
            round_number=round_number,
            candidate_count=self.config.candidates,
            mutation_parents=mutation_parents,
            mutation_axes=MUTATION_AXES,
            memory=memory,
        )
        if not assignments:
            assignments = self.deterministic_assignments(
                round_number=round_number,
                candidate_count=self.config.candidates,
                mutation_parents=mutation_parents,
            )

        for index, assignment in enumerate(assignments, start=1):
            module_name = f"RlmGeneratedFactorR{round_number:03d}C{index:03d}"
            workspace = CandidateWorkspaceFactory(round_dir).create(
                prefix=f"candidate_{index:03d}_"
            )
            jobs.append(
                CandidateJob(
                    round_number=round_number,
                    candidate_index=index,
                    candidate_count=self.config.candidates,
                    module_name=module_name,
                    workspace=workspace,
                    case_config=self.make_case_config(
                        workspace=workspace,
                        module_name=module_name,
                    ),
                    factor_library_path=self.config.factor_library_path,
                    memory_text=memory_text,
                    research_branch=assignment.research_branch,
                    model=self.config.model,
                    recursive_model=self.config.recursive_model,
                    max_iterations=self.config.max_iterations,
                    enable_rlm_logging=self.config.enable_rlm_logging,
                    candidate_mode=assignment.candidate_mode,
                    mutation_parent=assignment.mutation_parent,
                    mutation_axis=assignment.mutation_axis,
                )
            )
        return jobs

    def deterministic_assignments(
        self,
        *,
        round_number: int,
        candidate_count: int,
        mutation_parents: list[dict[str, Any]],
    ) -> list[CandidateAssignment]:
        assignments: list[CandidateAssignment] = []
        mutation_slots = self.mutation_slots(
            candidate_count=candidate_count,
            parent_count=len(mutation_parents),
        )
        for index in range(1, int(candidate_count) + 1):
            research_branch = branch_for_candidate(round_number, index)
            candidate_mode = self.candidate_mode(index, mutation_slots=mutation_slots)
            mutation_parent = None
            mutation_axis = ""
            if candidate_mode == "mutation":
                mutation_slot = index - NOVELTY_CANDIDATE_SLOTS
                mutation_parent = self.select_mutation_parent(
                    mutation_parents,
                    round_number=round_number,
                    mutation_slot=mutation_slot,
                )
                mutation_axis = self.select_mutation_axis(
                    round_number=round_number,
                    mutation_slot=mutation_slot,
                )
            assignments.append(
                CandidateAssignment(
                    research_branch=research_branch,
                    candidate_mode=candidate_mode,
                    mutation_parent=mutation_parent,
                    mutation_axis=mutation_axis,
                )
            )
        return assignments

    @staticmethod
    def mutation_slots(*, candidate_count: int, parent_count: int) -> int:
        if parent_count <= 0:
            return 0
        return max(
            0,
            min(
                MUTATION_CANDIDATE_SLOTS,
                max(0, int(candidate_count) - NOVELTY_CANDIDATE_SLOTS),
            ),
        )

    @staticmethod
    def candidate_mode(index: int, *, mutation_slots: int) -> str:
        mutation_start = NOVELTY_CANDIDATE_SLOTS + 1
        mutation_end = NOVELTY_CANDIDATE_SLOTS + max(0, int(mutation_slots))
        if mutation_start <= int(index) <= mutation_end:
            return "mutation"
        return "novelty"

    def mutation_parent_pool(self) -> list[dict[str, Any]]:
        existing = CandidatePromptBuilder().existing_factors_context(
            self.config.factor_library_path
        )
        items = existing.get("items") if isinstance(existing, dict) else None
        if not isinstance(items, list):
            return []
        parents: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            parent = {
                "name": item.get("name", ""),
                "signal_class": item.get("signal_class", ""),
                "description": item.get("description", ""),
                "summary": item.get("summary", ""),
                "occupied_idea_space": item.get("occupied_idea_space", ""),
                "rank_ic": item.get("rank_ic"),
                "rank_ic_name": item.get("rank_ic_name", ""),
            }
            parents.append(parent)
        return sorted(
            parents,
            key=lambda item: _finite_float(item.get("rank_ic")) or float("-inf"),
            reverse=True,
        )

    @staticmethod
    def select_mutation_parent(
        parents: list[dict[str, Any]],
        *,
        round_number: int,
        mutation_slot: int,
    ) -> dict[str, Any] | None:
        if not parents:
            return None
        index = (
            max(0, int(round_number) - 1)
            + max(1, int(mutation_slot))
            - 1
        ) % len(parents)
        return dict(parents[index])

    @staticmethod
    def select_mutation_axis(*, round_number: int, mutation_slot: int) -> str:
        index = (
            max(0, int(round_number) - 1)
            + max(1, int(mutation_slot))
            - 1
        ) % len(MUTATION_AXES)
        return MUTATION_AXES[index]

    def make_case_config(
        self,
        *,
        workspace: Path,
        module_name: str,
    ) -> FactorMinerCaseConfig:
        return FactorMinerCaseConfig(
            workspace=workspace,
            provider_uri=self.config.provider_uri,
            module_name=module_name,
            instruments=self.config.instruments,
            start=self.config.start,
            end=self.config.end,
            benchmark=self.config.benchmark,
            topk=self.config.topk,
            n_drop=self.config.n_drop,
            horizon=self.config.horizon,
            factor_shift=self.config.factor_shift,
            run_portfolio=self.config.run_portfolio,
            train_alpha158=False,
            use_case_signal=False,
            model=self.config.model,
            recursive_model=self.config.recursive_model,
            max_iterations=self.config.max_iterations,
            enable_rlm_logging=self.config.enable_rlm_logging,
        )

class CandidateWorkspaceFactory:
    """Create writable unique candidate workspaces under a round directory."""

    def __init__(self, round_dir: Path) -> None:
        self.round_dir = round_dir

    def create(self, *, prefix: str) -> Path:
        self.round_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            path = self._reserved_path(prefix=prefix)
            try:
                path.mkdir(parents=False, exist_ok=False)
                self._assert_writable(path)
                return path
            except FileExistsError:
                continue
        raise RuntimeError("could not create a unique writable candidate workspace")

    def _reserved_path(self, *, prefix: str) -> Path:
        return Path(tempfile.mktemp(prefix=prefix, dir=str(self.round_dir)))

    @staticmethod
    def _assert_writable(path: Path) -> None:
        probe = path / ".workspace_ready"
        probe.write_text("ok", encoding="utf-8")


def _candidate_query(module_name: str, memory_text: str = "") -> str:
    return CandidatePromptBuilder().query(module_name)


def _skills_manifest_for_context() -> list[dict[str, str]]:
    return CandidatePromptBuilder().skills_manifest()


def _candidate_context(job: CandidateJob) -> dict[str, Any]:
    return CandidatePromptBuilder().context(job)


class CandidateJobRunner:
    """Run one candidate generation, materialization, and IC extraction."""

    def __init__(
        self,
        job: CandidateJob,
        *,
        prompt_builder: CandidatePromptBuilder | None = None,
    ) -> None:
        self.job = job
        self.prompt_builder = prompt_builder or CandidatePromptBuilder()
        self.workspace = job.workspace.resolve()
        self.result = CandidateResult(
            round_number=job.round_number,
            candidate_index=job.candidate_index,
            module_name=job.module_name,
            workspace=self.workspace,
            research_branch=job.research_branch.name,
            research_branch_goal=job.research_branch.goal,
            candidate_mode=job.candidate_mode,
            mutation_parent=job.mutation_parent,
            mutation_axis=job.mutation_axis,
        )
        self.factor_df: Any | None = None

    def run(self) -> CandidateResult:
        _ensure_rlm_import_path()
        self.workspace.mkdir(parents=True, exist_ok=True)
        try:
            self._run_rlm()
            self._require_signal_source()
            self._require_run_signal_confirmation()
            self._compute_signal_analysis()
            self._maybe_run_portfolio()
            self._mark_ic_status()
        except Exception as exc:
            self.result.ok = False
            self.result.error_type = type(exc).__name__
            self.result.error = str(exc)
            self.result.traceback = traceback.format_exc()
            _write_json(
                self.workspace / "candidate_error.json",
                _result_payload(self.result),
            )
        finally:
            _write_candidate_result(self.result)
        return self.result

    def _run_rlm(self) -> None:
        initialize_qlib_for_factor_miner(self.job.case_config)
        from rlm.rlm_repl import RLM_REPL

        rlm = RLM_REPL(
            model=self.job.model,
            recursive_model=self.job.recursive_model,
            max_iterations=self.job.max_iterations,
            enable_logging=self.job.enable_rlm_logging,
            custom_tools=self._build_tools(),
            final_answer_validator=build_factor_miner_final_answer_validator(
                self.workspace,
                self.job.module_name,
            ),
        )
        started = time.perf_counter()
        rlm_result = rlm.completion(
            context=self.prompt_builder.context(self.job),
            query=self.prompt_builder.query(
                self.job.module_name,
                candidate_index=self.job.candidate_index,
                candidate_count=self.job.candidate_count,
                research_branch=self.job.research_branch,
                candidate_mode=self.job.candidate_mode,
                mutation_parent=self.job.mutation_parent,
                mutation_axis=self.job.mutation_axis,
            ),
        )
        elapsed = time.perf_counter() - started
        self.result.rlm_summary = str(rlm_result.response)
        trajectory_path = self.workspace / "rlm_trajectory.json"
        _write_json(
            trajectory_path,
            {
                "response": rlm_result.response,
                "metadata": rlm_result.metadata,
                "rlm_factor_generation_elapsed_seconds": elapsed,
                "rlm_summary_spec": RLM_SUMMARY_SPEC,
            },
        )
        self.result.trajectory_path = trajectory_path

    def _build_tools(self) -> dict[str, Any]:
        from quickbacktest import build_rlm_skill_tools

        tools = build_rlm_skill_tools(DEFAULT_SKILL_PATH, allow_write=False)
        tools.pop("list_skills", None)
        tools.update(
            build_factor_miner_custom_tools(self.workspace, config=self.job.case_config)
        )
        tools.pop("read_signal_template", None)
        return tools

    def _require_signal_source(self) -> None:
        signal_path = self.workspace / "signals" / f"{self.job.module_name}.py"
        self.result.signal_path = signal_path
        if not signal_path.exists():
            raise RuntimeError(
                f"RLM did not save {signal_path}. It must call submit_compute(compute_code)."
            )

    def _require_run_signal_confirmation(self) -> None:
        status_path = self.workspace / "run_signal_status.json"
        if not status_path.exists():
            raise MissingRunSignalConfirmation(
                "RLM must call submit_compute(compute_code) and receive ok=True before final answer."
            )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunSignalConfirmationFailed(
                f"Cannot read run_signal confirmation: {exc}"
            ) from exc
        if not isinstance(status, dict):
            raise RunSignalConfirmationFailed(
                "submit_compute confirmation must be a JSON object."
            )
        module_name = str(status.get("module_name", "") or "")
        if module_name and module_name != self.job.module_name:
            raise RunSignalConfirmationFailed(
                "submit_compute confirmed a different module: "
                f"{module_name} != {self.job.module_name}."
            )
        if status.get("ok") is not True:
            error_type = str(status.get("error_type", "") or "UnknownError")
            error = str(status.get("error", "") or "submit_compute returned ok=False")
            raise RunSignalConfirmationFailed(f"{error_type}: {error}")

    def _compute_signal_analysis(self) -> None:
        # Tool success only means a signal module exists. The candidate is not
        # considered usable until quickbacktest materializes scores and reports
        # a finite positive rank IC.
        factor_df, analysis = compute_signal_analysis(
            self.job.case_config,
            self.job.module_name,
        )
        self.factor_df = factor_df
        factor_csv = self.workspace / "factor_data.csv"
        factor_df.to_csv(factor_csv, index=False, encoding="utf-8")
        analysis_path = self.workspace / "analysis.json"
        _write_json(analysis_path, analysis)
        self.result.factor_data_csv = factor_csv
        self.result.analysis_json = analysis_path
        self.result.metrics = analysis.get("metrics", {}).get(self.job.module_name, {})

    def _maybe_run_portfolio(self) -> None:
        if not self.job.case_config.run_portfolio:
            return
        if self.factor_df is None:
            raise RuntimeError("factor data is missing before portfolio simulation")

        qlib_adapter = _import_qlib_adapter()
        pred = qlib_adapter.factor_df_to_qlib_signal(
            self.factor_df,
            score_column="score",
        )
        portfolio = qlib_adapter.simulate_qlib_portfolio(
            pred=pred,
            benchmark=self.job.case_config.benchmark,
            topk=self.job.case_config.topk,
            n_drop=self.job.case_config.n_drop,
            provider_uri=str(self.job.case_config.provider_uri.resolve()),
            output_dir=self.workspace / "portfolio",
        )
        portfolio_path = self.workspace / "portfolio.json"
        _write_json(portfolio_path, portfolio)
        self.result.portfolio_json = portfolio_path

    def _mark_ic_status(self) -> None:
        ic, ic_name = _candidate_ic(metrics=self.result.metrics)
        self.result.ic = ic
        self.result.ic_name = ic_name
        self.result.ok = ic is not None and ic > FAST_SCREEN_RANK_IC_THRESHOLD
        if not self.result.ok:
            if ic is None:
                self.result.error_type = "MissingIC"
                self.result.error = "Candidate ran but produced no finite rank IC."
            elif ic < 0:
                self.result.error_type = "NegativeRankICRequiresReverse"
                self.result.error = (
                    "Candidate ran but produced negative rank IC; reverse the "
                    "final signal by multiplying it by -1 before final answer."
                )
            elif ic == 0:
                self.result.error_type = "NonPositiveIC"
                self.result.error = "Candidate ran but produced rank IC == 0."
            else:
                self.result.error_type = "LowRankIC"
                self.result.error = (
                    "Candidate ran but produced rank IC <= "
                    f"{FAST_SCREEN_RANK_IC_THRESHOLD:g}."
                )


def run_candidate_job(job: CandidateJob) -> CandidateResult:
    """ProcessPool entrypoint; actual logic lives in CandidateJobRunner."""
    return CandidateJobRunner(job).run()


def _make_case_config(
    config: ParallelReflexionConfig,
    *,
    workspace: Path,
    module_name: str,
) -> FactorMinerCaseConfig:
    return CandidateJobFactory(config).make_case_config(
        workspace=workspace,
        module_name=module_name,
    )


def _make_jobs(
    config: ParallelReflexionConfig,
    *,
    round_number: int,
    round_dir: Path,
    memory_text: str,
    memory: dict[str, Any] | None = None,
) -> list[CandidateJob]:
    return CandidateJobFactory(config).make_jobs(
        round_number=round_number,
        round_dir=round_dir,
        memory_text=memory_text,
        memory=memory,
    )


class CandidateBatchExecutor:
    """Execute candidate jobs in a ProcessPool and preserve candidate order."""

    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))

    def run(self, jobs: list[CandidateJob]) -> list[CandidateResult]:
        if not jobs:
            return []
        results: list[CandidateResult | None] = [None] * len(jobs)
        worker_count = min(self.max_workers, len(jobs))

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_candidate_job, job): index
                for index, job in enumerate(jobs)
            }

            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    failed = self._failed_result(jobs[index], exc)
                    _write_candidate_result(failed)
                    results[index] = failed
        return [result for result in results if result is not None]

    @staticmethod
    def _failed_result(job: CandidateJob, exc: Exception) -> CandidateResult:
        return CandidateResult(
            round_number=job.round_number,
            candidate_index=job.candidate_index,
            module_name=job.module_name,
            workspace=job.workspace,
            research_branch=job.research_branch.name,
            research_branch_goal=job.research_branch.goal,
            candidate_mode=job.candidate_mode,
            mutation_parent=job.mutation_parent,
            mutation_axis=job.mutation_axis,
            label="failed",
            ok=False,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _run_candidate_jobs(
    jobs: list[CandidateJob],
    *,
    max_workers: int,
) -> list[CandidateResult]:
    return CandidateBatchExecutor(max_workers=max_workers).run(jobs)
