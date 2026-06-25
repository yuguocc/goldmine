from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .branches import PRICE_VOLUME_BRANCHES
from .models import ResearchBranch


@dataclass(frozen=True)
class CandidateAssignment:
    research_branch: ResearchBranch
    candidate_mode: str = "novelty"
    mutation_parent: dict[str, Any] | None = None
    mutation_axis: str = ""


@dataclass
class DirectionStats:
    attempts: int = 0
    successes: int = 0
    duplicates: int = 0
    failures: int = 0
    ic_sum: float = 0.0
    ic_count: int = 0

    def record(self, *, success: bool, label: str, ic: float | None) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        elif label == "duplicate":
            self.duplicates += 1
        else:
            self.failures += 1
        if ic is not None:
            self.ic_sum += ic
            self.ic_count += 1

    def score(
        self,
        *,
        total_attempts: int,
        alpha: float,
        beta: float,
        exploration_c: float,
    ) -> float:
        probability = (self.successes + alpha) / (self.attempts + alpha + beta)
        exploration = exploration_c * math.sqrt(
            math.log(total_attempts + 1.0) / (self.attempts + 1.0)
        )
        mean_ic = self.ic_sum / self.ic_count if self.ic_count else 0.0
        ic_bonus = max(0.0, min(mean_ic, 0.1)) * 2.0
        duplicate_penalty = (
            0.15 * self.duplicates / self.attempts if self.attempts else 0.0
        )
        failure_penalty = 0.10 * self.failures / self.attempts if self.attempts else 0.0
        return probability + exploration + ic_bonus - duplicate_penalty - failure_penalty


class ResearchDirectionScheduler:
    """Allocate candidate directions using smoothed historical success probability."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 4.0,
        exploration_c: float = 0.25,
        max_branch_per_round: int = 2,
        max_parent_per_round: int = 2,
        max_axis_per_round: int = 2,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.exploration_c = exploration_c
        self.max_branch_per_round = max(1, int(max_branch_per_round))
        self.max_parent_per_round = max(1, int(max_parent_per_round))
        self.max_axis_per_round = max(1, int(max_axis_per_round))

    def allocate(
        self,
        *,
        round_number: int,
        candidate_count: int,
        mutation_parents: list[dict[str, Any]],
        mutation_axes: tuple[str, ...],
        memory: dict[str, Any] | None,
    ) -> list[CandidateAssignment]:
        stats = self.stats_from_memory(memory)
        total_attempts = sum(item.attempts for item in stats.values())
        if total_attempts <= 0:
            return []

        mutation_slots = self.mutation_slots(
            candidate_count=candidate_count,
            parent_count=len(mutation_parents),
        )
        novelty_slots = max(0, int(candidate_count) - mutation_slots)
        branch_usage: dict[str, int] = {}
        parent_usage: dict[str, int] = {}
        axis_usage: dict[str, int] = {}

        assignments: list[CandidateAssignment] = []
        for _ in range(novelty_slots):
            branch = self.select_branch(
                mode="novelty",
                stats=stats,
                total_attempts=total_attempts,
                branch_usage=branch_usage,
            )
            self.bump(branch_usage, branch.name)
            assignments.append(CandidateAssignment(research_branch=branch))

        for _ in range(mutation_slots):
            branch = self.select_branch(
                mode="mutation",
                stats=stats,
                total_attempts=total_attempts,
                branch_usage=branch_usage,
            )
            parent = self.select_parent(
                parents=mutation_parents,
                stats=stats,
                total_attempts=total_attempts,
                parent_usage=parent_usage,
            )
            axis = self.select_axis(
                axes=mutation_axes,
                stats=stats,
                total_attempts=total_attempts,
                axis_usage=axis_usage,
            )
            self.bump(branch_usage, branch.name)
            self.bump(parent_usage, self.parent_name(parent))
            self.bump(axis_usage, axis)
            assignments.append(
                CandidateAssignment(
                    research_branch=branch,
                    candidate_mode="mutation",
                    mutation_parent=parent,
                    mutation_axis=axis,
                )
            )

        return assignments[: max(0, int(candidate_count))]

    @staticmethod
    def mutation_slots(*, candidate_count: int, parent_count: int) -> int:
        if parent_count <= 0:
            return 0
        return max(0, min(3, max(0, int(candidate_count) - 3)))

    def select_branch(
        self,
        *,
        mode: str,
        stats: dict[tuple[Any, ...], DirectionStats],
        total_attempts: int,
        branch_usage: dict[str, int],
    ) -> ResearchBranch:
        branches = list(PRICE_VOLUME_BRANCHES)
        ranked = sorted(
            branches,
            key=lambda branch: (
                -self.branch_score(
                    branch.name,
                    mode=mode,
                    stats=stats,
                    total_attempts=total_attempts,
                ),
                branch_usage.get(branch.name, 0),
                branch.name,
            ),
        )
        for branch in ranked:
            if branch_usage.get(branch.name, 0) < self.max_branch_per_round:
                return branch
        return ranked[0]

    def branch_score(
        self,
        branch_name: str,
        *,
        mode: str,
        stats: dict[tuple[Any, ...], DirectionStats],
        total_attempts: int,
    ) -> float:
        mode_branch = self.score_stats(
            stats.get(("mode_branch", mode, branch_name)),
            total_attempts=total_attempts,
        )
        branch = self.score_stats(
            stats.get(("branch", branch_name)),
            total_attempts=total_attempts,
        )
        mode_score = self.score_stats(
            stats.get(("mode", mode)),
            total_attempts=total_attempts,
        )
        return mode_branch + 0.4 * branch + 0.2 * mode_score

    def select_parent(
        self,
        *,
        parents: list[dict[str, Any]],
        stats: dict[tuple[Any, ...], DirectionStats],
        total_attempts: int,
        parent_usage: dict[str, int],
    ) -> dict[str, Any] | None:
        if not parents:
            return None
        ranked = sorted(
            parents,
            key=lambda parent: (
                -self.parent_score(parent, stats=stats, total_attempts=total_attempts),
                parent_usage.get(self.parent_name(parent), 0),
                self.parent_name(parent),
            ),
        )
        for parent in ranked:
            if parent_usage.get(self.parent_name(parent), 0) < self.max_parent_per_round:
                return dict(parent)
        return dict(ranked[0])

    def parent_score(
        self,
        parent: dict[str, Any],
        *,
        stats: dict[tuple[Any, ...], DirectionStats],
        total_attempts: int,
    ) -> float:
        name = self.parent_name(parent)
        rank_ic = max(0.0, min(_finite_float(parent.get("rank_ic")) or 0.0, 0.1)) * 2.0
        return self.score_stats(
            stats.get(("parent", name)),
            total_attempts=total_attempts,
        ) + rank_ic

    def select_axis(
        self,
        *,
        axes: tuple[str, ...],
        stats: dict[tuple[Any, ...], DirectionStats],
        total_attempts: int,
        axis_usage: dict[str, int],
    ) -> str:
        if not axes:
            return ""
        ranked = sorted(
            axes,
            key=lambda axis: (
                -self.axis_score(axis, stats=stats, total_attempts=total_attempts),
                axis_usage.get(axis, 0),
                axis,
            ),
        )
        for axis in ranked:
            if axis_usage.get(axis, 0) < self.max_axis_per_round:
                return axis
        return ranked[0]

    def axis_score(
        self,
        axis: str,
        *,
        stats: dict[tuple[Any, ...], DirectionStats],
        total_attempts: int,
    ) -> float:
        return self.score_stats(
            stats.get(("axis", axis)),
            total_attempts=total_attempts,
        ) + 0.3 * self.score_stats(
            stats.get(("mode_axis", "mutation", axis)),
            total_attempts=total_attempts,
        )

    def score_stats(
        self,
        stats: DirectionStats | None,
        *,
        total_attempts: int,
    ) -> float:
        item = stats or DirectionStats()
        return item.score(
            total_attempts=total_attempts,
            alpha=self.alpha,
            beta=self.beta,
            exploration_c=self.exploration_c,
        )

    def stats_from_memory(
        self,
        memory: dict[str, Any] | None,
    ) -> dict[tuple[Any, ...], DirectionStats]:
        state = memory.get("state") if isinstance(memory, dict) else None
        if not isinstance(state, dict):
            return {}

        stats: dict[tuple[Any, ...], DirectionStats] = {}
        for record in _as_dict_list(state.get("recent_admissions")):
            self.record_memory_item(stats, record, success=True)
        for record in _as_dict_list(state.get("recent_rejections")):
            self.record_memory_item(stats, record, success=False)
        return stats

    def record_memory_item(
        self,
        stats: dict[tuple[Any, ...], DirectionStats],
        record: dict[str, Any],
        *,
        success: bool,
    ) -> None:
        branch = str(record.get("research_branch", "") or "").strip()
        if not branch:
            return
        mode = str(record.get("candidate_mode", "") or "novelty").strip() or "novelty"
        axis = str(record.get("mutation_axis", "") or "").strip()
        parent = self.parent_name(record.get("mutation_parent"))
        label = "accepted" if success else str(record.get("label", "") or "failed")
        ic = _finite_float(record.get("candidate_ic", record.get("ic")))

        for key in [("branch", branch), ("mode", mode), ("mode_branch", mode, branch)]:
            self.stat(stats, key).record(success=success, label=label, ic=ic)
        if axis:
            for key in [("axis", axis), ("mode_axis", mode, axis)]:
                self.stat(stats, key).record(success=success, label=label, ic=ic)
        if parent:
            self.stat(stats, ("parent", parent)).record(
                success=success,
                label=label,
                ic=ic,
            )

    @staticmethod
    def stat(
        stats: dict[tuple[Any, ...], DirectionStats],
        key: tuple[Any, ...],
    ) -> DirectionStats:
        if key not in stats:
            stats[key] = DirectionStats()
        return stats[key]

    @staticmethod
    def bump(counter: dict[str, int], key: str) -> None:
        if key:
            counter[key] = counter.get(key, 0) + 1

    @staticmethod
    def parent_name(parent: Any) -> str:
        if isinstance(parent, dict):
            return str(parent.get("name", "") or parent.get("factor_name", "") or "").strip()
        return str(parent or "").strip()


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
