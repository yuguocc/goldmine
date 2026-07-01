from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.factor_miner import FactorMinerCaseConfig

from .constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OOS_END,
    DEFAULT_OOS_START,
    DEFAULT_PROVIDER_URI,
    DEFAULT_RLM_MODEL,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    MARGINAL_CONTRIBUTION_MIN_DELTA,
)


@dataclass(frozen=True)
class ResearchBranch:
    """One simple price/volume research style assigned to a candidate."""

    name: str
    goal: str
    must_use: tuple[str, ...] = ()
    must_avoid: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParallelReflexionConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    provider_uri: Path = DEFAULT_PROVIDER_URI
    factor_library_path: Path = DEFAULT_OUTPUT_DIR / "factor_library"
    instruments: str = "csi500"
    start: str = DEFAULT_TRAIN_START
    end: str = DEFAULT_TRAIN_END
    benchmark: str = "SH000905"
    topk: int = 50
    n_drop: int = 5
    horizon: int = 1
    factor_shift: int = 1
    candidates: int = 6
    rounds: int = 1
    max_workers: int = 3
    memory_size: int = 5
    run_portfolio: bool = False
    run_library_portfolio: bool = True
    marginal_contribution_gate: bool = True
    marginal_contribution_min_delta: float = MARGINAL_CONTRIBUTION_MIN_DELTA
    run_oos_test: bool = True
    oos_start: str | None = DEFAULT_OOS_START
    oos_end: str | None = DEFAULT_OOS_END
    oos_warmup_start: str | None = None
    model: str = DEFAULT_RLM_MODEL
    recursive_model: str = DEFAULT_RLM_MODEL
    max_iterations: int = 5
    enable_rlm_logging: bool = True


@dataclass(frozen=True)
class CandidateJob:
    round_number: int
    candidate_index: int
    candidate_count: int
    module_name: str
    workspace: Path
    case_config: FactorMinerCaseConfig
    factor_library_path: Path
    memory_text: str
    research_branch: ResearchBranch
    model: str
    recursive_model: str
    max_iterations: int
    enable_rlm_logging: bool
    candidate_mode: str = "novelty"
    mutation_parent: dict[str, Any] | None = None
    mutation_axis: str = ""


@dataclass
class CandidateResult:
    """Serializable result for one generated factor candidate."""

    round_number: int
    candidate_index: int
    module_name: str
    workspace: Path
    label: str = "unlabeled"
    ok: bool = False
    ic: float | None = None
    ic_name: str = ""
    rlm_summary: str = ""
    trajectory_path: Path | None = None
    signal_path: Path | None = None
    factor_data_csv: Path | None = None
    analysis_json: Path | None = None
    portfolio_json: Path | None = None
    candidate_result_json: Path | None = None
    metrics: dict[str, Any] | None = None
    dedup_of: str = ""
    dedup_correlation: float | None = None
    research_branch: str = ""
    research_branch_goal: str = ""
    candidate_mode: str = "novelty"
    mutation_parent: dict[str, Any] | None = None
    mutation_axis: str = ""
    error_type: str = ""
    error: str = ""
    traceback: str = ""


@dataclass(frozen=True)
class RoundEvaluation:
    """Post-generation state for one Ralph-style screening round."""

    batch_dedup: dict[str, Any]
    round_ic: dict[str, Any]
    best_signal: dict[str, Any] | None
    best_result: CandidateResult | None
    admitted_result: CandidateResult | None
    admitted_results: list[CandidateResult]
    factor_library_admission: dict[str, Any]
    factor_library_portfolio: dict[str, Any]


@dataclass(frozen=True)
class RoundReflexionInputs:
    """Artifacts passed into the two reflexion agents."""

    trajectory_context: dict[str, Any]
    trajectory_path: Path
    final_answer_context: str
    final_answer_path: Path
