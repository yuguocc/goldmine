from __future__ import annotations

import json
from pathlib import Path
import uuid

from quickbacktest import FactorLibrary
from rlm_factor_memory import (
    RlmExperienceMemory,
    RlmFactorMemoryManager,
    RlmForbiddenDirection,
    RlmMiningState,
    RlmStrategicInsight,
    RlmSuccessPattern,
)
from src.factor_miner_parallel_reflexion.branches import branch_for_candidate
from src.factor_miner_parallel_reflexion.candidate import CandidatePromptBuilder
from src.factor_miner_parallel_reflexion.candidate import CandidateJobFactory
from src.factor_miner_parallel_reflexion.cli import ParallelReflexionCLI
from src.factor_miner_parallel_reflexion.models import ParallelReflexionConfig
from src.factor_miner_parallel_reflexion.reflexion import RoundReflexionAgent
from src.factor_miner_parallel_reflexion.runner import ParallelReflexionRunner


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_parallel_reflexion_runner" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parallel_reflexion_skips_only_final_round_reflexion():
    assert ParallelReflexionRunner._should_run_reflexion(1, 4) is True
    assert ParallelReflexionRunner._should_run_reflexion(2, 4) is True
    assert ParallelReflexionRunner._should_run_reflexion(3, 4) is False
    assert ParallelReflexionRunner._should_run_reflexion(1, 2) is False


def test_runner_plots_portfolio_history_at_end(monkeypatch):
    output_dir = _workspace_tmp("plot_runner")
    history_path = output_dir / "portfolio_history.json"
    history_path.write_text(json.dumps({"rounds": [{"round": 1}]}), encoding="utf-8")
    runner = ParallelReflexionRunner(
        ParallelReflexionConfig(
            output_dir=output_dir,
            factor_library_path=output_dir / "factor_library",
        )
    )
    calls = []

    def fake_plot_portfolio_history(*, history_path, output_dir):
        calls.append((Path(history_path), Path(output_dir)))
        return {
            "history_json": str(history_path),
            "plots": {"round_return_curves": str(Path(output_dir) / "curves.png")},
            "round_count": 1,
            "round_curve_count": 1,
        }

    monkeypatch.setattr(
        "scripts.plot_portfolio_history.plot_portfolio_history",
        fake_plot_portfolio_history,
    )

    result = runner._plot_portfolio_history()

    assert result["status"] == "completed"
    assert result["round_curve_count"] == 1
    assert calls == [(history_path.resolve(), output_dir.resolve())]


def test_runner_skips_portfolio_history_plots_when_history_missing():
    output_dir = _workspace_tmp("plot_missing")
    runner = ParallelReflexionRunner(
        ParallelReflexionConfig(
            output_dir=output_dir,
            factor_library_path=output_dir / "factor_library",
        )
    )

    result = runner._plot_portfolio_history()

    assert result["status"] == "skipped"
    assert result["reason"] == "portfolio_history.json missing"


def test_memory_update_without_reflexion_records_state_only():
    manager = RlmFactorMemoryManager(max_entries=3)

    manager.update(
        round_number=3,
        reflexions=[],
        round_ic={"round": 3, "best_ic": 0.04, "improved": True},
        best_signal={"round": 3, "ic": 0.04, "admission_status": "accepted"},
        results=[],
        admission={
            "status": "accepted",
            "accepted_count": 1,
            "accepted_candidates": [
                {
                    "factor_name": "accepted-factor",
                    "candidate_module": "RlmGeneratedFactorR003C001",
                    "candidate_ic": 0.04,
                }
            ],
        },
    )

    payload = manager.to_dict()
    assert payload["state"]["latest_round"] == 3
    assert payload["state"]["latest_best_ic"] == 0.04
    assert payload["state"]["ic_history"][-1]["best_ic"] == 0.04
    assert payload["success_patterns"] == []
    assert payload["forbidden_directions"] == []
    assert payload["insights"] == []


def test_candidate_memory_prompt_keeps_only_four_compact_sections():
    manager = RlmFactorMemoryManager(
        RlmExperienceMemory(
            state=RlmMiningState(
                failure_type_stats={
                    "name_error": {
                        "error_type": "NameError",
                        "count": 3,
                        "rounds": [1, 2],
                    }
                }
            ),
            success_patterns=[
                RlmSuccessPattern(
                    name="accepted_momentum",
                    description="momentum worked",
                    example_factors=["factor-a"],
                    metadata={
                        "signal_family": "momentum",
                        "positive_pattern": "medium horizon continuation worked",
                        "recommended_hypotheses": [
                            "participation-supported continuation"
                        ],
                        "recommended_construction": [
                            "mutate the persistence story, not factor mechanics"
                        ],
                        "next_action": "add liquidity filter",
                    },
                )
            ],
            forbidden_directions=[
                RlmForbiddenDirection(
                    name="implementation_name_error",
                    description="avoid missing imports",
                    examples=["bad-factor"],
                    metadata={
                        "failure_types": ["NameError"],
                        "avoid_pattern": "referencing undefined pd",
                        "next_action": "use provided libraries only",
                    },
                )
            ],
            insights=[
                RlmStrategicInsight(
                    insight="prefer simple signal assignment",
                    evidence="The reflection found that shorter compute bodies repaired faster.",
                    batch_source=2,
                    phase="implementation_errors",
                    metadata={"next_action": "keep compute body minimal"},
                )
            ],
        )
    )

    text = manager.prompt_text()

    assert "## Factorminer Memory Priors" in text
    assert "### Current Library State" in text
    assert "### Recommended Directions (P_succ)" in text
    assert "### Forbidden Directions (P_fail)" in text
    assert "### Strategic Insights (I)" in text
    assert "P_succ_1:" in text
    assert "participation-supported continuation" in text
    assert "mutate the persistence story" in text
    assert "P_fail_1:" in text
    assert "I_1:" in text
    assert "high_frequency_failure_types" not in text
    assert "failure_type_stats" not in text


def test_economic_reflexion_updates_factorminer_memory_priors():
    manager = RlmFactorMemoryManager(max_entries=3)

    manager.update(
        round_number=1,
        reflexions=[
            {
                "phase": "economic_hypothesis",
                "reflexion": "\n".join(
                    [
                        "## Recommended Directions (P_succ)",
                        "- Select participation-supported continuation because it beat weak reversal; construct the next hypothesis around persistence rather than one-day activity.",
                        "## Forbidden Directions (P_fail)",
                        "- Avoid generic reversal because it was unstable; revisit only when the hypothesis explains why pressure is temporary.",
                    ]
                ),
            }
        ],
        round_ic={"round": 1, "best_ic": 0.03, "improved": True},
        best_signal=None,
        results=[],
        admission={
            "status": "accepted",
            "accepted_count": 1,
            "accepted_candidates": [
                {
                    "factor_name": "accepted-factor",
                    "candidate_module": "RlmGeneratedFactorR001C001",
                    "candidate_ic": 0.03,
                }
            ],
        },
    )

    memory_signal = manager.retrieve_memory_signal()
    assert (
        memory_signal["recommended_directions"][0]["metadata"][
            "recommended_hypotheses"
        ][0]
        == "Select participation-supported continuation because it beat weak reversal; construct the next hypothesis around persistence rather than one-day activity."
    )
    assert (
        memory_signal["forbidden_directions"][0]["metadata"]["forbidden_hypotheses"][
            0
        ]
        == "Avoid generic reversal because it was unstable; revisit only when the hypothesis explains why pressure is temporary."
    )
    assert "participation-supported continuation" in manager.prompt_text()
    assert "generic reversal" in manager.prompt_text()
    assert "hypothesis_slots" not in manager.to_dict()["state"]


def test_candidate_context_uses_factorminer_memory_priors_not_query():
    branch = branch_for_candidate(round_number=1, candidate_index=5)
    memory_text = "## Factorminer Memory Priors\n- none"
    job = CandidateJobFactory(ParallelReflexionConfig(candidates=3)).make_jobs(
        round_number=1,
        round_dir=_workspace_tmp("memory_context"),
        memory_text=memory_text,
    )[1]
    context = CandidatePromptBuilder().context(job)
    query = CandidatePromptBuilder().query(
        "RlmGeneratedFactorR001C001",
        candidate_index=2,
        candidate_count=3,
        research_branch=branch,
    )

    assert context["memory_priors"] == memory_text
    assert "Factorminer Memory Priors" not in query
    assert "Factorminer Memory Priors" not in CandidatePromptBuilder().query(
        "RlmGeneratedFactorR001C001",
        memory_text,
        candidate_index=2,
        candidate_count=3,
        research_branch=branch,
    )
    assert "Research Assignment:" in query
    assert "Candidate Mode:" in query
    assert "mode: novelty" in query
    assert f"- branch: {branch.name}" in query
    assert "must_use:" in query
    assert "must_avoid:" in query
    assert "Candidate index: 2 of 3" in query
    assert "Candidate class name: RlmGeneratedFactorR001C001" in query
    assert "context.existing_factors" in query
    assert "Avoid duplicating" in query
    assert "silently run a novelty check" in query
    assert "closest Existing Factor" in query
    assert "A high-IC candidate is still invalid" in query
    assert "diversity" not in query.lower()


def test_candidate_context_includes_existing_factor_library_entries():
    factor_library_path = _workspace_tmp("existing_factors")
    output_dir = _workspace_tmp("existing_context")
    library = FactorLibrary(factor_library_path)
    library.save_factor(
        name="accepted-momentum",
        signal_code="class AcceptedMomentum: pass\n",
        metrics={"daily_rank_ic_mean": 0.04, "coverage": 0.9},
        description="medium-horizon momentum with volume confirmation",
        rlm_summary=(
            "Hypothesis: medium-horizon return continuation works better when "
            "confirmed by volume expansion."
        ),
        signal_class="AcceptedMomentum",
        status="accepted",
    )
    library.save_factor(
        name="rejected-reversal",
        signal_code="class RejectedReversal: pass\n",
        metrics={"daily_rank_ic_mean": -0.01, "coverage": 0.9},
        description="short reversal",
        signal_class="RejectedReversal",
        status="rejected",
    )
    config = ParallelReflexionConfig(
        output_dir=output_dir,
        factor_library_path=factor_library_path,
        candidates=1,
    )
    job = CandidateJobFactory(config).make_jobs(
        round_number=1,
        round_dir=output_dir / "round_001",
        memory_text="memory",
    )[0]

    context = CandidatePromptBuilder().context(job)
    existing = context["existing_factors"]

    assert existing["label"] == "Existing Factors"
    assert existing["count"] == 1
    assert existing["items"][0]["name"] == "accepted-momentum"
    assert existing["items"][0]["signal_class"] == "AcceptedMomentum"
    assert existing["items"][0]["rank_ic"] == 0.04
    assert "volume expansion" in existing["items"][0]["summary"]
    assert "occupied_idea_space" in existing["items"][0]
    assert "Do not recreate" in existing["items"][0]["do_not_duplicate"]
    assert "rejected-reversal" not in json.dumps(existing)
    assert "Do not regenerate" in context["duplicate_avoidance_policy"]
    assert "window/threshold-only" in context["duplicate_avoidance_policy"]
    assert "high-IC duplicate is invalid" in context["duplicate_avoidance_policy"]


def test_candidate_job_factory_assigns_rotating_research_branches():
    config = ParallelReflexionConfig(candidates=3)
    jobs = CandidateJobFactory(config).make_jobs(
        round_number=2,
        round_dir=_workspace_tmp("branches"),
        memory_text="memory",
    )

    assert [job.research_branch.name for job in jobs] == [
        "Reversal",
        "Liquidity",
        "Volatility",
    ]


def test_candidate_job_factory_uses_three_novelty_plus_three_mutation_slots():
    factor_library_path = _workspace_tmp("mutation_parents")
    output_dir = _workspace_tmp("mutation_jobs")
    library = FactorLibrary(factor_library_path)
    library.save_factor(
        name="high-ic-parent",
        signal_code="class HighIcParent: pass\n",
        metrics={"daily_rank_ic_mean": 0.05, "coverage": 0.9},
        description="high IC parent",
        rlm_summary="type: momentum\nhypothesis:\n  hp1: high IC parent\n",
        signal_class="HighIcParent",
        status="accepted",
    )
    library.save_factor(
        name="second-parent",
        signal_code="class SecondParent: pass\n",
        metrics={"daily_rank_ic_mean": 0.03, "coverage": 0.9},
        description="second parent",
        rlm_summary="type: liquidity\nhypothesis:\n  hp1: second parent\n",
        signal_class="SecondParent",
        status="accepted",
    )
    config = ParallelReflexionConfig(
        output_dir=output_dir,
        factor_library_path=factor_library_path,
        candidates=6,
    )

    jobs = CandidateJobFactory(config).make_jobs(
        round_number=1,
        round_dir=output_dir / "round_001",
        memory_text="memory",
    )

    assert [job.candidate_mode for job in jobs[:3]] == ["novelty"] * 3
    assert [job.candidate_mode for job in jobs[3:]] == ["mutation"] * 3
    assert [job.mutation_axis for job in jobs[3:]] == [
        "replace_gate",
        "change_normalization",
        "add_interaction",
    ]
    assert [job.mutation_parent["name"] for job in jobs[3:]] == [
        "high-ic-parent",
        "second-parent",
        "high-ic-parent",
    ]

    context = CandidatePromptBuilder().context(jobs[3])
    query = CandidatePromptBuilder().query(
        jobs[3].module_name,
        candidate_index=jobs[3].candidate_index,
        candidate_count=jobs[3].candidate_count,
        research_branch=jobs[3].research_branch,
        candidate_mode=jobs[3].candidate_mode,
        mutation_parent=jobs[3].mutation_parent,
        mutation_axis=jobs[3].mutation_axis,
    )
    assert context["candidate_mode"] == "mutation"
    assert context["mutation_parent"]["name"] == "high-ic-parent"
    assert context["mutation_axis"] == "replace_gate"
    assert "Mutation candidates must preserve" in context["mutation_policy"]
    assert "mode: mutation" in query
    assert "parent_factor: high-ic-parent" in query


def test_candidate_job_factory_falls_back_to_novelty_without_mutation_parents():
    output_dir = _workspace_tmp("empty_mutation_jobs")
    config = ParallelReflexionConfig(
        output_dir=output_dir,
        factor_library_path=output_dir / "empty_factor_library",
        candidates=6,
    )

    jobs = CandidateJobFactory(config).make_jobs(
        round_number=1,
        round_dir=output_dir / "round_001",
        memory_text="memory",
    )

    assert [job.candidate_mode for job in jobs] == ["novelty"] * 6
    assert all(job.mutation_parent is None for job in jobs)
    assert all(job.mutation_axis == "" for job in jobs)


def test_candidate_job_factory_prioritizes_high_success_branch_from_memory():
    output_dir = _workspace_tmp("scheduled_novelty_jobs")
    config = ParallelReflexionConfig(
        output_dir=output_dir,
        factor_library_path=output_dir / "empty_factor_library",
        candidates=3,
    )
    memory = {
        "state": {
            "recent_admissions": [
                {
                    "research_branch": "Liquidity",
                    "candidate_mode": "novelty",
                    "candidate_ic": 0.06,
                }
            ],
            "recent_rejections": [
                {
                    "research_branch": "Momentum",
                    "candidate_mode": "novelty",
                    "label": "failed",
                    "ic": -0.01,
                }
            ],
        }
    }

    jobs = CandidateJobFactory(config).make_jobs(
        round_number=3,
        round_dir=output_dir / "round_003",
        memory_text="memory",
        memory=memory,
    )

    assert jobs[0].research_branch.name == "Liquidity"
    assert [job.candidate_mode for job in jobs] == ["novelty"] * 3


def test_candidate_job_factory_prioritizes_successful_mutation_axis():
    factor_library_path = _workspace_tmp("scheduled_mutation_parents")
    output_dir = _workspace_tmp("scheduled_mutation_jobs")
    library = FactorLibrary(factor_library_path)
    library.save_factor(
        name="high-ic-parent",
        signal_code="class HighIcParent: pass\n",
        metrics={"daily_rank_ic_mean": 0.05, "coverage": 0.9},
        description="high IC parent",
        rlm_summary="type: momentum\nhypothesis:\n  hp1: high IC parent\n",
        signal_class="HighIcParent",
        status="accepted",
    )
    memory = {
        "state": {
            "recent_admissions": [
                {
                    "research_branch": "Momentum",
                    "candidate_mode": "mutation",
                    "mutation_axis": "add_interaction",
                    "mutation_parent": {"name": "high-ic-parent"},
                    "candidate_ic": 0.055,
                }
            ],
            "recent_rejections": [
                {
                    "research_branch": "Momentum",
                    "candidate_mode": "mutation",
                    "mutation_axis": "replace_gate",
                    "mutation_parent": {"name": "high-ic-parent"},
                    "label": "duplicate",
                    "ic": 0.02,
                }
            ],
        }
    }
    config = ParallelReflexionConfig(
        output_dir=output_dir,
        factor_library_path=factor_library_path,
        candidates=6,
    )

    jobs = CandidateJobFactory(config).make_jobs(
        round_number=3,
        round_dir=output_dir / "round_003",
        memory_text="memory",
        memory=memory,
    )

    assert [job.candidate_mode for job in jobs[:3]] == ["novelty"] * 3
    assert [job.candidate_mode for job in jobs[3:]] == ["mutation"] * 3
    assert jobs[3].mutation_axis == "add_interaction"
    assert jobs[3].mutation_parent["name"] == "high-ic-parent"


def test_economic_reflexion_prompt_uses_factorminer_memory_sections():
    prompt = RoundReflexionAgent(
        ParallelReflexionConfig(candidates=4)
    ).economic_prompt()

    assert "candidate_count_observed: 4" in prompt
    assert "Recommended Directions (P_succ)" in prompt
    assert "Forbidden Directions (P_fail)" in prompt
    assert "Strategic Insights (I)" not in prompt
    assert "Search Metadata" not in prompt
    assert "factor_design:" not in prompt
    assert "hypothesis_selection:" not in prompt
    assert "Factor Design Analysis" not in prompt
    assert "Hypothesis Selection" not in prompt
    assert "Bullet points" in prompt
    assert "required_next_hypothesis_count" not in prompt


def test_round_reflexion_specs_run_only_economic_phase(monkeypatch):
    from src.factor_miner_parallel_reflexion.models import RoundReflexionInputs

    config = ParallelReflexionConfig(candidates=2)
    agent = RoundReflexionAgent(config)
    phases = []

    def fake_run_phase(*, phase, prompt, round_number, round_dir, context):
        phases.append(phase)
        return "reflection", f"{phase}.json"

    monkeypatch.setattr(agent, "run_phase", fake_run_phase)
    result = agent.run_round(
        round_number=1,
        round_dir=_workspace_tmp("reflection_phase"),
        inputs=RoundReflexionInputs(
            trajectory_context={"unused": True},
            trajectory_path=Path("unused_trajectory.json"),
            final_answer_context="final answers",
            final_answer_path=Path("final_answers.md"),
        ),
    )

    assert phases == ["economic_hypothesis"]
    assert [item["phase"] for item in result] == ["economic_hypothesis"]


def test_memory_seed_success_patterns_deduplicates():
    manager = RlmFactorMemoryManager()

    manager.seed_success_patterns(["  A hypothesis  ", "A hypothesis", "", "B"])

    descriptions = [item.description for item in manager.memory.success_patterns]
    assert descriptions == ["A hypothesis", "B"]


def test_cli_has_no_hypothesis_extraction_options():
    args = ParallelReflexionCLI().parse_args(
        [
            "--rounds",
            "1",
        ]
    )
    config = ParallelReflexionCLI().config_from_args(args)

    assert not hasattr(config, "bootstrap_hypotheses")
    assert not hasattr(config, "hypothesis_context_content")
    assert not hasattr(config, "hypothesis_context_paths")


def test_cli_enables_final_oos_by_default_and_can_skip_it():
    cli = ParallelReflexionCLI()

    default_config = cli.config_from_args(cli.parse_args(["--rounds", "1"]))
    skipped_config = cli.config_from_args(
        cli.parse_args(["--rounds", "1", "--skip-oos-test"])
    )

    assert default_config.run_oos_test is True
    assert skipped_config.run_oos_test is False


def test_cli_default_training_and_oos_dates():
    config = ParallelReflexionCLI().config_from_args(
        ParallelReflexionCLI().parse_args(["--rounds", "1"])
    )

    assert config.start == "2023-01-01"
    assert config.end == "2024-12-31"
    assert config.oos_start == "2025-01-01"
    assert config.oos_end == "2026-01-31"
    assert config.oos_warmup_start is None


def test_cli_accepts_configured_final_oos_dates():
    config = ParallelReflexionCLI().config_from_args(
        ParallelReflexionCLI().parse_args(
            [
                "--rounds",
                "1",
                "--oos-start",
                "2025-07-01",
                "--oos-end",
                "2026-06-30",
                "--oos-warmup-start",
                "2025-01-01",
            ]
        )
    )

    assert config.oos_start == "2025-07-01"
    assert config.oos_end == "2026-06-30"
    assert config.oos_warmup_start == "2025-01-01"


def test_package_top_level_main_remains_available():
    import src.factor_miner_parallel_reflexion as parallel_reflexion

    assert callable(parallel_reflexion.main)
    args = parallel_reflexion.parse_args(["--rounds", "1", "--candidates", "1"])
    assert args.rounds == 1
    assert args.candidates == 1
