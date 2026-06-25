# Factor Miner Parallel Reflexion Structure

This document describes the current `src/factor_miner_parallel_reflexion/`
pipeline.

## Entry Points

- `factor_miner_parallel_reflexion.py`: normal RLM factor-mining CLI wrapper.
- `run_parallel_reflexion_demo.py`: no-LLM demo that reuses the real evaluation,
  admission, memory, and artifact code with mock candidates.

## Package Layout

```text
src/factor_miner_parallel_reflexion/
  constants.py
  models.py
  memory.py
  utils.py
  candidate.py
  library.py
  evaluation.py
  portfolio.py
  reflexion.py
  runner.py
  cli.py
  demo.py
```

## Main Flow

Default date windows:

- Training: `2023-01-01` to `2024-12-31`
- OOS: `2025-01-01` to `2026-01-31`

```text
ParallelReflexionCLI
-> ParallelReflexionRunner
  -> load RLM memory
  -> for each round:
    -> retrieve compact memory patterns
    -> retrieve best accepted signal source
    -> generate N candidate jobs
      -> default N=6: 3 novelty candidates plus 3 mutation candidates
      -> mutation candidates fall back to novelty when no accepted parent exists
    -> run candidates in ProcessPool
      -> RLM_REPL.completion
      -> submit_compute(compute_code) to wrap, save, and confirm runnable
      -> reverse signal and submit again if rank IC is negative
      -> external compute_signal_analysis
      -> extract daily rank IC
    -> fast screen: finite daily rank IC > 0.02
    -> intra-batch dedup
    -> factor-library correlation and replacement checks
    -> admit every nonduplicate candidate that passes review
    -> run factor-library portfolio
    -> write round artifacts
    -> run economic and implementation reflexions
    -> update memory
  -> run final OOS composite rank IC and portfolio test
  -> plot portfolio history and per-round return curves
```

## Candidate Generation Contract

RLM does not choose or pass a signal name and does not write a full signal
file. Each candidate has an internal module/class name for Python import, and
the system wraps the RLM-provided compute body into that BaseSignal class.

The generator input is split to reduce tool calls and iterations:

- Context carries available fields, available libraries, helpers, compact
  `memory_priors`, compact `existing_factors`, a duplicate-avoidance policy,
  and a plain-text `parallel_generation`.
- Query carries only the short task, fixed runnable-check flow, output format,
  and instructions to read memory from context.

The default RLM generation cap is 5 iterations. This is intended to cover
compute-body generation, one `submit_compute` runnable check, and a small
number of execution error repairs without encouraging metric-search loops.

`parallel_generation` is a short string that states the required factor class
name, candidate index, candidate mode, and assigned research branch.

By default each round uses a 3+3 assignment when enough candidates are enabled:

- Candidates 1-3 are novelty candidates over the rotating research branches.
- Candidates 4-6 are mutation candidates. They receive a compact accepted
  factor-library parent and one assigned mutation axis:
  `replace_gate`, `change_normalization`, `add_interaction`, or
  `change_horizon_family`.
- If the factor library has no accepted parent yet, mutation slots fall back to
  novelty mode.

Mutation candidates preserve the parent factor's broad economic family but must
change the signal mechanism on the assigned axis. Simple sign flips,
parameter-only changes, and cosmetic rewrites are disallowed by prompt.

`existing_factors` is labeled "Existing Factors" and contains accepted
factor-library summaries only: name, signal class, description, rank IC, and
compact RLM summary. The candidate query and context explicitly tell the agent
to avoid duplicating these hypotheses, operator stacks, horizons, gates, field
interactions, or simple sign-flipped variants. External correlation and
intra-batch dedup checks still enforce duplicate control after generation.

The generator's job is to create one candidate, submit it through the real Qlib
path, and target positive rank IC direction. If `submit_compute()` fails, it
should repair execution/contract errors and submit again. If `submit_compute()`
reports negative rank IC, it must reverse the final factor output by multiplying
the returned `signal` by `-1` and submit again. It should not run an
IC-optimization loop or train machine-learning models inside compute.
External evaluation computes the final rank IC and decides whether the factor
passes the `0.02` admission threshold.

RLM-facing tools:

- `submit_compute(compute_code)`: wraps the compute body into the current
  candidate BaseSignal class, writes the signal file, and immediately runs the
  required quickbacktest/Qlib check. It returns
  only `ok` on success, and `ok`, `error_type`, `error` on failure.
  `NegativeRankICRequiresReverse` means the candidate must be reversed and
  submitted again before final answer.

`submit_signal`, `save_signal`, `run_signal`, `read_signal_template`, `list_factors`, and
`read_factor` are not exposed to the candidate generator because their data or
behavior is covered by `submit_compute`, context, and external evaluation.

The runner enforces this contract after RLM returns: `run_signal_status.json`
must exist, match the current module, and contain `ok: true`. Otherwise the
candidate is rejected before external IC analysis.

The RLM REPL also has a pre-submit final-answer gate. When the model sets
`answer["ready"] = True`, the gate checks that the current signal file exists
and `submit_compute` has produced `ok: true` for the current module. If not, the
REPL rejects that final answer, exposes the rejection reason, and continues the
loop. The outer runner keeps the same check as a backup.

The candidate context contains only the necessary runtime state:

- `available_data_fields`
- `available_libraries`
- `available_helpers`
- `memory_priors`, compact prior Reflexion memory retrieved before generation
- `parallel_generation`, a plain-text string with `factor_class_name` and
  research assignment details
- `candidate_mode`, either `novelty` or `mutation`
- `mutation_parent`, a compact accepted-factor card for mutation candidates
- `mutation_axis`, the required structural mutation axis
- `mutation_policy`
- `existing_factors`, accepted factor-library summaries to avoid duplicating
- `duplicate_avoidance_policy`

## Screening And Admission

The fast screen uses daily rank IC. A candidate is usable only when its rank IC
is finite and strictly greater than `0.02`.

Admission order is descending rank IC. In one round, multiple candidates can be
admitted if they pass:

- fast screen
- deterministic quickbacktest review
- accepted-factor-library correlation check
- replacement/blocking check
- intra-batch deduplication

`factor_library_admission.json` keeps backward-compatible top-level fields for
the first accepted candidate, and also records:

- `accepted_count`
- `accepted_candidates`
- `attempts`
- `candidate_pool`

## Factor-Library Portfolio

After each round, the library portfolio uses all accepted factors whose rank IC
is above the same threshold and whose `factor_data.csv` is available.

The composite factor is built by:

1. Loading each accepted factor score series.
2. Daily cross-sectional rank-normalizing each factor.
3. Weighting each factor by `daily_rank_ic_mean`.
4. Combining scores into one composite factor.
5. Running the portfolio backtest on the composite.

Round portfolio history is written to `portfolio_history.json` and
`portfolio_history.csv`.

At the end of the main runner, `scripts/plot_portfolio_history.py` is called
automatically to generate:

- `portfolio_history_rank_ic.png`
- `portfolio_history_admissions.png`
- `portfolio_history_returns.png`
- `portfolio_round_return_curves.png`
- `portfolio_round_excess_curves.png`

`portfolio_history_rank_ic.png` plots both the factor-library composite rank IC
and each round's best candidate rank IC. The per-round best IC is stored in
`portfolio_history.json` as `round_best_rank_ic`; for older runs the plotting
script falls back to `round_XXX/round_summary.json`.

The plotting result is embedded in `summary.json` as
`portfolio_history_plots`. Plotting errors are recorded there and do not abort
the factor-mining run.

After all rounds finish, the runner performs one final out-of-sample test. By
default it uses `2025-01-01` to `2026-01-31`. The OOS window can be configured
with `--oos-start`, `--oos-end`, and `--oos-warmup-start`.

It recomputes every accepted library factor on the OOS window, rebuilds the same
rank-IC-weighted composite, then runs composite rank IC analysis and the
portfolio backtest once. The result is stored under
`final_oos/factor_library_portfolio/library_portfolio.json` and also embedded in
`summary.json` as `final_oos`.

Use `--skip-oos-test` to disable only this final OOS test.

## Memory

Memory follows the FactorMiner-style structure:

```text
M = { S, P_succ, P_fail, I }
```

- `S`: loop state, recent admissions/rejections, admission logs, error-frequency
  counters, IC history, and best accepted signals.
- `P_succ`: accepted factor and economic hypothesis patterns.
- `P_fail`: implementation failures, duplicate directions, and rejected patterns.
- `I`: compact strategic insights from reflexion.

Only compact pattern text is injected into RLM. Debug metrics and raw JSON remain
in artifacts.

Multiple admissions in the same round update:

- `state.recent_admissions`
- `state.admission_log`
- `state.library_size`
- one success pattern per accepted factor

## Reflexion Inputs

Economic reflexion receives Markdown text built after factor-library admission:

```text
# Current Factor Library Hypotheses
## Library Factor 1: <name>
- status: <status>
- signal_class: <class>
- hypothesis:
<saved hypothesis>
- rlm_summary:
<saved RLM summary>

# Current Round Labeled Final Answers
## Candidate 1
- label: <best|average|failed|duplicate>
- module: <module>
- ok: <bool>
- ic: <float|null>
- failure_reason: <reason>
- failure_detail: <detail>
- final_answer:
<candidate final answer>
```

It does not receive previous-round reflection text. The factor-library
hypotheses are the prior admitted ideas.

Economic reflexion writes hypothesis memory only. It must output exactly two
sections:

```text
## Recommended Directions (P_succ)
- <economic hypothesis worth continuing, why to select it, and how to reshape
  the hypothesis next round; optional concise example hypothesis, not a formula>

## Forbidden Directions (P_fail)
- <economic hypothesis to avoid, why it was weak/redundant/unsupported, and
  when it may be revisited; optional concise example hypothesis, not a formula>
```

It must not include factor construction, formulas, code, fields, operators,
windows, gates, normalization, or implementation details. The memory manager
parses these two sections directly and injects the hypothesis selection and
hypothesis construction guidance into later `P_succ`/`P_fail` prompt sections.

Implementation reflexion receives a dictionary containing only labeled
trajectory code blocks:

```text
{
  "type": "round_labeled_trajectory_code_blocks",
  "schema_version": 2,
  "candidates": [
    {
      "candidate_index": int,
      "label": str,
      "module": str,
      "ok": bool,
      "error_type": str,
      "error": str,
      "code_blocks": [
        {
          "iteration": int,
          "block_index": int,
          "code": str,
          "stdout": str,
          "stderr": str,
          "error": str,
          "final_answer": str
        }
      ],
      "trajectory_parse_error": str
    }
  ]
}
```

The reflection prompt asks the agent to compute current-round error frequency
itself and use `llm_query` only for reasoning-level sanity checks.

## No-LLM Demo

Run:

```powershell
python -B .\run_parallel_reflexion_demo.py --rounds 2 --candidates 3
```

The demo creates mock candidate signals, metrics, factor data, trajectories,
and reflexions, then runs the real evaluator, admission service, memory manager,
and artifact writer.
