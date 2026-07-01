# Factor Miner Parallel Reflexion Structure

This document describes the current architecture, innovations, runbook, and artifacts for `src/factor_miner_parallel_reflexion/`.

## 1. Overview

Parallel Reflexion Factor Miner is a parallel factor-mining pipeline. In each round it asks multiple RLM candidates to generate only the body of `BaseSignal.compute()`, validates each candidate through the real Qlib / quickbacktest path, screens by rank IC, removes duplicates, checks factor-library correlation, applies a marginal composite-contribution gate, saves accepted factors into a run-local factor library, and compresses experience into Reflexion memory for later scheduling and prompting.

The goal is not only to find one high-IC factor. The goal is to grow a low-duplication, composable factor library that can be recomputed and tested out of sample.

## 2. Entry Points

- `factor_miner_parallel_reflexion.py`: real RLM mining CLI wrapper around `src.factor_miner_parallel_reflexion.cli.main()`.
- `python -m src.factor_miner_parallel_reflexion.demo`: no-LLM demo entry point. It does not call an RLM provider, but it reuses the real job assignment, evaluation, admission, factor-library, memory, and artifact code.
- `src/factor_miner_parallel_reflexion/__init__.py`: compatibility exports for `parse_args()`, `config_from_args()`, and `main()`.

There is currently no top-level `run_parallel_reflexion_demo.py` script in this repository.

## 3. Package Layout

```text
src/factor_miner_parallel_reflexion/
  __init__.py        # package exports and CLI compatibility helpers
  constants.py       # defaults, thresholds, reflexion prompts
  models.py          # dataclass configs, candidate/result/evaluation models
  memory.py          # re-export rlm_factor_memory.py memory manager
  branches.py        # fixed price/volume research branches
  scheduler.py       # memory-aware candidate assignment scheduler
  candidate.py       # prompt/context builder, job factory, ProcessPool execution
  library.py         # factor-library admission, correlation, replacement logic
  evaluation.py      # round labels, batch dedup, admission orchestration
  portfolio.py       # factor-library composite, portfolio, OOS test, history
  reflexion.py       # round artifacts and enabled Reflexion agent
  runner.py          # stateful end-to-end mining loop
  cli.py             # argparse adapter
  demo.py            # deterministic no-LLM demo
  utils.py           # shared JSON, IC, factor-data, memory helpers
```

Related files:

- `factor_miner.py`: single-factor RLM tools, `submit_compute()` contract, final-answer validator, Qlib initialization, and signal analysis.
- `rlm_factor_memory.py`: implementation of `M={S,P_succ,P_fail,I}` memory.
- `quickbacktest/`: BaseSignal, Qlib adapter, FactorLibrary, metric review, and portfolio simulator.
- `scripts/plot_portfolio_history.py`: end-of-run portfolio history plots.

## 4. Defaults And Thresholds

CLI defaults:

- Output dir: `runs/factor_miner_parallel_reflexion`
- Factor library: `<output-dir>/factor_library` unless `--factor-library-path` is passed explicitly
- Training window: `2023-01-01` to `2024-12-31`
- Final OOS window: `2025-01-01` to `2026-01-31`
- OOS warmup: one year before `--oos-start` by default
- Instruments: `csi500`
- Benchmark: `SH000905`
- `topk=50`, `n_drop=5`, `horizon=1`, `factor_shift=1`
- Candidates per round: `6`
- Rounds per CLI run: `2`
- Max workers: `3`
- RLM max iterations per candidate: `5`
- Model / recursive model: `z-ai/glm-5.2`
- RLM logging: enabled by default

Current thresholds:

- Fast-screen rank IC: `daily_rank_ic_mean > 0.01`
- Intra-batch duplicate threshold: absolute Spearman `>= 0.98`
- Factor-library duplicate / replacement threshold: absolute Spearman `>= 0.95`
- Minimum marginal composite contribution: `0.0`

## 5. Main Flow

```text
ParallelReflexionCLI
-> ParallelReflexionRunner
  -> create / load run-local memory.json
  -> use run-local factor_library unless a custom path is passed
  -> for each requested round, starting from memory.state.latest_round + 1:
    -> build compact Factorminer memory prompt
    -> build candidate jobs
      -> assign branch, mode, optional mutation parent, optional mutation axis
      -> create isolated workspace under round_XXX/
    -> run candidates in ProcessPool
      -> initialize Qlib in each process
      -> call RLM_REPL.completion(context, query)
      -> expose submit_compute(compute_code) plus read-only skill tools
      -> enforce final answer only after submit_compute ok=True
      -> externally compute factor_data.csv and analysis.json
      -> mark candidate usable only when finite rank IC > 0.01
    -> label results as best / average / duplicate / failed
    -> remove near-duplicates inside the batch
    -> admit admissible candidates by descending IC
      -> deterministic quickbacktest review
      -> factor-library correlation check
      -> replacement or blocking decision
      -> marginal contribution gate against current library composite
      -> save every passing candidate into factor_library
    -> build IC-weighted factor-library composite and run portfolio
    -> write round artifacts and portfolio history
    -> run economic Reflexion except on the final round of this CLI run
    -> update memory.json
  -> run final OOS composite test unless skipped
  -> plot portfolio history if portfolio_history.json exists
  -> write summary.json
```

Reusing the same `--output-dir` resumes from `memory.json` because the runner starts at `latest_round + 1`. Use a new `--output-dir` for a clean experiment.

The runner skips Reflexion on the final round of the current CLI run. The final round still updates state, IC history, admissions/rejections, and OOS artifacts; it just avoids spending another RLM call on reflection that no later round in the same run will consume.

## 6. Candidate Generation Contract

The RLM writes only the Python body of `BaseSignal.compute(**kwargs)`. It must not write imports, class definitions, `def compute`, markdown fences, signal names, or `return` statements. The final wide DataFrame must be assigned to `signal`; the system wrapper appends `return signal` and creates the current candidate class, such as `RlmGeneratedFactorR001C004`.

Candidate input is split into:

- `context`: available fields, allowed libraries/helpers, compact `memory_priors`, `existing_factors`, duplicate policy, candidate mode, mutation parent/axis, and `parallel_generation`.
- `query`: the short task, assignment text, compute examples, submit flow, novelty checks, and final summary format.

RLM-facing tools:

- `submit_compute(compute_code)`: wraps the compute body, writes `signals/<module>.py`, imports the signal, computes Qlib factor data, runs IC analysis, and writes `run_signal_status.json`.
- Read-only skill tools are available, but `list_skills` and `read_signal_template` are removed from the candidate tool surface.

The parallel generator does not expose `submit_signal`, `save_signal`, `run_signal`, `read_signal_template`, or factor-library write tools.

The contract is enforced twice: the RLM final-answer validator rejects answers before `submit_compute()` succeeds, and `CandidateJobRunner` repeats the check after RLM returns. If `run_signal_status.json` is missing, failed, or belongs to another module, the candidate is rejected before external IC analysis.

If `NegativeRankICRequiresReverse` appears, the expected repair is to multiply the final `signal` by `-1` and call `submit_compute()` again. The prompt discourages IC-search loops and ML training inside `compute()`.

## 7. Candidate Assignment

`branches.py` defines six research branches: Momentum, Reversal, Liquidity, Volatility, Volume-price, and Cross-sectional anomaly.

Without usable memory, assignment is deterministic: candidates 1-3 are novelty candidates; candidates 4-6 become mutation candidates only when accepted factor-library parents exist. Without accepted parents, mutation slots fall back to novelty.

Mutation mode receives an accepted parent card and one mutation axis: `replace_gate`, `change_normalization`, `add_interaction`, or `change_horizon_family`. Mutation must preserve the parent's broad economic family while changing the mechanism on the assigned axis. Simple sign flips, parameter-only changes, and cosmetic rewrites are forbidden.

When memory has admissions or rejections, `ResearchDirectionScheduler` replaces pure rotation. It scores branch, mode, mutation parent, and mutation axis using smoothed success probability, an exploration bonus, positive IC bonus, duplicate penalty, and failure penalty. Per-round caps prevent all candidates from collapsing onto one direction.

## 8. Screening And Admission

Candidate evaluation is deliberately outside the RLM loop:

1. `submit_compute()` proves the signal is runnable.
2. `compute_signal_analysis()` materializes `factor_data.csv` and `analysis.json`.
3. The runner extracts IC in this order: `daily_rank_ic_mean`, `rank_ic_distribution.mean`, `rank_ic`, `daily_ic_mean`, `ic_distribution.mean`.
4. A candidate is usable only when finite rank IC is strictly greater than `0.01`.

Round evaluation labels candidates as `best`, `average`, `failed`, or `duplicate`, and runs intra-batch Spearman deduplication on materialized factor scores. When absolute correlation is `>= 0.98`, the higher-IC candidate is kept.

Factor-library admission tries all positive nonduplicate candidates by descending IC. It requires the fast screen, deterministic metric review, accepted-library correlation check, replacement/blocking check, and marginal contribution gate. Every passing candidate is saved, not only the round best.

The marginal contribution gate builds the current training-window IC-weighted library composite as baseline, then simulates the composite after adding the candidate. For replacement candidates, the simulation removes the replaced high-correlation factors. The pass condition is:

```text
with_candidate_composite_rank_ic - baseline_composite_rank_ic
>= --marginal-contribution-min-delta
```

The first accepted component skips this gate because no library baseline exists yet.

## 9. Factor-Library Portfolio And OOS

After each round, `FactorLibraryPortfolioService` builds a composite from all accepted library factors whose rank IC passes the threshold and whose `factor_data.csv` is available. It loads factor score series, daily cross-sectionally rank-normalizes each factor to percentile minus `0.5`, weights each factor by rank IC, normalizes by available absolute weight, saves the composite factor, and runs the portfolio backtest.

Main artifacts:

```text
round_XXX/factor_library_portfolio/
  library_composite_factor.csv
  library_composite_analysis_factor.csv
  library_composite_weights.json
  library_composite_analysis.json
  library_portfolio.json
  portfolio/
```

Global history is written to:

```text
portfolio_history.json
portfolio_history.csv
```

At the end of the runner, `scripts/plot_portfolio_history.py` generates rank IC, admission, return, per-round return curve, and per-round excess curve plots. Plot failures are recorded in `summary.json` and do not abort the run.

The final OOS test is enabled by default. It recomputes every accepted signal on the OOS period, rebuilds the same rank-IC-weighted composite, runs composite rank IC analysis and portfolio once, and writes `final_oos/factor_library_portfolio/library_portfolio.json`. Use `--skip-oos-test` to disable it.

## 10. Memory And Reflexion

There are two separate concepts: the memory storage schema and the explicit output format of the currently enabled Reflexion agent.

The storage schema in `rlm_factor_memory.py` is still:

```text
M = { S, P_succ, P_fail, I }
```

- `S`: system state maintained by the runner / evaluator, including latest round, library size, recent admissions/rejections, admission log, IC history, accepted best signals, and global best IC.
- `P_succ`: successful economic hypothesis patterns parsed from accepted factors and the Economic Reflexion `Recommended Directions (P_succ)` section.
- `P_fail`: forbidden or failed directions parsed from rejected / duplicate / failed candidates and the Economic Reflexion `Forbidden Directions (P_fail)` section.
- `I`: optional derived insights. There is no separate current LLM output section for `I`; the memory manager may derive compact insights from existing reflexion text or library admission rejection, and those insights enter the retrieval prompt only when present.

So, from the current RLM Reflexion output contract, the explicit output has only two sections: `P_succ` and `P_fail`. `S` is system-maintained state, not LLM-written content; `I` is a derived memory-manager field, not a standalone section produced by the current economic Reflexion prompt.

Only compact retrieval text is injected into candidate generation through `context["memory_priors"]`. Debug-heavy raw state stays in artifacts and `memory.json`.

The only currently enabled Reflexion phase is `economic_hypothesis`. `IMPLEMENTATION_REFLEXION_PROMPT` still exists in `constants.py`, and the memory manager can parse implementation error frequency records, but `RoundReflexionAgent.run_round()` currently runs only the economic phase.

Economic Reflexion input is written to:

```text
round_XXX/round_economic_context.md
```

It contains current factor-library hypotheses, current-round labeled final answers, label, module, branch, mode, IC, failure reason, and failure detail. Output must contain exactly these two sections:

```text
## Recommended Directions (P_succ)
- <hypothesis-level guidance>

## Forbidden Directions (P_fail)
- <hypothesis-level guidance>
```

## 11. Innovation Points

1. Parallel candidate search with isolated execution: each candidate has its own workspace, deterministic module name, and ProcessPool worker.
2. Compute-body-only RLM contract: the model cannot write arbitrary signal files or skip validation; `submit_compute()` owns wrapping, saving, importing, and validation.
3. Branch-constrained novelty and mutation: every candidate has a research branch, and mutation candidates have a parent plus one structural axis.
4. Memory-aware scheduler: branch, mode, parent, and axis are allocated from historical admission/rejection statistics with exploration and penalty terms.
5. Library-centric admission: a round can admit multiple candidates, and admission optimizes factor-library value rather than only the round best.
6. Composite-aware marginal gate: high standalone IC is insufficient if the factor weakens the current composite.
7. OOS recomputation: final OOS recomputes accepted signals instead of reusing in-sample scores.
8. No-LLM demo: evaluation, admission, memory, and artifact flow can be tested without an LLM call.

## 12. How To Run

Run from the repository root:

```powershell
cd E:\ureca\RLMs\goldmine
```

No-LLM smoke test:

```powershell
python -B -m src.factor_miner_parallel_reflexion.demo --rounds 2 --candidates 3
```

Real RLM run with defaults:

```powershell
python -B .\factor_miner_parallel_reflexion.py
```

Recommended explicit run:

```powershell
python -B .\factor_miner_parallel_reflexion.py `
  --output-dir runs\factor_miner_parallel_reflexion_exp01 `
  --provider-uri .qlib\qlib_data\cn_data `
  --rounds 2 `
  --candidates 6 `
  --max-workers 3 `
  --max-iterations 5 `
  --disable-rlm-logging
```

Common switches:

```powershell
# Save candidate-level portfolio artifacts
python -B .\factor_miner_parallel_reflexion.py --run-portfolio

# Require stricter marginal composite contribution
python -B .\factor_miner_parallel_reflexion.py --marginal-contribution-min-delta 0.003

# Skip selected expensive steps
python -B .\factor_miner_parallel_reflexion.py --skip-library-portfolio --skip-oos-test

# Disable only the marginal contribution gate
python -B .\factor_miner_parallel_reflexion.py --skip-marginal-contribution-gate
```

Replot an existing history:

```powershell
python -B .\scripts\plot_portfolio_history.py `
  .\runs\factor_miner_parallel_reflexion_exp01\portfolio_history.json
```

Relevant tests:

```powershell
python -B -m pytest `
  quickbacktest\tests\test_parallel_reflexion_runner.py `
  quickbacktest\tests\test_parallel_reflexion_portfolio.py `
  quickbacktest\tests\test_parallel_reflexion_marginal_gate.py
```

## 13. Main Artifacts

```text
<output-dir>/
  memory.json
  summary.json
  portfolio_history.json
  portfolio_history.csv
  portfolio_history_rank_ic.png
  portfolio_history_admissions.png
  portfolio_history_returns.png
  portfolio_round_return_curves.png
  portfolio_round_excess_curves.png
  factor_library/
  final_oos/
  round_001/
  round_002/
```

```text
round_XXX/
  round_summary.json
  factor_library_admission.json
  round_economic_context.md
  round_trajectories.json
  economic_hypothesis_reflexion_trajectory.json
  factor_library_portfolio/
  marginal_contribution/
  candidate_001_<random>/
```

```text
candidate_XXX_<random>/
  rlm_trajectory.json
  run_signal_status.json
  candidate_result.json
  analysis.json
  factor_data.csv
  signals/<module>.py
  candidate_error.json        # only on candidate execution failure
  portfolio.json              # only when --run-portfolio is enabled
```

```text
factor_library/<factor-name>/
  FACTOR.md
  signal.py
  metrics.json
  review.json
  factor_data.csv
```
