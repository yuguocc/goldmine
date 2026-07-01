# 因子挖掘并行 Reflexion 架构

本文档按当前代码库说明 `src/factor_miner_parallel_reflexion/` 的架构、创新点、运行方式和主要产物。

## 1. 概览

Parallel Reflexion Factor Miner 是一个并行因子挖掘流水线：每轮同时让多个 RLM candidate 生成 `BaseSignal.compute()` 函数体，用真实 Qlib / quickbacktest 路径检查可运行性和 rank IC，再经过批内去重、因子库相关性审查、边际组合贡献门控后，把合格因子写入本轮隔离的 factor library，并把经验压缩进 Reflexion memory 供下一轮调度和提示使用。

核心目标不是只找到单个高 IC 因子，而是逐轮扩展一个低重复、可组合、可 OOS 复测的因子库。

## 2. 入口

- `factor_miner_parallel_reflexion.py`: 真实 RLM 挖掘入口，包装 `src.factor_miner_parallel_reflexion.cli.main()`。
- `python -m src.factor_miner_parallel_reflexion.demo`: no-LLM demo 入口，不调用 RLM provider，但复用真实的 job assignment、评价、准入、factor library、memory 和 artifact 代码。
- `src/factor_miner_parallel_reflexion/__init__.py`: 顶层导出兼容入口，保留 `parse_args()`、`config_from_args()`、`main()`。

当前仓库没有 `run_parallel_reflexion_demo.py` 顶层脚本。

## 3. 包结构

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

相关文件：

- `factor_miner.py`: 单因子 RLM 工具、`submit_compute()` 契约、final-answer validator、Qlib 初始化、signal analysis。
- `rlm_factor_memory.py`: `M={S,P_succ,P_fail,I}` memory 实现。
- `quickbacktest/`: BaseSignal、Qlib adapter、FactorLibrary、metric review、portfolio simulator。
- `scripts/plot_portfolio_history.py`: 运行结束后的组合历史图。

## 4. 默认配置和阈值

CLI 默认值：

- Output dir: `runs/factor_miner_parallel_reflexion`
- Factor library: 默认 `<output-dir>/factor_library`，除非显式传入 `--factor-library-path`
- Training window: `2023-01-01` to `2024-12-31`
- Final OOS window: `2025-01-01` to `2026-01-31`
- OOS warmup: 默认从 `--oos-start` 往前一年
- Instruments: `csi500`
- Benchmark: `SH000905`
- `topk=50`, `n_drop=5`, `horizon=1`, `factor_shift=1`
- Candidates per round: `6`
- Rounds per CLI run: `2`
- Max workers: `3`
- RLM max iterations per candidate: `5`
- Model / recursive model: `z-ai/glm-5.2`
- RLM logging: 默认开启

关键阈值：

- Fast screen rank IC: `daily_rank_ic_mean > 0.01`
- 批内重复阈值：absolute Spearman `>= 0.98`
- 因子库重复 / 替换阈值：absolute Spearman `>= 0.95`
- 边际组合贡献最小增量：`0.0`

## 5. 主流程

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

注意：复用同一个 `--output-dir` 会从 `memory.json` 续跑，因为 runner 从 `latest_round + 1` 开始。要做全新实验，应使用新的 `--output-dir`。

当前 runner 会跳过本次 CLI run 的最后一轮 Reflexion。最后一轮仍会更新 state、IC history、admission/rejection 和 OOS artifacts，只是不再消耗额外 RLM 调用生成下一轮才会用到的 reflection。

## 6. Candidate 生成契约

RLM 只写 `BaseSignal.compute(**kwargs)` 的 Python 函数体，不写 import、class、`def compute`、markdown fence、signal name 或 `return`。最终宽表 DataFrame 必须赋值给 `signal`；系统 wrapper 会追加 `return signal`，并创建当前候选类，例如 `RlmGeneratedFactorR001C004`。

候选输入被拆分为：

- `context`: available fields、allowed libraries/helpers、compact `memory_priors`、`existing_factors`、duplicate policy、candidate mode、mutation parent/axis、`parallel_generation`。
- `query`: 简短任务、分配说明、compute examples、submit flow、novelty checks、final summary format。

RLM-facing 工具：

- `submit_compute(compute_code)`: 包装 compute body，写入 `signals/<module>.py`，导入信号，计算 Qlib factor data，运行 IC analysis，并写 `run_signal_status.json`。
- read-only skill tools 也可用，但候选生成时移除了 `list_skills` 和 `read_signal_template`。

未暴露给候选生成器的工具包括 `submit_signal`、`save_signal`、`run_signal`、`read_signal_template` 和 factor-library 写工具。

契约由两层 gate 强制：RLM final-answer validator 会拒绝未通过 `submit_compute()` 的 final answer；`CandidateJobRunner` 在 RLM 返回后再次检查 `run_signal_status.json` 是否存在、是否 `ok: true`、是否匹配当前 module。

如果出现 `NegativeRankICRequiresReverse`，期望修复方式是将最终 `signal` 乘以 `-1` 后再次调用 `submit_compute()`。Prompt 明确禁止在 compute 中做 IC 搜索循环或训练 ML 模型。

## 7. Candidate 分配

`branches.py` 固定了六类研究分支：Momentum、Reversal、Liquidity、Volatility、Volume-price、Cross-sectional anomaly。

没有可用 memory 时采用确定性分配：候选 1-3 是 novelty；候选 4-6 在存在 accepted factor-library parent 时是 mutation，否则回退为 novelty。

Mutation mode 会收到一个 accepted parent card 和一个 mutation axis：`replace_gate`、`change_normalization`、`add_interaction`、`change_horizon_family`。Mutation 必须保留 parent 的大经济假设，但在指定轴上改变机制；禁止简单反号、纯参数调整或表面重写。

当 memory 中已有 admissions / rejections 时，`ResearchDirectionScheduler` 会替代纯轮转。它根据 branch、mode、mutation parent、mutation axis 统计历史成功率、探索 bonus、IC bonus、duplicate penalty 和 failure penalty，同时用 per-round caps 防止所有候选塌缩到同一方向。

## 8. 筛选与准入

候选评价在 RLM loop 外部完成：

1. `submit_compute()` 只证明信号可运行。
2. `compute_signal_analysis()` 物化 `factor_data.csv` 和 `analysis.json`。
3. runner 按顺序提取 IC：`daily_rank_ic_mean`、`rank_ic_distribution.mean`、`rank_ic`、`daily_ic_mean`、`ic_distribution.mean`。
4. 只有 finite rank IC 严格大于 `0.01` 才可用。

Round evaluation 会标注 `best`、`average`、`failed`、`duplicate`，并用 factor scores 的 Spearman correlation 做批内去重。absolute correlation `>= 0.98` 时保留 IC 更高者。

Factor-library admission 会按 IC 降序尝试所有正向非重复候选。准入条件包括 fast screen、deterministic metric review、accepted-library correlation check、replacement/blocking check 和 marginal contribution gate。通过的候选都会保存，不只保存 round best。

边际贡献门控会构建当前 training-window IC-weighted library composite 作为 baseline，再模拟加入 candidate 后的 composite。若是 replacement candidate，则模拟时移除被替换的高相关因子。通过条件为：

```text
with_candidate_composite_rank_ic - baseline_composite_rank_ic
>= --marginal-contribution-min-delta
```

首个 accepted component 因没有 baseline，会跳过该 gate。

## 9. Factor-Library Portfolio 和 OOS

每轮后，`FactorLibraryPortfolioService` 使用所有 accepted 且 rank IC 过阈值、存在 `factor_data.csv` 的因子构建组合：加载 score series，每日横截面 rank-normalize 到 percentile minus `0.5`，按 rank IC 加权，再按可用绝对权重归一，保存 composite factor，并运行组合回测。

主要产物：

```text
round_XXX/factor_library_portfolio/
  library_composite_factor.csv
  library_composite_analysis_factor.csv
  library_composite_weights.json
  library_composite_analysis.json
  library_portfolio.json
  portfolio/
```

全局 history 写入：

```text
portfolio_history.json
portfolio_history.csv
```

runner 结束时自动调用 `scripts/plot_portfolio_history.py`，生成 rank IC、admission、returns、per-round return curves、per-round excess curves 图。plot 失败会记录到 `summary.json`，不会中断主流程。

Final OOS test 默认开启。它会在 OOS period 重新计算每个 accepted signal，重建同样的 rank-IC-weighted composite，再运行 composite rank IC analysis 和 portfolio，并写入 `final_oos/factor_library_portfolio/library_portfolio.json`。可用 `--skip-oos-test` 关闭。

## 10. Memory 和 Reflexion

需要区分 memory 的存储 schema 和当前 Reflexion 的显式输出格式。

`rlm_factor_memory.py` 的存储 schema 仍然是：

```text
M = { S, P_succ, P_fail, I }
```

- `S`: runner / evaluator 维护的状态，包括 latest round、library size、recent admissions/rejections、admission log、IC history、accepted best signals、global best IC 等。
- `P_succ`: 从 accepted factors 和 Economic Reflexion 的 `Recommended Directions (P_succ)` 解析出的成功经济假设模式。
- `P_fail`: 从 rejected / duplicate / failed candidates 和 Economic Reflexion 的 `Forbidden Directions (P_fail)` 解析出的禁用或失败方向。
- `I`: 可选的派生 insights。当前没有单独的 LLM 输出 section 写 `I`；memory manager 会从已有 reflexion 文本或 library admission rejection 中提取 compact insight，存在时才会进入 retrieval prompt。

所以从当前 RLM Reflexion 输出契约看，显式要求只有两节：`P_succ` 和 `P_fail`。`S` 是系统状态，不是 LLM 写的；`I` 是 memory manager 派生字段，不是当前 economic Reflexion 单独产出的章节。

只有 compact retrieval text 会通过 `context["memory_priors"]` 注入候选生成。debug-heavy raw state 留在 artifacts 和 `memory.json`。

当前启用的 Reflexion phase 只有 `economic_hypothesis`。`IMPLEMENTATION_REFLEXION_PROMPT` 仍存在于 `constants.py`，memory manager 也能解析 implementation error frequency records，但 `RoundReflexionAgent.run_round()` 当前只运行 economic phase。

Economic Reflexion 输入写在：

```text
round_XXX/round_economic_context.md
```

它包含当前 factor-library hypotheses、当前轮 labeled final answers、label、module、branch、mode、IC、failure reason 和 failure detail。输出必须只有两节：

```text
## Recommended Directions (P_succ)
- <hypothesis-level guidance>

## Forbidden Directions (P_fail)
- <hypothesis-level guidance>
```

## 11. 创新点

1. 并行候选搜索与隔离执行：每个 candidate 独立 workspace、独立 module name、ProcessPool 执行，避免文件和 Qlib side effects 互相污染。
2. Compute-body-only RLM 契约：模型不能任意写 signal file 或跳过验证，只能提交 compute body，并由 `submit_compute()` 统一包装、保存、导入和验证。
3. Branch-constrained novelty 和 mutation：每个候选有明确研究分支，mutation 有 parent 和结构轴，novelty 是硬约束。
4. Memory-aware scheduler：根据历史 admission/rejection 对 branch、mode、parent、axis 进行带探索项的成功率调度。
5. Library-centric admission：每轮可接收多个候选，准入关注因子库增量价值，而不是只选 round best。
6. Composite-aware marginal gate：standalone IC 高但削弱当前 composite 的因子会被拒绝。
7. OOS recomputation：最终 OOS 不复用 in-sample scores，而是重新计算 accepted signals。
8. No-LLM demo：无需 LLM 即可测试评价、准入、memory 和 artifact 流程。

## 12. 如何运行

从仓库根目录运行：

```powershell
cd E:\ureca\RLMs\goldmine
```

No-LLM smoke test：

```powershell
python -B -m src.factor_miner_parallel_reflexion.demo --rounds 2 --candidates 3
```

真实 RLM 默认运行：

```powershell
python -B .\factor_miner_parallel_reflexion.py
```

推荐显式运行：

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

常用开关：

```powershell
# 保存 candidate-level portfolio artifacts
python -B .\factor_miner_parallel_reflexion.py --run-portfolio

# 更严格的边际贡献门控
python -B .\factor_miner_parallel_reflexion.py --marginal-contribution-min-delta 0.003

# 关闭部分昂贵步骤
python -B .\factor_miner_parallel_reflexion.py --skip-library-portfolio --skip-oos-test

# 关闭边际贡献门控
python -B .\factor_miner_parallel_reflexion.py --skip-marginal-contribution-gate
```

手动重画 history：

```powershell
python -B .\scripts\plot_portfolio_history.py `
  .\runs\factor_miner_parallel_reflexion_exp01\portfolio_history.json
```

相关测试：

```powershell
python -B -m pytest `
  quickbacktest\tests\test_parallel_reflexion_runner.py `
  quickbacktest\tests\test_parallel_reflexion_portfolio.py `
  quickbacktest\tests\test_parallel_reflexion_marginal_gate.py
```

## 13. 主要产物

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
