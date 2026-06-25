from __future__ import annotations

from factor_miner import (
    DEFAULT_FACTOR_LIBRARY_PATH,
    DEFAULT_PROVIDER_URI,
    DEFAULT_RLM_MODEL,
    DEFAULT_SKILL_PATH,
    PROJECT_ROOT,
    RLM_SUMMARY_SPEC,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "factor_miner_parallel_reflexion"
DEFAULT_TRAIN_START = "2023-01-01"
DEFAULT_TRAIN_END = "2024-12-31"
DEFAULT_OOS_START = "2025-01-01"
DEFAULT_OOS_END = "2026-01-31"
MEMORY_FILENAME = "memory.json"
MAX_SIGNAL_SOURCE_CHARS = 16000
BATCH_DUPLICATE_CORRELATION_THRESHOLD = 0.98
LIBRARY_CORRELATION_THRESHOLD = 0.95
FAST_SCREEN_RANK_IC_THRESHOLD = 0.01
ECONOMIC_REFLEXION_PROMPT = """Economic reflexion for one completed factor-generation round.

Input: Markdown text with two sections: current factor-library hypotheses and current-round labeled final answers. The library section contains saved hypotheses and summaries. The final-answer section contains candidate labels, rank IC, failure_reason, failure_detail, and final answer.
Task: write compact hypothesis memory only. Compare current-round outcomes against the existing factor-library hypotheses, then decide which economic hypotheses to continue, mutate, or avoid.

Output Markdown exactly:
## Recommended Directions (P_succ)
- Bullet points. Each bullet must state one economic hypothesis worth continuing or mutating, why it was selected, and how to reshape the hypothesis next round. Include a concise example hypothesis if useful, but no formula.

## Forbidden Directions (P_fail)
- Bullet points. Each bullet must state one economic hypothesis to avoid, why it was weak/redundant/unsupported, and when it may be revisited. Include a concise example hypothesis if useful, but no formula.

Rules: output only the two sections above and no extra headings. Do not include factor construction, formulas, code, fields, operators, windows, gates, normalization, or implementation details. Use rank IC only to compare hypotheses at a high level. Focus only on hypothesis selection and hypothesis construction. Keep each bullet compact.
"""

IMPLEMENTATION_REFLEXION_PROMPT = """Implementation reflexion for one completed factor-generation round.

Input: a dictionary with labeled trajectory code blocks only. The context is `type`, `schema_version`, and `candidates`. Each `candidates[i]` has label/module/ok/error fields and `code_blocks`; each code block is extracted from trajectory `metadata.iterations[*].code_blocks[*]` and includes only compact code/stdout/stderr/error/final_answer fields.
Task: use REPL code to count current error-type frequency from candidate labels, error_type/error fields, and trajectory code_blocks; then extract repeated implementation failures and robust patterns. Use llm_query only to sanity-check whether extracted features imply a fix; do not run signal or quickbacktest code.

Output Markdown exactly:
## Search Metadata
- memory_type: implementation_errors
- keywords: comma-separated search terms
- failure_types: error_type values from Error Frequency, ordered by descending count
- tools_or_functions: affected tools/functions
- data_contracts: required files, columns, schemas, returns
- robust_pattern: pattern worth reusing
- avoid_pattern: pattern to avoid
- next_action: one next implementation fix

## Overall Guidance
One short paragraph.

## Error Frequency
- error_type=<stable_error_type>; count=<int>; frequency=<0-1>; cause=<short cause>; evidence=<where observed>; fix=<short fix>

## Failed Function Calls
- Bullet points.

## Error Patterns
- Bullet points.

## Robust Patterns
- Bullet points.

## Next Round Fixes
- Bullet points.

Rules: no JSON, raw code, stack traces, temp paths, or performance metrics. `Error Frequency` must be computed by you from the current context and distinguish code errors from weak-factor failures. Use one bullet per error_type with the exact keys above; counts must sum repeated examples of the same error_type in the current round. Focus on submit_signal, imports/names/signatures, data access, alignment, NaN/inf, and output contracts. Keep brief.
"""
