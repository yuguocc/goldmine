from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBS_ROOT = PROJECT_ROOT / "libs"
DEFAULT_WORKSPACE = PROJECT_ROOT / "runs" / "factor_miner_case"
DEFAULT_PROVIDER_URI = PROJECT_ROOT / ".qlib" / "qlib_data" / "cn_data"
DEFAULT_RLM_MODEL = "google/gemini-3-flash-preview"
DEFAULT_SKILL_PATH = PROJECT_ROOT / "skills"
DEFAULT_FACTOR_LIBRARY_PATH = PROJECT_ROOT / "factors"

CASE_SIGNAL_MODULE = "RlmCaseMomentumVol"
GENERATED_SIGNAL_MODULE = "RlmGeneratedFactor"
RLM_SUMMARY_SPEC = """RLM summary output spec:
- The final answer['content'] is the RLM summary saved into FACTOR.md.
- Keep it concise and aligned with the quickbacktest signal template.
- Use exactly this Markdown field format:
  type: <momentum | mean_reversion | volatility | liquidity | hybrid>
  hypothesis:
    hp1: <one sentence primary predictive hypothesis>
    hp2: <one sentence condition, risk, or direction-control note>
  factor: <factor_name>: <short formula or pandas-level definition>
  explanation: <one concise economic intuition>
  validation: <submit_compute ok/error status; do not include numeric metrics>
- Do not include range, IC, ICIR, coverage, or rank_ic fields; metrics.json is the source of numeric evaluation.
"""
CASE_SIGNAL_CODE = """from quickbacktest.base_types import BaseSignal
import pandas as pd


class RlmCaseMomentumVol(BaseSignal):
    name = "RlmCaseMomentumVol"

    def compute(self, **kwargs):
        fast_window = int(kwargs.get("fast_window", 5))
        slow_window = int(kwargs.get("slow_window", 20))
        liquidity_floor = float(kwargs.get("liquidity_floor", 0.2))

        fast_return = self.close.pct_change(fast_window, fill_method=None)
        slow_return = self.close.pct_change(slow_window, fill_method=None)
        daily_return = self.close.pct_change(fill_method=None)
        volatility = daily_return.rolling(
            slow_window,
            min_periods=max(3, slow_window // 2),
        ).std()
        volatility = volatility.mask(volatility.abs() < 1e-12)

        liquidity_rank = (
            self.amount.rolling(slow_window, min_periods=max(3, slow_window // 2))
            .mean()
            .rank(axis=1, pct=True)
        )

        signal = (fast_return - slow_return) / volatility
        signal = signal.where(liquidity_rank >= liquidity_floor)
        signal.index.name = "trade_time"
        return signal
"""

COMPUTE_BODY_EXAMPLES: list[dict[str, str]] = [
    {
        "name": "momentum",
        "code": """window = int(kwargs.get("window", 20))
signal = self.close.pct_change(window, fill_method=None)
signal = signal.replace([np.inf, -np.inf], np.nan)""",
    },
    {
        "name": "mean_reversion",
        "code": """window = int(kwargs.get("window", 10))
ret = self.close.pct_change(window, fill_method=None)
signal = -ret
signal = signal.replace([np.inf, -np.inf], np.nan)""",
    },
]


def compute_examples_prompt_text() -> str:
    """Render the compact compute template/examples for RLM query text."""
    blocks = [
        "Compute template:",
        "```python",
        COMPUTE_BODY_EXAMPLES[0]["code"],
        "```",
        "",
        "Compute examples:",
    ]
    for item in COMPUTE_BODY_EXAMPLES:
        blocks.extend(
            [
                f"Example: {item['name']}",
                "```python",
                item["code"],
                "```",
            ]
        )
    blocks.extend(
        [
            "",
            "Implementation reminders:",
            "- Preserve the original DataFrame index/columns; the final signal must align with self.close.",
            "- If TA-Lib or another function returns a numpy array inside DataFrame.apply, wrap it as pd.Series(values, index=x.index) before combining with price/volume DataFrames.",
            "- Prefer pandas rolling/ewm operations when possible because they preserve alignment.",
        ]
    )
    return "\n".join(blocks)


@dataclass(frozen=True)
class FactorMinerCaseConfig:
    workspace: Path = DEFAULT_WORKSPACE
    provider_uri: Path = DEFAULT_PROVIDER_URI
    module_name: str = GENERATED_SIGNAL_MODULE
    instruments: str = "csi500"
    start: str = "2023-01-01"
    end: str = "2024-12-31"
    benchmark: str = "SH000905"
    topk: int = 50
    n_drop: int = 5
    horizon: int = 1
    factor_shift: int = 1
    run_portfolio: bool = True
    train_alpha158: bool = False
    use_case_signal: bool = False
    model: str = DEFAULT_RLM_MODEL
    recursive_model: str = DEFAULT_RLM_MODEL
    max_iterations: int = 5
    enable_rlm_logging: bool = True


@dataclass(frozen=True)
class GeneratedSignal:
    module_name: str
    signal_path: Path
    rlm_response: str
    trajectory_path: Path
    rlm_generation_elapsed_seconds: float
    rlm_metadata: dict[str, Any] | None = None


def _valid_module_name(module_name: str) -> bool:
    return (
        isinstance(module_name, str)
        and module_name.isidentifier()
        and not module_name.startswith("_")
    )


def save_signal_code(
    workspace: str | Path,
    module_name: str,
    code: str,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write a signal source into <workspace>/signals without pre-validation."""
    if not _valid_module_name(module_name):
        return {
            "ok": False,
            "errors": ["module_name must be a public Python identifier"],
        }
    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "errors": ["code must be a non-empty string"]}

    workspace_path = Path(workspace).resolve()
    signal_dir = workspace_path / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    signal_path = (signal_dir / f"{module_name}.py").resolve()
    if workspace_path not in signal_path.parents:
        return {"ok": False, "errors": ["resolved signal path escapes workspace"]}
    if signal_path.exists() and not overwrite:
        return {"ok": False, "errors": [f"signal already exists: {signal_path}"]}

    signal_path.write_text(code, encoding="utf-8")
    return {
        "ok": True,
        "module_name": module_name,
        "signal_path": str(signal_path),
    }


def _strip_markdown_code_fence(code: str) -> str:
    text = str(code or "").strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_compute_body(code: str) -> tuple[str, list[str]]:
    text = _strip_markdown_code_fence(code)
    if not text:
        return "", ["compute_code must be a non-empty string"]

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, []

    if any(isinstance(node, ast.ClassDef) for node in tree.body):
        return "", ["compute_code must not include class definitions"]

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions:
        if len(functions) == 1 and functions[0].name == "compute":
            function = functions[0]
            if not function.body:
                return "", ["compute() body must not be empty"]
            lines = text.splitlines()
            start = function.body[0].lineno - 1
            end = function.body[-1].end_lineno
            return textwrap.dedent("\n".join(lines[start:end])).strip(), []
        return "", ["compute_code must not define functions; provide only compute body"]

    return text, []


def _return_is_signal(node: ast.Return) -> bool:
    value = node.value
    return isinstance(value, ast.Name) and value.id == "signal"


def _remove_statement_from_compute_body(body: str, node: ast.AST) -> str:
    lines = body.splitlines()
    start = max(0, getattr(node, "lineno", 2) - 2)
    end = max(start + 1, getattr(node, "end_lineno", start + 2) - 1)
    del lines[start:end]
    return "\n".join(lines).strip()


def _target_assigns_signal(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return target.id == "signal"
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_assigns_signal(item) for item in target.elts)
    return False


def _function_assigns_signal(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if any(_target_assigns_signal(target) for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if _target_assigns_signal(node.target):
                return True
        elif isinstance(node, ast.AugAssign):
            if _target_assigns_signal(node.target):
                return True
    return False


def normalize_compute_body(compute_code: str) -> tuple[str, list[str]]:
    body, errors = _extract_compute_body(compute_code)
    if errors:
        return "", errors

    def parse_body(candidate: str) -> tuple[ast.FunctionDef | None, list[str]]:
        wrapper = "def compute(self, **kwargs):\n" + textwrap.indent(candidate, "    ")
        try:
            tree = ast.parse(wrapper)
        except SyntaxError as exc:
            return None, [f"compute_code syntax error: {exc.msg} at line {exc.lineno}"]
        function = tree.body[0] if tree.body else None
        if not isinstance(function, ast.FunctionDef):
            return None, ["internal parser error: compute wrapper was not a function"]
        return function, []

    function, errors = parse_body(body)
    if errors:
        return "", errors
    if function is None:
        return "", ["internal parser error: compute wrapper was not a function"]

    return_nodes = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    final_stmt = function.body[-1] if function.body else None
    if return_nodes:
        if (
            len(return_nodes) == 1
            and return_nodes[0] is final_stmt
            and _return_is_signal(return_nodes[0])
        ):
            body = _remove_statement_from_compute_body(body, return_nodes[0])
            if not body:
                return "", ["compute_code must assign the final DataFrame to signal"]
            function, errors = parse_body(body)
            if errors:
                return "", errors
            if function is None:
                return "", ["internal parser error: compute wrapper was not a function"]
        else:
            return "", [
                "do not write return statements; assign the final DataFrame to signal"
            ]

    errors = []
    for node in ast.walk(function):
        if node is function:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append("do not import libraries inside compute_code")
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.append("do not define classes or functions inside compute_code")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "fit":
                errors.append("do not call fit() or train models inside compute_code")
            elif isinstance(func, ast.Name) and func.id == "fit":
                errors.append("do not call fit() or train models inside compute_code")

    if not _function_assigns_signal(function):
        errors.append("compute_code must assign the final DataFrame to signal")

    if errors:
        return "", sorted(set(errors))
    return body, []


def build_signal_code_from_compute(module_name: str, compute_code: str) -> str:
    """Wrap a compute body into the standard quickbacktest BaseSignal file."""
    if not _valid_module_name(module_name):
        raise ValueError("module_name must be a public Python identifier")
    body, errors = normalize_compute_body(compute_code)
    if errors:
        raise ValueError("; ".join(errors))

    indented_body = textwrap.indent(body, "        ")
    return (
        "from quickbacktest.base_types import BaseSignal\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import talib as ta\n\n\n"
        f"class {module_name}(BaseSignal):\n"
        f'    name = "{module_name}"\n\n'
        "    def compute(self, **kwargs) -> pd.DataFrame:\n"
        f"{indented_body}\n"
        "        return signal\n"
    )


def build_factor_miner_custom_tools(
    workspace: str | Path,
    config: FactorMinerCaseConfig | None = None,
    *,
    expose_legacy_tools: bool = False,
) -> dict[str, Any]:
    """Build the custom tool set exposed to the RLM factor generator."""
    workspace_path = Path(workspace)
    tool_config = config or FactorMinerCaseConfig(
        workspace=workspace_path,
        run_portfolio=False,
        train_alpha158=False,
    )

    def read_factor_template(module_name: str | None = None) -> str:
        from quickbacktest import read_signal_template

        return read_signal_template(module_name or tool_config.module_name)

    def save_signal(
        code: str,
        module_name: str | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        target_module_name = tool_config.module_name
        warning = ""
        if module_name is not None and _valid_module_name(code):
            # Backward compatibility for old calls: save_signal(module_name, code).
            code, module_name = module_name, code
            warning = (
                "module_name positional argument is deprecated; save_signal(code) "
                "now writes the current candidate signal."
            )
        return save_signal_code(
            workspace_path,
            target_module_name,
            code,
            overwrite=overwrite,
        ) | ({"warning": warning} if warning else {})

    def run_signal(module_name: str | None = None) -> dict[str, Any]:
        return run_saved_signal(tool_config, module_name=module_name)

    def _submit_source_code(
        code: str,
        *,
        target_module_name: str,
        overwrite: bool = True,
        warning: str = "",
    ) -> dict[str, Any]:
        save_result = save_signal_code(
            workspace_path,
            target_module_name,
            code,
            overwrite=overwrite,
        )
        if not save_result.get("ok"):
            error = "; ".join(str(item) for item in save_result.get("errors", []))
            status = {
                "ok": False,
                "module_name": target_module_name,
                "error_type": "SaveSignalError",
                "error": error,
                "save": save_result,
            }
            if warning:
                status["warning"] = warning
            _write_json(workspace_path / "run_signal_status.json", status)
            return {
                "ok": False,
                "error_type": "SaveSignalError",
                "error": error,
            }

        run_result = run_saved_signal(tool_config, module_name=target_module_name)
        if run_result.get("ok") is True:
            return {"ok": True}
        return {
            "ok": False,
            "error_type": str(run_result.get("error_type") or "UnknownError"),
            "error": str(run_result.get("error") or "submit_compute returned ok=False"),
        }

    def submit_compute(
        compute_code: str,
        module_name: str | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        target_module_name = tool_config.module_name
        if module_name is not None and _valid_module_name(compute_code):
            # Backward compatibility for accidental submit_compute(module_name, body).
            compute_code, module_name = module_name, compute_code
        try:
            source_code = build_signal_code_from_compute(
                target_module_name,
                compute_code,
            )
        except ValueError as exc:
            status = {
                "ok": False,
                "module_name": target_module_name,
                "error_type": "ComputeCodeError",
                "error": str(exc),
            }
            _write_json(workspace_path / "run_signal_status.json", status)
            return {
                "ok": False,
                "error_type": "ComputeCodeError",
                "error": str(exc),
            }
        return _submit_source_code(
            source_code,
            target_module_name=target_module_name,
            overwrite=overwrite,
        )

    def submit_signal(
        code: str,
        module_name: str | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        target_module_name = tool_config.module_name
        warning = ""
        if module_name is not None and _valid_module_name(code):
            # Backward compatibility for accidental submit_signal(module_name, code).
            code, module_name = module_name, code
            warning = (
                "module_name positional argument is ignored; submit_signal(code) "
                "always writes and runs the current candidate signal."
            )

        return _submit_source_code(
            code,
            target_module_name=target_module_name,
            overwrite=overwrite,
            warning=warning,
        )

    tools = {
        "submit_compute": {
            "tool": submit_compute,
            "description": (
                "submit_compute(compute_code, module_name=None, overwrite=True) "
                "wraps a BaseSignal.compute body into the current candidate class, "
                "writes the signal file, imports it, computes Qlib factor data, "
                "and runs IC analysis. Do not write imports, class definitions, "
                "def compute, markdown fences, signal names, or return statements. "
                "Assign the final wide DataFrame to variable signal; the wrapper "
                "appends return signal. Final answer is allowed only when this "
                "returns ok=True. The return value is only ok plus error_type/error "
                "when ok=False. NegativeRankICRequiresReverse "
                "means multiply the final signal by -1 and submit again."
            ),
        },
    }
    if expose_legacy_tools:
        tools.update(
            {
                "read_signal_template": {
                    "tool": read_factor_template,
                    "description": (
                        "Legacy: read_signal_template(module_name=None) returns the "
                        "quickbacktest BaseSignal template rendered for the current "
                        "candidate by default. Prefer the compute examples in query."
                    ),
                },
                "submit_signal": {
                    "tool": submit_signal,
                    "description": (
                        "Legacy: submit_signal(code, module_name=None, overwrite=True) "
                        "writes and runs a complete signal source file. Prefer "
                        "submit_compute(compute_code)."
                    ),
                },
                "save_signal": {
                    "tool": save_signal,
                    "description": (
                        "Legacy: save_signal(code, module_name=None, overwrite=True) "
                        "writes the current candidate signal source without running "
                        "validation. Prefer submit_compute(compute_code)."
                    ),
                },
                "run_signal": {
                    "tool": run_signal,
                    "description": (
                        "Legacy: run_signal(module_name=None) imports the current "
                        "saved signal and runs the same Qlib factor-data and IC "
                        "analysis path used by factor_miner. Prefer submit_compute(compute_code)."
                    ),
                },
            }
        )
    return tools


def save_case_signal(
    workspace: str | Path = DEFAULT_WORKSPACE,
    *,
    overwrite: bool = True,
) -> Path:
    """Write the demo BaseSignal into <workspace>/signals."""
    result = save_signal_code(
        workspace,
        CASE_SIGNAL_MODULE,
        CASE_SIGNAL_CODE,
        overwrite=overwrite,
    )
    if not result.get("ok"):
        raise ValueError(f"case signal save failed: {result.get('errors')}")
    return Path(result["signal_path"])


def _import_qlib_adapter():
    try:
        from quickbacktest import qlib_adapter
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "Cannot import quickbacktest.qlib_adapter. Install the missing "
            f"dependency first: {missing}"
        ) from exc
    return qlib_adapter


def initialize_qlib_for_factor_miner(config: FactorMinerCaseConfig) -> str:
    """Initialize qlib for this process before the RLM starts iterating."""
    qlib_adapter = _import_qlib_adapter()
    return qlib_adapter.init_qlib_once(
        provider_uri=str(config.provider_uri.resolve()),
    )


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean_json_value(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank_ic_from_metrics(metrics: dict[str, Any] | None) -> tuple[float | None, str]:
    if not isinstance(metrics, dict):
        return None, ""
    daily_rank_ic = _finite_float(metrics.get("daily_rank_ic_mean"))
    if daily_rank_ic is not None:
        return daily_rank_ic, "daily_rank_ic_mean"
    rank_ic_distribution = metrics.get("rank_ic_distribution")
    if isinstance(rank_ic_distribution, dict):
        rank_ic = _finite_float(rank_ic_distribution.get("mean"))
        if rank_ic is not None:
            return rank_ic, "rank_ic_distribution.mean"
    rank_ic = _finite_float(metrics.get("rank_ic"))
    if rank_ic is not None:
        return rank_ic, "rank_ic"
    return None, ""


def compute_signal_analysis(
    config: FactorMinerCaseConfig,
    module_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the saved signal through the same Qlib factor and IC path."""
    qlib_adapter = _import_qlib_adapter()
    provider_uri = config.provider_uri.resolve()

    with qlib_adapter.suppress_qlib_console():
        factor_df = qlib_adapter.compute_qlib_factor_dataframe(
            signal_modules=[module_name],
            base_dir=config.workspace.resolve(),
            instruments=config.instruments,
            start=config.start,
            end=config.end,
            provider_uri=str(provider_uri),
            score_column="score",
            factor_shift=config.factor_shift,
        )
        analysis = qlib_adapter.analyze_qlib_factors(
            factor_df=factor_df,
            instruments=config.instruments,
            start=config.start,
            end=config.end,
            factor_columns=[module_name],
            score_column="score",
            horizon=config.horizon,
            provider_uri=str(provider_uri),
        )
    return factor_df, analysis


def run_saved_signal(
    config: FactorMinerCaseConfig,
    module_name: str | None = None,
) -> dict[str, Any]:
    """Run a saved signal and return a compact result for RLM tool use."""
    module_name = module_name or config.module_name
    workspace = config.workspace.resolve()
    status_path = workspace / "run_signal_status.json"
    try:
        factor_df, analysis = compute_signal_analysis(config, module_name)
    except Exception as exc:
        error = {
            "ok": False,
            "module_name": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(workspace / f"{module_name}_run_signal_error.json", error)
        _write_json(status_path, error)
        return error

    run_path = workspace / f"{module_name}_run_signal.json"
    metrics = analysis.get("metrics", {}).get(module_name, {})
    rank_ic, rank_ic_name = _rank_ic_from_metrics(metrics)
    payload = {
        "ok": True,
        "module_name": module_name,
        "rows": int(len(factor_df)),
        "factor_columns": analysis.get("factor_columns", []),
        "metrics": metrics,
        "analysis": analysis,
    }
    _write_json(run_path, payload)
    if rank_ic is not None and rank_ic < 0:
        reverse_status = {
            "ok": False,
            "runs_without_error": True,
            "module_name": module_name,
            "run_signal_json": str(run_path),
            "rows": int(len(factor_df)),
            "metrics": metrics,
            "rank_ic": rank_ic,
            "rank_ic_name": rank_ic_name,
            "error_type": "NegativeRankICRequiresReverse",
            "error": (
                f"{rank_ic_name}={rank_ic:.6g} is negative. Reverse the final "
                "signal by multiplying the returned factor by -1, then "
                "submit_compute(compute_code) again before final answer."
            ),
        }
        _write_json(status_path, reverse_status)
        return reverse_status
    status = {
        "ok": True,
        "module_name": module_name,
        "run_signal_json": str(run_path),
        "rows": int(len(factor_df)),
        "rank_ic": rank_ic,
        "rank_ic_name": rank_ic_name,
    }
    _write_json(status_path, status)
    return {
        "ok": True,
        "module_name": module_name,
        "rows": int(len(factor_df)),
        "factor_columns": analysis.get("factor_columns", []),
        "metrics": metrics,
        "rank_ic": rank_ic,
        "rank_ic_name": rank_ic_name,
        "run_signal_json": str(run_path),
    }


def _require_run_signal_confirmation(workspace: Path, module_name: str) -> None:
    rejection = _final_answer_submission_rejection(workspace, module_name)
    if rejection:
        raise RuntimeError(rejection)


def _final_answer_submission_rejection(workspace: Path, module_name: str) -> str:
    signal_path = workspace / "signals" / f"{module_name}.py"
    if not signal_path.exists():
        return (
            f"final answer rejected: missing signal file {signal_path}. "
            "Call submit_compute(compute_code) before setting answer['ready'] = True."
        )
    status_path = workspace / "run_signal_status.json"
    if not status_path.exists():
        return (
            "final answer rejected: missing run_signal_status.json. "
            "Call submit_compute(compute_code) and receive ok=True before final answer."
        )
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"final answer rejected: cannot read submit_compute confirmation: {exc}"
    if not isinstance(status, dict):
        return (
            "final answer rejected: submit_compute confirmation must be a JSON object."
        )
    confirmed_module = str(status.get("module_name", "") or "")
    if confirmed_module and confirmed_module != module_name:
        return (
            "final answer rejected: submit_compute confirmed a different module: "
            f"{confirmed_module} != {module_name}."
        )
    if status.get("ok") is not True:
        error_type = str(status.get("error_type", "") or "UnknownError")
        error = str(status.get("error", "") or "submit_compute returned ok=False")
        return (
            "final answer rejected: submit_compute confirmation failed: "
            f"{error_type}: {error}"
        )
    return ""


def build_factor_miner_final_answer_validator(
    workspace: str | Path,
    module_name: str,
):
    """Return an RLM final-answer gate for factor generation."""
    workspace_path = Path(workspace).resolve()

    def validate(_content: Any) -> str:
        return _final_answer_submission_rejection(workspace_path, module_name)

    return validate


def _factor_generation_context(config: FactorMinerCaseConfig) -> dict[str, Any]:
    return {
        "available_libraries": {
            "pd": "pandas",
            "np": "numpy",
            "ta": "talib",
        },
        "available_data_fields": [
            "self.open",
            "self.high",
            "self.low",
            "self.close",
            "self.volume",
            "self.amount",
        ],
        "available_helpers": [
            "self.rolling_vwap(window, min_periods=1)",
            "self.rolling_zscore(frame, window, min_periods=None)",
        ],
        "parallel_generation": (f"factor_class_name: {config.module_name}"),
    }


def _factor_library_name(module_name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", "-", module_name).replace("_", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", words).strip("-").lower()


def save_reviewed_factor(
    *,
    config: FactorMinerCaseConfig,
    module_name: str,
    signal_path: Path,
    analysis: dict[str, Any],
    rlm_summary: str,
    source: str = "rlm",
    library_root: Path = DEFAULT_FACTOR_LIBRARY_PATH,
) -> dict[str, Any]:
    """Persist a successful factor run and deterministic review into factors/."""
    from quickbacktest import FactorLibrary, review_factor_metrics

    metrics = analysis.get("metrics", {}).get(module_name)
    if not isinstance(metrics, dict):
        raise ValueError(f"analysis metrics missing for factor: {module_name}")

    factor_name = _factor_library_name(module_name)
    library = FactorLibrary(library_root)
    record = library.save_factor(
        name=factor_name,
        signal_code=signal_path.read_text(encoding="utf-8"),
        metrics=metrics,
        description=f"{source.upper()} generated factor signal.",
        rlm_summary=rlm_summary,
        signal_class=module_name,
        universe=config.instruments,
        horizon=config.horizon,
        factor_shift=config.factor_shift,
        status="candidate",
        source=source,
        tags=[source, "factor"],
    )
    review = review_factor_metrics(metrics, runs_without_error=True)
    library.save_review(record.name, review)
    status = "accepted" if review.get("verdict") == "accepted" else "rejected"
    library.update_status(record.name, status)
    factor_dir = library_root / record.name
    return {
        "factor_name": record.name,
        "factor_dir": str(factor_dir),
        "factor_card": str(factor_dir / "FACTOR.md"),
        "factor_signal": str(factor_dir / "signal.py"),
        "factor_metrics": str(factor_dir / "metrics.json"),
        "factor_review": str(factor_dir / "review.json"),
        "status": status,
        "review": review,
    }


def generate_signal_with_rlm(config: FactorMinerCaseConfig) -> GeneratedSignal:
    """Ask RLM to generate and save one factor signal."""
    libs_root = str(LIBS_ROOT)
    if libs_root not in sys.path:
        sys.path.insert(0, libs_root)

    from rlm.rlm_repl import RLM_REPL

    from quickbacktest import build_rlm_factor_tools, build_rlm_skill_tools

    workspace = config.workspace.resolve()
    initialize_qlib_for_factor_miner(config)
    tools = build_rlm_skill_tools(DEFAULT_SKILL_PATH, allow_write=False)
    tools.update(build_rlm_factor_tools(DEFAULT_FACTOR_LIBRARY_PATH))
    tools.update(build_factor_miner_custom_tools(workspace, config=config))
    query = (
        "Write only the Python body of BaseSignal.compute(**kwargs). "
        "Do not write imports, class definitions, def compute, markdown fences, "
        "or signal names; the system wraps your compute body into the correct "
        "BaseSignal class and appends return signal. Do not write return statements. "
        "Assign the final wide DataFrame to variable signal. Use the template/examples "
        "below, and context.available_data_fields / available_libraries for allowed "
        "inputs. The signal must be aligned to self.close.\n\n"
        f"{compute_examples_prompt_text()}\n\n"
        "Call submit_compute(compute_code). If submit_compute returns "
        "ok=False, repair using error_type/error and submit again. If error_type is "
        "NegativeRankICRequiresReverse, multiply the final signal by -1 and call "
        "submit_compute(compute_code) again. Do not train ML models or call fit(). "
        "Do not otherwise iterate to optimize IC magnitude. "
        "Use the available REPL tools, read relevant skills and existing factors first, "
        "Details are captured in the trajectory. "
        f"{RLM_SUMMARY_SPEC} "
        "When done, set answer['content'] to the full RLM summary following this spec, "
        "and set answer['ready'] = True."
    )
    rlm = RLM_REPL(
        model=config.model,
        recursive_model=config.recursive_model,
        max_iterations=config.max_iterations,
        enable_logging=config.enable_rlm_logging,
        custom_tools=tools,
        final_answer_validator=build_factor_miner_final_answer_validator(
            workspace,
            config.module_name,
        ),
    )
    generation_started_at = time.perf_counter()
    result = rlm.completion(
        context=_factor_generation_context(config),
        query=query,
    )
    generation_elapsed_seconds = time.perf_counter() - generation_started_at
    trajectory_path = workspace / "rlm_trajectory.json"
    _write_json(
        trajectory_path,
        {
            "response": result.response,
            "metadata": result.metadata,
            "rlm_factor_generation_elapsed_seconds": generation_elapsed_seconds,
            "rlm_summary_spec": RLM_SUMMARY_SPEC,
        },
    )

    signal_path = workspace / "signals" / f"{config.module_name}.py"
    if not signal_path.exists():
        raise RuntimeError(
            "RLM did not save the signal. It must call "
            f"submit_compute(compute_code). Final response: {result.response}. "
            f"Trajectory: {trajectory_path}"
        )
    _require_run_signal_confirmation(workspace, config.module_name)

    return GeneratedSignal(
        module_name=config.module_name,
        signal_path=signal_path,
        rlm_response=result.response,
        trajectory_path=trajectory_path,
        rlm_generation_elapsed_seconds=generation_elapsed_seconds,
        rlm_metadata=result.metadata,
    )


def run_factor_miner_case(config: FactorMinerCaseConfig) -> dict[str, Any]:
    """Run the packaged factor-miner example through the Qlib adapter."""
    workspace = config.workspace.resolve()
    provider_uri = config.provider_uri.resolve()

    generated: GeneratedSignal | None = None
    if config.use_case_signal:
        module_name = CASE_SIGNAL_MODULE
        signal_path = save_case_signal(workspace)
        generation_mode = "case_signal"
    else:
        generated = generate_signal_with_rlm(config)
        module_name = generated.module_name
        signal_path = generated.signal_path
        generation_mode = "rlm"

    factor_df, analysis = compute_signal_analysis(config, module_name)
    factor_csv = workspace / "factor_data.csv"
    factor_df.to_csv(factor_csv, index=False, encoding="utf-8")

    analysis_path = workspace / "analysis.json"
    _write_json(analysis_path, analysis)

    result: dict[str, Any] = {
        "workspace": str(workspace),
        "generation_mode": generation_mode,
        "signal_module": module_name,
        "signal_path": str(signal_path),
        "factor_data_csv": str(factor_csv),
        "analysis_json": str(analysis_path),
        "analysis": analysis,
        "factor_shift": config.factor_shift,
    }
    if generated is not None:
        result["rlm_response"] = generated.rlm_response
        result["rlm_summary"] = generated.rlm_response
        result["rlm_trajectory_json"] = str(generated.trajectory_path)
        result["rlm_factor_generation_elapsed_seconds"] = (
            generated.rlm_generation_elapsed_seconds
        )

    factor_record = save_reviewed_factor(
        config=config,
        module_name=module_name,
        signal_path=signal_path,
        analysis=analysis,
        rlm_summary=generated.rlm_response
        if generated is not None
        else "Built-in case signal.",
        source="rlm" if generated is not None else "case",
    )
    result["factor_library"] = factor_record

    if config.run_portfolio:
        qlib_adapter = _import_qlib_adapter()
        pred = qlib_adapter.factor_df_to_qlib_signal(factor_df, score_column="score")
        portfolio = qlib_adapter.simulate_qlib_portfolio(
            pred=pred,
            benchmark=config.benchmark,
            topk=config.topk,
            n_drop=config.n_drop,
            provider_uri=str(provider_uri),
            output_dir=workspace / "portfolio",
        )
        portfolio_path = workspace / "portfolio.json"
        _write_json(portfolio_path, portfolio)
        result["portfolio_json"] = str(portfolio_path)
        result["portfolio"] = portfolio

    if config.train_alpha158:
        qlib_adapter = _import_qlib_adapter()
        training = qlib_adapter.train_qlib_alpha158_augmented_model(
            signal_modules=[module_name],
            base_dir=workspace,
            instruments=config.instruments,
            start=config.start,
            end=config.end,
            provider_uri=str(provider_uri),
            factor_shift=config.factor_shift,
        )
        training_path = workspace / "alpha158_training.json"
        _write_json(training_path, training)
        result["alpha158_training_json"] = str(training_path)
        result["alpha158_training"] = training

    summary_path = workspace / "summary.json"
    _write_json(summary_path, result)
    result["summary_json"] = str(summary_path)
    return result


def _print_result(result: dict[str, Any]) -> None:
    module_name = result["signal_module"]
    print(f"workspace: {result['workspace']}")
    print(f"generation mode: {result['generation_mode']}")
    print(f"signal module: {module_name}")
    print(f"signal: {result['signal_path']}")
    if "rlm_trajectory_json" in result:
        print(f"rlm trajectory: {result['rlm_trajectory_json']}")
    if "rlm_factor_generation_elapsed_seconds" in result:
        elapsed = result["rlm_factor_generation_elapsed_seconds"]
        print(f"rlm factor generation seconds: {elapsed:.3f}")
    if "factor_library" in result:
        factor_library = result["factor_library"]
        print(f"factor library status: {factor_library.get('status')}")
        print(f"factor card: {factor_library.get('factor_card')}")
    print(f"factor data: {result['factor_data_csv']}")
    print(f"analysis: {result['analysis_json']}")

    metrics = result.get("analysis", {}).get("metrics", {}).get(module_name, {})
    if metrics:
        print("factor metrics:")
        for key in ("daily_rank_ic_mean", "rank_icir", "coverage", "missing_rate"):
            print(f"  {key}: {metrics.get(key)}")

    if "portfolio" in result:
        portfolio = result["portfolio"]
        print("portfolio:")
        for key in (
            "cumulative_return_after_cost",
            "cumulative_benchmark_return",
            "cumulative_excess_return_after_cost",
        ):
            print(f"  {key}: {portfolio.get(key)}")
        print(f"portfolio artifacts: {portfolio.get('artifacts')}")

    if "alpha158_training" in result:
        uplift = result["alpha158_training"].get("uplift", {})
        print(f"alpha158 uplift: {uplift}")

    print(f"summary: {result['summary_json']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal RLM-style factor-miner case on Qlib data.",
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--provider-uri", default=str(DEFAULT_PROVIDER_URI))
    parser.add_argument("--module-name", default=GENERATED_SIGNAL_MODULE)
    parser.add_argument("--instruments", default="csi500")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--benchmark", default="SH000905")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--factor-shift",
        type=int,
        default=1,
        help="Uniform lag applied by the Qlib adapter after signal computation.",
    )
    parser.add_argument(
        "--no-portfolio",
        action="store_true",
        help="Only compute factor data and IC analysis.",
    )
    parser.add_argument(
        "--train-alpha158",
        action="store_true",
        help="Also train a Qlib LightGBM model with Alpha158 plus the demo factor.",
    )
    parser.add_argument(
        "--use-case-signal",
        action="store_true",
        help="Use the built-in deterministic example factor instead of calling RLM.",
    )
    parser.add_argument("--model", default=DEFAULT_RLM_MODEL)
    parser.add_argument("--recursive-model", default=DEFAULT_RLM_MODEL)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--enable-rlm-logging",
        action="store_true",
        help="Enable verbose RLM trajectory logging.",
        default=True,
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Only generate/save the signal; do not import Qlib or run analysis.",
    )
    parser.add_argument(
        "--materialize-only",
        dest="save_only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.base_dir)

    config = FactorMinerCaseConfig(
        workspace=workspace,
        provider_uri=Path(args.provider_uri),
        module_name=args.module_name,
        instruments=args.instruments,
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        topk=args.topk,
        n_drop=args.n_drop,
        horizon=args.horizon,
        factor_shift=args.factor_shift,
        run_portfolio=not args.no_portfolio,
        train_alpha158=args.train_alpha158,
        use_case_signal=args.use_case_signal,
        model=args.model,
        recursive_model=args.recursive_model,
        max_iterations=args.max_iterations,
        enable_rlm_logging=args.enable_rlm_logging,
    )

    if args.save_only:
        try:
            if config.use_case_signal:
                signal_path = save_case_signal(workspace)
                print(f"wrote signal: {signal_path}")
            else:
                generated = generate_signal_with_rlm(config)
                print(f"wrote signal: {generated.signal_path}")
                print(f"rlm trajectory: {generated.trajectory_path}")
        except Exception as exc:
            print(
                f"factor save failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        return 0

    try:
        result = run_factor_miner_case(config)
    except Exception as exc:
        print(f"factor miner case failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


