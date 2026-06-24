from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import io
import importlib.util
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

from quickbacktest.base_types import BaseSignal, SignalData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QLIB_PROVIDER = PROJECT_ROOT / ".qlib" / "qlib_data" / "cn_data"
DECILE_LAYER_COUNT = 10
QLIB_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
QLIB_FIELD_MAP = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
    "$vwap": "vwap",
}


class _QlibKnownBacktestWarningFilter(logging.Filter):
    _PATTERNS = (
        "load calendar error: freq=day, future=True; return current calendar",
        "You can get future calendar by referring to the following document",
        "`common_infra` is not set for",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern in message for pattern in self._PATTERNS)


@contextmanager
def suppress_qlib_console():
    """Silence qlib console noise so RLM tool observations stay JSON-like."""
    previous_disable = logging.root.manager.disable
    sink = io.StringIO()
    logging.disable(max(previous_disable, logging.WARNING))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            logging.disable(previous_disable)


@contextmanager
def _suppress_known_qlib_backtest_warnings():
    loggers = [
        logging.getLogger("qlib.data"),
        logging.getLogger("qlib.BaseExecutor"),
    ]
    filt = _QlibKnownBacktestWarningFilter()
    for logger in loggers:
        logger.addFilter(filt)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(filt)


def _coerce_time(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _coerce_modules(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _coerce_instruments(value: str | Iterable[str]) -> str | list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("instruments must not be empty")
        return stripped
    return [str(item) for item in value]


def _looks_like_qlib_market(value: str, provider_uri: str | None = None) -> bool:
    provider = Path(provider_uri) if provider_uri is not None else DEFAULT_QLIB_PROVIDER
    return (provider.resolve() / "instruments" / f"{value}.txt").exists()


def _resolve_qlib_instruments(value: str | Iterable[str], provider_uri: str | None = None) -> Any:
    resolved = _coerce_instruments(value)
    if not isinstance(resolved, str):
        return resolved
    if _looks_like_qlib_market(resolved, provider_uri=provider_uri):
        from qlib.data import D

        return D.instruments(resolved)
    return [resolved]


def _load_class(file_path: Path, class_name: str) -> type:
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    spec = importlib.util.spec_from_file_location(
        f"_qlib_factor_{file_path.stem}_{hash(file_path)}",
        str(file_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")
    return getattr(module, class_name)


def _disable_qlib_git_code_logging() -> None:
    """Disable qlib recorder's git diff/status snapshot in non-git workspaces."""
    try:
        from qlib.workflow.recorder import MLflowRecorder, Recorder
    except Exception:
        return
    if getattr(Recorder, "_quickbacktest_git_logging_disabled", False):
        return

    def _noop_log_uncommitted_code(self):
        return None

    Recorder._log_uncommitted_code = _noop_log_uncommitted_code
    MLflowRecorder._log_uncommitted_code = _noop_log_uncommitted_code
    Recorder._quickbacktest_git_logging_disabled = True
    MLflowRecorder._quickbacktest_git_logging_disabled = True


@lru_cache(maxsize=8)
def init_qlib_once(
    provider_uri: str | None = None,
    region: str = "cn",
    kernels: int = 1,
) -> str:
    """Initialize qlib once and return the resolved provider path."""
    import qlib

    provider = Path(provider_uri) if provider_uri is not None else DEFAULT_QLIB_PROVIDER
    provider = provider.resolve()
    if not provider.exists():
        raise FileNotFoundError(f"qlib provider not found: {provider}")
    _disable_qlib_git_code_logging()
    with suppress_qlib_console():
        qlib.init(provider_uri=str(provider), region=region, kernels=kernels)
    _disable_qlib_git_code_logging()
    return str(provider)


def query_qlib_ohlcv(
    instruments: str | Iterable[str] = "csi300",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    provider_uri: str | None = None,
    freq: str = "day",
) -> pd.DataFrame:
    """Query qlib A-share OHLCV and return quickbacktest long-format data."""
    init_qlib_once(provider_uri=provider_uri)
    with suppress_qlib_console():
        from qlib.data import D

        resolved_instruments = _resolve_qlib_instruments(
            instruments,
            provider_uri=provider_uri,
        )
        data = D.features(
            resolved_instruments,
            QLIB_FIELDS,
            start_time=_coerce_time(start),
            end_time=_coerce_time(end),
            freq=freq,
        )
    if data.empty:
        raise ValueError("qlib returned no OHLCV data")

    df = data.rename(columns=QLIB_FIELD_MAP).reset_index()
    if "instrument" not in df.columns or "datetime" not in df.columns:
        raise ValueError("qlib OHLCV must have instrument/datetime index levels")
    df = df.rename(columns={"instrument": "code", "datetime": "trade_time"})
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    if "vwap" not in df.columns or df["vwap"].isna().all():
        df["vwap"] = df["close"]
    df["amount"] = df["close"] * df["volume"]
    cols = ["trade_time", "code", "open", "high", "low", "close", "volume", "vwap", "amount"]
    return df[cols].sort_values(["trade_time", "code"]).reset_index(drop=True)


def _normalize_signal_output_for_signal_data(
    output: pd.DataFrame | pd.Series,
    signal_data: SignalData,
    *,
    value_name: str,
) -> pd.DataFrame:
    """Normalize common factor output shapes to BaseSignal's required wide shape."""
    target_index = signal_data.close.index
    target_columns = signal_data.close.columns.astype(str)

    if isinstance(output, pd.Series):
        frame = output.to_frame(value_name)
    elif isinstance(output, pd.DataFrame):
        frame = output.copy()
    else:
        raise TypeError("signal compute output must be a pandas DataFrame or Series")

    if isinstance(frame.index, pd.MultiIndex):
        index_names = list(frame.index.names)
        if "trade_time" in index_names and "code" in index_names:
            value_columns = list(frame.columns)
            if len(value_columns) != 1:
                if value_name in frame.columns:
                    value_columns = [value_name]
                elif "score" in frame.columns:
                    value_columns = ["score"]
                else:
                    numeric = [
                        col for col in frame.columns
                        if pd.api.types.is_numeric_dtype(frame[col])
                    ]
                    if len(numeric) != 1:
                        raise ValueError(
                            "MultiIndex factor output must have exactly one value column "
                            "or a value_name/score column"
                        )
                    value_columns = numeric
            long_frame = frame[value_columns[0]].reset_index()
            frame = long_frame.pivot(index="trade_time", columns="code", values=value_columns[0])
        elif frame.index.nlevels == 2:
            value_columns = list(frame.columns)
            if len(value_columns) != 1:
                raise ValueError("2-level MultiIndex factor output must have exactly one value column")
            long_frame = frame[value_columns[0]].reset_index()
            long_frame.columns = ["trade_time", "code", value_name]
            frame = long_frame.pivot(index="trade_time", columns="code", values=value_name)

    if {"trade_time", "code"}.issubset(frame.columns):
        value_candidates = [
            col for col in frame.columns
            if col not in {"trade_time", "code"} and pd.api.types.is_numeric_dtype(frame[col])
        ]
        if value_name in frame.columns:
            value_col = value_name
        elif "score" in frame.columns:
            value_col = "score"
        elif len(value_candidates) == 1:
            value_col = value_candidates[0]
        else:
            raise ValueError(
                "long factor output must include trade_time, code, and exactly one numeric value column"
            )
        frame = frame.pivot(index="trade_time", columns="code", values=value_col)

    try:
        frame.index = pd.to_datetime(frame.index)
    except Exception as exc:
        raise ValueError("factor output index cannot be converted to DatetimeIndex") from exc

    frame.columns = frame.columns.astype(str)
    frame = frame.sort_index()

    if frame.index.intersection(target_index).empty:
        raise ValueError("factor output index has no overlap with data.close.index")
    if frame.columns.intersection(target_columns).empty:
        raise ValueError("factor output columns have no overlap with data.close.columns")

    return frame.reindex(index=target_index, columns=target_columns)


def compute_qlib_factor_dataframe(
    signal_modules: str | Iterable[str],
    base_dir: str | Path,
    instruments: str | Iterable[str] = "csi300",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    provider_uri: str | None = None,
    freq: str = "day",
    score_column: str = "score",
    factor_shift: int = 0,
) -> pd.DataFrame:
    """Compute existing BaseSignal modules on qlib A-share OHLCV.

    factor_shift applies a uniform lag after each signal is normalized to wide
    format. Use factor_shift=1 for the common "trade on yesterday's signal"
    convention and keep individual BaseSignal implementations unshifted.
    """
    modules = _coerce_modules(signal_modules)
    if not modules:
        raise ValueError("signal_modules must contain at least one module")
    if not isinstance(factor_shift, int) or factor_shift < 0:
        raise ValueError("factor_shift must be a non-negative integer")

    ohlcv = query_qlib_ohlcv(
        instruments=instruments,
        start=start,
        end=end,
        provider_uri=provider_uri,
        freq=freq,
    )
    signal_data = SignalData(ohlcv)
    result = ohlcv.copy()
    factor_columns: list[str] = []
    for module_name in modules:
        cls = _load_class(Path(base_dir) / "signals" / f"{module_name}.py", module_name)
        signal = cls(signal_data)
        if not isinstance(signal, BaseSignal):
            raise TypeError(f"{module_name} must inherit BaseSignal")
        factor_name = getattr(signal, "name", "") or module_name
        factor_name = str(factor_name).strip() or module_name
        wide = _normalize_signal_output_for_signal_data(
            signal.compute(),
            signal_data,
            value_name=factor_name,
        )
        if factor_shift:
            wide = wide.shift(factor_shift)
        signal.validate_wide_output(wide)
        long_factor = signal.wide_to_long(wide, factor_name)
        result = result.merge(long_factor, on=["trade_time", "code"], how="left")
        factor_columns.append(factor_name)

    if score_column not in result.columns:
        result[score_column] = result[factor_columns].mean(axis=1)
    return result


def _factor_columns(df: pd.DataFrame, factor_columns: Iterable[str] | None, score_column: str) -> list[str]:
    if factor_columns is not None:
        return [str(col) for col in factor_columns]
    excluded = {"trade_time", "code", "open", "high", "low", "close", "volume", "vwap", "amount"}
    cols = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]
    if score_column in cols:
        cols = [col for col in cols if col != score_column] + [score_column]
    return cols


def _assign_daily_score_layers(
    frame: pd.DataFrame,
    *,
    score_column: str,
) -> pd.Series:
    """Assign daily decile layers, with Decile1 as the highest score layer."""

    def _one_day(sub: pd.DataFrame) -> pd.Series:
        valid = sub[score_column].dropna()
        labels = pd.Series(index=sub.index, dtype=object)
        if len(valid) < DECILE_LAYER_COUNT:
            return labels
        ranks = valid.rank(method="first", ascending=False)
        bins = pd.qcut(
            ranks,
            q=DECILE_LAYER_COUNT,
            labels=[f"Decile{i}" for i in range(1, DECILE_LAYER_COUNT + 1)],
        )
        labels.loc[valid.index] = bins.astype(str)
        return labels

    return frame.groupby(level="datetime", group_keys=False).apply(_one_day)


def _qlib_layered_ic(
    joined: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
) -> dict[str, Any]:
    """Calculate decile-layer IC by applying qlib calc_ic inside each daily score layer."""
    from qlib.contrib.eva.alpha import calc_ic

    data = joined[[score_column, label_column]].dropna().copy()
    if data.empty:
        return {
            "layer_type": "decile",
            "deciles": {},
            "diagnostics": {"joined_rows": 0, "reason": "no non-null score/label rows"},
        }
    data["layer"] = _assign_daily_score_layers(
        data,
        score_column=score_column,
    )
    deciles: dict[str, Any] = {}
    for layer_name in [f"Decile{i}" for i in range(1, DECILE_LAYER_COUNT + 1)]:
        layer = data[data["layer"] == layer_name]
        if len(layer) >= 2:
            daily_ic, daily_rank_ic = calc_ic(
                layer[score_column],
                layer[label_column],
                date_col="datetime",
                dropna=True,
            )
        else:
            daily_ic = pd.Series(dtype=float)
            daily_rank_ic = pd.Series(dtype=float)

        daily_ic_mean = float(daily_ic.mean()) if len(daily_ic) else float("nan")
        daily_ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        daily_rank_ic_mean = float(daily_rank_ic.mean()) if len(daily_rank_ic) else float("nan")
        daily_rank_ic_std = float(daily_rank_ic.std(ddof=1)) if len(daily_rank_ic) > 1 else float("nan")
        deciles[layer_name] = {
            "rows": int(len(layer)),
            "daily_ic_mean": daily_ic_mean,
            "daily_ic_std": daily_ic_std,
            "daily_rank_ic_mean": daily_rank_ic_mean,
            "daily_rank_ic_std": daily_rank_ic_std,
            "pearson_icir": (
                float("nan")
                if not daily_ic_std or np.isnan(daily_ic_std)
                else daily_ic_mean / daily_ic_std
            ),
            "rank_icir": (
                float("nan")
                if not daily_rank_ic_std or np.isnan(daily_rank_ic_std)
                else daily_rank_ic_mean / daily_rank_ic_std
            ),
            "daily_ic_count": int(len(daily_ic)),
            "daily_rank_ic_count": int(len(daily_rank_ic)),
        }

    return {
        "layer_type": "decile",
        "deciles": deciles,
        "diagnostics": {
            "joined_rows": int(len(data)),
            "layered_rows": int(data["layer"].notna().sum()),
        },
    }


def _series_distribution(series: pd.Series) -> dict[str, Any]:
    """Summarize a daily IC series without storing the full time series."""
    values = pd.Series(series, dtype=float).dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "positive_rate": float("nan"),
            "negative_rate": float("nan"),
            "zero_rate": float("nan"),
        }
    quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(values.min()),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(values.max()),
        "positive_rate": float((values > 0).mean()),
        "negative_rate": float((values < 0).mean()),
        "zero_rate": float((values == 0).mean()),
    }


def factor_df_to_qlib_signal(
    factor_df: pd.DataFrame,
    score_column: str = "score",
) -> pd.Series:
    """Convert a canonical factor dataframe to qlib prediction Series."""
    required = {"trade_time", "code", score_column}
    missing = required - set(factor_df.columns)
    if missing:
        raise ValueError(f"factor_df is missing required columns: {sorted(missing)}")
    pred = factor_df[["trade_time", "code", score_column]].copy()
    pred = pred.rename(columns={"trade_time": "datetime", "code": "instrument"})
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred = pred.set_index(["datetime", "instrument"]).sort_index()[score_column]
    pred.name = "score"
    return pred


def analyze_qlib_factors(
    signal_modules: str | Iterable[str] | None = None,
    base_dir: str | Path | None = None,
    factor_df: pd.DataFrame | None = None,
    instruments: str | Iterable[str] = "csi300",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    factor_columns: Iterable[str] | None = None,
    score_column: str = "score",
    horizon: int = 1,
    provider_uri: str | None = None,
    factor_shift: int = 0,
) -> dict[str, Any]:
    """Run standalone qlib-style factor analysis without strategy simulation."""
    from qlib.contrib.eva.alpha import calc_ic

    if factor_df is None:
        if signal_modules is None or base_dir is None:
            raise ValueError("signal_modules and base_dir are required when factor_df is not provided")
        factor_df = compute_qlib_factor_dataframe(
            signal_modules=signal_modules,
            base_dir=base_dir,
            instruments=instruments,
            start=start,
            end=end,
            provider_uri=provider_uri,
            score_column=score_column,
            factor_shift=factor_shift,
        )

    cols = _factor_columns(factor_df, factor_columns, score_column)
    if not cols:
        raise ValueError("no numeric factor columns found")

    df = factor_df.sort_values(["code", "trade_time"]).copy()
    df["future_return"] = df.groupby("code")["close"].shift(-horizon) / df["close"] - 1
    rows: dict[str, Any] = {}
    for col in cols:
        qlib_frame = (
            df[["trade_time", "code", col, "future_return"]]
            .rename(columns={"trade_time": "datetime", "code": "instrument"})
            .assign(datetime=lambda x: pd.to_datetime(x["datetime"]))
            .set_index(["datetime", "instrument"])
            .sort_index()
        )
        joined = qlib_frame[[col, "future_return"]].dropna()
        if len(joined) >= 2:
            daily_ic, daily_rank_ic = calc_ic(
                joined[col],
                joined["future_return"],
                date_col="datetime",
                dropna=True,
            )
        else:
            daily_ic = pd.Series(dtype=float)
            daily_rank_ic = pd.Series(dtype=float)

        daily_ic_mean = float(daily_ic.mean()) if len(daily_ic) else float("nan")
        daily_ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        daily_rank_ic_mean = float(daily_rank_ic.mean()) if len(daily_rank_ic) else float("nan")
        daily_rank_ic_std = float(daily_rank_ic.std(ddof=1)) if len(daily_rank_ic) > 1 else float("nan")
        rows[col] = {
            "ic": float(joined[col].corr(joined["future_return"], method="pearson"))
            if len(joined) >= 2
            else float("nan"),
            "rank_ic": float(joined[col].corr(joined["future_return"], method="spearman"))
            if len(joined) >= 2
            else float("nan"),
            "daily_ic_mean": daily_ic_mean,
            "daily_ic_std": daily_ic_std,
            "daily_rank_ic_mean": daily_rank_ic_mean,
            "daily_rank_ic_std": daily_rank_ic_std,
            "icir": (
                float("nan")
                if not daily_rank_ic_std or np.isnan(daily_rank_ic_std)
                else daily_rank_ic_mean / daily_rank_ic_std
            ),
            "rank_icir": (
                float("nan")
                if not daily_rank_ic_std or np.isnan(daily_rank_ic_std)
                else daily_rank_ic_mean / daily_rank_ic_std
            ),
            "pearson_icir": (
                float("nan")
                if not daily_ic_std or np.isnan(daily_ic_std)
                else daily_ic_mean / daily_ic_std
            ),
            "daily_ic_count": int(len(daily_ic)),
            "daily_rank_ic_count": int(len(daily_rank_ic)),
            "ic_distribution": _series_distribution(daily_ic),
            "rank_ic_distribution": _series_distribution(daily_rank_ic),
            "layered_ic": _qlib_layered_ic(
                joined.rename(columns={col: "score", "future_return": "label"}),
                score_column="score",
                label_column="label",
            ),
            "coverage": float(df[col].notna().mean()),
            "missing_rate": float(df[col].isna().mean()),
        }

    corr = factor_df[cols].corr(method="spearman").replace({np.nan: None}).to_dict()
    return {
        "rows": int(len(factor_df)),
        "instruments": instruments,
        "start": str(factor_df["trade_time"].min()),
        "end": str(factor_df["trade_time"].max()),
        "factor_columns": cols,
        "metrics": rows,
        "factor_correlation": corr,
    }


def _to_qlib_feature_frame(
    factor_df: pd.DataFrame,
    factor_columns: Iterable[str],
    label: pd.DataFrame,
) -> pd.DataFrame:
    rlm = factor_df[["trade_time", "code", *factor_columns]].copy()
    rlm = rlm.rename(columns={"trade_time": "datetime", "code": "instrument"})
    rlm["datetime"] = pd.to_datetime(rlm["datetime"])
    rlm = rlm.set_index(["datetime", "instrument"]).sort_index()
    rlm.columns = pd.MultiIndex.from_product([["feature"], list(rlm.columns)])
    label_frame = label.copy()
    if not isinstance(label_frame.columns, pd.MultiIndex):
        label_frame.columns = pd.MultiIndex.from_product([["label"], list(label_frame.columns)])
    return pd.concat([rlm, label_frame], axis=1).dropna(subset=label_frame.columns)


def build_alpha158_augmented_handler(
    factor_df: pd.DataFrame,
    instruments: str | Iterable[str] = "csi300",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    fit_start: datetime | str | None = None,
    fit_end: datetime | str | None = None,
    factor_columns: Iterable[str] | None = None,
    score_column: str = "score",
    provider_uri: str | None = None,
):
    """Build a DataHandlerLP from Alpha158 features plus RLM factor columns."""
    init_qlib_once(provider_uri=provider_uri)
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DataHandlerLP

    alpha = Alpha158(
        instruments=instruments,
        start_time=_coerce_time(start),
        end_time=_coerce_time(end),
        fit_start_time=_coerce_time(fit_start or start),
        fit_end_time=_coerce_time(fit_end or end),
    )
    alpha_feature = alpha.fetch(col_set="feature")
    label = alpha.fetch(col_set="label")

    cols = _factor_columns(factor_df, factor_columns, score_column)
    augmented = pd.concat(
        [
            pd.concat({"feature": alpha_feature}, axis=1),
            _to_qlib_feature_frame(factor_df, cols, label).drop(columns=pd.MultiIndex.from_product([["label"], label.columns]), errors="ignore"),
            pd.concat({"label": label}, axis=1),
        ],
        axis=1,
    ).dropna(subset=pd.MultiIndex.from_product([["label"], label.columns]))
    return DataHandlerLP.from_df(augmented)


def _default_segments(start: datetime | str, end: datetime | str) -> dict[str, tuple[str, str]]:
    dates = pd.date_range(_coerce_time(start), _coerce_time(end), freq="D")
    if len(dates) < 10:
        raise ValueError("at least 10 calendar days are required for default train/valid/test segments")
    train_end = dates[int(len(dates) * 0.6)].strftime("%Y-%m-%d")
    valid_end = dates[int(len(dates) * 0.8)].strftime("%Y-%m-%d")
    return {
        "train": (_coerce_time(start), train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, _coerce_time(end)),
    }


def train_qlib_alpha158_augmented_model(
    signal_modules: str | Iterable[str],
    base_dir: str | Path,
    instruments: str | Iterable[str] = "csi300",
    start: datetime | str = "2020-01-01",
    end: datetime | str = "2023-01-01",
    segments: dict[str, tuple[str, str]] | None = None,
    compare_baseline: bool = True,
    provider_uri: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    factor_shift: int = 0,
) -> dict[str, Any]:
    """Train qlib LightGBM with Alpha158+RLM factors and optionally compare baseline."""
    init_qlib_once(provider_uri=provider_uri)
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    segments = segments or _default_segments(start, end)
    factor_df = compute_qlib_factor_dataframe(
        signal_modules=signal_modules,
        base_dir=base_dir,
        instruments=instruments,
        start=start,
        end=end,
        provider_uri=provider_uri,
        factor_shift=factor_shift,
    )
    augmented_handler = build_alpha158_augmented_handler(
        factor_df=factor_df,
        instruments=instruments,
        start=start,
        end=end,
        provider_uri=provider_uri,
    )
    model_kwargs = {
        "num_boost_round": 100,
        "early_stopping_rounds": 20,
        "num_leaves": 8,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1,
        "verbose": -1,
        **(model_kwargs or {}),
    }
    augmented_dataset = DatasetH(handler=augmented_handler, segments=segments)
    augmented_model = LGBModel(**model_kwargs)
    augmented_model.fit(augmented_dataset)
    augmented_pred = augmented_model.predict(augmented_dataset, segment="test")
    augmented_analysis = _analyze_prediction_series(augmented_pred, augmented_dataset)

    result: dict[str, Any] = {
        "mode": "alpha158_augmented",
        "instruments": instruments,
        "period": [_coerce_time(start), _coerce_time(end)],
        "segments": segments,
        "augmented": augmented_analysis,
    }

    if compare_baseline:
        alpha = Alpha158(
            instruments=instruments,
            start_time=_coerce_time(start),
            end_time=_coerce_time(end),
            fit_start_time=_coerce_time(start),
            fit_end_time=_coerce_time(end),
        )
        baseline_dataset = DatasetH(handler=alpha, segments=segments)
        baseline_model = LGBModel(**model_kwargs)
        baseline_model.fit(baseline_dataset)
        baseline_pred = baseline_model.predict(baseline_dataset, segment="test")
        baseline_analysis = _analyze_prediction_series(baseline_pred, baseline_dataset)
        result["baseline"] = baseline_analysis
        result["uplift"] = {
            key: _safe_subtract(result["augmented"].get(key), baseline_analysis.get(key))
            for key in ("rank_ic", "daily_rank_ic_mean", "icir")
        }
    return result


def _safe_subtract(left: Any, right: Any) -> float:
    try:
        left_f = float(left)
        right_f = float(right)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(left_f) or np.isnan(right_f):
        return float("nan")
    return left_f - right_f


def _jsonable_frame(frame: pd.DataFrame | pd.Series | None) -> Any:
    if frame is None:
        return None
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    out = frame.copy()
    out.index = out.index.map(str)
    return out.replace({np.nan: None}).to_dict(orient="index")


def _risk_dict(series: pd.Series, freq: str = "day") -> dict[str, Any]:
    from qlib.contrib.evaluate import risk_analysis

    try:
        risk = risk_analysis(series.dropna(), freq=freq)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(risk, pd.Series):
        return risk.replace({np.nan: None}).to_dict()
    if isinstance(risk, pd.DataFrame):
        if "risk" in risk.columns:
            return risk["risk"].replace({np.nan: None}).to_dict()
        if "value" in risk.columns:
            return risk["value"].replace({np.nan: None}).to_dict()
        return _jsonable_frame(risk)
    return {"value": risk}


def _save_return_artifacts(
    report_normal: pd.DataFrame,
    positions_normal: Any,
    output_dir: str | Path | None,
) -> dict[str, str]:
    if output_dir is None:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "qlib_report_normal.csv"
    curve_path = out / "qlib_return_curve.csv"
    plot_path = out / "qlib_return_curve.png"
    positions_path = out / "qlib_positions_normal.pkl"

    report_normal.to_csv(report_path, encoding="utf-8")
    if positions_normal is not None:
        try:
            positions_normal.to_pickle(positions_path)
        except Exception:
            positions_path = None

    curve = pd.DataFrame(index=report_normal.index)
    cost = report_normal["cost"].fillna(0.0) if "cost" in report_normal else pd.Series(0.0, index=report_normal.index)
    bench = report_normal["bench"].fillna(0.0) if "bench" in report_normal else pd.Series(0.0, index=report_normal.index)
    curve["strategy_return"] = report_normal["return"].fillna(0.0)
    curve["strategy_return_after_cost"] = report_normal["return"].fillna(0.0) - cost
    curve["benchmark_return"] = bench
    curve["excess_return_after_cost"] = (
        curve["strategy_return_after_cost"] - curve["benchmark_return"]
    )
    cum_curve = (1.0 + curve).cumprod() - 1.0
    cum_curve.to_csv(curve_path, encoding="utf-8")

    ax = cum_curve[
        ["strategy_return_after_cost", "benchmark_return", "excess_return_after_cost"]
    ].plot(figsize=(11, 6), linewidth=1.2)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Qlib Portfolio Return Curve")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    artifacts = {
        "report_normal_csv": str(report_path),
        "return_curve_csv": str(curve_path),
        "return_curve_png": str(plot_path),
    }
    if positions_path is not None:
        artifacts["positions_normal_pkl"] = str(positions_path)
    return artifacts


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    if joined["left"].nunique(dropna=True) < 2 or joined["right"].nunique(dropna=True) < 2:
        return float("nan")
    return float(joined["left"].corr(joined["right"], method="spearman"))


def _nan_reason(joined: pd.DataFrame, daily_count: int) -> str | None:
    if joined.empty:
        return "prediction and label have no overlapping non-null rows"
    if joined["pred"].nunique(dropna=True) < 2:
        return "prediction is constant or has fewer than 2 unique values"
    if joined["label"].nunique(dropna=True) < 2:
        return "label is constant or has fewer than 2 unique values"
    if daily_count == 0:
        return "no date has at least 2 non-null instruments with non-constant pred/label"
    return None


def _analyze_prediction_series(pred: pd.Series, dataset) -> dict[str, Any]:
    test = dataset.prepare("test", col_set="label")
    label = test.iloc[:, 0]
    pred = pred.reindex(label.index)
    joined = pd.concat([pred.rename("pred"), label.rename("label")], axis=1).dropna()
    if joined.empty:
        return {
            "rank_ic": float("nan"),
            "daily_rank_ic_mean": float("nan"),
            "icir": float("nan"),
            "diagnostics": {
                "test_label_rows": int(len(label)),
                "prediction_rows": int(len(pred)),
                "joined_rows": 0,
                "nan_reason": "prediction and label have no overlapping non-null rows",
            },
        }
    daily = joined.groupby(level="datetime").apply(
        lambda sub: _safe_spearman(sub["pred"], sub["label"])
    ).dropna()
    mean = float(daily.mean()) if len(daily) else float("nan")
    std = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    rank_ic = _safe_spearman(joined["pred"], joined["label"])
    return {
        "rank_ic": rank_ic,
        "daily_rank_ic_mean": mean,
        "daily_rank_ic_std": std,
        "icir": float("nan") if not std or np.isnan(std) else mean / std,
        "diagnostics": {
            "test_label_rows": int(len(label)),
            "prediction_rows": int(len(pred)),
            "joined_rows": int(len(joined)),
            "prediction_unique": int(joined["pred"].nunique(dropna=True)),
            "label_unique": int(joined["label"].nunique(dropna=True)),
            "daily_ic_count": int(len(daily)),
            "nan_reason": _nan_reason(joined, int(len(daily)))
            if np.isnan(rank_ic) or not len(daily)
            else None,
        },
    }


def simulate_qlib_portfolio(
    pred: pd.Series | pd.DataFrame,
    benchmark: str = "SH000300",
    topk: int = 50,
    n_drop: int = 5,
    provider_uri: str | None = None,
    output_dir: str | Path | None = None,
    account: float = 1e8,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
    min_cost: float = 5.0,
) -> dict[str, Any]:
    """Run qlib native TopkDropoutStrategy return backtest and save return curves."""
    init_qlib_once(provider_uri=provider_uri)
    with suppress_qlib_console():
        from qlib.contrib.evaluate import backtest_daily
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    if isinstance(pred, pd.DataFrame):
        if pred.shape[1] != 1:
            raise ValueError("pred DataFrame must have exactly one column")
        pred = pred.iloc[:, 0]
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError("pred must use qlib MultiIndex ['datetime', 'instrument']")
    pred = pred.dropna().sort_index()
    if pred.empty:
        raise ValueError("pred contains no non-null scores")

    dates = pred.index.get_level_values("datetime")
    start_time = str(pd.Timestamp(dates.min()).date())
    end_time = str(pd.Timestamp(dates.max()).date())
    min_daily_instruments = int(pred.groupby(level="datetime").size().min())
    effective_topk = max(1, min(int(topk), min_daily_instruments))
    effective_n_drop = max(0, min(int(n_drop), effective_topk))

    strategy = TopkDropoutStrategy(
        signal=pred,
        topk=effective_topk,
        n_drop=effective_n_drop,
    )
    with suppress_qlib_console(), _suppress_known_qlib_backtest_warnings():
        report_normal, positions_normal = backtest_daily(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy,
            benchmark=benchmark,
            account=account,
            exchange_kwargs={
                "freq": "day",
                "limit_threshold": None,
                "deal_price": None,
                "open_cost": open_cost,
                "close_cost": close_cost,
                "min_cost": min_cost,
            },
        )
    if report_normal is None or report_normal.empty:
        raise ValueError("qlib backtest returned an empty report_normal")

    ret = report_normal["return"].fillna(0.0)
    cost = report_normal.get("cost", pd.Series(0.0, index=report_normal.index)).fillna(0.0)
    bench = report_normal.get("bench", pd.Series(0.0, index=report_normal.index)).fillna(0.0)
    after_cost = ret - cost
    excess_after_cost = after_cost - bench
    artifacts = _save_return_artifacts(report_normal, positions_normal, output_dir)

    return {
        "backend": "qlib",
        "strategy": "TopkDropoutStrategy",
        "executor": "SimulatorExecutor",
        "benchmark": benchmark,
        "requested_topk": int(topk),
        "requested_n_drop": int(n_drop),
        "topk": effective_topk,
        "n_drop": effective_n_drop,
        "min_daily_instruments": min_daily_instruments,
        "prediction_rows": int(pred.shape[0]),
        "prediction_start": start_time,
        "prediction_end": end_time,
        "report_rows": int(len(report_normal)),
        "latest_return": float(ret.iloc[-1]),
        "latest_return_after_cost": float(after_cost.iloc[-1]),
        "latest_benchmark_return": float(bench.iloc[-1]),
        "cumulative_return": float((1.0 + ret).prod() - 1.0),
        "cumulative_return_after_cost": float((1.0 + after_cost).prod() - 1.0),
        "cumulative_benchmark_return": float((1.0 + bench).prod() - 1.0),
        "cumulative_excess_return_after_cost": float((1.0 + excess_after_cost).prod() - 1.0),
        "risk": {
            "strategy": _risk_dict(ret),
            "strategy_after_cost": _risk_dict(after_cost),
            "excess_after_cost": _risk_dict(excess_after_cost),
            "benchmark": _risk_dict(bench),
        },
        "report_tail": _jsonable_frame(report_normal.tail(10)),
        "artifacts": artifacts,
        "suppressed_warnings": [
            "qlib future calendar fallback warning",
            "qlib SimulatorExecutor common_infra warning",
        ],
    }
