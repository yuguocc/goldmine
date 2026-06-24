from __future__ import annotations

import logging
from pathlib import Path
import sys
import uuid
import warnings

import pandas as pd

from quickbacktest import qlib_adapter


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_qlib_adapter_output_normalization" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_ohlcv() -> pd.DataFrame:
    rows = []
    for dt in pd.date_range("2020-01-01", periods=4, freq="D"):
        for code, close in (("SH600000", 10.0), ("SH600004", 20.0)):
            rows.append(
                {
                    "trade_time": dt,
                    "code": code,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000.0,
                    "vwap": close,
                    "amount": close * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_suppress_qlib_console_hides_stdout_stderr_warnings_and_logs(capsys, caplog):
    logger = logging.getLogger("qlib.test_quiet")

    with caplog.at_level(logging.WARNING):
        with qlib_adapter.suppress_qlib_console():
            print("hidden stdout")
            print("hidden stderr", file=sys.stderr)
            warnings.warn("hidden warning", RuntimeWarning)
            logger.warning("hidden qlib warning")

    captured = capsys.readouterr()
    assert "hidden stdout" not in captured.out
    assert "hidden stderr" not in captured.err
    assert "hidden qlib warning" not in caplog.text


def _fake_cross_section_factor_df() -> pd.DataFrame:
    rows = []
    codes = ["SH600000", "SH600004", "SH600009", "SH600010"]
    for day_idx, dt in enumerate(pd.date_range("2020-01-01", periods=5, freq="D")):
        for code_idx, code in enumerate(codes):
            close = 100.0 + day_idx * (code_idx + 1)
            rows.append(
                {
                    "trade_time": dt,
                    "code": code,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000.0,
                    "vwap": close,
                    "amount": close * 1000.0,
                    "score": float(code_idx),
                }
            )
    return pd.DataFrame(rows)


def _fake_decile_factor_df() -> pd.DataFrame:
    rows = []
    codes = [f"SH60{i:04d}" for i in range(20)]
    for day_idx, dt in enumerate(pd.date_range("2020-01-01", periods=5, freq="D")):
        for code_idx, code in enumerate(codes):
            close = 100.0 + day_idx * (code_idx + 1)
            rows.append(
                {
                    "trade_time": dt,
                    "code": code,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000.0,
                    "vwap": close,
                    "amount": close * 1000.0,
                    "score": float(code_idx),
                }
            )
    return pd.DataFrame(rows)


def _write_signal(base: Path, module_name: str, body: str) -> None:
    signals = base / "signals"
    signals.mkdir(parents=True, exist_ok=True)
    (signals / f"{module_name}.py").write_text(
        "from quickbacktest.base_types import BaseSignal\n\n"
        f"class {module_name}(BaseSignal):\n"
        f"    name = \"{module_name}\"\n\n"
        "    def compute(self, **kwargs):\n"
        f"{body}",
        encoding="utf-8",
    )


def test_qlib_adapter_accepts_long_factor_output(monkeypatch):
    base = _workspace_tmp("long")
    module_name = "LongOutputFactor"
    _write_signal(
        base,
        module_name,
        "        out = self.ohlcv[['trade_time', 'code', 'close']].copy()\n"
        "        out['score'] = out['close'].pct_change().fillna(0.0)\n"
        "        return out[['trade_time', 'code', 'score']]\n",
    )
    monkeypatch.setattr(qlib_adapter, "query_qlib_ohlcv", lambda *args, **kwargs: _fake_ohlcv())

    result = qlib_adapter.compute_qlib_factor_dataframe(
        signal_modules=[module_name],
        base_dir=base,
        instruments=["SH600000", "SH600004"],
        start="2020-01-01",
        end="2020-01-04",
    )

    assert module_name in result.columns
    assert "score" in result.columns
    assert len(result) == 8


def test_qlib_adapter_reindexes_wide_output_with_string_dates(monkeypatch):
    base = _workspace_tmp("wide")
    module_name = "StringIndexWideFactor"
    _write_signal(
        base,
        module_name,
        "        out = self.close.pct_change().fillna(0.0)\n"
        "        out.index = out.index.astype(str)\n"
        "        return out\n",
    )
    monkeypatch.setattr(qlib_adapter, "query_qlib_ohlcv", lambda *args, **kwargs: _fake_ohlcv())

    result = qlib_adapter.compute_qlib_factor_dataframe(
        signal_modules=[module_name],
        base_dir=base,
        instruments=["SH600000", "SH600004"],
        start="2020-01-01",
        end="2020-01-04",
    )

    assert module_name in result.columns
    assert result[module_name].notna().all()


def test_qlib_adapter_applies_uniform_factor_shift(monkeypatch):
    base = _workspace_tmp("shift")
    module_name = "ShiftedFactor"
    _write_signal(
        base,
        module_name,
        "        return self.close\n",
    )
    monkeypatch.setattr(qlib_adapter, "query_qlib_ohlcv", lambda *args, **kwargs: _fake_ohlcv())

    result = qlib_adapter.compute_qlib_factor_dataframe(
        signal_modules=[module_name],
        base_dir=base,
        instruments=["SH600000", "SH600004"],
        start="2020-01-01",
        end="2020-01-04",
        factor_shift=1,
    )

    first_day = pd.Timestamp("2020-01-01")
    second_day = pd.Timestamp("2020-01-02")
    assert result.loc[result["trade_time"] == first_day, module_name].isna().all()
    shifted = result.loc[result["trade_time"] == second_day].sort_values("code")
    assert shifted[module_name].tolist() == [10.0, 20.0]


def test_analyze_qlib_factors_returns_qlib_layered_ic():
    result = qlib_adapter.analyze_qlib_factors(
        factor_df=_fake_decile_factor_df(),
    )

    metrics = result["metrics"]["score"]
    assert "daily_ic_mean" in metrics
    assert "daily_rank_ic_mean" in metrics
    assert "ic_distribution" in metrics
    assert "rank_ic_distribution" in metrics
    assert metrics["ic_distribution"]["count"] == metrics["daily_ic_count"]
    assert metrics["rank_ic_distribution"]["count"] == metrics["daily_rank_ic_count"]
    assert "layered_ic" in metrics
    assert metrics["layered_ic"]["layer_type"] == "decile"
    assert set(metrics["layered_ic"]["deciles"]) == {
        f"Decile{i}" for i in range(1, 11)
    }
    assert metrics["layered_ic"]["deciles"]["Decile1"]["rows"] > 0


def test_analyze_qlib_factors_defaults_to_decile_layers():
    result = qlib_adapter.analyze_qlib_factors(
        factor_df=_fake_decile_factor_df(),
    )

    metrics = result["metrics"]["score"]
    assert metrics["layered_ic"]["layer_type"] == "decile"
    assert set(metrics["layered_ic"]["deciles"]) == {
        f"Decile{i}" for i in range(1, 11)
    }
    assert metrics["rank_ic_distribution"]["count"] == metrics["daily_rank_ic_count"]
