from __future__ import annotations

from datetime import datetime

from quickbacktest import run as qbt_run
from BinanceDatabase.src.core import BinanceDatabase


def test_binance_database_imports():
    assert BinanceDatabase.__name__ == "BinanceDatabase"


def test_quickbacktest_binance_query_returns_required_columns():
    data = qbt_run.query_quickbacktest_ohlcv(
        venue="binance_um",
        symbol="BTCUSDT",
        start_ms=qbt_run.utc_ms(datetime(2020, 1, 1)),
        end_ms=qbt_run.utc_ms(datetime(2020, 1, 2)),
    )

    assert not data.empty
    assert {
        "trade_time",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }.issubset(data.columns)
