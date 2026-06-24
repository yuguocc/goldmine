from functools import lru_cache
from typing import Tuple

import backtrader as bt
from backtrader.feeds import PandasDirectData

__all__ = ["CryptoDataFeed", "make_crypto_datafeed"]


BASE_LINES: Tuple[str, ...] = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


class CryptoDataFeed(PandasDirectData):
    """Backward-compatible crypto feed with signal_1 through signal_5."""

    lines = BASE_LINES + (
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_5",
        "vwap",
    )
    params = tuple((name, idx) for idx, name in enumerate(lines)) + (
        ("dtformat", "%Y-%m-%d %H:%M:%S"),
        ("timeframe", bt.TimeFrame.Minutes),
    )


@lru_cache(maxsize=32)
def make_crypto_datafeed(number_of_signals: int) -> type[PandasDirectData]:
    """Create a PandasDirectData subclass for signal_1..signal_N."""
    if not isinstance(number_of_signals, int) or number_of_signals <= 0:
        raise ValueError("number_of_signals must be a positive integer")

    signal_lines = tuple(f"signal_{idx}" for idx in range(1, number_of_signals + 1))
    lines = BASE_LINES + signal_lines + ("vwap",)
    params = tuple((name, idx) for idx, name in enumerate(lines)) + (
        ("dtformat", "%Y-%m-%d %H:%M:%S"),
        ("timeframe", bt.TimeFrame.Minutes),
    )
    class_name = f"CryptoDataFeed{number_of_signals}"
    return type(class_name, (PandasDirectData,), {"lines": lines, "params": params})
