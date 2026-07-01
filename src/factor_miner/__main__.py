"""Command-line entry point for python -m src.factor_miner."""

from __future__ import annotations

from .core import main

if __name__ == "__main__":
    raise SystemExit(main())