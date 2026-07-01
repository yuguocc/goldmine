"""Compatibility wrapper for the factor miner CLI and imports.

The implementation lives in src.factor_miner.core. Importing this module returns
that implementation module so existing monkeypatches against factor_miner keep
patching the globals used by the migrated functions.
"""

from __future__ import annotations

import sys

from src.factor_miner import core as _core

if __name__ == "__main__":
    raise SystemExit(_core.main())

sys.modules[__name__] = _core