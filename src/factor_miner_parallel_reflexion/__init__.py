from __future__ import annotations

from . import branches as _branches
from . import candidate as _candidate
from . import constants as _constants
from . import evaluation as _evaluation
from . import library as _library
from . import memory as _memory
from . import models as _models
from . import portfolio as _portfolio
from . import reflexion as _reflexion
from . import runner as _runner
from . import scheduler as _scheduler
from . import utils as _utils

_MODULES = (
    _constants,
    _models,
    _memory,
    _branches,
    _portfolio,
    _utils,
    _candidate,
    _library,
    _evaluation,
    _reflexion,
    _scheduler,
    _runner,
)

for _module in _MODULES:
    globals().update(
        {
            _name: _value
            for _name, _value in vars(_module).items()
            if not _name.startswith("__")
        }
    )


def parse_args(argv: list[str] | None = None):
    from .cli import parse_args as _parse_args

    return _parse_args(argv)


def config_from_args(args):
    from .cli import config_from_args as _config_from_args

    return _config_from_args(args)


def main(argv: list[str] | None = None) -> None:
    from .cli import main as _main

    _main(argv)


__all__ = sorted(
    _name for _name in globals() if not _name.startswith("__") and _name != "_MODULES"
)
