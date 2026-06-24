from __future__ import annotations

from src import factor_miner_parallel_reflexion as _impl

for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


if __name__ == "__main__":
    main()
