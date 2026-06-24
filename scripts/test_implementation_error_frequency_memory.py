from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rlm_factor_memory import RlmFactorMemoryManager


def implementation_reflexion(lines: list[str]) -> dict[str, str]:
    return {
        "phase": "implementation_errors",
        "reflexion": "\n".join(
            [
                "## Search Metadata",
                "- memory_type: implementation_errors",
                "- keywords: mock, implementation, error_frequency",
                "- failure_types: will_be_replaced_by_error_frequency",
                "- tools_or_functions: save_signal, run_signal",
                "- data_contracts: signal.py, factor_data.csv, analysis.json",
                "- robust_pattern: validate artifacts before admission",
                "- avoid_pattern: repeated missing imports or missing saves",
                "- next_action: fix the highest-frequency implementation error first",
                "",
                "## Overall Guidance",
                "Use the counted implementation failures to prioritize next fixes.",
                "",
                "## Error Frequency",
                *lines,
                "",
                "## Failed Function Calls",
                "- Synthetic script only; no tool calls.",
                "",
                "## Error Patterns",
                "- Repeated implementation errors should become stronger P_fail signals.",
                "",
                "## Robust Patterns",
                "- Keep save_signal and imports explicit.",
                "",
                "## Next Round Fixes",
                "- Address the highest cumulative count first.",
            ]
        ),
    }


def failed_result(module: str, error_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        module_name=module,
        label="failed",
        error_type=error_type,
        error=f"synthetic {error_type}",
        ic=None,
    )


def main() -> None:
    manager = RlmFactorMemoryManager(max_entries=5)
    manager.update(
        round_number=1,
        reflexions=[
            implementation_reflexion(
                [
                    "- error_type=NameError; count=2; frequency=0.67; cause=pd not imported; evidence=NameError: pd; fix=import pandas as pd",
                    "- error_type=MissingSaveSignal; count=1; frequency=0.33; cause=save_signal not called; evidence=missing signal.py; fix=call save_signal before final answer",
                ]
            )
        ],
        round_ic={"round": 1, "best_ic": None, "improved": False},
        best_signal=None,
        results=[
            failed_result("R001C001", "NameError"),
            failed_result("R001C002", "NameError"),
            failed_result("R001C003", "MissingSaveSignal"),
        ],
        admission={"status": "skipped"},
    )
    manager.update(
        round_number=2,
        reflexions=[
            implementation_reflexion(
                [
                    "- error_type=MissingSaveSignal; count=3; frequency=0.75; cause=save_signal path missing; evidence=RLM did not save signal.py; fix=always call save_signal",
                    "- error_type=NameError; count=1; frequency=0.25; cause=np not imported; evidence=NameError: np; fix=import numpy as np",
                ]
            )
        ],
        round_ic={"round": 2, "best_ic": None, "improved": False},
        best_signal=None,
        results=[
            failed_result("R002C001", "MissingSaveSignal"),
            failed_result("R002C002", "MissingSaveSignal"),
            failed_result("R002C003", "NameError"),
        ],
        admission={"status": "skipped"},
    )

    state = manager.to_dict()["state"]
    stats = {
        item["error_type"]: item
        for item in state["failure_type_stats"].values()
    }
    assert stats["MissingSaveSignal"]["count"] == 4, stats
    assert stats["NameError"]["count"] == 3, stats
    assert stats["MissingSaveSignal"]["rounds"] == [1, 2], stats
    assert stats["NameError"]["rounds"] == [1, 2], stats

    prompt = manager.prompt_text()
    missing_index = prompt.index("MissingSaveSignal: count=4")
    name_index = prompt.index("NameError: count=3")
    assert missing_index < name_index, prompt

    print("PASS implementation error frequency accumulation")
    print(
        json.dumps(
            {
                "failure_type_stats": state["failure_type_stats"],
                "prompt_excerpt": prompt[
                    prompt.index("### high_frequency_failure_types") :
                ].split("\n\n", 1)[0],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
