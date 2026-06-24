from __future__ import annotations

import re
from pathlib import Path


QUICKBACKTEST_ROOT = Path(__file__).resolve().parent

TEMPLATE_FILES = {
    "signal": "signal_template.py",
    "strategy": "strategy_template.py",
    "signal_evaluator": "signal_benchmark.py",
    "strategy_evaluator": "strategy_benchmark._template.py",
}


def _validate_class_name(class_name: str) -> None:
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", class_name):
        raise ValueError(f"invalid Python class name: {class_name}")


def read_quickbacktest_template(template_name: str) -> str:
    """Return a raw template source file bundled with quickbacktest."""
    try:
        filename = TEMPLATE_FILES[template_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown template_name: {template_name}; "
            f"available templates: {sorted(TEMPLATE_FILES)}"
        ) from exc

    path = QUICKBACKTEST_ROOT / filename
    return path.read_text(encoding="utf-8")


def read_signal_template(module_name: str | None = None) -> str:
    """Return the BaseSignal template, optionally rendered for module_name."""
    template = read_quickbacktest_template("signal")
    if module_name is None:
        return template

    _validate_class_name(module_name)
    template = re.sub(
        r"class\s+AgentSignal\s*\(\s*BaseSignal\s*\):",
        f"class {module_name}(BaseSignal):",
        template,
        count=1,
    )
    template = re.sub(
        r'name\s*=\s*["\'][^"\']*["\']',
        f'name = "{module_name}"',
        template,
        count=1,
    )
    return template
