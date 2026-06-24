"""
Helpers for injecting custom tools into the minimal REPL environment.
"""

from dataclasses import dataclass
from typing import Any


RESERVED_TOOL_NAMES = frozenset(
    {
        "llm_query",
        "SHOW_VARS",
        "answer",
        "context",
        "context_0",
        "_stdout",
        "_stderr",
        "__builtins__",
        "__name__",
    }
)


@dataclass
class ToolInfo:
    name: str
    value: Any
    description: str | None = None

    @property
    def is_callable(self) -> bool:
        return callable(self.value)


def parse_tool_entry(name: str, entry: Any) -> ToolInfo:
    """Parse a plain tool value or {"tool": value, "description": "..."} entry."""
    if isinstance(entry, dict) and "tool" in entry:
        description = entry.get("description")
        if not isinstance(description, str):
            description = None
        return ToolInfo(name=name, value=entry["tool"], description=description)
    return ToolInfo(name=name, value=entry)


def parse_custom_tools(custom_tools: dict[str, Any] | None) -> list[ToolInfo]:
    if not custom_tools:
        return []
    return [parse_tool_entry(name, entry) for name, entry in custom_tools.items()]


def extract_tool_value(entry: Any) -> Any:
    if isinstance(entry, dict) and "tool" in entry:
        return entry["tool"]
    return entry


def format_tools_for_prompt(custom_tools: dict[str, Any] | None) -> str | None:
    tool_infos = parse_custom_tools(custom_tools)
    if not tool_infos:
        return None

    lines = []
    for tool in tool_infos:
        if tool.description:
            lines.append(f"- `{tool.name}`: {tool.description}")
        elif tool.is_callable:
            lines.append(f"- `{tool.name}`: A custom function")
        else:
            lines.append(f"- `{tool.name}`: A custom {type(tool.value).__name__} value")
    return "\n".join(lines)


def validate_custom_tools(custom_tools: dict[str, Any] | None) -> None:
    if not custom_tools:
        return

    conflicts = set(custom_tools.keys()) & RESERVED_TOOL_NAMES
    if conflicts:
        raise ValueError(
            "Custom tools cannot override reserved REPL names: "
            f"{sorted(conflicts)}. Reserved names: {sorted(RESERVED_TOOL_NAMES)}"
        )
