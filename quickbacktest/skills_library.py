"""Claude-style skills library and RLM tool adapter for quickbacktest."""

from __future__ import annotations

import json
import re
import shutil
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DELETED_METADATA_VALUE = "true"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("skill name must be a string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("skill name must be non-empty")
    if not _SKILL_NAME_RE.match(normalized):
        raise ValueError(
            "skill name must be a Claude-style directory name: "
            "letters, numbers, dot, underscore, or hyphen; no spaces or slashes"
        )
    return normalized


def _quote_yaml_scalar(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value.strip("'\"")
    return str(decoded)


def _format_frontmatter(
    *,
    name: str,
    description: str,
    tags: list[str],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "---",
        f"name: {_quote_yaml_scalar(name)}",
        f"description: {_quote_yaml_scalar(description)}",
    ]
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {_quote_yaml_scalar(tag)}")
    if metadata:
        for key in sorted(metadata):
            if key in {"name", "description", "tags"}:
                continue
            lines.append(f"{key}: {_quote_yaml_scalar(metadata[key])}")
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter")

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")

    end_index = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = idx
            break
    if end_index is None:
        raise ValueError("SKILL.md frontmatter is missing closing ---")

    meta: dict[str, Any] = {}
    section: str | None = None
    for raw_line in lines[1:end_index]:
        if not raw_line.strip():
            continue
        if raw_line.startswith("  - ") and section == "tags":
            meta.setdefault("tags", []).append(_unquote_yaml_scalar(raw_line[4:]))
            continue
        if raw_line.startswith("  ") and section == "metadata":
            key, sep, value = raw_line.strip().partition(":")
            if sep:
                meta.setdefault("metadata", {})[key.strip()] = _unquote_yaml_scalar(value)
            continue

        key, sep, value = raw_line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        section = key if key in {"tags", "metadata"} and not value else None
        if key == "tags":
            meta["tags"] = []
        elif key == "metadata":
            meta["metadata"] = {}
        else:
            meta[key] = _unquote_yaml_scalar(value)

    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return meta, body


def _is_deleted_metadata(meta: dict[str, Any]) -> bool:
    if str(meta.get("deleted", "")).lower() == _DELETED_METADATA_VALUE:
        return True
    metadata = meta.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("deleted", "")).lower() == _DELETED_METADATA_VALUE


def _is_deleted_skill(text: str) -> bool:
    meta, _body = _parse_frontmatter(text)
    return _is_deleted_metadata(meta)


def _deleted_skill_markdown(name: str) -> str:
    return (
        _format_frontmatter(
            name=name,
            description="Deleted skill tombstone.",
            tags=[],
            metadata={"deleted": _DELETED_METADATA_VALUE},
        )
        + "\n\nDeleted.\n"
    )


def _as_text_list(value: str | list[str], *, field_name: str) -> list[str]:
    if isinstance(value, str):
        items = [line.strip() for line in value.splitlines() if line.strip()]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise TypeError(f"{field_name} must be a string or list of strings")
    if not items:
        raise ValueError(f"{field_name} must be non-empty")
    return items


def _format_markdown_list(items: list[str]) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def _build_autonomous_skill_content(
    *,
    when_to_use: str | list[str],
    procedure: str | list[str],
    success_criteria: str | list[str] | None = None,
    notes: str | list[str] | None = None,
) -> str:
    sections = [
        "## When To Use",
        _format_markdown_list(_as_text_list(when_to_use, field_name="when_to_use")),
        "",
        "## Procedure",
        _format_markdown_list(_as_text_list(procedure, field_name="procedure")),
    ]
    if success_criteria:
        sections.extend(
            [
                "",
                "## Success Criteria",
                _format_markdown_list(
                    _as_text_list(success_criteria, field_name="success_criteria")
                ),
            ]
        )
    if notes:
        sections.extend(
            [
                "",
                "## Notes",
                _format_markdown_list(_as_text_list(notes, field_name="notes")),
            ]
        )
    return "\n".join(sections)


def _normalize_skill_file_path(relative_path: str) -> Path:
    if not isinstance(relative_path, str):
        raise TypeError("skill file path must be a string")
    path = Path(relative_path.strip())
    if not str(path):
        raise ValueError("skill file path must be non-empty")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("skill file path must stay inside the skill directory")
    if path.name == "":
        raise ValueError("skill file path must point to a file")
    return path


def _assert_inside_root(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
        raise ValueError("resolved path escapes library root")
    return resolved_path


def _remove_tree(path: Path) -> None:
    def _onerror(func, failed_path, _exc_info):
        try:
            Path(failed_path).chmod(stat.S_IWRITE)
            func(failed_path)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_onerror)


@dataclass
class Skill:
    """One Claude/Codex-style skill entry backed by SKILL.md."""

    name: str
    description: str
    content: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.name = _normalize_name(self.name)
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("skill description must be a non-empty string")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("skill content must be a non-empty string")
        self.description = self.description.strip()
        self.tags = [str(tag) for tag in self.tags]
        self.metadata = dict(self.metadata or {})
        self.files = {
            str(_normalize_skill_file_path(path).as_posix()): str(content)
            for path, content in dict(self.files or {}).items()
        }

    @property
    def markdown(self) -> str:
        body = self.content.strip()
        return (
            _format_frontmatter(
                name=self.name,
                description=self.description,
                tags=self.tags,
                metadata=self.metadata,
            )
            + "\n\n"
            + body
            + "\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "files": dict(self.files),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        return cls(
            name=data["name"],
            description=data["description"],
            content=data["content"],
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
            files=dict(data.get("files", {})),
            created_at=data.get("created_at") or _utc_now(),
            updated_at=data.get("updated_at") or _utc_now(),
        )

    @classmethod
    def from_markdown(cls, text: str) -> "Skill":
        meta, body = _parse_frontmatter(text)
        metadata = dict(meta.get("metadata", {}))
        for key, value in meta.items():
            if key not in {"name", "description", "tags", "metadata"}:
                metadata[key] = value
        return cls(
            name=meta["name"],
            description=meta["description"],
            content=body,
            tags=list(meta.get("tags", [])),
            metadata=metadata,
        )


class SkillsLibrary:
    """CRUD library using Claude-style skill folders.

    Layout:

    ``root/<skill-name>/SKILL.md``

    Optional support files can live beside it, for example:

    ``root/<skill-name>/references/*.md``
    ``root/<skill-name>/scripts/*.py``
    ``root/<skill-name>/assets/*``
    """

    def __init__(
        self,
        root: str | Path | None = None,
        skills: list[Skill] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self._skills: dict[str, Skill] = {}
        if self.root is not None and self.root.exists():
            self._load_from_root()
        for skill in skills or []:
            self.add(skill, overwrite=False)

    def _skill_dir(self, name: str) -> Path:
        if self.root is None:
            raise ValueError("skills library has no root directory")
        normalized = _normalize_name(name)
        return _assert_inside_root(self.root / normalized, self.root)

    def _skill_path(self, name: str) -> Path:
        return self._skill_dir(name) / "SKILL.md"

    def _load_from_root(self) -> None:
        assert self.root is not None
        for skill_path in sorted(self.root.glob("*/SKILL.md")):
            text = skill_path.read_text(encoding="utf-8")
            if _is_deleted_skill(text):
                meta, _body = _parse_frontmatter(text)
                deleted_name = meta.get("name")
                if isinstance(deleted_name, str):
                    self._skills.pop(deleted_name, None)
                continue
            skill = Skill.from_markdown(text)
            self._skills[skill.name] = skill

    def _write_skill(self, skill: Skill) -> None:
        if self.root is None:
            return
        skill_dir = self._skill_dir(skill.name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        self._skill_path(skill.name).write_text(skill.markdown, encoding="utf-8")
        for relative_path, content in skill.files.items():
            target = _assert_inside_root(skill_dir / relative_path, skill_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def add(
        self,
        skill: Skill | dict[str, Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        files: dict[str, str] | None = None,
        overwrite: bool = True,
    ) -> Skill:
        """Add a skill and write ``<name>/SKILL.md`` when rooted."""
        if skill is None:
            if name is None or description is None or content is None:
                raise ValueError(
                    "name, description, and content are required when skill is not provided"
                )
            skill_obj = Skill(
                name=name,
                description=description,
                content=content,
                tags=tags or [],
                metadata=metadata or {},
                files=files or {},
            )
        elif isinstance(skill, Skill):
            skill_obj = skill
        elif isinstance(skill, dict):
            skill_obj = Skill.from_dict(skill)
        else:
            raise TypeError("skill must be a Skill, dict, or None")

        existing = self._skills.get(skill_obj.name)
        if existing is not None and not overwrite:
            raise ValueError(f"skill already exists: {skill_obj.name}")
        if existing is not None:
            skill_obj.created_at = existing.created_at
            skill_obj.updated_at = _utc_now()

        self._skills[skill_obj.name] = skill_obj
        self._write_skill(skill_obj)
        return skill_obj

    def read(self, name: str) -> Skill:
        """Read a skill by name, loading from ``SKILL.md`` if needed."""
        normalized = _normalize_name(name)
        if normalized not in self._skills and self.root is not None:
            path = self._skill_path(normalized)
            if path.exists():
                text = path.read_text(encoding="utf-8")
                if _is_deleted_skill(text):
                    raise KeyError(f"skill not found: {normalized}")
                self._skills[normalized] = Skill.from_markdown(text)
        try:
            return self._skills[normalized]
        except KeyError as exc:
            raise KeyError(f"skill not found: {normalized}") from exc

    def list_files(self, name: str) -> list[str]:
        """List support files under a skill directory, excluding ``SKILL.md``."""
        skill = self.read(name)
        if self.root is None:
            return sorted(skill.files)

        skill_dir = self._skill_dir(skill.name)
        if not skill_dir.exists():
            return []

        files: list[str] = []
        for path in skill_dir.rglob("*"):
            if not path.is_file() or path.name == "SKILL.md":
                continue
            files.append(path.relative_to(skill_dir).as_posix())
        return sorted(files)

    def read_file(
        self,
        name: str,
        relative_path: str,
        *,
        max_chars: int | None = 20000,
    ) -> str:
        """Read ``SKILL.md`` or a support file inside a skill directory."""
        skill = self.read(name)
        normalized_path = _normalize_skill_file_path(relative_path)
        if max_chars is not None and max_chars <= 0:
            raise ValueError("max_chars must be positive or None")

        if self.root is None:
            key = normalized_path.as_posix()
            if key == "SKILL.md":
                text = skill.markdown
            else:
                try:
                    text = skill.files[key]
                except KeyError as exc:
                    raise FileNotFoundError(key) from exc
        else:
            skill_dir = self._skill_dir(skill.name)
            path = _assert_inside_root(skill_dir / normalized_path, skill_dir)
            if not path.is_file():
                raise FileNotFoundError(normalized_path.as_posix())
            text = path.read_text(encoding="utf-8")

        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text

    def delete(self, name: str) -> Skill:
        """Delete a skill from memory and remove its folder when rooted."""
        normalized = _normalize_name(name)
        skill = self.read(normalized)
        self._skills.pop(normalized, None)
        if self.root is not None:
            skill_dir = self._skill_dir(normalized)
            if skill_dir.exists():
                try:
                    _remove_tree(skill_dir)
                except PermissionError:
                    skill_dir.mkdir(parents=True, exist_ok=True)
                    self._skill_path(normalized).write_text(
                        _deleted_skill_markdown(normalized),
                        encoding="utf-8",
                    )
        return skill

    def list(self) -> list[Skill]:
        """Return all skills sorted by name."""
        if self.root is not None and self.root.exists():
            self._load_from_root()
        return [self._skills[name] for name in sorted(self._skills)]

    def manifest(self) -> list[dict[str, Any]]:
        """Return lightweight skill metadata for RLM tool discovery."""
        return [skill.to_manifest() for skill in self.list()]

    def read_for_rlm(
        self,
        name: str,
        *,
        max_file_chars: int | None = 20000,
    ) -> dict[str, Any]:
        """Read one skill as a compact payload suitable for RLM tools."""
        skill = self.read(name)
        support_files = []
        for path in self.list_files(skill.name):
            support_files.append(
                {
                    "path": path,
                    "content": self.read_file(
                        skill.name,
                        path,
                        max_chars=max_file_chars,
                    ),
                }
            )
        return {
            **skill.to_manifest(),
            "content": skill.content,
            "support_files": support_files,
        }

    def save(self, root: str | Path | None = None) -> Path:
        """Write all loaded skills as Claude/Codex-style skill folders."""
        if root is not None:
            self.root = Path(root)
        if self.root is None:
            raise ValueError("root is required to save a skills library")
        self.root.mkdir(parents=True, exist_ok=True)
        for skill in self.list():
            self._write_skill(skill)
        return self.root

    @classmethod
    def load(cls, root: str | Path) -> "SkillsLibrary":
        """Load a directory containing ``*/SKILL.md`` skills."""
        return cls(root=root)

    def to_rlm_tools(
        self,
        *,
        allow_write: bool = True,
        include_management_tools: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Expose this skills library as RLM custom tools."""

        def list_skills() -> list[dict[str, Any]]:
            """List available Claude-style skills with name and description."""
            return self.manifest()

        def read_skill(
            name: str,
            max_file_chars: int | None = 20000,
        ) -> dict[str, Any]:
            """Read a skill's SKILL.md content and support files."""
            return self.read_for_rlm(name, max_file_chars=max_file_chars)

        tools: dict[str, dict[str, Any]] = {
            "list_skills": {
                "tool": list_skills,
                "description": (
                    "list_skills() -> list available RLM skills from the "
                    "Claude-style skills library folder"
                ),
            },
            "read_skill": {
                "tool": read_skill,
                "description": (
                    "read_skill(name, max_file_chars=20000) -> read a skill's "
                    "SKILL.md instructions and support files"
                ),
            },
        }

        if allow_write:

            def create_skill(
                name: str,
                description: str,
                when_to_use: str | list[str],
                procedure: str | list[str],
                success_criteria: str | list[str] | None = None,
                notes: str | list[str] | None = None,
                tags: list[str] | None = None,
                allowed_tools: list[str] | str | None = None,
                files: dict[str, str] | None = None,
                overwrite: bool = False,
            ) -> dict[str, Any]:
                """Create a reusable Claude-style skill from structured guidance."""
                metadata: dict[str, Any] = {"created-by": "rlm"}
                if allowed_tools:
                    if isinstance(allowed_tools, str):
                        metadata["allowed-tools"] = allowed_tools
                    else:
                        metadata["allowed-tools"] = ", ".join(
                            str(tool).strip()
                            for tool in allowed_tools
                            if str(tool).strip()
                        )
                skill = self.add(
                    name=name,
                    description=description,
                    content=_build_autonomous_skill_content(
                        when_to_use=when_to_use,
                        procedure=procedure,
                        success_criteria=success_criteria,
                        notes=notes,
                    ),
                    tags=tags or [],
                    metadata=metadata,
                    files=files,
                    overwrite=overwrite,
                )
                return {
                    "ok": True,
                    "skill": skill.to_manifest(),
                    "files": self.list_files(skill.name),
                }

            def add_skill(
                name: str,
                description: str,
                content: str,
                files: dict[str, str] | None = None,
                overwrite: bool = True,
            ) -> dict[str, Any]:
                """Add or update a Claude-style skill in the library folder."""
                skill = self.add(
                    name=name,
                    description=description,
                    content=content,
                    files=files,
                    overwrite=overwrite,
                )
                return {"ok": True, "skill": skill.to_manifest()}

            def delete_skill(name: str) -> dict[str, Any]:
                """Delete a skill from the library by name."""
                skill = self.delete(name)
                return {"ok": True, "deleted": skill.name}

            tools["create_skill"] = {
                "tool": create_skill,
                "description": (
                    "create_skill(name, description, when_to_use, "
                    "procedure, success_criteria=None, notes=None, "
                    "tags=None, allowed_tools=None, files=None, "
                    "overwrite=False) -> autonomously create a reusable "
                    "Claude-style RLM skill without manually formatting "
                    "SKILL.md"
                ),
            }

            if include_management_tools:

                def list_skill_files(name: str) -> list[str]:
                    """List support files for one skill."""
                    return self.list_files(name)

                def read_skill_file(
                    name: str,
                    relative_path: str,
                    max_chars: int | None = 20000,
                ) -> str:
                    """Read SKILL.md or a support file inside one skill directory."""
                    return self.read_file(name, relative_path, max_chars=max_chars)

                tools.update(
                    {
                        "list_skill_files": {
                            "tool": list_skill_files,
                            "description": (
                                "list_skill_files(name) -> list references/scripts/assets "
                                "inside a skill directory"
                            ),
                        },
                        "read_skill_file": {
                            "tool": read_skill_file,
                            "description": (
                                "read_skill_file(name, relative_path, max_chars=20000) -> "
                                "read SKILL.md or a support file safely"
                            ),
                        },
                        "add_skill": {
                            "tool": add_skill,
                            "description": (
                                "add_skill(name, description, content, files=None, "
                                "overwrite=True) -> add or update a Claude-style skill"
                            ),
                        },
                        "delete_skill": {
                            "tool": delete_skill,
                            "description": (
                                "delete_skill(name) -> remove a skill from the "
                                "skills library"
                            ),
                        },
                    }
                )

        return tools

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip() in {skill.name for skill in self.list()}

    def __len__(self) -> int:
        return len(self.list())


def build_rlm_skill_tools(
    root: str | Path,
    *,
    allow_write: bool = True,
    include_management_tools: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build RLM custom tools from a Claude-style skills library folder."""
    return SkillsLibrary(root).to_rlm_tools(
        allow_write=allow_write,
        include_management_tools=include_management_tools,
    )
