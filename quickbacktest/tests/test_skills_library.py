from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from quickbacktest import Skill, SkillsLibrary, build_rlm_skill_tools


def _workspace_tmp(name: str) -> Path:
    path = Path("runs") / "test_skills_library" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_skills_library_add_read_delete():
    root = _workspace_tmp("crud")
    library = SkillsLibrary(root)

    stored = library.add(
        name="rank-ic-review",
        content="Check rank IC, coverage, and correlation before admission.",
        description="Review a candidate factor evaluation.",
        tags=["evaluation", "ic"],
    )

    skill_path = root / "rank-ic-review" / "SKILL.md"
    assert stored.name == "rank-ic-review"
    assert skill_path.exists()
    assert skill_path.read_text(encoding="utf-8").startswith("---\n")
    assert len(library) == 1
    assert library.read("rank-ic-review").content.startswith("Check rank IC")
    assert library.delete("rank-ic-review").name == "rank-ic-review"
    assert len(library) == 0

    with pytest.raises(KeyError):
        library.read("rank-ic-review")

    deleted_path = root / "rank-ic-review" / "SKILL.md"
    if deleted_path.exists():
        assert 'deleted: "true"' in deleted_path.read_text(encoding="utf-8")
    else:
        assert not (root / "rank-ic-review").exists()

    reloaded = SkillsLibrary.load(root)
    assert len(reloaded) == 0
    with pytest.raises(KeyError):
        reloaded.read("rank-ic-review")


def test_skills_library_overwrite_preserves_created_at():
    library = SkillsLibrary()
    first = library.add(name="skill", description="First skill", content="first")
    second = library.add(name="skill", description="Second skill", content="second")

    assert second.content == "second"
    assert second.description == "Second skill"
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_skills_library_save_load():
    root = _workspace_tmp("save_load")
    library = SkillsLibrary(
        skills=[
            Skill(
                name="amplitude-factor",
                description="Use relative amplitude as a volatility feature.",
                content="Use (high - low) / (abs(close) + eps) as relative amplitude.",
                tags=["volatility"],
                metadata={"source": "test"},
            )
        ]
    )

    library.save(root)
    loaded = SkillsLibrary.load(root)

    skill = loaded.read("amplitude-factor")
    assert "relative amplitude" in skill.content
    assert skill.description == "Use relative amplitude as a volatility feature."
    assert skill.tags == ["volatility"]
    assert skill.metadata == {"source": "test"}
    assert (root / "amplitude-factor" / "SKILL.md").exists()


def test_skills_library_loads_claude_style_skill_folder():
    root = _workspace_tmp("claude_style")
    skill_dir = root / "factor-review"
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: factor-review
description: Review a generated factor before accepting it.
allowed-tools: read_signal_template, evaluate_signal_rank_ic
---

# Factor Review

Check rank IC, coverage, decay, turnover, and correlation.
""",
        encoding="utf-8",
    )
    (references_dir / "checklist.md").write_text(
        "- rank ic\n- coverage\n",
        encoding="utf-8",
    )

    library = SkillsLibrary.load(root)
    skill = library.read("factor-review")

    assert skill.name == "factor-review"
    assert skill.description == "Review a generated factor before accepting it."
    assert skill.metadata["allowed-tools"] == (
        "read_signal_template, evaluate_signal_rank_ic"
    )
    assert "rank IC" in skill.content
    assert library.list_files("factor-review") == ["references/checklist.md"]
    assert "coverage" in library.read_file(
        "factor-review",
        "references/checklist.md",
    )


def test_skills_library_exposes_rlm_tools():
    root = _workspace_tmp("rlm_tools")
    library = SkillsLibrary(root)
    library.add(
        name="rlm-factor-review",
        description="Review factor output during RLM runs.",
        content="Use this when deciding whether a generated factor is admissible.",
        tags=["factor", "review"],
        metadata={"allowed-tools": "run_signal, save_signal"},
        files={"references/checklist.md": "Check IC, coverage, and turnover."},
    )
    tools = build_rlm_skill_tools(root)

    assert set(tools) == {
        "list_skills",
        "read_skill",
        "create_skill",
    }
    for name, entry in tools.items():
        assert set(entry) == {"tool", "description"}
        assert callable(entry["tool"])
        assert name in entry["description"]

    assert tools["list_skills"]["tool"]() == [
        {
            "name": "rlm-factor-review",
            "description": "Review factor output during RLM runs.",
            "tags": ["factor", "review"],
            "metadata": {"allowed-tools": "run_signal, save_signal"},
        }
    ]
    skill = tools["read_skill"]["tool"]("rlm-factor-review")
    assert skill["description"] == "Review factor output during RLM runs."
    assert skill["support_files"] == [
        {
            "path": "references/checklist.md",
            "content": "Check IC, coverage, and turnover.",
        }
    ]


def test_skills_library_can_expose_management_tools_when_requested():
    root = _workspace_tmp("management_tools")
    tools = build_rlm_skill_tools(root, include_management_tools=True)

    assert {
        "list_skill_files",
        "read_skill_file",
        "add_skill",
        "delete_skill",
    }.issubset(tools)

    tools["add_skill"]["tool"](
        name="managed-skill",
        description="Manage skill with explicit tools.",
        content="Manual content.",
        files={"references/checklist.md": "check"},
    )
    assert tools["list_skill_files"]["tool"]("managed-skill") == [
        "references/checklist.md"
    ]
    assert tools["read_skill_file"]["tool"](
        "managed-skill",
        "references/checklist.md",
    ) == "check"
    assert tools["delete_skill"]["tool"]("managed-skill") == {
        "ok": True,
        "deleted": "managed-skill",
    }


def test_skills_library_create_skill_tool_builds_claude_style_skill():
    root = _workspace_tmp("create_skill")
    tools = build_rlm_skill_tools(root)

    result = tools["create_skill"]["tool"](
        name="factor-debugging",
        description="Debug failed factor generation runs.",
        when_to_use=[
            "A generated factor fails to run.",
            "The failure can become reusable debugging knowledge.",
        ],
        procedure=[
            "Read the traceback and identify the first user-code frame.",
            "Record the missing import, variable, or contract mismatch.",
            "Add the smallest reusable fix pattern.",
        ],
        success_criteria=[
            "The saved skill explains when to apply the pattern.",
            "The skill is not tied to one temporary run directory.",
        ],
        notes="Do not save secrets or one-off data paths.",
        tags=["rlm", "debugging"],
        allowed_tools=["list_skills", "read_skill", "run_signal"],
        files={"references/common-errors.md": "NameError: import the missing symbol."},
    )

    assert result["ok"] is True
    assert result["skill"]["name"] == "factor-debugging"
    skill_path = root / "factor-debugging" / "SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")
    assert 'created-by: "rlm"' in skill_text
    assert 'allowed-tools: "list_skills, read_skill, run_signal"' in skill_text
    assert "## When To Use" in skill_text
    assert "## Procedure" in skill_text
    assert "## Success Criteria" in skill_text
    assert "Do not save secrets" in skill_text
    assert tools["read_skill"]["tool"]("factor-debugging")["support_files"] == [
        {
            "path": "references/common-errors.md",
            "content": "NameError: import the missing symbol.",
        }
    ]

    with pytest.raises(ValueError, match="already exists"):
        tools["create_skill"]["tool"](
            name="factor-debugging",
            description="Duplicate.",
            when_to_use="same",
            procedure="same",
        )


def test_skills_library_rejects_unsafe_support_file_path():
    library = SkillsLibrary()

    with pytest.raises(ValueError, match="inside the skill directory"):
        library.add(
            name="bad",
            description="Bad support file path.",
            content="content",
            files={"../escape.md": "no"},
        )


def test_skills_library_rejects_duplicate_without_overwrite():
    library = SkillsLibrary()
    library.add(name="skill", description="First skill", content="first")

    with pytest.raises(ValueError, match="already exists"):
        library.add(
            name="skill",
            description="Second skill",
            content="second",
            overwrite=False,
        )
