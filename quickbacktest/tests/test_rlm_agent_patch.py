from __future__ import annotations

import json
from pathlib import Path

import pytest

from factor_mining.agent import (
    DEFAULT_RLM_TIMEOUT,
    RLMFactorAgent,
)
from factor_mining.patching import (
    PatchError,
    parse_git_patch,
)
from factor_mining.quickbacktest_tools import QuickBacktestToolbox
from factor_mining.research_skill import FACTOR_RESEARCH_SKILL
from factor_mining.validation import REQUIRED_HTML_SECTIONS, validate_report_html


SIGNAL_SOURCE = """from quickbacktest.base_types import BaseSignal

class TestSignal(BaseSignal):
    name = "test"

    def compute(self, **kwargs):
        return kwargs["close"]
"""


RETURN_PATCH = """diff --git a/signals/TestSignal.py b/signals/TestSignal.py
--- a/signals/TestSignal.py
+++ b/signals/TestSignal.py
@@ -4,4 +4,5 @@ class TestSignal(BaseSignal):
     name = "test"
 
     def compute(self, **kwargs):
-        return kwargs["close"]
+        close = kwargs["close"]
+        return close.pct_change()
"""


def _report_html(missing_section: str | None = None):
    sections = [
        section for section in REQUIRED_HTML_SECTIONS if section != missing_section
    ]
    body = "\n".join(
        f'<section id="{section}"><h2>{section}</h2><p>ok</p></section>'
        for section in sections
    )
    return f"<!doctype html><html><body>{body}</body></html>"


def _write_signal(base_dir):
    signals_dir = base_dir / "signals"
    signals_dir.mkdir()
    target = signals_dir / "TestSignal.py"
    target.write_text(SIGNAL_SOURCE, encoding="utf-8")
    return target


def test_default_timeout_is_enabled_and_validated():
    agent = RLMFactorAgent()
    assert DEFAULT_RLM_TIMEOUT == 300.0
    assert agent.max_timeout == 300.0
    assert agent.custom_sub_tools == {}

    assert RLMFactorAgent(max_timeout=None).max_timeout is None
    with pytest.raises(ValueError, match="max_timeout"):
        RLMFactorAgent(max_timeout=0)


def test_parse_git_patch_accepts_replace_hunk():
    patches = parse_git_patch(RETURN_PATCH)

    assert len(patches) == 1
    assert patches[0].path == "signals/TestSignal.py"
    assert len(patches[0].hunks) == 1


def test_apply_patch_dry_run_does_not_write(tmp_path):
    target = _write_signal(tmp_path)
    agent = RLMFactorAgent()

    result = agent.apply_patch(tmp_path, RETURN_PATCH, dry_run=True)

    assert result["dry_run"] is True
    assert result["changed_paths"] == ["signals/TestSignal.py"]
    assert target.read_text(encoding="utf-8") == SIGNAL_SOURCE
    assert "return close.pct_change()" in result["previews"]["signals/TestSignal.py"]


def test_apply_patch_updates_signal(tmp_path):
    target = _write_signal(tmp_path)
    agent = RLMFactorAgent()

    result = agent.apply_patch(tmp_path, RETURN_PATCH)

    assert result["changed_paths"] == ["signals/TestSignal.py"]
    assert "return close.pct_change()" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "patch_text",
    [
        """diff --git a/C:/x.py b/C:/x.py
--- a/C:/x.py
+++ b/C:/x.py
@@ -1 +1 @@
-a
+b
""",
        """diff --git a/../signals/TestSignal.py b/../signals/TestSignal.py
--- a/../signals/TestSignal.py
+++ b/../signals/TestSignal.py
@@ -1 +1 @@
-a
+b
""",
        """diff --git a/signals/TestSignal.py b/signals/TestSignal.py
deleted file mode 100644
--- a/signals/TestSignal.py
+++ /dev/null
@@ -1 +0,0 @@
-from quickbacktest.base_types import BaseSignal
""",
        """diff --git a/signals/NewSignal.py b/signals/NewSignal.py
new file mode 100644
--- /dev/null
+++ b/signals/NewSignal.py
@@ -0,0 +1 @@
+from quickbacktest.base_types import BaseSignal
""",
        """diff --git a/signals/TestSignal.py b/signals/RenamedSignal.py
similarity index 100%
rename from signals/TestSignal.py
rename to signals/RenamedSignal.py
""",
        """diff --git a/signals/TestSignal.py b/signals/TestSignal.py
old mode 100644
new mode 100755
--- a/signals/TestSignal.py
+++ b/signals/TestSignal.py
@@ -1 +1 @@
-from quickbacktest.base_types import BaseSignal
+from quickbacktest.base_types import BaseSignal
""",
        """diff --git a/signals/TestSignal.py b/signals/TestSignal.py
Binary files a/signals/TestSignal.py and b/signals/TestSignal.py differ
""",
    ],
)
def test_apply_patch_rejects_unsafe_patch_metadata(tmp_path, patch_text):
    _write_signal(tmp_path)
    agent = RLMFactorAgent()

    with pytest.raises(PatchError):
        agent.apply_patch(tmp_path, patch_text, dry_run=True)


def test_apply_patch_rejects_context_mismatch(tmp_path):
    _write_signal(tmp_path)
    agent = RLMFactorAgent()
    patch = """diff --git a/signals/TestSignal.py b/signals/TestSignal.py
--- a/signals/TestSignal.py
+++ b/signals/TestSignal.py
@@ -1,3 +1,3 @@
 from quickbacktest.base_types import BaseSignal
 
-class Missing(BaseSignal):
+class Nope(BaseSignal):
"""

    with pytest.raises(PatchError, match="mismatch"):
        agent.apply_patch(tmp_path, patch, dry_run=True)


def test_apply_patch_validates_signal_contract(tmp_path):
    _write_signal(tmp_path)
    agent = RLMFactorAgent()
    patch = """diff --git a/signals/TestSignal.py b/signals/TestSignal.py
--- a/signals/TestSignal.py
+++ b/signals/TestSignal.py
@@ -1,6 +1,6 @@
 from quickbacktest.base_types import BaseSignal
 
-class TestSignal(BaseSignal):
+class OtherSignal(BaseSignal):
     name = "test"
 
     def compute(self, **kwargs):
"""

    with pytest.raises(ValueError, match="class TestSignal"):
        agent.apply_patch(tmp_path, patch, dry_run=True)


def test_quickbacktest_toolbox_exposes_approved_tools(tmp_path):
    _write_signal(tmp_path)
    toolbox = QuickBacktestToolbox(tmp_path)

    tools = toolbox.to_rlm_tools()

    assert set(tools) == {
        "list_agent_files",
        "read_agent_file",
        "summary",
        "read_signal_template",
        "read_strategy_template",
        "read_signal_evaluator_template",
        "read_strategy_evaluator_template",
        "query_market_summary",
        "evaluate_signal_rank_ic",
        "describe_signal_quantile",
        "run_strategy_backtest",
        "run_signal_evaluation",
        "run_strategy_evaluation",
        "query_qlib_ohlcv",
        "analyze_qlib_factors",
        "train_qlib_alpha158_augmented_model",
        "simulate_qlib_portfolio",
        "get_unstructured_factor_db_info",
        "register_unstructured_source",
        "list_unstructured_sources",
        "read_unstructured_source",
        "preview_unstructured_factor_records",
        "save_unstructured_factor_records",
        "query_unstructured_factor_records",
        "build_unstructured_factor_dataframe",
        "evaluate_unstructured_factor",
        "list_factors",
        "read_factor",
        "read_latest_result",
        "list_hypotheses",
        "read_hypothesis",
        "save_factor",
        "save_factor_result",
        "preview_git_patch",
        "validate_signal_code",
        "materialize_signal",
        "validate_strategy_code",
        "validate_evaluator_code",
        "materialize_strategy",
        "materialize_evaluator",
        "validate_report_html",
        "format_tool_result_for_report",
    }
    assert not any(
        banned in name
        for name in tools
        for banned in ("apply", "delete", "shell")
    )
    for name, entry in tools.items():
        assert set(entry) == {"tool", "description"}
        assert callable(entry["tool"])
        assert name in entry["description"]
        assert entry["description"].strip()
        assert entry["tool"].__doc__

    assert toolbox.list_agent_files() == ["signals/TestSignal.py"]
    content = toolbox.read_agent_file("signals/TestSignal.py", start=1, end=3)["content"]
    assert "1: from quickbacktest.base_types import BaseSignal" in content


def test_quickbacktest_toolbox_previews_patch(tmp_path):
    target = _write_signal(tmp_path)
    toolbox = QuickBacktestToolbox(tmp_path)

    result = toolbox.preview_git_patch(RETURN_PATCH)

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == SIGNAL_SOURCE
    assert result["changed_paths"] == ["signals/TestSignal.py"]


def test_factor_research_skill_contract():
    assert FACTOR_RESEARCH_SKILL["name"] == "rlm_native_factor_research"
    assert FACTOR_RESEARCH_SKILL["research_protocol"]["use_rlm_query_for_complex_subtasks"]
    assert FACTOR_RESEARCH_SKILL["html_sections"] == REQUIRED_HTML_SECTIONS
    assert "apply_patch" not in FACTOR_RESEARCH_SKILL["available_tools"]
    assert "validate_signal_code" in FACTOR_RESEARCH_SKILL["available_tools"]


def test_quickbacktest_does_not_import_factor_mining():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "factor_mining" in text:
            offenders.append(path.name)

    assert offenders == []


def test_validate_report_html_requires_fixed_sections():
    validate_report_html(_report_html())

    with pytest.raises(ValueError, match="missing required section"):
        validate_report_html(_report_html(missing_section="decision"))


def test_generate_signal_applies_fake_rlm_code(tmp_path, monkeypatch):
    agent = RLMFactorAgent()

    generated_code = '''from quickbacktest.base_types import BaseSignal

class GeneratedSignal(BaseSignal):
    name = "generated"

    def compute(self, **kwargs):
        return self.close.pct_change()
'''

    class FakeCompletion:
        response = json.dumps(
            {
                "module_name": "GeneratedSignal",
                "class_name": "GeneratedSignal",
                "hypothesis": "short-term momentum",
                "signal_description": "close percent change",
                "code": generated_code,
                "validation_notes": ["ok"],
            }
        )
        usage_summary = None

    class FakeRLM:
        def completion(self, context, *, root_prompt):
            assert context["module_name"] == "GeneratedSignal"
            assert "signal_template" in context
            return FakeCompletion()

    class FakeToolbox:
        def list_agent_files(self):
            return []

        def read_signal_template(self):
            return "from quickbacktest.base_types import BaseSignal"

        def query_market_summary(self, symbol, start=None, end=None):
            return {"stats": {"symbol": symbol, "rows": 10}, "columns": [], "sample_rows": []}

        def to_rlm_tools(self):
            return {}

    monkeypatch.setattr(agent, "_toolbox", lambda *args, **kwargs: FakeToolbox())
    monkeypatch.setattr(agent, "_make_rlm", lambda custom_tools=None: FakeRLM())

    result = agent.generate_signal(
        module_name="GeneratedSignal",
        base_dir=tmp_path,
        symbol="BTCUSDT",
    )

    target = tmp_path / "signals" / "GeneratedSignal.py"
    assert result["module_name"] == "GeneratedSignal"
    assert target.exists()
    assert "class GeneratedSignal(BaseSignal)" in target.read_text(encoding="utf-8")


def test_run_research_applies_fake_rlm_signal_code(tmp_path, monkeypatch):
    agent = RLMFactorAgent(max_depth=2)
    signal_code = '''from quickbacktest.base_types import BaseSignal

class ResearchSignal(BaseSignal):
    name = "research"

    def compute(self, **kwargs):
        return self.close.pct_change()
'''

    class FakeCompletion:
        response = json.dumps(
            {
                "status": "accepted",
                "module_name": "ResearchSignal",
                "signal_code": signal_code,
                "patches": [],
                "research_trace": [
                    {
                        "node": "hypothesis_validation",
                        "tool": "evaluate_signal_rank_ic",
                        "evidence": {"rank_ic_1h": 0.1},
                    }
                ],
                "report_html": _report_html(),
            }
        )
        usage_summary = None

    class FakeRLM:
        def completion(self, context, *, root_prompt):
            assert context["skill"]["research_protocol"]["use_rlm_query_for_complex_subtasks"]
            assert "validate_signal_code" in context["tool_manifest"]
            assert "Validation-only" in context["tool_manifest"]["validate_signal_code"]
            assert "Evidence tool" in context["tool_manifest"]["evaluate_signal_rank_ic"]
            assert "apply_patch" not in context["tool_manifest"]
            assert "rlm_query" in root_prompt
            return FakeCompletion()

    monkeypatch.setattr(agent, "_make_rlm", lambda custom_tools=None: FakeRLM())

    result = agent.run_research(
        base_dir=tmp_path,
        context={
            "research_reports": ["raw report text"],
            "data": {"source": "binance"},
            "ideas": ["momentum"],
            "memory": [],
            "objective": "validate momentum",
            "constraints": {},
            "universe": ["BTCUSDT"],
            "time_range": {"start": "2020-01-01", "end": "2020-02-01"},
        },
        module_name="ResearchSignal",
        symbol="BTCUSDT",
    )

    target = tmp_path / "signals" / "ResearchSignal.py"
    assert result["status"] == "accepted"
    assert result["signal_path"] == str(target)
    assert target.exists()
    assert "class ResearchSignal(BaseSignal)" in target.read_text(encoding="utf-8")


def test_run_research_applies_fake_rlm_patch(tmp_path, monkeypatch):
    target = _write_signal(tmp_path)
    agent = RLMFactorAgent(max_depth=2)

    class FakeCompletion:
        response = json.dumps(
            {
                "status": "needs_review",
                "module_name": "TestSignal",
                "signal_code": "",
                "patches": [{"patch": RETURN_PATCH}],
                "research_trace": [{"node": "patch_after_weak_ic"}],
                "report_html": _report_html(),
            }
        )
        usage_summary = None

    class FakeRLM:
        def completion(self, context, *, root_prompt):
            assert context["module_name"] == "TestSignal"
            return FakeCompletion()

    monkeypatch.setattr(agent, "_make_rlm", lambda custom_tools=None: FakeRLM())

    result = agent.run_research(
        base_dir=tmp_path,
        context={"research_reports": [], "ideas": ["repair"]},
        module_name="TestSignal",
    )

    assert result["status"] == "needs_review"
    assert result["patch_results"][0]["changed_paths"] == ["signals/TestSignal.py"]
    assert "return close.pct_change()" in target.read_text(encoding="utf-8")


def test_run_research_rejects_malformed_html(tmp_path, monkeypatch):
    agent = RLMFactorAgent()
    signal_code = '''from quickbacktest.base_types import BaseSignal

class BadReportSignal(BaseSignal):
    name = "bad"

    def compute(self, **kwargs):
        return self.close
'''

    class FakeCompletion:
        response = json.dumps(
            {
                "status": "accepted",
                "module_name": "BadReportSignal",
                "signal_code": signal_code,
                "patches": [],
                "research_trace": [],
                "report_html": _report_html(missing_section="diagnostics"),
            }
        )
        usage_summary = None

    class FakeRLM:
        def completion(self, context, *, root_prompt):
            return FakeCompletion()

    monkeypatch.setattr(agent, "_make_rlm", lambda custom_tools=None: FakeRLM())

    with pytest.raises(ValueError, match="missing required section"):
        agent.run_research(
            base_dir=tmp_path,
            context={"research_reports": []},
            module_name="BadReportSignal",
        )


def test_edit_signal_applies_fake_rlm_patch(tmp_path, monkeypatch):
    target = _write_signal(tmp_path)
    agent = RLMFactorAgent()

    class FakeCompletion:
        response = json.dumps(
            {
                "summary": "use returns",
                "patch": RETURN_PATCH,
                "validation_notes": ["ok"],
            }
        )
        usage_summary = None

    class FakeRLM:
        def completion(self, context, *, root_prompt):
            assert context["target_path"] == "signals/TestSignal.py"
            assert "git unified diff" in root_prompt
            return FakeCompletion()

    monkeypatch.setattr(agent, "_make_rlm", lambda custom_tools=None: FakeRLM())

    result = agent.edit_signal(
        module_name="TestSignal",
        base_dir=tmp_path,
        instruction="Use percent change.",
    )

    assert result["summary"] == "use returns"
    assert result["patch_result"]["changed_paths"] == ["signals/TestSignal.py"]
    assert "return close.pct_change()" in target.read_text(encoding="utf-8")
