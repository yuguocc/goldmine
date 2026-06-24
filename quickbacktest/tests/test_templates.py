from quickbacktest import read_quickbacktest_template, read_signal_template


def test_read_signal_template_renders_module_name():
    template = read_signal_template("RlmTemplateFactor")

    assert "class RlmTemplateFactor(BaseSignal):" in template
    assert 'name = "RlmTemplateFactor"' in template
    assert "class AgentSignal(BaseSignal):" not in template


def test_read_quickbacktest_template_rejects_unknown_name():
    try:
        read_quickbacktest_template("missing")
    except ValueError as exc:
        assert "unknown template_name" in str(exc)
    else:
        raise AssertionError("expected ValueError")
