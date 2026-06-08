import pytest

from mlflow.genai.agents.trace_view_agent import _ElementIntent, _ViewIntent


def test_view_intent_parses_flat_elements():
    intent = _ViewIntent(
        name="SME labeling view",
        elements=[
            {"kind": "text", "text": "Trace overview"},
            {"kind": "span_detail", "span_type": "LLM", "title": "LLM call"},
            {"kind": "feedback_thumbs", "feedback_name": "overall_quality", "label": "Overall"},
        ],
    )
    assert intent.name == "SME labeling view"
    assert [e.kind for e in intent.elements] == ["text", "span_detail", "feedback_thumbs"]
    assert intent.elements[1].span_type == "LLM"
    assert intent.elements[2].feedback_name == "overall_quality"


def test_element_intent_rejects_unknown_kind():
    with pytest.raises(ValueError):
        _ElementIntent(kind="table")
