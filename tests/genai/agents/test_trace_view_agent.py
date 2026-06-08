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


from mlflow.genai.agents.trace_view_agent import expand_to_a2ui


def _component_by_id(doc, element_id):
    return next(c for c in doc["components"] if c["id"] == element_id)


def test_expand_builds_column_root_referencing_all_children():
    intent = _ViewIntent(
        name="View",
        elements=[
            {"kind": "text", "text": "Overview"},
            {"kind": "span_detail", "span_type": "LLM", "title": "LLM call"},
            {"kind": "feedback_thumbs", "feedback_name": "overall", "label": "Overall"},
        ],
    )

    doc = expand_to_a2ui(intent)

    assert doc["root"] == "col-root"
    root = _component_by_id(doc, "col-root")
    child_ids = root["component"]["Column"]["children"]["explicitList"]
    # root + 3 children
    assert len(doc["components"]) == 4
    assert len(child_ids) == 3
    # every referenced child id exists as a component
    component_ids = {c["id"] for c in doc["components"]}
    assert set(child_ids).issubset(component_ids)
    # ids are unique
    assert len(component_ids) == len(doc["components"])


def test_expand_text_element():
    doc = expand_to_a2ui(_ViewIntent(name="V", elements=[{"kind": "text", "text": "Hello"}]))
    child_id = doc["components"][0]["component"]["Column"]["children"]["explicitList"][0]
    text = _component_by_id(doc, child_id)
    assert text["component"]["Text"] == {"text": {"literal": "Hello"}, "usageHint": "h3"}


def test_expand_span_detail_desugars_span_type_to_selector():
    doc = expand_to_a2ui(
        _ViewIntent(name="V", elements=[{"kind": "span_detail", "span_type": "TOOL", "title": "Tool"}])
    )
    child_id = doc["components"][0]["component"]["Column"]["children"]["explicitList"][0]
    detail = _component_by_id(doc, child_id)
    assert detail["component"]["GenAISpanDetail"] == {
        "selector": {"span_type": "TOOL"},
        "title": "Tool",
    }


def test_expand_feedback_thumbs_targets_trace():
    doc = expand_to_a2ui(
        _ViewIntent(
            name="V",
            elements=[{"kind": "feedback_thumbs", "feedback_name": "quality", "label": "Quality"}],
        )
    )
    child_id = doc["components"][0]["component"]["Column"]["children"]["explicitList"][0]
    fb = _component_by_id(doc, child_id)
    assert fb["component"]["FeedbackThumbs"] == {
        "name": "quality",
        "target": "trace",
        "label": "Quality",
    }


def test_expand_omits_optional_fields_when_absent():
    doc = expand_to_a2ui(
        _ViewIntent(
            name="V",
            elements=[
                {"kind": "span_detail", "span_type": "LLM"},
                {"kind": "feedback_thumbs", "feedback_name": "q"},
            ],
        )
    )
    detail = _component_by_id(doc, "detail-0")
    fb = _component_by_id(doc, "feedback-1")
    assert detail["component"]["GenAISpanDetail"] == {"selector": {"span_type": "LLM"}}
    assert fb["component"]["FeedbackThumbs"] == {"name": "q", "target": "trace"}
