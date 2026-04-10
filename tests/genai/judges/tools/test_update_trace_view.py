import json
from unittest import mock

import pytest

from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.genai.judges.tools.update_trace_view import UpdateTraceViewTool


@pytest.fixture
def tool():
    return UpdateTraceViewTool()


def test_tool_name(tool):
    assert tool.name == "update_trace_view"


def test_tool_definition_has_required_params(tool):
    defn = tool.get_definition()
    props = defn.function.parameters.properties
    assert "view_id" in props
    assert "name" in props
    assert "ranges_json" in props
    assert defn.function.parameters.required == ["view_id"]


@mock.patch("mlflow.genai.judges.tools.update_trace_view.TracingClient")
def test_invoke_update_name(mock_client_cls, tool):
    mock_client = mock_client_cls.return_value
    mock_client.update_trace_view.return_value = TraceView(
        name="Updated",
        trace_id="tr-123",
        view_id="tv-456",
    )

    trace = mock.MagicMock()
    trace.info.trace_id = "tr-123"

    result = tool.invoke(trace=trace, view_id="tv-456", name="Updated")

    mock_client.update_trace_view.assert_called_once_with(
        view_id="tv-456",
        name="Updated",
        ranges=None,
    )
    assert result == {"view_id": "tv-456", "trace_id": "tr-123", "name": "Updated"}


@mock.patch("mlflow.genai.judges.tools.update_trace_view.TracingClient")
def test_invoke_update_ranges(mock_client_cls, tool):
    mock_client = mock_client_cls.return_value
    mock_client.update_trace_view.return_value = TraceView(
        name="View",
        trace_id="tr-123",
        view_id="tv-456",
        ranges=[
            SpanRange(
                from_selector=SpanSelector(span_id="span-1"),
                label="Phase 1",
            )
        ],
    )

    trace = mock.MagicMock()
    trace.info.trace_id = "tr-123"

    ranges_json = json.dumps([{"from_selector": {"span_id": "span-1"}, "label": "Phase 1"}])

    result = tool.invoke(trace=trace, view_id="tv-456", ranges_json=ranges_json)

    mock_client.update_trace_view.assert_called_once()
    call_kwargs = mock_client.update_trace_view.call_args[1]
    assert call_kwargs["view_id"] == "tv-456"
    assert len(call_kwargs["ranges"]) == 1
    assert call_kwargs["ranges"][0].label == "Phase 1"
    assert result["view_id"] == "tv-456"
