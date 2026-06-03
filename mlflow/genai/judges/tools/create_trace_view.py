from __future__ import annotations

import json
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.entities.trace_view import SpanRange, SpanSelector
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.types.llm import (
    FunctionToolDefinition,
    ParamProperty,
    ToolDefinition,
    ToolParamsSchema,
)


def _parse_ranges_json(ranges_json: str) -> list[SpanRange]:
    raw_ranges = json.loads(ranges_json)
    ranges = []
    for i, r in enumerate(raw_ranges):
        from_sel = SpanSelector.from_dict(r["from_selector"])
        to_sel = SpanSelector.from_dict(r["to_selector"]) if r.get("to_selector") else None
        ranges.append(
            SpanRange(
                from_selector=from_sel,
                to_selector=to_sel,
                label=r.get("label", ""),
                description=r.get("description", ""),
                input_path=r.get("input_path"),
                output_path=r.get("output_path"),
                position=r.get("position", i),
            )
        )
    return ranges


class CreateTraceViewTool(JudgeTool):
    @property
    def name(self) -> str:
        return "create_trace_view"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="create_trace_view",
                description=(
                    "Create a multi-range trace view that highlights key parts of the trace. "
                    "Each range in ranges_json maps to a milestone or phase in the agent "
                    "trajectory. Use span IDs from list_spans/get_span to build selectors."
                ),
                parameters=ToolParamsSchema(
                    properties={
                        "name": ParamProperty(
                            type="string",
                            description="Name for the trace view",
                        ),
                        "ranges_json": ParamProperty(
                            type="string",
                            description=(
                                'JSON array of SpanRange objects. Each has: '
                                '"from_selector" (required, object with "span_id"), '
                                '"to_selector" (optional, object with "span_id"), '
                                '"label" (string), "description" (string), '
                                '"position" (integer, 0-indexed order). '
                                "Example: "
                                '[{"from_selector": {"span_id": "abc"}, '
                                '"label": "Step 1", "description": "...", "position": 0}]'
                            ),
                        ),
                    },
                    required=["name", "ranges_json"],
                ),
            ),
        )

    def invoke(self, trace: Trace, name: str, ranges_json: str, **kwargs) -> Any:
        from mlflow.tracing.client import TracingClient

        ranges = _parse_ranges_json(ranges_json)
        client = TracingClient()
        view = client.create_trace_view(
            trace_id=trace.info.trace_id,
            name=name,
            ranges=ranges,
            created_by="trace_view_agent",
        )
        return {"view_id": view.view_id, "trace_id": view.trace_id, "name": view.name}
