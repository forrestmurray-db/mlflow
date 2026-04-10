from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.create_trace_view import _parse_ranges_json
from mlflow.tracing.client import TracingClient
from mlflow.types.llm import (
    FunctionToolDefinition,
    ParamProperty,
    ToolDefinition,
    ToolParamsSchema,
)


class UpdateTraceViewTool(JudgeTool):
    @property
    def name(self) -> str:
        return "update_trace_view"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="update_trace_view",
                description=(
                    "Update an existing trace view. Can rename it, replace its ranges, "
                    "or both. Use this to iteratively refine a view based on user feedback."
                ),
                parameters=ToolParamsSchema(
                    properties={
                        "view_id": ParamProperty(
                            type="string",
                            description="The ID of the trace view to update (tv-prefixed UUID)",
                        ),
                        "name": ParamProperty(
                            type="string",
                            description="New name for the trace view (optional)",
                        ),
                        "ranges_json": ParamProperty(
                            type="string",
                            description=(
                                "JSON array of SpanRange objects to replace existing ranges. "
                                "Same format as create_trace_view. Optional — omit to keep "
                                "existing ranges."
                            ),
                        ),
                    },
                    required=["view_id"],
                ),
            ),
        )

    def invoke(
        self,
        trace: Trace,
        view_id: str,
        name: str | None = None,
        ranges_json: str | None = None,
        **kwargs,
    ) -> Any:
        ranges = _parse_ranges_json(ranges_json) if ranges_json else None
        client = TracingClient()
        view = client.update_trace_view(
            view_id=view_id,
            name=name,
            ranges=ranges,
        )
        return {"view_id": view.view_id, "trace_id": view.trace_id, "name": view.name}
