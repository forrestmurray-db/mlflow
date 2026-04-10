from __future__ import annotations

import json
import logging
from pathlib import Path

import pydantic

from mlflow.entities.trace_summary import Milestone, TraceSummary

_logger = logging.getLogger(__name__)

_SKILL_DIR = Path(__file__).resolve().parents[2] / "assistant" / "skills" / "analyze-mlflow-trace"

_SUMMARIZE_SYSTEM_PROMPT = """\
You are an expert at analyzing MLflow traces of AI agent trajectories.

Your task is to create an executive summary of the trace, identifying the key
milestones and phases of the agent's execution.

First, read the analyze-mlflow-trace skill to understand trace structure and
analysis techniques. Then explore the trace using the available tools.

Focus on:
- What the agent was trying to accomplish (root span inputs/outputs)
- The major phases/steps in the trajectory
- Key decisions, tool calls, and their outcomes
- Any errors or notable events

Return a structured summary with:
- A concise executive summary (2-4 sentences)
- A list of milestones representing the key phases, each with a descriptive
  label and explanation of what happened
"""

_CREATE_VIEW_SYSTEM_PROMPT = """\
You are an expert at creating MLflow trace views that highlight key parts of
an agent trajectory.

You have been given a TraceSummary that describes the key milestones of this
trace. Your job is to explore the trace and create a multi-range trace view
where each range corresponds to a milestone from the summary.

TraceSummary:
{summary_json}

For each milestone, find the span(s) that best represent that phase of
execution and create a SpanRange with:
- from_selector: the span where this phase starts (use span_id)
- to_selector: the span where this phase ends (use span_id), or omit for
  single span + subtree
- label: the milestone label
- description: the milestone description
- position: the order (0-indexed)

Use list_spans and get_span to explore the trace structure and find the right
spans for each milestone. Then call create_trace_view with the ranges.
"""


class _MilestoneSchema(pydantic.BaseModel):
    label: str = pydantic.Field(description="Short label for this phase, e.g. 'Planning & Template Lookup'")
    description: str = pydantic.Field(description="What happened in this phase and why it matters")


class _TraceSummarySchema(pydantic.BaseModel):
    summary: str = pydantic.Field(description="Executive summary of the agent trajectory (2-4 sentences)")
    milestones: list[_MilestoneSchema] = pydantic.Field(
        description="Key phases/steps identified in the trace, in chronological order"
    )


def _get_skills():
    from mlflow.genai.skills.parsing import SkillSet

    if _SKILL_DIR.exists():
        return SkillSet([_SKILL_DIR])
    return None


def summarize_trace(trace, model: str = "openai:/gpt-4o") -> TraceSummary:
    from mlflow.genai.judges.utils.invocation_utils import (
        get_chat_completions_with_structured_output,
    )
    from mlflow.types.llm import ChatMessage

    skills = _get_skills()

    system_msg = _SUMMARIZE_SYSTEM_PROMPT
    if skills:
        skill_descriptions = "\n".join(
            f"- {s.name}: {s.description}" for s in skills.skills
        )
        system_msg += f"\n\nAvailable skills:\n{skill_descriptions}"

    messages = [
        ChatMessage(role="system", content=system_msg),
        ChatMessage(role="user", content="Analyze this trace and provide a structured summary."),
    ]

    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_TraceSummarySchema,
        trace=trace,
        skills=skills,
    )

    return TraceSummary(
        trace_id=trace.info.trace_id,
        summary=result.summary,
        milestones=[Milestone(label=m.label, description=m.description) for m in result.milestones],
        model=model,
    )


def create_view_from_summary(
    trace_id: str,
    summary: TraceSummary,
    model: str = "openai:/gpt-4o",
    name: str | None = None,
):
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.tracking import MlflowClient
    from mlflow.types.llm import ChatMessage

    trace = MlflowClient().get_trace(trace_id)

    summary_json = json.dumps(summary.to_dict(), indent=2)
    system_msg = _CREATE_VIEW_SYSTEM_PROMPT.format(summary_json=summary_json)

    if not name:
        name = f"Milestone View — {summary.milestones[0].label}" if summary.milestones else "Milestone View"
    system_msg += f"\n\nUse this name for the view: {name}"

    messages = [
        ChatMessage(role="system", content=system_msg),
        ChatMessage(role="user", content="Explore the trace and create a trace view mapping each milestone to span ranges."),
    ]

    model_provider, model_name = _parse_model_uri(model)

    _invoke_litellm_and_handle_tools(
        provider=model_provider,
        model_name=model_name,
        messages=messages,
        trace=trace,
        num_retries=10,
        skills=_get_skills(),
    )

    # The view was created as a side effect of the CreateTraceViewTool.
    # Return the most recently created view.
    views = MlflowClient().list_trace_views(trace_id=trace_id)
    if views:
        return views[-1]
    return None


_CONVERSATIONAL_SYSTEM_PROMPT = """\
You are an expert trace analyst for MLflow traces of AI agent trajectories.

You help users understand, explore, and create filtered views of their traces.
You have tools to explore spans and create/update trace views.

When the user asks you to analyze a trace, use list_spans and get_span to explore
the trace structure, then explain what you find.

When asked to create or modify views, use create_trace_view and update_trace_view.
Each view contains ranges that highlight specific phases or milestones.

You can also summarize traces — identify key milestones and phases of execution.
"""


def create_conversational_agent(trace_id: str, model: str = "openai:/gpt-4o"):
    """Create a conversational trace analysis agent.

    Returns a callable that accepts a messages list and returns the agent's response.
    """
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.tracking import MlflowClient
    from mlflow.types.llm import ChatMessage

    trace = MlflowClient().get_trace(trace_id)
    model_provider, model_name = _parse_model_uri(model)

    def invoke(messages: list[dict]) -> str:
        chat_messages = [
            ChatMessage(role="system", content=_CONVERSATIONAL_SYSTEM_PROMPT),
        ]
        for msg in messages:
            chat_messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        result = _invoke_litellm_and_handle_tools(
            provider=model_provider,
            model_name=model_name,
            messages=chat_messages,
            trace=trace,
            num_retries=10,
            skills=_get_skills(),
        )
        return result.response

    return invoke
