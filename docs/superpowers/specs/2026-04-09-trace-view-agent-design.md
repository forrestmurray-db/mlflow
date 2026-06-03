# Trace View Agent Design Spec

**Date:** 2026-04-09
**Status:** Approved

## Overview

Build a two-step agentic pipeline that programmatically summarizes MLflow traces and creates trace views from those summaries. The key insight is that `TraceSummary` is a standalone, useful artifact — client applications can consume it directly for dashboards, alerts, and reports. View creation is one downstream consumer.

## API Surface

### `trace.summarize(model="openai:/gpt-4o") -> TraceSummary`

Runs an agentic tool-calling loop that explores a trace and returns a structured summary with identified milestones. This is the primary value — a programmatic summary of an agent trajectory.

### `summary.create_view(model=None, name=None) -> TraceView`

Runs a second agentic tool-calling loop that receives the full `TraceSummary` as context, explores the trace to map milestones to concrete span ranges, and creates a `TraceView` via a `CreateTraceViewTool`.

### Usage

```python
# Single trace
summary = trace.summarize(model="openai:/gpt-4o")
print(summary.summary)      # "The agent processed a customer order by..."
print(summary.milestones)   # [Milestone(label="Planning", description="..."), ...]

# Create a view from the summary
view = summary.create_view()

# Parallelized across an experiment
from concurrent.futures import ThreadPoolExecutor

traces = mlflow.search_traces(experiment_ids=["1"])
with ThreadPoolExecutor() as pool:
    summaries = list(pool.map(lambda t: t.summarize(model="openai:/gpt-4o"), traces))

# Batch create views
views = [s.create_view() for s in summaries]
```

## Data Model

### `TraceSummary`

Location: `mlflow/entities/trace_summary.py`

```python
@dataclass
class Milestone:
    label: str          # e.g. "Planning & Template Lookup"
    description: str    # What happened and why it matters

@dataclass
class TraceSummary:
    trace_id: str
    summary: str                   # Executive summary of the agent trajectory
    milestones: list[Milestone]    # Key steps/phases identified
    model: str                     # Model used to generate this summary

    def create_view(self, model: str | None = None, name: str | None = None) -> TraceView:
        """Second agentic loop: maps milestones to span ranges and creates a view.
        
        model defaults to self.model if not provided.
        name defaults to "Summary: {first 50 chars of summary}" if not provided.
        """
```

`Milestone` is intentionally simple — natural language label + description, no span IDs. The mapping from milestones to spans is deferred to the `create_view` step where the LLM has tool access.

## Architecture

### Step 1: `summarize_trace(trace, model, skills)` 

Location: `mlflow/genai/agents/trace_view_agent.py`

Uses `get_chat_completions_with_structured_output` from the existing judge infrastructure:

1. Build system message instructing the LLM to analyze the trace and identify key milestones
2. Pass the trace object so the LLM gets access to trace exploration tools (list_spans, get_span, get_root_span, search_trace_regex, get_trace_info, get_span_performance_and_timing_report)
3. Pass a `SkillSet` with the `analyze-mlflow-trace` skill so the LLM can read domain knowledge via `ReadSkillTool` / `ReadSkillFileTool`
4. LLM explores the trace via tool calls, then returns structured output matching a pydantic `TraceSummarySchema`
5. Convert to `TraceSummary` dataclass and return

### Step 2: `create_view_from_summary(trace, summary, model)`

Location: `mlflow/genai/agents/trace_view_agent.py`

Uses `_invoke_litellm_and_handle_tools` from the existing judge infrastructure:

1. Build system message that includes the full serialized `TraceSummary` as context
2. Instruct the LLM to explore the trace and create a multi-range view mapping each milestone to span ranges
3. Tools available: existing trace exploration tools + new `CreateTraceViewTool`
4. LLM explores spans, identifies the right span IDs for each milestone, calls `CreateTraceViewTool` with ranges_json
5. Return the created `TraceView`

### New Tool: `CreateTraceViewTool`

Location: `mlflow/genai/judges/tools/create_trace_view.py`

A `JudgeTool` subclass that:
- Accepts `name` (str) and `ranges_json` (str — JSON array of SpanRange objects, same format as CLI `--ranges-json`)
- Calls `TracingClient().create_trace_view(trace_id=trace.info.trace_id, name=name, ranges=parsed_ranges)`
- Returns the view ID and trace ID on success

Tool definition parameters:
- `name` (string, required): Name for the trace view
- `ranges_json` (string, required): JSON array of SpanRange objects. Each range has `from_selector` (required, with `span_id`/`span_name`/`span_type`), optional `to_selector`, `label`, `description`, `position`, `input_path`, `output_path`

This tool is NOT registered in the global `_judge_tool_registry` — it's only available in the `create_view` agentic loop. The agent function constructs a local tool list that includes the global registry tools plus this one.

### Tool Registration Strategy

The `CreateTraceViewTool` should not be globally registered since it's only relevant in the view creation context. Instead:

- `summarize_trace` uses the standard global registry tools (trace exploration + skill reading)
- `create_view_from_summary` constructs a custom tool list: global registry tools + `CreateTraceViewTool`

Both functions pass tool definitions to the LiteLLM adapter. The existing `_process_tool_calls` in `tool_calling_utils.py` resolves tools from the global registry, so `CreateTraceViewTool` will need to either:
- Be temporarily registered/unregistered (not thread-safe), OR
- Be handled by passing a custom registry or tool list to the invocation

The simplest prototype approach: register `CreateTraceViewTool` globally but the LLM only sees it when it's included in the tool definitions passed to the adapter. The registry lookup will find it regardless.

## Prompt Design

### Summarize Prompt (System Message)

```
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
```

### Create View Prompt (System Message)

```
You are an expert at creating MLflow trace views that highlight key parts of
an agent trajectory.

You have been given a TraceSummary that describes the key milestones of this
trace. Your job is to explore the trace and create a multi-range trace view
where each range corresponds to a milestone from the summary.

TraceSummary:
{serialized_summary}

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
```

## Files to Create/Modify

### New Files

1. **`mlflow/entities/trace_summary.py`** — `TraceSummary` and `Milestone` dataclasses
2. **`mlflow/genai/agents/__init__.py`** — package init
3. **`mlflow/genai/agents/trace_view_agent.py`** — `summarize_trace()` and `create_view_from_summary()` functions
4. **`mlflow/genai/judges/tools/create_trace_view.py`** — `CreateTraceViewTool` JudgeTool subclass

### Modified Files

5. **`mlflow/entities/trace.py`** — Update `Trace.summarize()` to call `summarize_trace()` from the new agent module
6. **`mlflow/genai/judges/tools/registry.py`** — Register `CreateTraceViewTool` globally

## Parallelization

`trace.summarize()` and `summary.create_view()` are both stateless per-trace. Callers parallelize using `ThreadPoolExecutor` or `asyncio`. No internal parallelization needed.

## Prototype Scope

### In Scope
- `TraceSummary` and `Milestone` data model
- `trace.summarize(model=...)` agentic loop
- `summary.create_view(model=..., name=...)` agentic loop
- `CreateTraceViewTool`
- Skill integration (analyze-mlflow-trace)

### Out of Scope
- Experiment-level orchestration API
- View editing/iteration
- Custom prompt customization
- Streaming/progress callbacks
- Caching of summaries
