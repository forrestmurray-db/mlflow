# Trace View UI Pathways Design

**Date:** 2026-04-10
**Status:** Draft

## Overview

Two new UI pathways for invoking trace summarization and trace view creation:

1. **Batch button** in the experiment traces toolbar — select traces, click a button, run summarize → create_view across all of them as an async job (mirrors the existing Detect Issues pattern).
2. **Interactive assistant mode** — a "Trace Analysis" mode in the assistant chat panel that connects to the trace view agent via the agent server, allowing conversational trace exploration and view creation.

## 1. Batch Trace View Creation

### Button & Toolbar

- New "Create Views" button in `GenAITracesTableToolbar` alongside "Detect Issues"
- Same visual style: gradient border, AI sparkle icon
- Works with selected traces or all traces in the experiment

### Modal (2-Step)

**Step 1 — Confirmation:**
- Shows trace count and brief explanation of what summarize + create view does
- "This will analyze N traces and create milestone views for each"

**Step 2 — Model Selection:**
- Provider, model, API key fields
- Reuse or extract shared component from `IssueDetectionModelSelection`

### Backend

**New endpoint:** `POST /ajax-api/3.0/mlflow/traces/views/invoke`

**Request:**
```json
{
  "experiment_id": "string",
  "trace_ids": ["string"],
  "model": "string",
  "provider": "string",
  "secret_id": "string (optional)",
  "endpoint_name": "string (optional)"
}
```

**Response:**
```json
{
  "job_id": "string",
  "run_id": "string"
}
```

**Handler behavior:**
- Creates an MLflow run with `MLFLOW_RUN_TYPE = "trace_view_creation"`
- Submits async job to executor
- Job iterates traces: `summarize_trace(trace, model)` → `summary.create_view()` for each
- Logs progress to the run (traces processed, views created, errors)

### Progress Component

- Adapts the `IssueDetectionProgress` component pattern
- Displays: traces processed / total, views created, error count
- Cancel button to terminate the job

## 2. Assistant Trace Analysis Mode

### Entry Point: `+` Button Menu

- Small `+` button to the left of the text input in `AssistantChatPanel`
- Opens a popover/dropdown menu with one option: **"Trace Analysis"**
- Selecting it switches the assistant into trace analysis mode

### Mode Indicator & Toggle

- When active, a "Trace Analysis" badge/chip appears on or near the input area
- Clicking the chip or the `+` menu toggles back to normal assistant mode
- Switching modes starts a fresh session (no cross-mode conversation continuity)

### Routing

- In normal mode: messages route to Claude Code via existing `AssistantService`
- In trace analysis mode: messages route to the agent server (proxied through the MLflow server)
- New proxy endpoint: `POST /ajax-api/3.0/mlflow/assistant/trace-analysis/message`
- SSE streaming uses the same `EventSource` pattern the assistant already implements

### Request Format

```json
{
  "messages": [
    {"role": "user", "content": "Summarize this trace"},
    {"role": "assistant", "content": "..."}
  ],
  "context": {
    "trace_id": "string (optional)",
    "experiment_id": "string (optional)"
  },
  "model": "string (optional)",
  "stream": true
}
```

`AssistantPageContext` provides the `trace_id` and `experiment_id` automatically.

### Message History

- Stateless server, stateful client — full message history sent with each request
- Matches the existing assistant communication pattern

## 3. Agent Server Integration

### Trace View Agent as Conversational Agent

The existing trace view agent (`mlflow/genai/agents/trace_view_agent.py`) currently has hardcoded system prompts and a one-shot tool loop. It needs to be wrapped as a conversational agent served by the agent server:

- Registered on the agent server, accepting user messages and streaming responses
- System prompt includes the analyze-mlflow-trace skill context and trace exploration instructions
- Multi-turn: the agent maintains conversation context via message history passed from the client

### Tool Set

The agent has four tools available:

| Tool | Purpose |
|------|---------|
| `list_spans` | Explore trace structure — returns span tree with IDs, names, types |
| `get_span` | Fetch detailed span data — inputs, outputs, attributes, timing |
| `create_trace_view` | Create a new trace view with ranges |
| `update_trace_view` | Modify an existing trace view (add/remove/reorder ranges, update labels, rename) |

### Proxy Architecture

- MLflow server proxies `/ajax-api/3.0/mlflow/assistant/trace-analysis/message` to the agent server
- Keeps auth, CORS, and credential handling centralized
- Agent server streams SSE events back through the proxy

### Frontend Markers

- When the agent calls `create_trace_view` or `update_trace_view`, the response includes the `[trace_view_created: {...}]` or `[trace_view_updated: {...}]` marker
- Existing frontend regex extraction logic picks up the marker, hides it from rendered text, and calls `invalidateTraceViews(trace_id)` to refresh the view selector
- `list_spans` and `get_span` calls are internal reasoning — streamed as text, no special frontend handling

## 4. New: `update_trace_view` Tool

### Capabilities

- Add, remove, or reorder ranges within a view
- Update range labels and descriptions
- Update range selectors (from_selector, to_selector) and JSONPath extraction paths
- Rename the view

### REST API

**Endpoint:** `PATCH /ajax-api/3.0/mlflow/traces/{trace_id}/views/{view_id}`

**Request:** Partial update — only fields present are modified.

```json
{
  "name": "string (optional)",
  "ranges": [
    {
      "from_selector": {},
      "to_selector": {},
      "label": "string",
      "description": "string",
      "position": 0,
      "input_path": "string (optional)",
      "output_path": "string (optional)"
    }
  ]
}
```

### Python Client

```python
client = mlflow.MlflowClient()
client.update_trace_view(
    trace_id="tr-abc",
    view_id="tv-xyz",
    name="Updated View",
    ranges=[...],
)
```

### Agent Tool

The `update_trace_view` tool wraps the Python client method. The agent can call it during conversation to iteratively refine views based on user feedback.

## Component Reference

### Existing Components to Reuse/Extend

| Component | Location | Reuse |
|-----------|----------|-------|
| `DetectIssuesButton` | `mlflow/server/js/src/.../DetectIssuesButton.tsx` | Pattern for button style and guidance popover |
| `IssueDetectionModal` | `mlflow/server/js/src/.../IssueDetectionModal.tsx` | Pattern for 2-step modal |
| `IssueDetectionModelSelection` | `mlflow/server/js/src/.../IssueDetectionModelSelection.tsx` | Extract shared model selection component |
| `IssueDetectionProgress` | `mlflow/server/js/src/.../IssueDetectionProgress.tsx` | Pattern for progress tracking |
| `AssistantChatPanel` | `mlflow/server/js/src/assistant/AssistantChatPanel.tsx` | Add `+` button and mode switching |
| `AssistantService` | `mlflow/server/js/src/assistant/AssistantService.ts` | Add trace analysis routing |
| `AssistantContext` | `mlflow/server/js/src/assistant/AssistantContext.tsx` | Add mode state and provider switching |
| `GenAITracesTableToolbar` | `mlflow/server/js/src/.../GenAITracesTableToolbar.tsx` | Add "Create Views" button |

### New Components

| Component | Purpose |
|-----------|---------|
| `CreateViewsButton` | "Create Views" button in traces toolbar |
| `ViewCreationModal` | 2-step modal (confirmation → model selection) |
| `ViewCreationProgress` | Progress display for batch view creation job |
| `AssistantModeMenu` | `+` button popover with mode options |
| `AssistantModeBadge` | Mode indicator chip on input area |
| `TraceAnalysisService` | Service layer for routing to agent server |
