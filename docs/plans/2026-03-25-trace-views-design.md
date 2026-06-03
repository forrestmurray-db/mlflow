# Trace Views, Summaries & Analysis — Design Spec

**Author:** Forrest Murray
**Date:** 2026-03-25
**Status:** DRAFT

---

## Objective

Make it easier for humans and AI to navigate, analyze, and evaluate traces in MLflow. Traces contain a lot of information that's not necessary for understanding what happened in an agent's trajectory. This feature introduces **trace views** — named, shareable configurations that filter and extract key elements from a trace — along with natural language interfaces to summarize and analyze traces.

## Background

### Current State

- Developers must invest effort creating reviewable datasets, often going off-platform (Excel, custom apps) or writing detailed instructions for trace analysis
- All summarization and analysis are DIY and left to developers
- The MLflow assistant can already analyze traces via the `analyze-mlflow-trace` skill, but cannot update the UI or persist analysis configurations
- A prototype of span filtering + JSONPath extraction exists in an internal project (`project-0xfffff`) used for SME review workshops

### Prior Art

- **Braintrust Loop** / **Arize Alyx**: Natural language interfaces in the developer UI
- **Langsmith**: Input/output preview filters
- **MLflow Assistant**: General-purpose chat that receives trace context but has no structured view-update capability
- **MLflow Judge Infrastructure**: `invoke_judge_model()` with trace tools (`ListSpansTool`, `GetSpanTool`, etc.) for agentic trace analysis

---

## Design

### 1. TraceView Data Model

`TraceView` is a new first-class entity in the tracking store, following the same patterns as `Assessment`.

#### Entity

```python
@dataclass
class TraceView:
    name: str                              # e.g., "Tool Calls Only", "SME Review"
    trace_id: str | None = None            # Set for trace-scoped views
    experiment_id: str | None = None       # Set for experiment-scoped views (templates)
    span_filter: SpanFilter | None = None  # Which spans to show
    input_path: str | None = None          # JSONPath for input extraction
    output_path: str | None = None         # JSONPath for output extraction
    created_by: str | None = None          # Attribution ("forrest", "judge:gpt-4o")
    description: str | None = None         # What this view is for
    view_id: str | None = None             # Server-generated
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None

@dataclass
class SpanFilter:
    span_name: str | None = None           # Match by span name
    span_type: str | None = None           # Match by type (TOOL, LLM, etc.)
    attribute_key: str | None = None       # Match by attribute existence/value
    attribute_value: str | None = None     # Requires attribute_key
```

#### Scope

Views can be scoped to either a trace or an experiment:

- **Trace-scoped**: Attached to a specific trace. "This trace should always be viewed this way."
- **Experiment-scoped**: A template that applies across all traces in an experiment. "All traces from this agent should show tool calls with this JSONPath."

**Scope resolution** when rendering a trace:
1. If a specific view is selected by the user, use it
2. No view selected = show raw trace data (today's behavior)

**List endpoint behavior:** `GET /mlflow/traces/{trace_id}/views` returns both trace-scoped and experiment-scoped views. When views share the same name, both are returned with a `scope` indicator (`"trace"` or `"experiment"`). The UI groups them by scope in the dropdown and shows trace-scoped views first.

#### Database Table

```sql
CREATE TABLE trace_views (
    view_id                 VARCHAR(50) PRIMARY KEY,
    name                    VARCHAR(256) NOT NULL,
    trace_id                VARCHAR(50) REFERENCES trace_info(request_id) ON DELETE CASCADE,
    experiment_id           INTEGER REFERENCES experiments(experiment_id),
    span_filter             TEXT,        -- JSON-serialized SpanFilter
    input_path              TEXT,        -- JSONPath expression
    output_path             TEXT,        -- JSONPath expression
    created_by              VARCHAR(256),
    description             TEXT,
    created_timestamp       BIGINT,      -- milliseconds since epoch
    last_updated_timestamp  BIGINT,      -- milliseconds since epoch
    CHECK (
        (trace_id IS NOT NULL AND experiment_id IS NULL)
        OR (trace_id IS NULL AND experiment_id IS NOT NULL)
    )
);
```

**Note:** `experiment_id` is `INTEGER` to match `experiments.experiment_id` in the existing schema. `trace_id` references `trace_info.request_id` (the physical column name), following the same pattern as the `assessments` table. View IDs are server-generated UUIDs with a `tv-` prefix.

Indexes: `(trace_id, created_timestamp)`, `(experiment_id, created_timestamp)`, `last_updated_timestamp`.

---

### 2. REST API & Python Client

#### REST Endpoints

Trace-scoped:
```
POST   /mlflow/traces/{trace_id}/views              → create trace-scoped view
GET    /mlflow/traces/{trace_id}/views               → list views (trace-scoped + experiment-scoped)
GET    /mlflow/traces/{trace_id}/views/{view_id}     → get a specific view
PATCH  /mlflow/traces/{trace_id}/views/{view_id}     → update a view
DELETE /mlflow/traces/{trace_id}/views/{view_id}     → delete a view
```

Experiment-scoped:
```
POST   /mlflow/experiments/{experiment_id}/views              → create experiment-scoped view
GET    /mlflow/experiments/{experiment_id}/views               → list experiment-scoped views
PATCH  /mlflow/experiments/{experiment_id}/views/{view_id}     → update an experiment-scoped view
DELETE /mlflow/experiments/{experiment_id}/views/{view_id}     → delete an experiment-scoped view
```

#### Python API

View CRUD lives on `MlflowClient`, consistent with how assessments work. The `Trace` entity provides convenience accessors that delegate to the client.

**MlflowClient methods (primary API):**

```python
client = mlflow.MlflowClient()

client.create_trace_view(
    trace_id="tr-abc123",
    name="Tool Calls Only",
    span_filter=SpanFilter(span_type="TOOL"),
    input_path="$.query",
    output_path="$.result",
    description="Shows only tool invocations for SME review",
) -> TraceView

client.list_trace_views(trace_id="tr-abc123") -> list[TraceView]

client.get_trace_view(trace_id="tr-abc123", view_id="tv-xyz") -> TraceView

client.update_trace_view(
    trace_id="tr-abc123", view_id="tv-xyz", name="...", ...
) -> TraceView

client.delete_trace_view(trace_id="tr-abc123", view_id="tv-xyz") -> None

client.create_experiment_view(
    experiment_id="123",
    name="SME Review",
    span_filter=SpanFilter(span_type="CHAT_MODEL"),
    output_path="$.choices[0].message.content",
) -> TraceView

client.list_experiment_views(experiment_id="123") -> list[TraceView]
```

**Trace convenience methods:**

```python
# View accessors (delegate to MlflowClient)
trace.create_view(name="...", span_filter=..., ...) -> TraceView
trace.views -> list[TraceView]  # All views (trace-scoped + inherited experiment-scoped)
trace.delete_view(view_id="...") -> None

# Summarize & Analyze (judge infrastructure)
trace.summarize(
    model="openai:/gpt-4o-mini",
    view: TraceView | None = None,   # optionally scope to a view
) -> str

trace.analyze(
    question: str,
    model="openai:/gpt-4o-mini",
    view: TraceView | None = None,
) -> str
```

The `Trace` convenience methods instantiate an `MlflowClient` internally (same pattern as `search_assessments` which operates on in-memory data; `create_view` is a new pattern where the Trace entity calls back to the server). This is a pragmatic choice for API ergonomics — the canonical CRUD interface remains on `MlflowClient`.

`summarize()` and `analyze()` return plain text strings. Errors from the judge model (invalid model URI, rate limits, etc.) are raised as `MlflowException`.

#### Experiment-level Convenience

```python
import mlflow

mlflow.create_experiment_view(
    experiment_id="123",
    name="SME Review",
    span_filter=SpanFilter(span_type="CHAT_MODEL"),
    output_path="$.choices[0].message.content",
) -> TraceView
```

#### Summarize/Analyze Implementation

Both call `invoke_judge_model()` from `mlflow.genai.judges` with the existing trace tools (`ListSpansTool`, `GetSpanTool`, etc.). When a `view` is provided, the trace data is pre-filtered (span filter applied, JSONPath extracted) before being passed as context, so the LLM reasons over the focused data rather than the full trace.

---

### 3. Span Filter & JSONPath Utilities

Ported from the `project-0xfffff` prototype into MLflow as pure functions.

#### Location

`mlflow/tracing/utils/view_utils.py`

#### Span Filter

```python
def find_first_matching_span(spans: list[Span], filter_config: SpanFilter) -> Span | None:
    """Find first span matching all filter criteria (AND-combined).
    Returns the first match in span order. For traces with multiple matching
    spans (e.g., multiple TOOL spans), only the first is returned. Future work
    may add multi-match support."""

def apply_span_filter(
    trace: Trace, filter_config: SpanFilter | None
) -> tuple[str | None, str | None]:
    """Apply span filter to a trace, returning (filtered_inputs, filtered_outputs).
    Returns root span inputs/outputs if no filter or no match."""
```

Matching logic:
- `span_name`: exact match on `span.name`
- `span_type`: match on `span.span_type` or `attributes["mlflow.spanType"]`
- `attribute_key` + `attribute_value`: match on span attributes (key must be present, value optional)
- All criteria are AND-combined; first matching span wins

#### JSONPath Extraction

```python
def apply_jsonpath(data: str, jsonpath_expr: str | None) -> tuple[str | None, bool]:
    """Apply JSONPath expression to a JSON string.
    Returns (extracted_value, success). Falls back gracefully on any error."""

def validate_jsonpath(expr: str) -> tuple[bool, str | None]:
    """Validate JSONPath syntax. Returns (is_valid, error_message)."""
```

#### Pipeline

When applying a view to a trace, the two-stage pipeline is:

```
Trace
  -> apply_span_filter(trace, view.span_filter)
    -> (inputs_json, outputs_json)
      -> apply_jsonpath(inputs_json, view.input_path)
      -> apply_jsonpath(outputs_json, view.output_path)
        -> (display_input, display_output)
```

A convenience function wraps the pipeline:

```python
def apply_view(trace: Trace, view: TraceView) -> tuple[str, str]:
    """Apply a complete view (span filter + JSONPath) to a trace.
    Returns (display_input, display_output), falling back to raw data on any failure."""
```

#### Dependency

`jsonpath-ng` added as an optional dependency in `pyproject.toml`. `apply_jsonpath` gracefully returns `(None, False)` if the library is not installed.

**JSONPath library parity:** The Python backend uses `jsonpath-ng` and the JS frontend uses `jsonpath-plus`. These libraries have subtly different JSONPath dialect support. JSONPath expressions are authored once (when creating a view) and applied in both contexts. For the prototype, we restrict to the common subset (dot notation, bracket notation, array indexing, wildcard `[*]`). A parity test suite validates that the same expressions produce the same results in both libraries.

---

### 4. Assistant Integration

#### How the Assistant Creates Views

The existing `analyze-mlflow-trace` skill in `mlflow/skills` is extended with instructions for creating views. The assistant:

1. Fetches the trace (already knows how via the existing skill)
2. Analyzes the user's request ("show me just the tool calls", "focus on the LLM decision points")
3. Calls Python to create the view via the MLflow API:
   ```python
   import mlflow
   from mlflow.entities.trace_view import SpanFilter
   trace = mlflow.get_trace("$TRACE_ID")
   view = trace.create_view(
       name="Tool Calls",
       span_filter=SpanFilter(span_type="TOOL"),
       output_path="$.result",
   )
   print(f"[trace_view_created: {{\"view_id\": \"{view.view_id}\", \"trace_id\": \"{trace.info.trace_id}\"}}]")
   ```
4. Outputs a structured marker `[trace_view_created: {...}]` that the frontend detects

#### Frontend Handling

The message parser in `AssistantContext.tsx` watches for `[trace_view_created: {...}]` blocks in streamed assistant messages. When detected:

1. Parse the view ID and trace ID from the marker
2. If the user is currently viewing that trace, fetch the new view via `GET /mlflow/traces/{trace_id}/views/{view_id}`
3. Apply it to the trace viewer (set as active view)
4. The marker itself is hidden from the rendered chat message (similar to how internal skill messages are already filtered)

**Marker contract:**
- Regex: `\[trace_view_created:\s*(\{.*?\})\]`
- Payload: JSON with required fields `view_id` (string) and `trace_id` (string)
- Malformed markers (invalid JSON, missing fields) are ignored and left visible in the message as-is
- This is a purpose-built convention for trace views; if other assistant-to-UI signals are needed in the future, a generalized action marker system should be designed

#### Skill Changes

The `analyze-mlflow-trace` skill lives in the external `mlflow/skills` repository (https://github.com/mlflow/skills), which is referenced as a submodule at `mlflow/assistant/skills/` and installed at runtime via the skill installer. The SKILL.md in that repository gets a new section teaching the assistant how to create views, with examples of user requests that should result in view creation ("show me only the tool calls", "focus on the LLM spans", "filter to just the retriever inputs and outputs", "show key decision points").

#### Judge Integration

When `trace.summarize()` or `trace.analyze()` is called with a `view` parameter, the judge receives pre-filtered trace data. The judge's `SkillSet` can also include a trace-view skill via `ReadSkillTool` so it understands the concept of views and can recommend creating them in its feedback.

---

### 5. UI Changes

#### Trace Viewer — View Selector

`ModelTraceExplorerContent.tsx` gets a view selector dropdown in its header:

- **Default state**: "Raw Trace" (no view applied, today's behavior)
- **When views exist**: Dropdown lists all views for this trace (trace-scoped + experiment-scoped), grouped by scope
- **Active view badge**: When a view is active, a small badge shows the view name
- **"Clear view" action**: Returns to raw trace display

#### Trace Viewer — Filtered Display

When a view is active, `ModelTraceExplorerDetailView.tsx` changes behavior:

- **Span tree**: Non-matching spans are dimmed/collapsed (not hidden — the user should still see the full structure for orientation)
- **Right pane (inputs/outputs)**: JSONPath extraction is applied. The matched span's filtered inputs/outputs replace the default display
- **Visual indicator**: A subtle banner: "Viewing: Tool Calls Only — showing TOOL spans with output $.result" with a dismiss button

#### Trace Viewer — Applied by Assistant

When the assistant creates a view (detected via the `[trace_view_created]` marker):

- The view selector auto-switches to the new view
- A toast notification: "Assistant applied view: Tool Calls Only"
- The user can switch back to raw or another view at any time

#### No New Pages or Modals

View creation from the UI is out of scope for the prototype. Views are created via:
- The assistant ("show me only tool calls")
- Python API (`trace.create_view(...)`)

---

## Implementation Layers

### New Files

| File | Purpose |
|------|---------|
| `mlflow/entities/trace_view.py` | `TraceView` and `SpanFilter` dataclasses |
| `mlflow/protos/trace_views.proto` | Protobuf messages |
| `mlflow/tracing/utils/view_utils.py` | `apply_span_filter`, `apply_jsonpath`, `apply_view` |
| `mlflow/store/db_migrations/versions/xxx_add_trace_views_table.py` | Alembic migration |
| `mlflow/server/js/src/shared/web-shared/model-trace-explorer/TraceViewSelector.tsx` | View dropdown component |

### Modified Files

| File | Change |
|------|--------|
| `mlflow/store/tracking/dbmodels/models.py` | Add `SqlTraceView` model |
| `mlflow/store/tracking/abstract_store.py` | Add abstract CRUD methods |
| `mlflow/store/tracking/sqlalchemy_store.py` | Implement CRUD |
| `mlflow/store/tracking/sqlalchemy_workspace_store.py` | Workspace-aware overrides |
| `mlflow/store/tracking/rest_store.py` | REST proxy implementation |
| `mlflow/server/handlers.py` | Register new endpoints |
| `mlflow/entities/trace.py` | Add convenience methods (`create_view`, `views`, `summarize`, `analyze`) that delegate to `MlflowClient` |
| `mlflow/tracking/client.py` | Add canonical CRUD methods (`create_trace_view`, `list_trace_views`, `get_trace_view`, `update_trace_view`, `delete_trace_view`, `create_experiment_view`, `list_experiment_views`) |
| `mlflow/tracking/_tracking_service/client.py` | Add service-layer methods |
| `mlflow/server/js/.../ModelTraceExplorerContent.tsx` | View selector integration |
| `mlflow/server/js/.../ModelTraceExplorerDetailView.tsx` | Filtered display logic |
| `mlflow/server/js/src/assistant/AssistantContext.tsx` | Parse `[trace_view_created]` markers |
| `mlflow/skills` repo: `analyze-mlflow-trace/SKILL.md` | Add view creation instructions (external repo) |

### Dependencies

| Dependency | Where |
|------------|-------|
| `jsonpath-ng` | Optional Python dependency in `pyproject.toml` |
| `jsonpath-plus` | Added to `mlflow/server/js/package.json` |

---

## Testing

- **Unit tests** for `view_utils.py`: span filter matching, JSONPath extraction, pipeline, edge cases (ported from project-0xfffff test suites)
- **Store tests** for CRUD: `tests/store/tracking/test_sqlalchemy_store.py` — create, get, list, update, delete for both scopes
- **Workspace-aware tests**: `tests/store/tracking/test_sqlalchemy_store_workspace.py` — workspace variants per CLAUDE.md guidance
- **REST endpoint tests**: handler tests for all view endpoints
- **JSONPath parity tests**: same expressions validated against both `jsonpath-ng` (Python) and `jsonpath-plus` (JS)
- **Frontend component tests**: TraceViewSelector rendering, view application, assistant marker parsing

---

## Out of Scope (Future Work)

- UI form for creating/editing views manually
- View permissions / role-based access control
- View versioning or history
- Recursive span filter nesting (mentioned in Heilmeier doc as open question — would allow filtering nested spans within a matched parent)
- `trace.update_view(prompt="...")` — natural language view creation via Python API (would call judge to generate the SpanFilter from a prompt)
- Per-view custom separators for multi-match JSONPath results (always uses newline)
- Export with view applied (exports use raw data)
