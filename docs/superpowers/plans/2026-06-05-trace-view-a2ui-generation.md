# Trace View a2ui Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a renderable a2ui trace-view document from a real trace — replacing the hardcoded `buildSampleSpec()` — via an LLM-emitted flat intent that a deterministic Python compiler expands into the `{root, components}` document the existing `A2UIViewer` already renders.

**Architecture:** Per [ADR 0001](../../adr/0001-flat-intent-a2ui-compiler.md): the LLM emits a flat, catalog-constrained `_ViewIntent` via constrained structured output; a pure function `expand_to_a2ui` compiles it into the verbose a2ui document; `generate_view_spec` orchestrates (reuse `summarize_trace` → one structured-output call → compile). A localhost-gated `POST /trace-analysis/view-spec` endpoint serves the document; the frontend fetches it on a Generate/Regenerate action.

**Tech Stack:** Python (Pydantic structured output, FastAPI), React/TypeScript (`@a2ui/react`, React Query), pytest, Jest.

**Out of scope (this slice):** persistence, REST entity changes, the `propose_trace_view`/`update_trace_view` tools, the editor. (ADR §Decision.)

---

## File Structure

**Backend**
- `mlflow/genai/agents/trace_view_agent.py` (MODIFY) — add `_ElementIntent`, `_ViewIntent` schemas, `expand_to_a2ui()` compiler, `_GENERATE_VIEW_SPEC_SYSTEM_PROMPT`, and `generate_view_spec()`. These live alongside the existing `summarize_trace()` they reuse.
- `tests/genai/agents/test_trace_view_agent.py` (CREATE) — unit tests for the compiler (pure, no mocks) and the generator (mocked LLM calls).
- `mlflow/server/assistant/api.py` (MODIFY) — add `ViewSpecRequest`/`ViewSpecResponse` models and the `POST /trace-analysis/view-spec` route, mirroring the existing `trace-analysis/message` route.
- `tests/server/assistant/test_api.py` (MODIFY) — endpoint tests using the existing `client` fixture, patching `generate_view_spec`.

**Frontend**
- `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.ts` (CREATE) — `ViewSpec` type + `fetchTraceViewSpec()` calling the endpoint via the existing `fetchAPI`/`getAjaxUrl` helpers.
- `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.test.ts` (CREATE) — Jest test mocking `fetchAPI`.
- `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/TraceViewA2UIPrototype.tsx` (MODIFY) — add a Generate/Regenerate action (React Query `useMutation`) that fetches the spec and renders it; fall back to `buildSampleSpec()` until first generation.

### The a2ui wire shape (the compiler's output contract)

`A2UIViewer` consumes `{ root: string, components: ComponentInstance[] }`. Each component is `{ id, component: { <CatalogType>: <props> } }`. The compiler MUST produce exactly the shape the existing catalog renders. Reference: `a2ui/buildSampleSpec.ts` and the catalog components.

- Root: `{ "id": "col-root", "component": { "Column": { "children": { "explicitList": [<child ids>] } } } }`
- `Text`: `{ "id": <id>, "component": { "Text": { "text": { "literal": <str> }, "usageHint": "h3" } } }`
- `GenAISpanDetail` (props read by `GenAISpanDetail.tsx`: `selector`, `title`, optional `inputPath`/`outputPath`): `{ "id": <id>, "component": { "GenAISpanDetail": { "selector": { "span_type": <str> }, "title": <str> } } }`
- `FeedbackThumbs` (props read by `FeedbackThumbs.tsx`: `name`, `target`, optional `label`/`spanSelector`): `{ "id": <id>, "component": { "FeedbackThumbs": { "name": <str>, "target": "trace", "label": <str> } } }`

---

## Task 1: Flat intent schemas (`_ElementIntent`, `_ViewIntent`)

**Files:**
- Modify: `mlflow/genai/agents/trace_view_agent.py` (add after the existing `_TraceSummarySchema`, ~line 71)
- Test: `tests/genai/agents/test_trace_view_agent.py`

- [ ] **Step 1: Write the failing test**

Create `tests/genai/agents/test_trace_view_agent.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py -v`
Expected: FAIL with `ImportError: cannot import name '_ElementIntent'`

- [ ] **Step 3: Write minimal implementation**

In `mlflow/genai/agents/trace_view_agent.py`, add `from typing import Literal` to the imports at the top (after `from pathlib import Path`), then add after `_TraceSummarySchema` (line 71):

```python
class _ElementIntent(pydantic.BaseModel):
    kind: Literal["text", "span_detail", "feedback_thumbs"] = pydantic.Field(
        description="Which catalog component this element renders as"
    )
    text: str | None = pydantic.Field(
        default=None, description="Header/prose text. Required when kind='text'."
    )
    span_type: str | None = pydantic.Field(
        default=None,
        description=(
            "OTel-GenAI span type shorthand to bind to, e.g. 'LLM', 'TOOL', "
            "'RETRIEVER', 'AGENT'. Required when kind='span_detail'."
        ),
    )
    title: str | None = pydantic.Field(
        default=None, description="Card title for kind='span_detail'."
    )
    feedback_name: str | None = pydantic.Field(
        default=None,
        description="Feedback assessment name to record. Required when kind='feedback_thumbs'.",
    )
    label: str | None = pydantic.Field(
        default=None, description="Human-facing label for kind='feedback_thumbs'."
    )


class _ViewIntent(pydantic.BaseModel):
    name: str = pydantic.Field(description="Short descriptive name for this trace view")
    elements: list[_ElementIntent] = pydantic.Field(
        description="Ordered, flat list of view elements rendered top to bottom"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add mlflow/genai/agents/trace_view_agent.py tests/genai/agents/test_trace_view_agent.py
git commit -s -m "feat(trace-views): add flat a2ui view-intent schemas

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Deterministic `expand_to_a2ui` compiler

**Files:**
- Modify: `mlflow/genai/agents/trace_view_agent.py` (add after `_ViewIntent`)
- Test: `tests/genai/agents/test_trace_view_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/genai/agents/test_trace_view_agent.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py -v`
Expected: FAIL with `ImportError: cannot import name 'expand_to_a2ui'`

- [ ] **Step 3: Write minimal implementation**

In `mlflow/genai/agents/trace_view_agent.py`, add after `_ViewIntent`:

```python
_ROOT_ID = "col-root"
_ID_PREFIX = {"text": "text", "span_detail": "detail", "feedback_thumbs": "feedback"}


def _compile_element(element: _ElementIntent, element_id: str) -> dict:
    match element.kind:
        case "text":
            return {
                "id": element_id,
                "component": {"Text": {"text": {"literal": element.text}, "usageHint": "h3"}},
            }
        case "span_detail":
            props = {"selector": {"span_type": element.span_type}}
            if element.title is not None:
                props["title"] = element.title
            return {"id": element_id, "component": {"GenAISpanDetail": props}}
        case "feedback_thumbs":
            props = {"name": element.feedback_name, "target": "trace"}
            if element.label is not None:
                props["label"] = element.label
            return {"id": element_id, "component": {"FeedbackThumbs": props}}


def expand_to_a2ui(intent: _ViewIntent) -> dict:
    """Compile a flat view intent into an a2ui document ({root, components}).

    Pure function: mints deterministic element ids, builds the root Column whose
    explicitList references each child in order, and desugars the flat element
    fields into catalog component props. See ADR 0001.
    """
    children = []
    components = []
    for index, element in enumerate(intent.elements):
        element_id = f"{_ID_PREFIX[element.kind]}-{index}"
        children.append(element_id)
        components.append(_compile_element(element, element_id))

    root = {
        "id": _ROOT_ID,
        "component": {"Column": {"children": {"explicitList": children}}},
    }
    return {"root": _ROOT_ID, "components": [root, *components]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py -v`
Expected: PASS (all compiler tests green)

- [ ] **Step 5: Commit**

```bash
git add mlflow/genai/agents/trace_view_agent.py tests/genai/agents/test_trace_view_agent.py
git commit -s -m "feat(trace-views): add deterministic expand_to_a2ui compiler

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: `generate_view_spec` generator

**Files:**
- Modify: `mlflow/genai/agents/trace_view_agent.py` (add after `expand_to_a2ui`)
- Test: `tests/genai/agents/test_trace_view_agent.py`

This reuses the existing `summarize_trace()` milestone stage, then makes one structured-output call (mirroring `summarize_trace`'s own use of `get_chat_completions_with_structured_output`) producing a `_ViewIntent`, then compiles it. Returns `{"name", "root", "components"}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/genai/agents/test_trace_view_agent.py`:

```python
from unittest import mock

from mlflow.entities.trace_summary import Milestone, TraceSummary


def test_generate_view_spec_summarizes_then_compiles_intent():
    fake_trace = mock.MagicMock()
    fake_summary = TraceSummary(
        trace_id="tr-1",
        summary="An agent answered a question.",
        milestones=[Milestone(label="Planning", description="planned")],
        model="openai:/gpt-4o",
    )
    fake_intent = _ViewIntent(
        name="Generated view",
        elements=[{"kind": "text", "text": "Overview"}],
    )

    with (
        mock.patch(
            "mlflow.genai.agents.trace_view_agent.MlflowClient"
        ) as mock_client_cls,
        mock.patch(
            "mlflow.genai.agents.trace_view_agent.summarize_trace",
            return_value=fake_summary,
        ) as mock_summarize,
        mock.patch(
            "mlflow.genai.judges.utils.invocation_utils.get_chat_completions_with_structured_output",
            return_value=fake_intent,
        ) as mock_structured,
    ):
        mock_client_cls.return_value.get_trace.return_value = fake_trace
        doc = generate_view_spec("tr-1", model="openai:/gpt-4o")

    mock_client_cls.return_value.get_trace.assert_called_once_with("tr-1")
    mock_summarize.assert_called_once_with(fake_trace, "openai:/gpt-4o")
    mock_structured.assert_called_once()
    # output schema passed to the structured call is the view intent
    assert mock_structured.call_args.kwargs["output_schema"] is _ViewIntent
    assert doc["name"] == "Generated view"
    assert doc["root"] == "col-root"
    assert any("Text" in c["component"] for c in doc["components"])
```

Add the import for `generate_view_spec` to the existing import line at the top of the test file:

```python
from mlflow.genai.agents.trace_view_agent import (
    _ElementIntent,
    _ViewIntent,
    expand_to_a2ui,
    generate_view_spec,
)
```

(Remove the now-redundant per-test imports of `_ElementIntent`/`_ViewIntent`/`expand_to_a2ui` if you prefer a single import block — not required for correctness.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py::test_generate_view_spec_summarizes_then_compiles_intent -v`
Expected: FAIL with `ImportError: cannot import name 'generate_view_spec'`

- [ ] **Step 3: Write minimal implementation**

In `mlflow/genai/agents/trace_view_agent.py`, add a prompt constant near the other prompts (after `_CREATE_VIEW_SYSTEM_PROMPT`, ~line 58):

```python
_GENERATE_VIEW_SPEC_SYSTEM_PROMPT = """\
You are an expert at composing MLflow trace views from a small UI catalog.

You have a TraceSummary describing this trace's key milestones. Design a concise,
review-focused view as an ordered, flat list of elements. Each element has a
'kind':

- text: a header or short prose line (set 'text').
- span_detail: shows one bound span's input/output. Set 'span_type' to the
  OTel-GenAI span type to bind to ('LLM', 'TOOL', 'RETRIEVER', 'AGENT', ...) and
  an optional 'title'.
- feedback_thumbs: a thumbs up/down feedback widget at the trace level. Set
  'feedback_name' (the assessment name) and an optional 'label'.

Lead with a short text header, surface the spans that matter for the milestones,
and end with at least one feedback_thumbs so reviewers can label the trace.
Do not invent element ids or layout containers — only emit the flat element list.

TraceSummary:
{summary_json}
"""
```

Then add the generator after `expand_to_a2ui`:

```python
def generate_view_spec(trace_id: str, model: str = "openai:/gpt-4o") -> dict:
    """Generate an a2ui trace-view document for a trace.

    Reuses summarize_trace() for milestones, makes one structured-output call to
    produce a flat _ViewIntent, then compiles it with expand_to_a2ui(). Returns
    {"name", "root", "components"}. See ADR 0001.
    """
    from mlflow.genai.judges.utils.invocation_utils import (
        get_chat_completions_with_structured_output,
    )
    from mlflow.tracking import MlflowClient
    from mlflow.types.llm import ChatMessage

    trace = MlflowClient().get_trace(trace_id)
    summary = summarize_trace(trace, model)

    summary_json = json.dumps(summary.to_dict(), indent=2)
    system_msg = _GENERATE_VIEW_SPEC_SYSTEM_PROMPT.format(summary_json=summary_json)

    messages = [
        ChatMessage(role="system", content=system_msg),
        ChatMessage(
            role="user",
            content="Design the trace view as a flat list of catalog elements.",
        ),
    ]

    intent = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_ViewIntent,
        trace=trace,
        skills=_get_skills(),
    )

    return {"name": intent.name, **expand_to_a2ui(intent)}
```

Note: `MlflowClient` is referenced by the test's patch target `mlflow.genai.agents.trace_view_agent.MlflowClient`. The import above is local to the function (matching the file's existing lazy-import style), which still makes the name patchable at module scope only if imported at module load. To keep the patch target valid, add a module-level import instead — at the top of the file add:

```python
from mlflow.tracking import MlflowClient
```

and remove the `from mlflow.tracking import MlflowClient` line from inside `generate_view_spec`. (The other functions import it lazily; module-level here is required so `mock.patch("...trace_view_agent.MlflowClient")` resolves.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py -v`
Expected: PASS (all tests in the file green)

- [ ] **Step 5: Commit**

```bash
git add mlflow/genai/agents/trace_view_agent.py tests/genai/agents/test_trace_view_agent.py
git commit -s -m "feat(trace-views): add generate_view_spec generator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: `POST /trace-analysis/view-spec` endpoint

**Files:**
- Modify: `mlflow/server/assistant/api.py` (add after the `trace_analysis_message` route, ~line 201)
- Test: `tests/server/assistant/test_api.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/server/assistant/test_api.py`:

```python
def test_view_spec_returns_generated_document(client):
    fake_doc = {
        "name": "Generated view",
        "root": "col-root",
        "components": [
            {"id": "col-root", "component": {"Column": {"children": {"explicitList": ["text-0"]}}}},
            {"id": "text-0", "component": {"Text": {"text": {"literal": "Hi"}, "usageHint": "h3"}}},
        ],
    }
    with patch(
        "mlflow.genai.agents.trace_view_agent.generate_view_spec",
        return_value=fake_doc,
    ) as mock_generate:
        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec",
            json={"trace_id": "tr-123"},
        )

    assert response.status_code == 200
    assert response.json() == fake_doc
    mock_generate.assert_called_once_with("tr-123", "openai:/gpt-4o")


def test_view_spec_requires_trace_id(client):
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec",
        json={},
    )
    assert response.status_code == 422


def test_view_spec_returns_500_on_generation_error(client):
    with patch(
        "mlflow.genai.agents.trace_view_agent.generate_view_spec",
        side_effect=RuntimeError("boom"),
    ) as mock_generate:
        response = client.post(
            "/ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec",
            json={"trace_id": "tr-123"},
        )

    assert response.status_code == 500
    assert "boom" in response.json()["detail"]
    mock_generate.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/server/assistant/test_api.py::test_view_spec_returns_generated_document -v`
Expected: FAIL with status 404 (route not yet registered)

- [ ] **Step 3: Write minimal implementation**

In `mlflow/server/assistant/api.py`, add after the `trace_analysis_message` route (after line 201):

```python
class ViewSpecRequest(BaseModel):
    trace_id: str
    model: str = "openai:/gpt-4o"


class ViewSpecResponse(BaseModel):
    name: str
    root: str
    components: list[dict[str, Any]]


@assistant_router.post("/trace-analysis/view-spec")
async def trace_analysis_view_spec(request: ViewSpecRequest) -> ViewSpecResponse:
    from mlflow.genai.agents.trace_view_agent import generate_view_spec

    loop = asyncio.get_event_loop()
    try:
        doc = await loop.run_in_executor(
            None, lambda: generate_view_spec(request.trace_id, request.model)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ViewSpecResponse(**doc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/server/assistant/test_api.py -k view_spec -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add mlflow/server/assistant/api.py tests/server/assistant/test_api.py
git commit -s -m "feat(trace-views): add /trace-analysis/view-spec endpoint

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Frontend `fetchTraceViewSpec` service

**Files:**
- Create: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.ts`
- Test: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.test.ts`

All `yarn` commands run from `mlflow/server/js`.

- [ ] **Step 1: Write the failing test**

Create `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.test.ts`:

```typescript
import { fetchTraceViewSpec } from './fetchViewSpec';
import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';

jest.mock('../ModelTraceExplorer.request.utils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: jest.fn((url: string) => url),
}));

describe('fetchTraceViewSpec', () => {
  it('POSTs trace_id and model to the view-spec endpoint and returns the spec', async () => {
    const spec = { name: 'V', root: 'col-root', components: [] };
    (fetchAPI as jest.Mock).mockResolvedValue(spec);

    const result = await fetchTraceViewSpec({ traceId: 'tr-1', model: 'openai:/gpt-4o' });

    expect(getAjaxUrl).toHaveBeenCalledWith('ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec');
    expect(fetchAPI).toHaveBeenCalledWith(
      'ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec',
      'POST',
      { trace_id: 'tr-1', model: 'openai:/gpt-4o' },
    );
    expect(result).toBe(spec);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pushd mlflow/server/js && yarn test fetchViewSpec; popd`
Expected: FAIL — cannot find module `./fetchViewSpec`

- [ ] **Step 3: Write minimal implementation**

Create `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.ts`:

```typescript
import type { ComponentInstance } from '@a2ui/react';

import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';

export interface ViewSpec {
  name: string;
  root: string;
  components: ComponentInstance[];
}

const VIEW_SPEC_URL = 'ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec';

export const fetchTraceViewSpec = ({
  traceId,
  model,
}: {
  traceId: string;
  model?: string;
}): Promise<ViewSpec> =>
  fetchAPI(getAjaxUrl(VIEW_SPEC_URL), 'POST', { trace_id: traceId, model });
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pushd mlflow/server/js && yarn test fetchViewSpec; popd`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.ts \
        mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/fetchViewSpec.test.ts
git commit -s -m "feat(trace-views): add fetchTraceViewSpec service

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Wire Generate/Regenerate into the prototype

**Files:**
- Modify: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/TraceViewA2UIPrototype.tsx`

Replace the static `buildSampleSpec()` render with a fetched spec triggered by a button. Use `useMutation` from the same query-client the codebase uses (`useCreateAssessment.tsx` imports it from `'../../query-client/queryClient'`; from the `a2ui/` folder that path is `'../../query-client/queryClient'` — verify by matching `useCreateAssessment.tsx`'s relative depth, which is one folder shallower, so from `a2ui/` use `'../../query-client/queryClient'`). Fall back to `buildSampleSpec()` until the first successful generation.

- [ ] **Step 1: Add imports and mutation state**

Edit the import block at the top of `TraceViewA2UIPrototype.tsx`. Add `useState` to the React import and add the new imports:

```typescript
import { useCallback, useMemo, useState } from 'react';

import { A2UIViewer, litTheme } from '@a2ui/react';
import { Button, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { useMutation } from '../../query-client/queryClient';
import type { CreateAssessmentPayload } from '../api';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { useCreateAssessment } from '../hooks/useCreateAssessment';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { buildSampleSpec } from './buildSampleSpec';
import { registerMlflowCatalog } from './catalog/registerMlflowCatalog';
import { FeedbackActionProvider } from './FeedbackActionContext';
import type { SubmitFeedbackArgs } from './FeedbackActionContext';
import { fetchTraceViewSpec } from './fetchViewSpec';
import type { ViewSpec } from './fetchViewSpec';
import { TraceDataProvider } from './TraceDataContext';
```

If `yarn type-check` (final step) reports the `queryClient` path is wrong, correct it to match the path `useCreateAssessment.tsx` actually resolves to from this folder.

- [ ] **Step 2: Replace the `sampleSpec` memo with fetched-spec state**

Find:

```typescript
  const sampleSpec = useMemo(() => buildSampleSpec(), []);
```

Replace with:

```typescript
  const [generatedSpec, setGeneratedSpec] = useState<ViewSpec | null>(null);

  const { mutate: generate, isLoading: isGenerating } = useMutation({
    mutationFn: () => fetchTraceViewSpec({ traceId }),
    onSuccess: (spec: ViewSpec) => setGeneratedSpec(spec),
  });

  const fallbackSpec = useMemo(() => buildSampleSpec(), []);
  const activeSpec = generatedSpec ?? fallbackSpec;
```

- [ ] **Step 3: Replace the `A2UIViewer` usage to add the action bar and use `activeSpec`**

Find the returned JSX block from `<div css={{ padding: theme.spacing.lg, ... }}>` through the closing `</div>` that wraps `<A2UIViewer .../>`. Replace that inner `<div>...</div>` (the one inside `FeedbackActionProvider`) with:

```typescript
        <div
          css={{
            padding: theme.spacing.lg,
            overflow: 'auto',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Button
              componentId="a2ui-prototype.generate-view"
              type="primary"
              loading={isGenerating}
              disabled={!traceId || isGenerating}
              onClick={() => generate()}
            >
              {generatedSpec ? 'Regenerate view' : 'Generate view'}
            </Button>
            {isGenerating && <Spinner size="small" />}
            <Typography.Text color="secondary" size="sm">
              {generatedSpec ? generatedSpec.name : 'Showing sample template'}
            </Typography.Text>
          </div>
          <A2UIViewer
            root={activeSpec.root}
            components={activeSpec.components}
            data={{}}
            theme={litTheme}
            onAction={(action) => {
              // a2ui's own action plumbing is not used in this prototype; feedback
              // routes through React context. Logged here so we can see what a2ui
              // emits when interacting with built-in components.
              // eslint-disable-next-line no-console
              console.debug('[a2ui prototype] action', action);
            }}
          />
        </div>
```

- [ ] **Step 4: Lint, format, type-check**

Run from `mlflow/server/js`:

```bash
pushd mlflow/server/js
yarn lint --fix src/shared/web-shared/model-trace-explorer/a2ui
yarn prettier:fix
yarn type-check
popd
```

Expected: no errors. If `type-check` flags the `'../../query-client/queryClient'` import path or `Spinner`/`Button` props, correct against the design-system types and re-run.

- [ ] **Step 5: Run the a2ui frontend tests**

Run: `pushd mlflow/server/js && yarn test src/shared/web-shared/model-trace-explorer/a2ui; popd`
Expected: PASS (existing a2ui tests + `fetchViewSpec` test green)

- [ ] **Step 6: Commit**

```bash
git add mlflow/server/js/src/shared/web-shared/model-trace-explorer/a2ui/TraceViewA2UIPrototype.tsx
git commit -s -m "feat(trace-views): fetch generated a2ui spec on Generate action

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: End-to-end manual verification

**Files:** none (verification only). REQUIRED SUB-SKILL: superpowers:verification-before-completion.

- [ ] **Step 1: Run the full backend test suite for the touched modules**

```bash
uv run --no-sync pytest tests/genai/agents/test_trace_view_agent.py tests/server/assistant/test_api.py -v
```
Expected: all PASS. Paste the summary line as evidence.

- [ ] **Step 2: Start the dev server**

```bash
LOG=$(mktemp) && echo "Logs: $LOG"
uv run dev/run_dev_server.py > "$LOG" 2>&1 &
tail -f "$LOG"
```
Wait for the frontend + backend URLs to print.

- [ ] **Step 3: Open a trace with the a2ui gate and generate**

Open a trace in the UI with `?a2ui=1` appended (see `FeatureUtils.shouldEnableA2UITraceViews`). Confirm: the sample template renders, the "Generate view" button is enabled, clicking it shows the spinner, and on success the view re-renders from the generated spec (button label becomes "Regenerate view" and the spec name shows next to it). Use a trace id that exists in the running server's backend.

- [ ] **Step 4: Confirm the endpoint directly (optional sanity check)**

```bash
curl -s -X POST http://127.0.0.1:5000/ajax-api/3.0/mlflow/assistant/trace-analysis/view-spec \
  -H 'Content-Type: application/json' \
  -d '{"trace_id":"<a real trace id>"}' | python -m json.tool
```
Expected: a JSON document with `name`, `root: "col-root"`, and a `components` list whose first entry is the `Column` root. (Adjust host/port to the dev server's printed backend URL.)

- [ ] **Step 5: Final commit / status**

Report results honestly (tests passing, manual generation working). If any step failed, capture the output and debug with superpowers:systematic-debugging before claiming completion.

---

## Self-Review Notes

- **Spec coverage (ADR §Decision):** intent schema → Task 1; compiler → Task 2; generator reusing `summarize_trace` → Task 3; localhost-gated endpoint → Task 4 (inherits `Depends(_require_localhost)` from `assistant_router`); frontend fetch on Generate/Regenerate replacing `buildSampleSpec()` → Tasks 5–6. Out-of-scope items (persistence, REST entity, propose/update tools, editor) are intentionally untouched.
- **Type consistency:** `expand_to_a2ui(intent) -> {root, components}` (Task 2) is consumed by `generate_view_spec` which adds `name` (Task 3); endpoint `ViewSpecResponse` mirrors `{name, root, components}` (Task 4); frontend `ViewSpec` mirrors the same three fields (Task 5) and is consumed in Task 6. Wire-shape props (`Text.usageHint`, `GenAISpanDetail.selector.span_type`/`title`, `FeedbackThumbs.name`/`target`/`label`) match the catalog components verified in `GenAISpanDetail.tsx`/`FeedbackThumbs.tsx`/`buildSampleSpec.ts`.
- **Patch-target caveat (Task 3):** `MlflowClient` must be a module-level import in `trace_view_agent.py` for the test's `mock.patch` to resolve — called out explicitly in the task.
