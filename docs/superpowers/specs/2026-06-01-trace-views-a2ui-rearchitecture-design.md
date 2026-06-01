# Trace Views — a2ui Rearchitecture Design

**Date:** 2026-06-01
**Status:** Draft (supersedes the `SpanRange`-based design in `docs/rfcs/0004-trace-views/0004-trace-views.md`)
**Owner:** Forrest Murray

## Overview

A `TraceView` is an **a2ui document** (json-render compatible) describing a composable, agent-generatable, human-editable review surface on top of an OTel-GenAI–shaped trace. The surface composes a small set of typed components from a published catalog. Two of those components — feedback widgets — emit MLflow `Feedback` records, making labeling a first-class concern of the view rather than a separate UI surface.

The persisted schema, catalog, and event model are aligned to existing open protocols so the same view document can be rendered by any a2ui-compatible client against any OTel-GenAI trace. MLflow ships the first renderer and the first GenAI catalog extension.

## Motivation

The original RFC modeled a view as an ordered `SpanRange[]`, each rendered as a uniform "range card" with extracted I/O. Two limits drove this rearchitecture:

1. **Render model too rigid.** Range cards are one fixed layout. Real review experiences want tables of tool calls, scoring rubrics, raw JSON, mixed prose + data — the agent and the developer should be able to pick the right shape per range.
2. **Feedback isn't first-class.** Labeling/scoring lives outside the view today, bolted onto the trace explorer toolbar. Widgets that bake in their own feedback affordances make a view *also* a labeling surface.

A bespoke widget DSL was rejected; aligning to a2ui and json-render gives ecosystem leverage that an internal DSL cannot. The strategic posture: be the first evals + observability provider to expose a generative-UI API for traces.

## Architecture

```
TraceView
  view_id, name, trace_id | experiment_id, created_by
  spec: A2UIDocument
        ├── root: str                       # element id
        └── elements: dict[str, Element]    # flat dict, tree via children refs
                ├── type: str               # catalog component type
                ├── props: dict             # validated per type by catalog schema
                └── children: list[str] | None
```

`SpanSelector` and JSONPath survive as **typed prop values** inside `props`; they are no longer top-level entities. The flat `elements` dictionary keyed by id is a2ui's "flat streaming JSON" shape — composition emerges through `children` id references, not nested JSON.

## Catalog (v1)

The catalog is the contract between agent, editor, and renderer. Three buckets:

**a2ui standard, reused as-is** — `Stack`, `Row`, `Card`, `Heading`, `Text`, `Markdown`, `Code`.

**GenAI trace-data components (MLflow-contributed)** — anchored to OTel GenAI semantic conventions, not MLflow-internal shapes:

| Component | Binding | Notes |
|---|---|---|
| `GenAISpanDetail` | one span via `selector` | Optional `inputPath` / `outputPath` JSONPath; renders extracted I/O. Default-extraction helper handles common OTel `gen_ai.*` event shapes when paths are omitted. |
| `GenAISpanTable`  | many spans via `selector` | `columns: [{label, path}]` or `{label, feedback: <FeedbackComponent>}` for per-row feedback. |
| `GenAISpanRef`    | one span by id | Inline deeplink primitive for `[text](spans/{id})` in `Markdown`. |

**Feedback components (MLflow-contributed)** — high-level wrappers only in v1:

| Component | Emits |
|---|---|
| `FeedbackThumbs` | boolean `Feedback` |
| `FeedbackRubric` | one `Feedback` per criterion |
| `FeedbackText`   | string `Feedback` |

Each declares a typed `submitFeedback` action. Host wires it to MLflow's `Feedback` REST. Records carry `metadata: {view_id, element_id}` for provenance.

## OTel GenAI alignment

`SpanSelector` keys map to OTel GenAI semantic conventions:

```python
@dataclass
class SpanSelector:
    span_id: str | None
    name: str | None
    kind: SpanKind | None
    attributes: dict[str, str | int | float | bool] | None
        # e.g., {"gen_ai.operation.name": "chat"}
```

MLflow's familiar `span_type` ("LLM" / "TOOL" / "RETRIEVER") becomes shorthand that desugars to OTel attribute matches, not wire identity. JSONPath extraction operates over the OTel span shape (attributes, events, status, name) — making the catalog portable to any OTel-GenAI–shaped trace, not just MLflow's.

## Authoring

**Two paths, one API.** Both write through `POST/PATCH /views` with full-document replacement (no JSON-patch in v1).

**Agent path** — one-shot generation in v1:
- `propose_trace_view(name, spec)` — emits a complete document; server validates against the catalog and returns structured errors keyed to element ids.
- `update_trace_view(view_id, ops)` — element-level patches for iterative refinement: `AddElement`, `RemoveElement`, `UpdateProps`, `SetChildren`.
- Catalog manifest is rendered into the agent's system prompt at boot (~1.5–2k tokens). No separate `get_catalog` tool.
- The elements-by-id document shape stays streaming-ready; partial-render streaming is deferred to v2.

**Editor (WYSIWYG-lite)** — flat-only in v1:
- Sidebar with the root's children as an ordered list. `Stack`/`Row` are agent-only in v1; editor-authored views are always one root `Stack` with leaf children.
- "Add widget" menu picks from the catalog.
- Per-component prop forms generated from the catalog's typed schemas: text inputs for strings, dropdowns for enums, `SpanSelector` picker for selector props, `JsonFieldSelector` checkbox tree for path props.
- Drag to reorder, duplicate, delete.
- Cancel discards; Save persists.

**Drag-select on the timeline** (CUJ 3 from the original RFC) survives — it opens the editor with one `GenAISpanDetail` pre-populated per selected span.

## Timeline highlight derivation

The left timeline still dims non-matching spans and color-codes matching ones; derivation, not storage:

1. Each catalog component declares which props are **span-binding**.
2. Renderer walks `spec`, resolves every span-binding prop, assigns a deterministic color per element by document position.
3. Spans matched by multiple elements take the color of the first element in document order.
4. Non-span-binding widgets contribute nothing.

## Persistence and REST

Single `trace_views` table; JSON `spec` column. Routes unchanged from the original RFC; only the body shape changes (`ranges` → `spec`):

```
POST   /mlflow/traces/{trace_id}/views
GET    /mlflow/traces/{trace_id}/views
GET    /mlflow/traces/{trace_id}/views/{view_id}
PATCH  /mlflow/traces/{trace_id}/views/{view_id}
DELETE /mlflow/traces/{trace_id}/views/{view_id}

POST   /mlflow/experiments/{exp_id}/views    (+ GET / PATCH / DELETE)
```

Server validates `spec.root` exists, every `children` id resolves, every `props` block matches the catalog schema. Unknown component `type` → 400 with structured error. Catalog manifest is published with MLflow; responses include `catalog_version: "1"`. Older clients render unknown components as a placeholder.

## Migration (single PR on `impl/trace-views`)

Nothing has shipped, so this is a wholesale rewrite of the branch's view internals, not a versioned migration.

**Backend (Python):**
- Replace `ranges: list[SpanRange]` with `spec: A2UIDocument`. Add `A2UIDocument` / `A2UIElement`. Keep `SpanSelector`.
- Catalog: Pydantic models per component; JSON Schema emission for the JS manifest.
- REST validators check `spec` against the catalog.
- Agent tools: `create_trace_view` → `propose_trace_view` + `update_trace_view`.
- Default-extraction helpers for common OTel GenAI event shapes.
- Storage column: `ranges` → `spec`.

**Frontend (JS):**
- Add `json-render` as a dependency (Databricks npm proxy).
- Register the MLflow catalog with json-render's renderer.
- Implement React components per catalog type, reusing `JsonFieldSelector` / `SpanSelector` picker.
- `submitFeedback` action handler → existing MLflow Feedback REST.
- Drag-select on timeline produces `GenAISpanDetail` elements.

**Agent / prompts:**
- Catalog manifest embedded in the system prompt; remove the `SpanRange`-shaped prompts and tools.

**CLI:**
- `mlflow traces create-view --spec-json '...'` (was `--ranges-json`).

## Out of scope (v1)

- Nested-layout authoring in the editor (agent can produce nested layouts; editor stays flat).
- Chat/thread and Comparison/side-by-side widgets.
- Streaming agent generation (the document shape is streaming-ready; the renderer isn't, yet).
- Third-party catalog plugin model.
- JSON-patch endpoints; only whole-spec replacement in v1.
- Per-view permissions beyond existing experiment scope.

## Open questions

1. **Catalog versioning policy.** Semver, and what counts as a breaking change for downstream renderers — since we're publishing for ecosystem reuse, not just MLflow's own UI.
2. **OTel GenAI convention pinning.** Conventions are still evolving; pin a specific revision and document our default-extraction assumptions.
3. **Catalog package location.** v1 lives inside `mlflow.genai` (or `mlflow.trace_views`); split into a standalone package or contribute upstream to a2ui as a `gen_ai` observability extension once it stabilizes and external interest materializes.
4. **Catalog editing affordances.** Whether the editor exposes `Card`-style grouping (single-level) without exposing full nesting controls. v1 says no; revisit after first dogfooding.

## Strategic posture

The catalog is the public artifact, not MLflow's renderer. Any a2ui-compatible client can render an MLflow trace view; MLflow can render any trace view authored against the same catalog. This positions MLflow as the first evals + observability provider with a generative-UI API.
