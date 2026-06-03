start_date: 2026-04-22
mlflow_issue: TODO (link MLflow enhancement issue)
rfc_pr:

# Summary

Add **Trace Views** to MLflow Tracing so users can review long traces through a saved, reusable view instead of repeatedly scanning raw span trees. A trace view captures simple filters (for example, "tool calls only") and focused input/output extraction, and can be applied from the trace UI or by the assistant.

# Basic example

A reviewer opens a trace and switches from `Raw Trace` to `Tool Calls Only`:

1. The timeline keeps full structure for orientation, but non-matching spans are visually de-emphasized.
2. The details panel shows only the extracted fields the reviewer cares about (for example, tool input query and tool output result).
3. A banner clearly shows the active view and provides one-click `Clear`.
4. The same view can be reused for other traces in the experiment when saved as an experiment-scoped template.

## Motivation

Trace review is currently high-friction for both humans and AI-assisted workflows:

- Important signals are buried in large, nested traces.
- Teams recreate ad-hoc review logic outside MLflow (spreadsheets, notes, manual instructions).
- Assistant guidance helps interpret traces, but cannot persist a focused review lens in the UI.

Trace Views reduce repeated cognitive load and make review workflows consistent across teams.

### Out of scope

- Rich visual query builder for complex boolean filters.
- Role-based access control and sharing policies beyond existing experiment permissions.
- View version history and rollback.
- Export formats that embed view-transformed payloads.

## Detailed design

### UX goals (avoid config hell)

1. **Safe default**: `Raw Trace` remains the default and requires zero setup.
2. **Progressive disclosure**: common review presets first; advanced options only when needed.
3. **Always recoverable**: clear active view state and one-click return to raw trace.
4. **Readable over clever**: views emphasize decision-critical data, not all possible transformations.

### View model

Each view stores:

- Name and optional description.
- Scope: trace-scoped or experiment-scoped.
- Span match criteria (for example by span type or name).
- Optional JSONPath extraction for input/output focus.

### Core user flow

1. Open trace detail page.
2. Select a view from the `View` dropdown (`Raw Trace` + saved views grouped by scope).
3. Review filtered timeline + extracted details panel.
4. Clear view or switch to another view.
5. Optional: ask assistant ("show only tool calls"), which creates/applies a view and confirms with a toast.

### UI states

- **No saved views**: dropdown shows only `Raw Trace`.
- **View active**: info banner shows the applied view and extraction summary.
- **No span match**: timeline remains visible; details panel explains no match and suggests switching/clearing.
- **Invalid extraction path**: fallback to full span payload with non-blocking warning.

### Screenshots (to include in RFC PR)

Add these images under the RFC directory (for example `rfcs/0000-trace-views/assets/`):

1. **View selector on trace page**
  - Shows `Raw Trace`, trace-scoped views, and experiment-scoped views.
  - Placeholder: `assets/01-view-selector.png`
2. **Active view with filtered timeline + focused output**
  - Shows active-view banner and de-emphasized non-matching spans.
  - Placeholder: `assets/02-active-view.png`
3. **Assistant-created view flow**
  - Shows assistant response plus toast: "Assistant applied view: Tool Calls Only".
  - Placeholder: `assets/03-assistant-applied-view.png`

## Drawbacks

- Adds another concept ("view") that users must learn.
- Poor defaults could create confusion if teams over-customize.
- Requires consistency between backend and frontend extraction behavior.

# Alternatives

- **Keep raw-only traces**: no additional complexity, but review cost remains high.
- **Full custom query language first**: more power, but much higher UX and maintenance complexity.
- **Assistant-only transient filters**: fast to build, but no reusable/shared review standard.

# Adoption strategy

- No breaking changes; raw trace workflow remains unchanged.
- Start with lightweight presets and assistant-driven creation.
- Add manual editing/authoring only after observing real usage patterns.
- Document recommended "review recipes" for common agent debugging tasks.

# Open questions

- Should experiment-scoped views auto-suggest based on span distribution?
- How should name collisions between trace-scoped and experiment-scoped views be presented?
- Should we support multi-span aggregation in v1, or strictly first-match behavior?
- What telemetry best indicates that views improve review speed and quality?