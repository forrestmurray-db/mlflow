# ADR 0001 — Generate trace views via a flat intent + deterministic A2UI compiler

**Date:** 2026-06-03
**Status:** Accepted (prototype scope)
**Deciders:** Forrest Murray
**Context docs:** [a2ui rearchitecture design](../superpowers/specs/2026-06-01-trace-views-a2ui-rearchitecture-design.md)

## Context

The a2ui rearchitecture design specifies an agent that generates a `TraceView`
as an a2ui document (`A2UIDocument`: `{root, elements}`). The design's literal
framing is a `propose_trace_view(name, spec)` tool whose `spec` argument is a
complete a2ui document emitted by the model.

A shipped prototype already renders such documents: `TraceViewA2UIPrototype`
feeds a hardcoded `buildSampleSpec()` into `@a2ui/react`'s `A2UIViewer`, with an
MLflow catalog (`GenAISpanDetail`, `FeedbackThumbs`) that resolves `SpanSelector`
bindings client-side against `TraceDataContext`. The next slice is to *generate*
that document from a real trace instead of hardcoding it.

Before committing to a generation mechanism we surveyed how generative-UI
applications are built today (CopilotKit, Vercel AI SDK, Vercel `json-render`,
the A2UI vs. Open-JSON-UI discussion). Two findings are decisive:

1. **Raw A2UI is token-heavy and hallucination-prone for direct LLM
   generation.** The wire shape — deeply nested trees, stable component IDs,
   `explicitList`/`literalString` wrappers — is exactly what our
   `buildSampleSpec()` already looks like. Making the model emit that directly is
   the known-hard path: high token cost, high malformed-output rate.

2. **The pattern that works is a compile layer.** SimpleA2UI, Open-JSON-UI, and
   `json-render` all have the agent emit a *flat, content-first, catalog-
   constrained* schema, then a *deterministic function* expands it into the
   verbose renderable form. "Agents decide **what**; the compiler decides
   **how**; clients decide **where**." `json-render` — which our design doc
   claims compatibility with — operates this way (Zod catalog → flat typed JSON
   tree → Renderer maps to components).

Our catalog is small (`Text`, `GenAISpanDetail`, `FeedbackThumbs`) and
selector-based, so a flat intent is nearly trivial and the compiler is ~15 lines.

## Decision

For the generation prototype, **the LLM emits a flat intent via constrained
structured output, and a deterministic Python compiler expands it into the
A2UIDocument.** The model never emits raw a2ui, never invents element IDs, and
never produces the `Column`/`explicitList` envelope.

- **Intent (`_ViewIntent`)**: `name` + ordered `elements`, each a flat
  `_ElementIntent` with `kind ∈ {text, span_detail, feedback_thumbs}` and the
  few fields that kind needs (`span_type` shorthand, `title`, `text`,
  `feedback_name`, `label`). No nesting, no IDs.
- **Compiler (`expand_to_a2ui`)**: pure function, intent → `{root, components}`.
  Mints element IDs, builds the root `Column` with an `explicitList` of child
  IDs, desugars `span_type` → `selector`. Fully unit-testable.
- **Generator (`generate_view_spec`)**: reuse the existing `summarize_trace()`
  milestone stage, then one structured-output call (mirroring
  `summarize_trace`'s own pattern) producing `_ViewIntent`, then
  `expand_to_a2ui()`.
- **Delivery**: a localhost-gated `POST /trace-analysis/view-spec` endpoint in
  `mlflow/server/assistant/api.py`; the frontend fetches the spec on a
  Generate/Regenerate action instead of calling `buildSampleSpec()`.

Explicitly out of scope for this slice: persistence, REST entity changes, the
`propose_trace_view`/`update_trace_view` tools, and the editor.

## Consequences

**Positive**
- Reliability: flat schema under JSON-schema constraints has a low malformed
  rate; no deep-tree or ID bookkeeping for the model.
- Token efficiency: the model emits a fraction of the raw-document tokens.
- Ecosystem alignment: matches `json-render`/SimpleA2UI, the framework family we
  claim compatibility with.
- The compiler is a reusable v1 asset, not throwaway prototype glue, and is
  deterministically testable in isolation.

**Negative / costs**
- An extra indirection layer (intent → document) the design doc didn't name.
- The intent schema must evolve alongside the catalog; a new component kind
  touches both the intent and the compiler.
- This slice does **not** exercise the design doc's literal
  `propose_trace_view(spec)` tool path, so that contract stays unvalidated for now.

## Revisit triggers

Reopen this decision if any of the following hold:
- The catalog grows layout primitives (nested `Stack`/`Row`/`Card`) that a flat
  intent can't express, and the agent must author real nested layouts.
- We adopt SimpleA2UI (or an upstream a2ui compile layer) directly, in which case
  our bespoke `expand_to_a2ui` should be replaced rather than extended.
- Direct A2UI generation reliability improves enough (better constrained
  decoding / larger models) that the compile layer is no longer worth its
  indirection.

## References

- [CopilotKit — Developer's Guide to Generative UI in 2026](https://www.copilotkit.ai/blog/the-developer-s-guide-to-generative-ui-in-2026)
- [A2UI vs. Open-JSON-UI: Bridging the Gap](https://dev.to/vishalmysore/a2ui-vs-open-json-ui-bridging-the-gap-5gk0)
- [InfoQ — Vercel Releases JSON-Render](https://www.infoq.com/news/2026/03/vercel-json-render/)
- [Vercel — Introducing AI SDK 3.0 with Generative UI](https://vercel.com/blog/ai-sdk-3-generative-ui)
- [Stop Parsing Text: How JSON Render Turns Model Output Into UI](https://medium.com/@kenzic/stop-parsing-text-how-json-render-turns-model-output-into-ui-0cd01b59dfa9)
