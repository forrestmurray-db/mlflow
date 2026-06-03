# Screenshot capture list — Trace Views RFC

Dev server: http://localhost:3000 (running from the `impl/trace-views` worktree, which has the full feature set including AI generation).

Save all images to `docs/rfcs/0004-trace-views/images/` with the exact filenames below. The RFC references them via relative paths.

## General guidance

- Use 100% browser zoom for consistency
- Pick a single theme (light or dark) and use it for every shot
- Width target: ~1400–1600px so it fits in a GitHub RFC view without wrapping
- PNG, not JPG, for text legibility
- Anonymize anything with real customer data
- A RAG-style trace (planning → retrieval → generation) makes the most photogenic ranges; demo data should have at least one. If not, seed one via `tmp/create_trace_view.py` or `trace.summarize() → summary.create_view()`

## Shots

### 01. Default state — raw trace, no view active

**File:** `images/01-default-state.png`

**Setup:**
1. Open any trace in the trace explorer
2. Verify the view selector reads "Raw trace" (no view applied)

**Capture:** Full trace explorer panel — both left timeline (nested spans visible) and right pane (raw span details visible).

---

### 02. View selector open

**File:** `images/02-view-selector-open.png`

**Setup:**
1. On a trace that has at least one trace-scoped view and at least one experiment template (create one of each if needed)
2. Click the view selector dropdown in the header

**Capture:** Focused on the dropdown showing the groupings: Raw trace, Trace views, Experiment templates, "+ Create view".

---

### 03. View active — range summary ✓ captured

**File:** `images/03-view-active-range-summary.png` (already populated from `~/Desktop/milestone_summary.png`)

**Setup:**
1. Select a view from the dropdown
2. Right pane shows range cards; left timeline shows matching spans highlighted with range colors and others dimmed

**Capture:** Full trace explorer panel showing both the dimmed timeline and the right-pane range cards. Aim for a view with 3–5 ranges visible.

**Ideal trace:** RAG agent with planning, retrieval, and generation phases. Range cards should have visible extracted I/O so the JSONPath story reads clearly.

---

### 04. Range detail ✓ covered by `edit-mode-detail.png`

**Setup:**
1. From the range-summary state in shot 03, click one of the range cards

**Capture:** Range detail view showing the matched span(s) in focus on the left and the extracted I/O on the right. "← Back to ranges" button visible.

---

### 05. Edit mode — toolbar visible ✓ covered by `edit-mode-detail.png`

**Setup:**
1. With a view active, click the Edit button (✎ icon) next to the view name
2. Edit toolbar appears at the top with the name input + Save / Cancel
3. Range overlay should be visible on the timeline (badges, colored bars)

**Capture:** Top portion of the explorer showing the toolbar prominently; include the timeline with range badges so the editing context is clear.

---

### 06. Drag-to-select mid-action ✓ captured as GIF

**File:** `images/06-drag-to-select.gif` (already populated from `Screen Recording 2026-05-11 at 3.24.03 PM.mov`)

**Setup:**
1. In edit mode with at least one range already defined
2. Start a drag selection across spans in the timeline (use the edit-mode gesture — typically click-drag in the span gutter, or via the checkbox overlay)
3. Capture mid-drag or immediately after release, with the new-range configuration panel about to appear

**Capture:** Left timeline showing the selection rectangle / span highlights; right panel showing the new-range form starting to populate.

---

### 07. JsonFieldSelector — checkbox tree ✓ captured (still + GIF)

**Files:**
- `images/edit-mode-detail.png` (from `~/Desktop/milestone_detail.png`) — also covers shots 04 and 05
- `images/07-output-path-selection.gif` (4.4MB, ~43s, from `Screen Recording 2026-05-11 at 3.35.19 PM.mov`) — supplementary workflow demo

**Setup:**
1. In edit mode with a range under construction
2. Open the input or output JSON field selector (the checkbox tree component) — typically by clicking into the JSONPath field on the range config panel
3. Have at least one leaf field checked so the generated JSONPath is visible

**Capture:** The JsonFieldSelector tree with at least one branch expanded, one or more checkboxes checked, and the resulting JSONPath visible at the bottom or in an output field.

---

### 08. (Optional) AI-generated view creation

**File:** `images/08-ai-view-creation.png`

**Status:** Not referenced in the current RFC. Capture only if you decide to inline an example of the AI flow.

**Setup:**
1. Trigger the AI view creation flow via the assistant panel or `ViewCreationModal`
2. Capture mid-flow showing batch progress

**Note:** Requires the original `impl/trace-views` worktree (the assistant integration is not in the slim PR).

## After capture

Once the images are in `docs/rfcs/0004-trace-views/images/`, the RFC renders without further edits. If any shot doesn't match what the RFC describes, ping me and I'll adjust the prose to match the screenshot rather than the other way around.
