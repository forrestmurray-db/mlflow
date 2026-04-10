# Trace View UI Pathways Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two UI pathways for invoking trace summarization and view creation: a batch button in the experiment traces toolbar and an interactive "Trace Analysis" mode in the assistant chat panel.

**Architecture:** The batch button mirrors the existing "Detect Issues" pattern — a toolbar button → modal → async job endpoint → progress tracking. The assistant mode adds a `+` menu to the chat input that switches the backend from Claude Code to the agent server, enabling conversational trace analysis with tools (list_spans, get_span, create_trace_view, update_trace_view).

**Tech Stack:** Python (Flask handlers, job framework), TypeScript/React (DuBois design system), FastAPI (agent server), SSE streaming.

**Spec:** `docs/superpowers/specs/2026-04-10-trace-view-ui-pathways-design.md`

---

## File Structure

### Backend (Python)

| File | Responsibility |
|------|---------------|
| `mlflow/genai/agents/job.py` (create) | Batch trace view creation job function |
| `mlflow/genai/judges/tools/update_trace_view.py` (create) | UpdateTraceViewTool for the agent |
| `mlflow/genai/agents/trace_view_agent.py` (modify) | Add conversational agent entry point |
| `mlflow/server/handlers.py` (modify) | Add `/traces/views/invoke` endpoint |
| `mlflow/server/jobs/__init__.py` (modify) | Register new job function |
| `mlflow/utils/mlflow_tags.py` (modify) | Add `MLFLOW_RUN_TYPE_TRACE_VIEW_CREATION` |

### Frontend (TypeScript/React)

| File | Responsibility |
|------|---------------|
| `mlflow/server/js/src/shared/web-shared/genai-traces-table/components/CreateViewsButton.tsx` (create) | "Create Views" button in toolbar |
| `mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/ViewCreationModal.tsx` (create) | 2-step modal for batch view creation |
| `mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/hooks/useInvokeViewCreation.ts` (create) | React Query mutation for batch endpoint |
| `mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/ViewCreationProgress.tsx` (create) | Progress display for batch job |
| `mlflow/server/js/src/assistant/AssistantModeMenu.tsx` (create) | `+` button popover with mode options |
| `mlflow/server/js/src/assistant/TraceAnalysisService.ts` (create) | Service for routing to agent server |
| `mlflow/server/js/src/assistant/AssistantChatPanel.tsx` (modify) | Add `+` button and mode badge to input area |
| `mlflow/server/js/src/assistant/AssistantContext.tsx` (modify) | Add mode state, dual-provider routing |
| `mlflow/server/js/src/assistant/types.ts` (modify) | Add mode type and trace analysis types |
| `mlflow/server/js/src/shared/web-shared/genai-traces-table/GenAITracesTableToolbar.tsx` (modify) | Add "Create Views" button |

---

## Task 1: UpdateTraceViewTool

**Files:**
- Create: `mlflow/genai/judges/tools/update_trace_view.py`
- Test: `tests/genai/judges/tools/test_update_trace_view.py`

- [ ] **Step 1: Write failing test for UpdateTraceViewTool**

```python
# tests/genai/judges/tools/test_update_trace_view.py
import json
from unittest import mock

import pytest

from mlflow.entities.trace_view import SpanRange, SpanSelector, TraceView
from mlflow.genai.judges.tools.update_trace_view import UpdateTraceViewTool


@pytest.fixture
def tool():
    return UpdateTraceViewTool()


def test_tool_name(tool):
    assert tool.name == "update_trace_view"


def test_tool_definition_has_required_params(tool):
    defn = tool.get_definition()
    props = defn.function.parameters.properties
    assert "view_id" in props
    assert "name" in props
    assert "ranges_json" in props
    assert defn.function.parameters.required == ["view_id"]


@mock.patch("mlflow.genai.judges.tools.update_trace_view.TracingClient")
def test_invoke_update_name(mock_client_cls, tool):
    mock_client = mock_client_cls.return_value
    mock_client.update_trace_view.return_value = TraceView(
        name="Updated",
        trace_id="tr-123",
        view_id="tv-456",
    )

    trace = mock.MagicMock()
    trace.info.trace_id = "tr-123"

    result = tool.invoke(trace=trace, view_id="tv-456", name="Updated")

    mock_client.update_trace_view.assert_called_once_with(
        view_id="tv-456",
        name="Updated",
        ranges=None,
    )
    assert result == {"view_id": "tv-456", "trace_id": "tr-123", "name": "Updated"}


@mock.patch("mlflow.genai.judges.tools.update_trace_view.TracingClient")
def test_invoke_update_ranges(mock_client_cls, tool):
    mock_client = mock_client_cls.return_value
    mock_client.update_trace_view.return_value = TraceView(
        name="View",
        trace_id="tr-123",
        view_id="tv-456",
        ranges=[
            SpanRange(
                from_selector=SpanSelector(span_id="span-1"),
                label="Phase 1",
            )
        ],
    )

    trace = mock.MagicMock()
    trace.info.trace_id = "tr-123"

    ranges_json = json.dumps([
        {"from_selector": {"span_id": "span-1"}, "label": "Phase 1"}
    ])

    result = tool.invoke(trace=trace, view_id="tv-456", ranges_json=ranges_json)

    mock_client.update_trace_view.assert_called_once()
    call_kwargs = mock_client.update_trace_view.call_args[1]
    assert call_kwargs["view_id"] == "tv-456"
    assert len(call_kwargs["ranges"]) == 1
    assert call_kwargs["ranges"][0].label == "Phase 1"
    assert result["view_id"] == "tv-456"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/genai/judges/tools/test_update_trace_view.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlflow.genai.judges.tools.update_trace_view'`

- [ ] **Step 3: Implement UpdateTraceViewTool**

```python
# mlflow/genai/judges/tools/update_trace_view.py
from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.create_trace_view import _parse_ranges_json
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
        from mlflow.tracing.client import TracingClient

        ranges = _parse_ranges_json(ranges_json) if ranges_json else None
        client = TracingClient()
        view = client.update_trace_view(
            view_id=view_id,
            name=name,
            ranges=ranges,
        )
        return {"view_id": view.view_id, "trace_id": view.trace_id, "name": view.name}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/genai/judges/tools/test_update_trace_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlflow/genai/judges/tools/update_trace_view.py tests/genai/judges/tools/test_update_trace_view.py
git commit -s -m "feat: add UpdateTraceViewTool for iterative view refinement"
```

---

## Task 2: Register UpdateTraceViewTool in the Agent's Tool Set

**Files:**
- Modify: `mlflow/genai/judges/tools/registry.py`

- [ ] **Step 1: Read the current tool registry**

Read `mlflow/genai/judges/tools/registry.py` to see how tools are registered and which tools currently exist in the registry.

- [ ] **Step 2: Add UpdateTraceViewTool to the registry**

Add the import and registration of `UpdateTraceViewTool` alongside `CreateTraceViewTool` in the registry. The exact change depends on the registry pattern found in Step 1.

- [ ] **Step 3: Verify the tool appears in the agent's available tools**

Run: `uv run --no-sync python -c "from mlflow.genai.judges.tools.registry import get_all_tools; print([t.name for t in get_all_tools()])"`
Expected: Output includes `"update_trace_view"`

- [ ] **Step 4: Commit**

```bash
git add mlflow/genai/judges/tools/registry.py
git commit -s -m "feat: register UpdateTraceViewTool in tool registry"
```

---

## Task 3: Batch Trace View Creation Job

**Files:**
- Create: `mlflow/genai/agents/job.py`
- Modify: `mlflow/server/jobs/__init__.py`
- Modify: `mlflow/utils/mlflow_tags.py`
- Test: `tests/genai/agents/test_job.py`

- [ ] **Step 1: Add the run type tag**

In `mlflow/utils/mlflow_tags.py`, add after line 40 (`MLFLOW_RUN_TYPE_ISSUE_DETECTION = "issue_detection"`):

```python
MLFLOW_RUN_TYPE_TRACE_VIEW_CREATION = "trace_view_creation"
```

- [ ] **Step 2: Write failing test for the job function**

```python
# tests/genai/agents/test_job.py
from unittest import mock

import pytest

from mlflow.genai.agents.job import invoke_trace_view_creation_job


@mock.patch("mlflow.genai.agents.job.MlflowClient")
@mock.patch("mlflow.genai.agents.job.summarize_trace")
@mock.patch("mlflow.genai.agents.job.TraceSummary")
def test_invoke_trace_view_creation_job(mock_summary_cls, mock_summarize, mock_client_cls):
    mock_client = mock_client_cls.return_value

    # Set up mock trace
    mock_trace = mock.MagicMock()
    mock_trace.info.trace_id = "tr-1"
    mock_client.get_trace.return_value = mock_trace

    # Set up mock summary
    mock_summary = mock.MagicMock()
    mock_summary.create_view.return_value = mock.MagicMock(view_id="tv-1")
    mock_summarize.return_value = mock_summary

    result = invoke_trace_view_creation_job(
        experiment_id="1",
        trace_ids=["tr-1"],
        run_id="run-123",
        model="openai:/gpt-4o",
    )

    mock_client.get_trace.assert_called_once_with("tr-1")
    mock_summarize.assert_called_once_with(trace=mock_trace, model="openai:/gpt-4o")
    mock_summary.create_view.assert_called_once()
    assert result["views_created"] == 1
    assert result["traces_processed"] == 1
    assert result["errors"] == 0


@mock.patch("mlflow.genai.agents.job.MlflowClient")
@mock.patch("mlflow.genai.agents.job.summarize_trace")
def test_job_handles_individual_trace_errors(mock_summarize, mock_client_cls):
    mock_client = mock_client_cls.return_value

    mock_trace = mock.MagicMock()
    mock_trace.info.trace_id = "tr-1"
    mock_client.get_trace.return_value = mock_trace
    mock_summarize.side_effect = Exception("LLM error")

    result = invoke_trace_view_creation_job(
        experiment_id="1",
        trace_ids=["tr-1"],
        run_id="run-123",
        model="openai:/gpt-4o",
    )

    assert result["views_created"] == 0
    assert result["traces_processed"] == 1
    assert result["errors"] == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/genai/agents/test_job.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the job function**

```python
# mlflow/genai/agents/job.py
import logging

from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS
from mlflow.server.jobs import job

_logger = logging.getLogger(__name__)


@job(name="invoke_trace_view_creation", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
def invoke_trace_view_creation_job(
    experiment_id: str,
    trace_ids: list[str],
    run_id: str,
    model: str = "openai:/gpt-4o",
):
    from mlflow.genai.agents.trace_view_agent import summarize_trace

    client = MlflowClient()
    views_created = 0
    errors = 0

    try:
        for i, trace_id in enumerate(trace_ids):
            try:
                trace = client.get_trace(trace_id)
                summary = summarize_trace(trace=trace, model=model)
                view = summary.create_view()
                if view:
                    views_created += 1
                _logger.info(f"[{i + 1}/{len(trace_ids)}] Created view for trace {trace_id}")
            except Exception:
                errors += 1
                _logger.exception(f"Failed to create view for trace {trace_id}")

        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        return {
            "traces_processed": len(trace_ids),
            "views_created": views_created,
            "errors": errors,
        }
    except Exception:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise
```

- [ ] **Step 5: Register the job function**

In `mlflow/server/jobs/__init__.py`, add to `_SUPPORTED_JOB_FUNCTION_LIST` (line 20-27):

```python
"mlflow.genai.agents.job.invoke_trace_view_creation_job",
```

Add to `_ALLOWED_JOB_NAME_LIST` (line 33-40):

```python
"invoke_trace_view_creation",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/genai/agents/test_job.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add mlflow/genai/agents/job.py mlflow/server/jobs/__init__.py mlflow/utils/mlflow_tags.py tests/genai/agents/test_job.py
git commit -s -m "feat: add batch trace view creation job function"
```

---

## Task 4: Batch Trace View Creation REST Endpoint

**Files:**
- Modify: `mlflow/server/handlers.py`

- [ ] **Step 1: Add the handler function**

Add after the `_invoke_issue_detection_handler` function (around line 4427) in `mlflow/server/handlers.py`:

```python
@catch_mlflow_exception
@_disable_if_artifacts_only
def _invoke_trace_view_creation_handler():
    from mlflow.genai.agents.job import invoke_trace_view_creation_job
    from mlflow.genai.discovery.job import _fetch_provider_credentials
    from mlflow.server.jobs import submit_job
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE_TRACE_VIEW_CREATION

    _validate_content_type(request, ["application/json"])

    request_json = _get_validated_flask_request_json(
        schema={
            "experiment_id": [_assert_required, _assert_string],
            "trace_ids": [_assert_required, _assert_array],
            "provider": [_assert_required, _assert_string],
            "model": [_assert_string],
            "secret_id": [_assert_string],
            "endpoint_name": [_assert_string],
        }
    )

    experiment_id = request_json.get("experiment_id")
    trace_ids = request_json.get("trace_ids", [])
    provider = request_json.get("provider")
    model = request_json.get("model")
    secret_id = request_json.get("secret_id")
    endpoint_name = request_json.get("endpoint_name")

    if not endpoint_name and not (provider and model):
        raise MlflowException(
            "Either 'endpoint_name' or both 'provider' and 'model' must be provided"
        )

    if secret_id:
        store = _get_tracking_store()
        credentials = _fetch_provider_credentials(store, provider, secret_id)
    else:
        credentials = None

    model_name = f"gateway:/{endpoint_name}" if endpoint_name else f"{provider}:/{model}"
    run = mlflow.start_run(
        experiment_id=experiment_id,
        tags={
            MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_TRACE_VIEW_CREATION,
            "model": model_name,
            "total_traces": len(trace_ids),
        },
    )
    run_id = run.info.run_id

    job = submit_job(
        function=invoke_trace_view_creation_job,
        params={
            "experiment_id": experiment_id,
            "trace_ids": trace_ids,
            "run_id": run_id,
            "model": model_name,
        },
        extra_envs=credentials,
    )
    mlflow.end_run(RunStatus.to_string(RunStatus.RUNNING))

    return jsonify({"job_id": job.job_id, "run_id": run_id})
```

- [ ] **Step 2: Register the endpoint**

Find the `get_issues_detection_endpoints` function (around line 5970) and add a new endpoint registration function after it:

```python
def get_trace_view_creation_endpoints():
    return [
        (
            _get_ajax_path("/mlflow/traces/views/invoke", version=3),
            _invoke_trace_view_creation_handler,
            ["POST"],
        ),
    ]
```

Then find where `get_issues_detection_endpoints()` is called in the app setup (in `mlflow/server/__init__.py`) and add `get_trace_view_creation_endpoints()` alongside it. Read `mlflow/server/__init__.py` to find the exact pattern.

- [ ] **Step 3: Commit**

```bash
git add mlflow/server/handlers.py mlflow/server/__init__.py
git commit -s -m "feat: add REST endpoint for batch trace view creation"
```

---

## Task 5: Frontend — useInvokeViewCreation Hook

**Files:**
- Create: `mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/hooks/useInvokeViewCreation.ts`

- [ ] **Step 1: Implement the hook**

Model this after `useInvokeIssueDetection.ts`:

```typescript
// mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/hooks/useInvokeViewCreation.ts
import { useMutation } from '@tanstack/react-query';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

interface InvokeViewCreationParams {
  experimentId: string;
  traceIds: string[];
  provider: string;
  model: string;
  secretId?: string;
  endpointName?: string;
}

interface InvokeViewCreationResponse {
  job_id: string;
  run_id: string;
}

export const useInvokeViewCreation = () => {
  return useMutation<InvokeViewCreationResponse, Error, InvokeViewCreationParams>({
    mutationFn: async (params) => {
      const response = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/traces/views/invoke'), {
        method: 'POST',
        body: {
          experiment_id: params.experimentId,
          trace_ids: params.traceIds,
          provider: params.provider,
          model: params.model,
          secret_id: params.secretId,
          endpoint_name: params.endpointName,
        },
      });
      return response;
    },
  });
};
```

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/hooks/useInvokeViewCreation.ts
git commit -s -m "feat: add useInvokeViewCreation React Query hook"
```

---

## Task 6: Frontend — CreateViewsButton Component

**Files:**
- Create: `mlflow/server/js/src/shared/web-shared/genai-traces-table/components/CreateViewsButton.tsx`

- [ ] **Step 1: Implement the button**

Model after `DetectIssuesButton.tsx` — same gradient border style, uses SparkleIcon:

```typescript
// mlflow/server/js/src/shared/web-shared/genai-traces-table/components/CreateViewsButton.tsx
import { Button, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';

interface CreateViewsButtonProps {
  componentId: string;
  onClick: () => void;
}

export const CreateViewsButton = ({ componentId, onClick }: CreateViewsButtonProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Button
      componentId={componentId}
      type="tertiary"
      onClick={onClick}
      icon={<SparkleIcon color="ai" />}
      css={{
        border: `1px solid transparent`,
        backgroundImage: `linear-gradient(${theme.colors.backgroundPrimary}, ${theme.colors.backgroundPrimary}), linear-gradient(135deg, ${theme.colors.purple400}, ${theme.colors.blue400})`,
        backgroundOrigin: 'border-box',
        backgroundClip: 'padding-box, border-box',
      }}
    >
      Create Views
    </Button>
  );
};
```

Note: Read `DetectIssuesButton.tsx` to get the exact CSS pattern for the gradient border before implementing. The code above is approximate — match the existing pattern exactly.

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/shared/web-shared/genai-traces-table/components/CreateViewsButton.tsx
git commit -s -m "feat: add CreateViewsButton component"
```

---

## Task 7: Frontend — ViewCreationModal Component

**Files:**
- Create: `mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/ViewCreationModal.tsx`

- [ ] **Step 1: Implement the modal**

Model after `IssueDetectionModal.tsx` — 2-step modal. Step 1 is a simple confirmation with trace count. Step 2 reuses `IssueDetectionModelSelection` for provider/model/key selection.

Read `IssueDetectionModal.tsx` fully before implementing. The modal should:
1. Accept `traceIds: string[]`, `experimentId: string`, `onClose: () => void`, `onSubmitSuccess?: (runId: string) => void` as props
2. Step 1: Show "Create milestone views for N traces" with a description explaining what summarize + create view does
3. Step 2: Reuse `IssueDetectionModelSelection` component (forwardRef pattern) for model config
4. On submit: call `useInvokeViewCreation` mutation, then `onSubmitSuccess(response.run_id)`

```typescript
// Skeleton — fill in by reading IssueDetectionModal.tsx patterns
import { Modal } from '@databricks/design-system';
import { useState, useRef } from 'react';
import { useInvokeViewCreation } from './hooks/useInvokeViewCreation';
import { IssueDetectionModelSelection } from './IssueDetectionModelSelection';

interface ViewCreationModalProps {
  traceIds: string[];
  experimentId: string;
  onClose: () => void;
  onSubmitSuccess?: (runId: string) => void;
}

export const ViewCreationModal = ({
  traceIds,
  experimentId,
  onClose,
  onSubmitSuccess,
}: ViewCreationModalProps) => {
  const [step, setStep] = useState<1 | 2>(1);
  const modelSelectionRef = useRef<any>(null);
  const { mutateAsync: invokeViewCreation, isPending } = useInvokeViewCreation();

  const handleSubmit = async () => {
    const values = modelSelectionRef.current?.getValues();
    if (!values) return;

    // Follow IssueDetectionModal's secret handling pattern exactly
    // (endpoint mode vs direct mode with new/existing key)
    const response = await invokeViewCreation({
      experimentId,
      traceIds,
      provider: values.provider,
      model: values.model,
      endpointName: values.endpointName,
      // secretId handling...
    });
    onSubmitSuccess?.(response.run_id);
    onClose();
  };

  return (
    <Modal
      componentId="mlflow.traces.create-views-modal"
      title="Create Trace Views"
      visible
      onCancel={onClose}
      onOk={step === 1 ? () => setStep(2) : handleSubmit}
      okText={step === 1 ? 'Next' : 'Create Views'}
      confirmLoading={isPending}
    >
      {step === 1 ? (
        <div>
          <p>
            This will analyze {traceIds.length} trace{traceIds.length !== 1 ? 's' : ''} and create
            milestone views for each. Each trace will be summarized to identify key phases, then a
            view will be created highlighting those phases.
          </p>
        </div>
      ) : (
        <IssueDetectionModelSelection ref={modelSelectionRef} />
      )}
    </Modal>
  );
};
```

Note: The above is a skeleton. The implementer MUST read `IssueDetectionModal.tsx` to replicate the exact secret creation flow, model selection ref pattern, and error handling.

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/ViewCreationModal.tsx
git commit -s -m "feat: add ViewCreationModal for batch trace view creation"
```

---

## Task 8: Frontend — ViewCreationProgress Component

**Files:**
- Create: `mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/ViewCreationProgress.tsx`

- [ ] **Step 1: Implement progress component**

Model after `IssueDetectionProgress.tsx`. Read it fully before implementing.

The component should:
- Accept `jobId`, `jobStatus`, `totalTraces` and similar props from the parent
- Show a progress spinner while running
- Display "N/M traces processed, K views created" when complete
- Show errors if any
- Include a cancel button

Read `IssueDetectionProgress.tsx` to understand the exact polling and status display patterns.

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/experiment-tracking/components/experiment-page/components/traces-v3/ViewCreationProgress.tsx
git commit -s -m "feat: add ViewCreationProgress component"
```

---

## Task 9: Frontend — Wire CreateViewsButton into Toolbar

**Files:**
- Modify: `mlflow/server/js/src/shared/web-shared/genai-traces-table/GenAITracesTableToolbar.tsx`

- [ ] **Step 1: Read the toolbar component**

Read `GenAITracesTableToolbar.tsx` to understand the current props and where `DetectIssuesButton` is rendered.

- [ ] **Step 2: Add CreateViewsButton to the toolbar**

Add a new optional prop `onCreateViews?: () => void` to the toolbar props (alongside `onDetectIssues`).

Render the `CreateViewsButton` next to `DetectIssuesButton`:

```typescript
{onCreateViews && (
  <CreateViewsButton
    componentId="mlflow.traces-table.create-views-button"
    onClick={onCreateViews}
  />
)}
```

- [ ] **Step 3: Wire up in the parent experiment page**

Find where `GenAITracesTableToolbar` is used and pass the `onCreateViews` callback that opens the `ViewCreationModal`. This follows the same pattern as `onDetectIssues`. Read the parent component to find the exact integration point.

- [ ] **Step 4: Commit**

```bash
git add mlflow/server/js/src/shared/web-shared/genai-traces-table/GenAITracesTableToolbar.tsx
git commit -s -m "feat: add Create Views button to traces toolbar"
```

---

## Task 10: Frontend — Assistant Mode Types

**Files:**
- Modify: `mlflow/server/js/src/assistant/types.ts`

- [ ] **Step 1: Add mode type and trace analysis types**

Add to `mlflow/server/js/src/assistant/types.ts`:

```typescript
/**
 * Assistant chat modes.
 * 'assistant' = normal Claude Code assistant
 * 'trace_analysis' = trace view agent via agent server
 */
export type AssistantMode = 'assistant' | 'trace_analysis';

/**
 * Request body for trace analysis mode messages.
 * Sent to the agent server instead of the assistant backend.
 */
export interface TraceAnalysisRequest {
  messages: Array<{ role: string; content: string }>;
  context: {
    trace_id?: string;
    experiment_id?: string;
  };
  stream: boolean;
}
```

Add `mode` to `AssistantAgentState`:

```typescript
export interface AssistantAgentState {
  // ... existing fields ...
  /** Current chat mode */
  mode: AssistantMode;
}
```

Add `setMode` to `AssistantAgentActions`:

```typescript
export interface AssistantAgentActions {
  // ... existing fields ...
  /** Switch the assistant mode */
  setMode: (mode: AssistantMode) => void;
}
```

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/assistant/types.ts
git commit -s -m "feat: add assistant mode types for trace analysis"
```

---

## Task 11: Frontend — TraceAnalysisService

**Files:**
- Create: `mlflow/server/js/src/assistant/TraceAnalysisService.ts`

- [ ] **Step 1: Implement the service**

This service routes messages to the agent server. It connects to the agent server's SSE streaming endpoint via the MLflow server proxy.

```typescript
// mlflow/server/js/src/assistant/TraceAnalysisService.ts
import type { SendMessageStreamCallbacks, SendMessageStreamResult } from './AssistantService';
import { getAjaxUrl, getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';

const TRACE_ANALYSIS_API = getAjaxUrl('ajax-api/3.0/mlflow/assistant/trace-analysis');

export interface TraceAnalysisMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Send a message to the trace analysis agent and stream the response.
 * The agent server uses SSE with "data: {json}\n\n" format.
 */
export const sendTraceAnalysisStream = async (
  messages: TraceAnalysisMessage[],
  context: { traceId?: string; experimentId?: string },
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  const { onMessage, onError, onDone, onToolUse } = callbacks;

  try {
    const response = await fetch(`${TRACE_ANALYSIS_API}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDefaultHeaders(document.cookie),
      },
      body: JSON.stringify({
        messages,
        context: {
          trace_id: context.traceId,
          experiment_id: context.experimentId,
        },
        stream: true,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return { eventSource: null };
    }

    // The agent server streams SSE directly in the response body
    const reader = response.body?.getReader();
    if (!reader) {
      onError('No response body');
      return { eventSource: null };
    }

    const decoder = new TextDecoder();
    let buffer = '';

    const processStream = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                onToolUse?.([]);
                onDone();
                return;
              }
              try {
                const parsed = JSON.parse(data);
                if (parsed.error) {
                  onError(parsed.error);
                  return;
                }
                // Agent server streams response chunks as text
                if (typeof parsed === 'string') {
                  onMessage(parsed);
                } else if (parsed.content) {
                  onMessage(parsed.content);
                } else if (parsed.delta?.text) {
                  onMessage(parsed.delta.text);
                }
              } catch {
                // Skip malformed chunks
              }
            }
          }
        }
        onDone();
      } catch (err) {
        onError(err instanceof Error ? err.message : 'Stream error');
      }
    };

    // Process in background (not awaited intentionally)
    processStream();

    // Return a fake EventSource-like object for cancellation
    return {
      eventSource: {
        close: () => reader.cancel(),
        readyState: EventSource.OPEN,
      } as unknown as EventSource,
    };
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
    return { eventSource: null };
  }
};
```

Note: The exact SSE format from the agent server is `data: {json}\n\n` followed by `data: [DONE]\n\n`. The implementer should verify the agent server response format by reading `mlflow/genai/agent_server/server.py:349-397` and adapt the parsing accordingly. The code above handles the standard format but the response structure of each chunk depends on what the agent server returns.

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/assistant/TraceAnalysisService.ts
git commit -s -m "feat: add TraceAnalysisService for agent server communication"
```

---

## Task 12: Frontend — AssistantModeMenu Component

**Files:**
- Create: `mlflow/server/js/src/assistant/AssistantModeMenu.tsx`

- [ ] **Step 1: Implement the mode menu**

```typescript
// mlflow/server/js/src/assistant/AssistantModeMenu.tsx
import { useState } from 'react';
import { Button, PlusIcon, Popover, useDesignSystemTheme } from '@databricks/design-system';
import type { AssistantMode } from './types';

interface AssistantModeMenuProps {
  currentMode: AssistantMode;
  onModeChange: (mode: AssistantMode) => void;
}

export const AssistantModeMenu = ({ currentMode, onModeChange }: AssistantModeMenuProps) => {
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);

  const handleSelect = (mode: AssistantMode) => {
    onModeChange(mode);
    setIsOpen(false);
  };

  return (
    <Popover.Root open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <Button
          componentId="mlflow.assistant.mode-menu"
          type="tertiary"
          size="small"
          icon={<PlusIcon />}
          aria-label="Switch assistant mode"
        />
      </Popover.Trigger>
      <Popover.Content align="start" side="top">
        <div css={{ padding: theme.spacing.xs, minWidth: 180 }}>
          <button
            type="button"
            onClick={() => handleSelect('trace_analysis')}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              width: '100%',
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              border: 'none',
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: currentMode === 'trace_analysis' ? theme.colors.actionTertiaryBackgroundHover : 'transparent',
              cursor: 'pointer',
              fontSize: theme.typography.fontSizeBase,
              color: theme.colors.textPrimary,
              '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover,
              },
            }}
          >
            Trace Analysis
          </button>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
```

Note: Read the DuBois design system exports to verify `Popover` API. The implementer should check `@databricks/design-system` exports and use the correct Popover/DropdownMenu pattern from the codebase. The code above is approximate — adapt to match the actual component API.

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/assistant/AssistantModeMenu.tsx
git commit -s -m "feat: add AssistantModeMenu component"
```

---

## Task 13: Frontend — Mode Badge Component

**Files:**
- Create: `mlflow/server/js/src/assistant/AssistantModeBadge.tsx`

- [ ] **Step 1: Implement the mode badge**

```typescript
// mlflow/server/js/src/assistant/AssistantModeBadge.tsx
import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import type { AssistantMode } from './types';

interface AssistantModeBadgeProps {
  mode: AssistantMode;
  onClear: () => void;
}

export const AssistantModeBadge = ({ mode, onClear }: AssistantModeBadgeProps) => {
  if (mode === 'assistant') return null;

  return (
    <Tag
      componentId="mlflow.assistant.mode-badge"
      closable
      onClose={onClear}
    >
      Trace Analysis
    </Tag>
  );
};
```

Note: Check `@databricks/design-system` for the correct `Tag` component API (it may be `Tag`, `Badge`, or `Chip`). Read existing usage in the codebase to match.

- [ ] **Step 2: Commit**

```bash
git add mlflow/server/js/src/assistant/AssistantModeBadge.tsx
git commit -s -m "feat: add AssistantModeBadge component"
```

---

## Task 14: Frontend — Wire Mode into AssistantContext

**Files:**
- Modify: `mlflow/server/js/src/assistant/AssistantContext.tsx`

- [ ] **Step 1: Add mode state**

Add to the state declarations in `AssistantProvider` (after line 67):

```typescript
const [mode, setMode] = useState<AssistantMode>('assistant');
```

Import `AssistantMode` from `./types`.

- [ ] **Step 2: Modify sendMessage to route by mode**

In the `startChat` and `handleSendMessage` functions, add mode-aware routing. When `mode === 'trace_analysis'`, use `sendTraceAnalysisStream` instead of `sendMessageStream`.

In `startChat` (around line 241), replace the `sendMessageStream` call with:

```typescript
if (mode === 'trace_analysis') {
  // Build message history from current messages
  const messageHistory = messages
    .filter((m) => !m.isStreaming)
    .map((m) => ({ role: m.role, content: m.content }));
  messageHistory.push({ role: 'user', content: prompt || '' });

  const pageContext = getPageContext();
  const result = await sendTraceAnalysisStream(
    messageHistory,
    {
      traceId: pageContext['traceId'] as string | undefined,
      experimentId: pageContext['experimentId'] as string | undefined,
    },
    {
      onMessage: appendToStreamingMessage,
      onError: handleStreamError,
      onDone: finalizeStreamingMessage,
      onStatus: handleStatus,
      onSessionId: handleSessionId,
      onToolUse: handleToolUse,
      onInterrupted: handleInterrupted,
    },
  );
  eventSourceRef.current = result.eventSource;
} else {
  // Existing Claude Code path
  const pageContext = getPageContext();
  const result = await sendMessageStream(
    {
      message: prompt || '',
      session_id: sessionId ?? undefined,
      experiment_id: pageContext['experimentId'] as string | undefined,
      context: pageContext,
    },
    {
      onMessage: appendToStreamingMessage,
      onError: handleStreamError,
      onDone: finalizeStreamingMessage,
      onStatus: handleStatus,
      onSessionId: handleSessionId,
      onToolUse: handleToolUse,
      onInterrupted: handleInterrupted,
    },
  );
  eventSourceRef.current = result.eventSource;
}
```

Apply the same pattern to `handleSendMessage`.

- [ ] **Step 3: Reset on mode switch**

Add a `handleSetMode` callback that resets the conversation when switching modes:

```typescript
const handleSetMode = useCallback(
  (newMode: AssistantMode) => {
    if (newMode !== mode) {
      reset();
      setMode(newMode);
    }
  },
  [mode, reset],
);
```

- [ ] **Step 4: Update trace view marker regex to also handle update markers**

In the `TRACE_VIEW_MARKER_REGEX` constant (line 28), update to also match `trace_view_updated` markers:

```typescript
const TRACE_VIEW_MARKER_REGEX = /\[trace_view_(?:created|updated):\s*(\{.*?\})\]/g;
```

- [ ] **Step 5: Expose mode and setMode in the context value**

Add to the `value` object (around line 450):

```typescript
const value: AssistantAgentContextType = {
  // ... existing fields ...
  mode,
  setMode: handleSetMode,
};
```

Update the `disabledAssistantContext` default (around line 477):

```typescript
mode: 'assistant',
setMode: () => {},
```

- [ ] **Step 6: Commit**

```bash
git add mlflow/server/js/src/assistant/AssistantContext.tsx
git commit -s -m "feat: add mode-aware routing to AssistantContext"
```

---

## Task 15: Frontend — Wire Mode Menu and Badge into ChatPanel

**Files:**
- Modify: `mlflow/server/js/src/assistant/AssistantChatPanel.tsx`

- [ ] **Step 1: Read the chat panel**

Read `AssistantChatPanel.tsx` fully to understand the input area layout.

- [ ] **Step 2: Add mode menu and badge to the input area**

In the `ChatPanelContent` component, import the mode from context:

```typescript
const { messages, isStreaming, error, activeTools, sendMessage, regenerateLastMessage, cancelSession, mode, setMode } =
  useAssistant();
```

Add the `+` button to the left of the textarea (inside the `display: 'flex'` div at line 366):

```typescript
<div css={{ display: 'flex', alignItems: 'flex-end' }}>
  <AssistantModeMenu currentMode={mode} onModeChange={setMode} />
  <textarea
    ref={textareaRef}
    // ... existing textarea props ...
  />
  <Button
    // ... existing send button ...
  />
</div>
```

Add the mode badge between the textarea row and `<AssistantContextTags />`:

```typescript
<AssistantModeBadge mode={mode} onClear={() => setMode('assistant')} />
<AssistantContextTags />
```

- [ ] **Step 3: Update the placeholder text based on mode**

```typescript
placeholder={mode === 'trace_analysis' ? 'Ask about this trace...' : 'Ask a question...'}
```

- [ ] **Step 4: Commit**

```bash
git add mlflow/server/js/src/assistant/AssistantChatPanel.tsx
git commit -s -m "feat: add mode menu and badge to assistant chat panel"
```

---

## Task 16: Backend — Trace Analysis Proxy Endpoint

**Files:**
- Modify: `mlflow/server/handlers.py` (or wherever the assistant API routes are defined)

- [ ] **Step 1: Find the assistant backend**

The assistant endpoints (`/ajax-api/3.0/mlflow/assistant/message`, `/sessions/{id}/stream`, etc.) are not in `handlers.py`. Search for where they are defined — likely in a separate assistant module. Read that code to understand the proxy pattern.

Run: `grep -r "assistant/message\|assistant/sessions" mlflow/server/ --include="*.py" -l`

- [ ] **Step 2: Add the trace analysis proxy endpoint**

Add a new endpoint `POST /ajax-api/3.0/mlflow/assistant/trace-analysis/message` that:
1. Accepts the request body (`messages`, `context`, `stream`)
2. Forwards to the agent server (either locally running or configured URL)
3. Streams the SSE response back to the client

The exact implementation depends on how the assistant backend is structured (found in Step 1). The proxy needs to forward the request to the agent server and relay the SSE stream.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -s -m "feat: add trace analysis proxy endpoint"
```

---

## Task 17: Backend — Conversational Trace View Agent

**Files:**
- Modify: `mlflow/genai/agents/trace_view_agent.py`

- [ ] **Step 1: Read the current agent**

The current agent has two one-shot functions (`summarize_trace` and `create_view_from_summary`). For the assistant's trace analysis mode, we need a conversational entry point that:
- Accepts a message history (not hardcoded prompts)
- Has all four tools available (list_spans, get_span, create_trace_view, update_trace_view)
- Streams responses back

- [ ] **Step 2: Add a conversational agent function**

Add to `mlflow/genai/agents/trace_view_agent.py`:

```python
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
    """Create a conversational trace analysis agent for the agent server.

    Returns a callable that accepts a messages list and yields streaming chunks.
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
        return result

    return invoke
```

Note: The exact return type of `_invoke_litellm_and_handle_tools` needs to be checked. The implementer should read `mlflow/genai/judges/adapters/litellm_adapter.py` to understand what it returns and how to extract the text response. The streaming behavior also depends on whether litellm supports streaming — if not, the agent server can return the full response as a single SSE chunk.

- [ ] **Step 3: Commit**

```bash
git add mlflow/genai/agents/trace_view_agent.py
git commit -s -m "feat: add conversational agent entry point for trace analysis"
```

---

## Task 18: Integration Testing — Batch Flow

**Files:**
- Test: Manual verification

- [ ] **Step 1: Start the dev server**

```bash
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &
tail -f /tmp/mlflow-dev-server.log
```

- [ ] **Step 2: Verify the Create Views button appears**

Navigate to `http://localhost:3000`, open an experiment with traces. Verify "Create Views" button appears in the toolbar next to "Detect Issues".

- [ ] **Step 3: Verify the modal flow**

Click "Create Views" → verify 2-step modal appears. Step 1 shows trace count. Step 2 shows model selection.

- [ ] **Step 4: Verify the batch endpoint**

Submit the modal and verify:
- POST to `/ajax-api/3.0/mlflow/traces/views/invoke` returns `{job_id, run_id}`
- Progress component shows the job status
- Views appear on the traces after completion

---

## Task 19: Integration Testing — Assistant Trace Analysis Mode

**Files:**
- Test: Manual verification

- [ ] **Step 1: Verify the `+` menu**

Open the assistant panel. Verify the `+` button appears to the left of the text input.

- [ ] **Step 2: Verify mode switching**

Click `+` → select "Trace Analysis" → verify:
- Mode badge "Trace Analysis" appears
- Conversation resets
- Placeholder changes to "Ask about this trace..."

- [ ] **Step 3: Verify trace analysis mode**

Navigate to a trace detail page. In trace analysis mode, send "Summarize this trace". Verify:
- Request goes to the agent server (check network tab)
- Response streams back with trace analysis
- Trace view markers trigger view refresh

- [ ] **Step 4: Verify mode toggle back**

Click the "Trace Analysis" badge to dismiss → verify:
- Mode returns to normal assistant
- Conversation resets
- Badge disappears

---

## Task Dependencies

```
Task 1 → Task 2 (UpdateTraceViewTool → register in registry)
Task 3 → Task 4 (Job function → REST endpoint)
Task 5 → Task 7 (Hook → Modal uses hook)
Task 6 → Task 9 (Button → wire into toolbar)
Task 7 → Task 9 (Modal → wire into toolbar parent)
Task 10 → Task 11, 12, 13, 14 (Types → Service, Menu, Badge, Context)
Task 11, 12, 13, 14 → Task 15 (All assistant pieces → wire into ChatPanel)
Task 16 → Task 17 (Proxy endpoint → agent serves it)
Tasks 1-9 → Task 18 (All batch pieces → integration test)
Tasks 10-17 → Task 19 (All assistant pieces → integration test)
```

Tasks 1-2 and Tasks 3-4 can run in parallel.
Tasks 5-9 (frontend batch) can run in parallel with Tasks 10-17 (frontend assistant).
