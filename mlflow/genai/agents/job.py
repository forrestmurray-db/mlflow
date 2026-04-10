import logging

from mlflow.client import MlflowClient
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS
from mlflow.genai.agents.trace_view_agent import summarize_trace
from mlflow.server.jobs import job
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST

_logger = logging.getLogger(__name__)


@job(name="invoke_trace_view_creation", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
def invoke_trace_view_creation_job(
    experiment_id: str,
    trace_ids: list[str],
    run_id: str,
    model: str | None = None,
):
    client = MlflowClient()
    traces_processed = 0
    views_created = 0
    errors = 0

    for i in range(0, len(trace_ids), MAX_TRACE_LINKS_PER_REQUEST):
        batch = trace_ids[i : i + MAX_TRACE_LINKS_PER_REQUEST]
        client.link_traces_to_run(batch, run_id)

    try:
        traces = client._tracing_client.batch_get_traces(trace_ids)
    except Exception:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise

    for trace in traces:
        traces_processed += 1
        try:
            summary = summarize_trace(trace, model=model)
            summary.create_view(model=model)
            views_created += 1
        except Exception:
            _logger.exception("Failed to process trace %s", getattr(trace, "trace_id", trace))
            errors += 1

    client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
    return {
        "traces_processed": traces_processed,
        "views_created": views_created,
        "errors": errors,
    }
