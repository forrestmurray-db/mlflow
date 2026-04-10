from unittest import mock

import pytest

from mlflow.entities.run_status import RunStatus
from mlflow.genai.agents.job import invoke_trace_view_creation_job


def test_invoke_trace_view_creation_job_has_metadata():
    assert hasattr(invoke_trace_view_creation_job, "_job_fn_metadata")
    metadata = invoke_trace_view_creation_job._job_fn_metadata
    assert metadata.name == "invoke_trace_view_creation"


def test_invoke_trace_view_creation_job_success():
    mock_client = mock.MagicMock()
    mock_trace_1 = mock.MagicMock()
    mock_trace_2 = mock.MagicMock()
    mock_traces = [mock_trace_1, mock_trace_2]

    mock_summary_1 = mock.MagicMock()
    mock_summary_2 = mock.MagicMock()

    with (
        mock.patch("mlflow.genai.agents.job.MlflowClient", return_value=mock_client),
        mock.patch(
            "mlflow.genai.agents.job.summarize_trace",
            side_effect=[mock_summary_1, mock_summary_2],
        ) as mock_summarize,
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        result = invoke_trace_view_creation_job(
            experiment_id="exp-123",
            trace_ids=["trace-1", "trace-2"],
            run_id="run-123",
            model="openai:/gpt-4o",
        )

        mock_client._tracing_client.batch_get_traces.assert_called_once_with(["trace-1", "trace-2"])
        assert mock_summarize.call_count == 2
        mock_summary_1.create_view.assert_called_once_with(model="openai:/gpt-4o")
        mock_summary_2.create_view.assert_called_once_with(model="openai:/gpt-4o")
        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FINISHED)
        )

        assert result["traces_processed"] == 2
        assert result["views_created"] == 2
        assert result["errors"] == 0


def test_invoke_trace_view_creation_job_batches_link_traces():
    mock_client = mock.MagicMock()
    trace_ids = [f"trace-{i}" for i in range(250)]
    mock_traces = [mock.MagicMock() for _ in range(250)]
    mock_summaries = [mock.MagicMock() for _ in range(250)]

    with (
        mock.patch("mlflow.genai.agents.job.MlflowClient", return_value=mock_client),
        mock.patch(
            "mlflow.genai.agents.job.summarize_trace",
            side_effect=mock_summaries,
        ),
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        result = invoke_trace_view_creation_job(
            experiment_id="exp-123",
            trace_ids=trace_ids,
            run_id="run-123",
            model="openai:/gpt-4o",
        )

        # Should batch link_traces_to_run calls (100 traces per call)
        assert mock_client.link_traces_to_run.call_count == 3
        call_args_list = mock_client.link_traces_to_run.call_args_list
        assert len(call_args_list[0][0][0]) == 100
        assert len(call_args_list[1][0][0]) == 100
        assert len(call_args_list[2][0][0]) == 50

        assert result["traces_processed"] == 250
        assert result["views_created"] == 250
        assert result["errors"] == 0


def test_invoke_trace_view_creation_job_handles_individual_trace_errors():
    mock_client = mock.MagicMock()
    mock_trace_1 = mock.MagicMock()
    mock_trace_2 = mock.MagicMock()
    mock_traces = [mock_trace_1, mock_trace_2]

    mock_summary_1 = mock.MagicMock()

    def summarize_side_effect(trace, model):
        if trace is mock_trace_1:
            return mock_summary_1
        raise Exception("Summarization failed")

    with (
        mock.patch("mlflow.genai.agents.job.MlflowClient", return_value=mock_client),
        mock.patch(
            "mlflow.genai.agents.job.summarize_trace",
            side_effect=summarize_side_effect,
        ),
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        result = invoke_trace_view_creation_job(
            experiment_id="exp-123",
            trace_ids=["trace-1", "trace-2"],
            run_id="run-123",
            model="openai:/gpt-4o",
        )

        # Job should succeed overall despite individual trace failure
        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FINISHED)
        )
        assert result["traces_processed"] == 2
        assert result["views_created"] == 1
        assert result["errors"] == 1


def test_invoke_trace_view_creation_job_handles_create_view_error():
    mock_client = mock.MagicMock()
    mock_trace_1 = mock.MagicMock()
    mock_traces = [mock_trace_1]

    mock_summary_1 = mock.MagicMock()
    mock_summary_1.create_view.side_effect = Exception("View creation failed")

    with (
        mock.patch("mlflow.genai.agents.job.MlflowClient", return_value=mock_client),
        mock.patch(
            "mlflow.genai.agents.job.summarize_trace",
            return_value=mock_summary_1,
        ),
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        result = invoke_trace_view_creation_job(
            experiment_id="exp-123",
            trace_ids=["trace-1"],
            run_id="run-123",
            model="openai:/gpt-4o",
        )

        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FINISHED)
        )
        assert result["traces_processed"] == 1
        assert result["views_created"] == 0
        assert result["errors"] == 1


def test_invoke_trace_view_creation_job_failure_marks_run_failed():
    mock_client = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.side_effect = Exception("API error")

    with mock.patch("mlflow.genai.agents.job.MlflowClient", return_value=mock_client):
        with pytest.raises(Exception, match="API error"):
            invoke_trace_view_creation_job(
                experiment_id="exp-123",
                trace_ids=["trace-1"],
                run_id="run-123",
                model="openai:/gpt-4o",
            )

        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FAILED)
        )
