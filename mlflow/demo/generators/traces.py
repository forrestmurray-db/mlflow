from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import mlflow
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    DEMO_PROMPT_PREFIX,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.demo.data import (
    AGENT_TRACES,
    MULTI_AGENT_TRACES,
    PROMPT_TRACES,
    RAG_TRACES,
    SESSION_TRACES,
    DemoTrace,
)
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

DEMO_VERSION_TAG = "mlflow.demo.version"
DEMO_TRACE_TYPE_TAG = "mlflow.demo.trace_type"
DEMO_START_TIME_TAG = "mlflow.demo.start_time_ms"
DEMO_END_TIME_TAG = "mlflow.demo.end_time_ms"

_TOTAL_TRACES_PER_VERSION = 20


@dataclass(frozen=True)
class _TraceSetResult:
    """Result from generating a set of traces.

    Attributes:
        trace_ids: List of generated trace IDs.
        start_time_ns: Earliest trace start time in nanoseconds.
        end_time_ns: Latest trace end time in nanoseconds.
    """

    trace_ids: list[str]
    start_time_ns: int
    end_time_ns: int


def _get_trace_timestamps(trace_index: int, version: str) -> tuple[int, int]:
    """Get deterministic start and end timestamps for a trace.

    Distributes traces over the last 7 days with a deterministic pattern
    based on the trace index and version. This ensures the demo dashboard
    shows activity across the time range.

    Args:
        trace_index: Index of the trace (0-based) within its version set.
        version: "v1" or "v2" - v1 traces are earlier, v2 traces are later.

    Returns:
        Tuple of (start_time_ns, end_time_ns).
    """
    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)

    if version == "v1":
        day_offset = (trace_index * 3.5) / _TOTAL_TRACES_PER_VERSION
    else:
        day_offset = 3.5 + (trace_index * 3.5) / _TOTAL_TRACES_PER_VERSION

    hash_input = f"{trace_index}:{version}"
    hash_val = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    hour_offset = (hash_val % 24) / 24
    minute_offset = ((hash_val >> 8) % 60) / (60 * 24)

    trace_time = seven_days_ago + timedelta(days=day_offset + hour_offset + minute_offset)

    duration_ms = 50 + (hash_val % 1950)

    start_ns = int(trace_time.timestamp() * 1_000_000_000)
    end_ns = start_ns + (duration_ms * 1_000_000)

    return start_ns, end_ns


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation: ~4 chars per token)."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class _Model:
    """Model configuration with name, provider, and pricing."""

    name: str
    provider: str
    pricing: tuple[float, float]  # (input $/1M tokens, output $/1M tokens)


# Using three distinct models so the cost breakdown chart shows a nice distribution.
GPT_5_2 = _Model(name="gpt-5.2", provider="openai", pricing=(1.75, 14.00))
CLAUDE_SONNET_4_5 = _Model(name="claude-sonnet-4-5", provider="anthropic", pricing=(3.00, 15.00))
GEMINI_3_PRO = _Model(name="gemini-3-pro", provider="google", pricing=(2.00, 12.00))

_DEMO_MODELS = (GPT_5_2, CLAUDE_SONNET_4_5, GEMINI_3_PRO)


def _compute_cost(model: _Model, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    """Compute synthetic cost using approximate per-model pricing."""
    input_rate, output_rate = model.pricing
    input_cost = prompt_tokens * input_rate / 1_000_000
    output_cost = completion_tokens * output_rate / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


class TracesDemoGenerator(BaseDemoGenerator):
    """Generates demo traces for the MLflow UI.

    Creates two sets of traces showing agent improvement:
    - V1 traces: Initial/baseline agent (uses v1_response)
    - V2 traces: Improved agent after updates (uses v2_response)

    Both versions use the same inputs but produce different outputs,
    simulating an agent improvement workflow.

    Trace types generated:
    - RAG: Document retrieval and generation pipeline
    - Agent: Tool-using agent with function calls
    - Prompt: Prompt template-based generation
    - Session: Multi-turn conversation sessions
    """

    name = DemoFeature.TRACES
    version = 1

    def generate(self) -> DemoResult:
        self._restore_experiment_if_deleted()
        experiment = mlflow.set_experiment(DEMO_EXPERIMENT_NAME)
        mlflow.MlflowClient().set_experiment_tag(
            experiment.experiment_id, "mlflow.experimentKind", "genai_development"
        )
        mlflow.set_experiment_tag(
            "mlflow.note.content",
            "Sample experiment with pre-populated demo data including traces, evaluations, "
            "and prompts. Explore MLflow's GenAI features with this experiment.",
        )

        v1_result = self._generate_trace_set("v1")
        v2_result = self._generate_trace_set("v2")

        all_trace_ids = v1_result.trace_ids + v2_result.trace_ids

        # Store the overall time range of demo data as experiment tags
        overall_start_ms = min(v1_result.start_time_ns, v2_result.start_time_ns) // 1_000_000
        overall_end_ms = max(v1_result.end_time_ns, v2_result.end_time_ns) // 1_000_000
        mlflow.set_experiment_tag(DEMO_START_TIME_TAG, str(overall_start_ms))
        mlflow.set_experiment_tag(DEMO_END_TIME_TAG, str(overall_end_ms))

        return DemoResult(
            feature=self.name,
            entity_ids=all_trace_ids,
            navigation_url=f"#/experiments/{experiment.experiment_id}",
        )

    def _generate_trace_set(self, version: Literal["v1", "v2"]) -> _TraceSetResult:
        """Generate a complete set of traces for the given version."""
        trace_ids = []
        trace_index = 0
        min_start_ns = float("inf")
        max_end_ns = 0

        for trace_def in RAG_TRACES:
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            min_start_ns = min(min_start_ns, start_ns)
            max_end_ns = max(max_end_ns, end_ns)
            if trace_id := self._create_rag_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        for trace_def in AGENT_TRACES:
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            min_start_ns = min(min_start_ns, start_ns)
            max_end_ns = max(max_end_ns, end_ns)
            if trace_id := self._create_agent_trace(trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        for idx, trace_def in enumerate(PROMPT_TRACES):
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            min_start_ns = min(min_start_ns, start_ns)
            max_end_ns = max(max_end_ns, end_ns)
            prompt_version_num = str(idx % 2 + 1) if version == "v1" else str(idx % 2 + 3)
            if trace_id := self._create_prompt_trace(
                trace_def, version, start_ns, end_ns, prompt_version_num
            ):
                trace_ids.append(trace_id)
            trace_index += 1

        for idx, trace_def in enumerate(MULTI_AGENT_TRACES):
            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            min_start_ns = min(min_start_ns, start_ns)
            max_end_ns = max(max_end_ns, end_ns)
            creators = [
                self._create_supervisor_worker_trace,
                self._create_triage_handoff_trace,
                self._create_hierarchical_multi_agent_trace,
            ]
            if trace_id := creators[idx](trace_def, version, start_ns, end_ns):
                trace_ids.append(trace_id)
            trace_index += 1

        session_result = self._create_session_traces(version, trace_index)
        trace_ids.extend(session_result.trace_ids)
        min_start_ns = min(min_start_ns, session_result.start_time_ns)
        max_end_ns = max(max_end_ns, session_result.end_time_ns)

        return _TraceSetResult(
            trace_ids=trace_ids,
            start_time_ns=int(min_start_ns),
            end_time_ns=int(max_end_ns),
        )

    def _data_exists(self) -> bool:
        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None or experiment.lifecycle_stage != "active":
                return False
            traces = mlflow.search_traces(
                locations=[experiment.experiment_id],
                max_results=1,
            )
            return len(traces) > 0
        except Exception:
            _logger.debug("Failed to check if demo data exists", exc_info=True)
            return False

    def delete_demo(self) -> None:
        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None:
                return
            client = mlflow.MlflowClient()
            traces = client.search_traces(
                locations=[experiment.experiment_id],
                max_results=200,
            )
            if trace_ids := [trace.info.trace_id for trace in traces]:
                try:
                    client.delete_traces(
                        experiment_id=experiment.experiment_id,
                        trace_ids=trace_ids,
                    )
                except Exception:
                    pass
        except Exception:
            _logger.debug("Failed to delete demo traces", exc_info=True)

    def _restore_experiment_if_deleted(self) -> None:
        """Restore the demo experiment if it was soft-deleted."""
        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is not None and experiment.lifecycle_stage == "deleted":
                _logger.info("Restoring soft-deleted demo experiment")
                client = mlflow.MlflowClient()
                client.restore_experiment(experiment.experiment_id)
        except Exception:
            _logger.debug("Failed to check/restore demo experiment", exc_info=True)

    def _get_response(self, trace_def: DemoTrace, version: Literal["v1", "v2"]) -> str:
        """Get the appropriate response based on version."""
        return trace_def.v1_response if version == "v1" else trace_def.v2_response

    def _create_rag_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a RAG pipeline trace: embed -> retrieve -> generate."""
        response = self._get_response(trace_def, version)
        prompt_tokens = _estimate_tokens(trace_def.query) + 50
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        embed_end = start_ns + int(total_duration * 0.1)
        retrieve_end = embed_end + int(total_duration * 0.2)
        llm_start = retrieve_end
        llm_end = end_ns - int(total_duration * 0.05)

        root = mlflow.start_span_no_context(
            name="rag_pipeline",
            span_type=SpanType.CHAIN,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "rag"},
            start_time_ns=start_ns,
        )

        embed = mlflow.start_span_no_context(
            name="embed_query",
            span_type=SpanType.EMBEDDING,
            parent_span=root,
            inputs={"text": trace_def.query},
            start_time_ns=start_ns + 1000,
        )
        embedding = [random.uniform(-1, 1) for _ in range(384)]
        embed.set_outputs({"embedding": embedding[:5], "dimensions": 384})
        embed.end(end_time_ns=embed_end)

        retrieve = mlflow.start_span_no_context(
            name="retrieve_docs",
            span_type=SpanType.RETRIEVER,
            parent_span=root,
            inputs={"embedding": embedding[:5], "top_k": 3},
            start_time_ns=embed_end + 1000,
        )
        docs = [
            {"id": f"doc_{i}", "score": round(0.7 + random.uniform(0, 0.25), 2)} for i in range(3)
        ]
        retrieve.set_outputs({"documents": docs})
        retrieve.end(end_time_ns=retrieve_end)

        model = GPT_5_2
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are an MLflow assistant."},
                    {"role": "user", "content": trace_def.query},
                ],
                "context": docs,
                "model": model.name,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model.name,
                SpanAttributeKey.MODEL_PROVIDER: model.provider,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=llm_end)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id

    def _create_agent_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        response = self._get_response(trace_def, version)
        prompt_tokens = _estimate_tokens(trace_def.query) + 100
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        tool_duration = int(total_duration * 0.3)
        llm_start = start_ns + tool_duration + 10000

        root = mlflow.start_span_no_context(
            name="agent",
            span_type=SpanType.AGENT,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "agent"},
            start_time_ns=start_ns,
        )

        tool_start = start_ns + 5000
        for tool in trace_def.tools:
            tool_span = mlflow.start_span_no_context(
                name=tool.name,
                span_type=SpanType.TOOL,
                parent_span=root,
                inputs=tool.input,
                start_time_ns=tool_start,
            )
            tool_span.set_outputs(tool.output)
            tool_span.end(end_time_ns=tool_start + tool_duration // len(trace_def.tools))
            tool_start += tool_duration // len(trace_def.tools) + 1000

        model = CLAUDE_SONNET_4_5
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant with tools."},
                    {"role": "user", "content": trace_def.query},
                ],
                "tool_results": [t.output for t in trace_def.tools],
                "model": model.name,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model.name,
                SpanAttributeKey.MODEL_PROVIDER: model.provider,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id

    def _create_prompt_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
        prompt_version: str = "1",
    ) -> str | None:
        """Create a prompt-based trace showing template rendering and generation.

        Fetches the actual registered prompt template and renders it with appropriate
        variables to ensure trace contents match the linked prompt version.
        """
        response = self._get_response(trace_def, version)

        if trace_def.prompt_template is None:
            return None

        full_prompt_name = f"{DEMO_PROMPT_PREFIX}.prompts.{trace_def.prompt_template.prompt_name}"
        try:
            client = mlflow.MlflowClient()
            prompt_version_obj = client.get_prompt_version(
                name=full_prompt_name,
                version=prompt_version,
            )
            actual_template = prompt_version_obj.template
        except Exception:
            actual_template = trace_def.prompt_template.template

        variables = self._get_prompt_variables(
            trace_def.prompt_template.prompt_name,
            trace_def.query,
            trace_def.prompt_template.variables,
        )

        rendered_prompt = self._render_template(actual_template, variables)
        prompt_tokens = _estimate_tokens(rendered_prompt) + 20
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        render_end = start_ns + int(total_duration * 0.1)
        llm_start = render_end + 1000

        root = mlflow.start_span_no_context(
            name="prompt_chain",
            span_type=SpanType.CHAIN,
            inputs={
                "query": trace_def.query,
                "template_variables": variables,
            },
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "prompt"},
            start_time_ns=start_ns,
        )

        render = mlflow.start_span_no_context(
            name="render_prompt",
            span_type=SpanType.CHAIN,
            parent_span=root,
            inputs={
                "template": actual_template,
                "variables": variables,
            },
            start_time_ns=start_ns + 1000,
        )
        render.set_outputs({"rendered_prompt": rendered_prompt})
        render.end(end_time_ns=render_end)

        model = GEMINI_3_PRO
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "user", "content": rendered_prompt},
                ],
                "model": model.name,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model.name,
                SpanAttributeKey.MODEL_PROVIDER: model.provider,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"response": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        trace_id = root.trace_id

        self._link_prompt_to_trace(trace_def.prompt_template.prompt_name, trace_id, prompt_version)

        return trace_id

    def _link_prompt_to_trace(
        self, short_prompt_name: str, trace_id: str, prompt_version: str = "1"
    ) -> None:
        full_prompt_name = f"{DEMO_PROMPT_PREFIX}.prompts.{short_prompt_name}"
        try:
            client = mlflow.MlflowClient()
            prompt_version_obj = client.get_prompt_version(
                name=full_prompt_name,
                version=prompt_version,
            )
            client.link_prompt_versions_to_trace(
                prompt_versions=[prompt_version_obj],
                trace_id=trace_id,
            )
        except Exception:
            _logger.debug(
                "Failed to link prompt %s v%s to trace %s",
                full_prompt_name,
                prompt_version,
                trace_id,
                exc_info=True,
            )

    def _get_prompt_variables(
        self, prompt_name: str, query: str, base_variables: dict[str, str]
    ) -> dict[str, str]:
        """Get complete variable set for a prompt type.

        Combines base variables from the trace definition with additional
        variables that may be needed for more advanced prompt versions.
        """
        variables = dict(base_variables)

        if "query" not in variables:
            variables["query"] = query

        if prompt_name == "customer-support":
            variables.setdefault("company_name", "TechCorp")
            variables.setdefault("context", "Customer has been with us for 2 years, premium tier.")
        elif prompt_name == "document-summarizer":
            variables.setdefault("max_words", "150")
            variables.setdefault("audience", "technical professionals")
            variables.setdefault(
                "document",
                variables.get("query", "Sample document content for summarization."),
            )
        elif prompt_name == "code-reviewer":
            variables.setdefault("language", "python")
            variables.setdefault("focus_areas", "security, performance, readability")
            variables.setdefault("severity_levels", "critical, warning, suggestion")
            variables.setdefault("code", variables.get("query", "def example(): pass"))

        return variables

    def _render_template(
        self, template: str | list[dict[str, str]], variables: dict[str, str]
    ) -> str:
        """Render a prompt template with variables.

        Handles both string templates and chat-format templates (list of messages).
        """

        def substitute(text: str, vars_dict: dict[str, str]) -> str:
            for key, value in vars_dict.items():
                text = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), text)
            return text

        if isinstance(template, str):
            return substitute(template, variables)
        elif isinstance(template, list):
            rendered_parts = []
            for msg in template:
                role = msg.get("role", "user")
                content = substitute(msg.get("content", ""), variables)
                rendered_parts.append(f"[{role}]: {content}")
            return "\n\n".join(rendered_parts)
        else:
            return str(template)

    def _create_supervisor_worker_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a LangGraph-style supervisor/worker trace.

        Supervisor delegates to a researcher agent (web_search + analyze tools),
        then to a writer agent (format_markdown tool), then finishes.

        Span tree (~18 spans, depth 3-4):
            LangGraph (CHAIN)
              ├── supervisor (CHAIN) → ChatOpenAI (CHAT_MODEL)
              ├── researcher (CHAIN)
              │     ├── agent (CHAIN) → ChatOpenAI (CHAT_MODEL)
              │     ├── tools (CHAIN) → web_search (TOOL)
              │     ├── agent (CHAIN) → ChatOpenAI (CHAT_MODEL)
              │     ├── tools (CHAIN) → analyze_results (TOOL)
              │     └── agent (CHAIN) → ChatOpenAI (CHAT_MODEL)
              ├── supervisor (CHAIN) → ChatOpenAI (CHAT_MODEL)
              ├── writer (CHAIN)
              │     ├── agent (CHAIN) → ChatOpenAI (CHAT_MODEL)
              │     ├── tools (CHAIN) → format_markdown (TOOL)
              │     └── agent (CHAIN) → ChatOpenAI (CHAT_MODEL)
              └── supervisor (CHAIN) → ChatOpenAI (CHAT_MODEL)
        """
        response = self._get_response(trace_def, version)
        total = end_ns - start_ns

        # Allocate time across phases (supervisor1, researcher, supervisor2, writer, supervisor3)
        phase_pcts = [0.05, 0.35, 0.05, 0.35, 0.05]
        phase_boundaries = []
        t = start_ns
        for pct in phase_pcts:
            phase_start = t
            t += int(total * pct)
            phase_boundaries.append((phase_start, t))

        model = GPT_5_2

        # Root: LangGraph
        root = mlflow.start_span_no_context(
            name="LangGraph",
            span_type=SpanType.CHAIN,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "multi_agent"},
            start_time_ns=start_ns,
        )

        def _llm_span(parent, name, messages, start, end, prompt_tok, comp_tok):
            s = mlflow.start_span_no_context(
                name=name,
                span_type=SpanType.CHAT_MODEL,
                parent_span=parent,
                inputs={"messages": messages},
                attributes={
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": prompt_tok,
                        "output_tokens": comp_tok,
                        "total_tokens": prompt_tok + comp_tok,
                    },
                    SpanAttributeKey.MODEL: model.name,
                    SpanAttributeKey.MODEL_PROVIDER: model.provider,
                    SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tok, comp_tok),
                },
                start_time_ns=start,
            )
            s.set_outputs({"message": {"role": "assistant", "content": "..."}})
            s.end(end_time_ns=end)

        # --- Phase 1: Supervisor decides to delegate to researcher ---
        p1_s, p1_e = phase_boundaries[0]
        sup1 = mlflow.start_span_no_context(
            name="supervisor", span_type=SpanType.CHAIN, parent_span=root,
            inputs={"messages": [{"role": "user", "content": trace_def.query}]},
            start_time_ns=p1_s,
        )
        _llm_span(
            sup1, "ChatOpenAI",
            [{"role": "system", "content": "Route to researcher or writer."},
             {"role": "user", "content": trace_def.query}],
            p1_s + 1000, p1_e - 1000, 85, 12,
        )
        sup1.set_outputs({"next": "researcher"})
        sup1.end(end_time_ns=p1_e)

        # --- Phase 2: Researcher sub-agent (agent→tool→agent→tool→agent loop) ---
        p2_s, p2_e = phase_boundaries[1]
        researcher = mlflow.start_span_no_context(
            name="researcher", span_type=SpanType.CHAIN, parent_span=root,
            inputs={"task": "research quantum computing developments"},
            start_time_ns=p2_s,
        )
        sub_dur = (p2_e - p2_s) // 5

        # researcher: agent → decides to search
        ra1 = mlflow.start_span_no_context(
            name="agent", span_type=SpanType.CHAIN, parent_span=researcher,
            inputs={"messages": [{"role": "user", "content": "Research quantum computing"}]},
            start_time_ns=p2_s,
        )
        _llm_span(ra1, "ChatOpenAI", [{"role": "user", "content": "Research quantum computing"}],
                   p2_s + 500, p2_s + sub_dur - 500, 120, 25)
        ra1.set_outputs({"tool_calls": [{"name": "web_search"}]})
        ra1.end(end_time_ns=p2_s + sub_dur)

        # researcher: tools → web_search
        rt1 = mlflow.start_span_no_context(
            name="tools", span_type=SpanType.CHAIN, parent_span=researcher,
            inputs={"tool_calls": [{"name": "web_search"}]},
            start_time_ns=p2_s + sub_dur,
        )
        ws = mlflow.start_span_no_context(
            name="web_search", span_type=SpanType.TOOL, parent_span=rt1,
            inputs={"query": "quantum computing breakthroughs 2025"},
            start_time_ns=p2_s + sub_dur + 500,
        )
        ws.set_outputs({
            "results": [
                {"title": "Google Willow chip achieves error correction milestone", "url": "..."},
                {"title": "IBM Heron processor demonstrates quantum advantage", "url": "..."},
                {"title": "Qiskit 2.0 launches with hybrid workflow support", "url": "..."},
            ]
        })
        ws.end(end_time_ns=p2_s + 2 * sub_dur - 500)
        rt1.set_outputs({"results": "3 search results"})
        rt1.end(end_time_ns=p2_s + 2 * sub_dur)

        # researcher: agent → decides to analyze
        ra2 = mlflow.start_span_no_context(
            name="agent", span_type=SpanType.CHAIN, parent_span=researcher,
            inputs={"messages": [{"role": "tool", "content": "3 search results"}]},
            start_time_ns=p2_s + 2 * sub_dur,
        )
        _llm_span(ra2, "ChatOpenAI", [{"role": "tool", "content": "search results..."}],
                   p2_s + 2 * sub_dur + 500, p2_s + 3 * sub_dur - 500, 280, 30)
        ra2.set_outputs({"tool_calls": [{"name": "analyze_results"}]})
        ra2.end(end_time_ns=p2_s + 3 * sub_dur)

        # researcher: tools → analyze_results
        rt2 = mlflow.start_span_no_context(
            name="tools", span_type=SpanType.CHAIN, parent_span=researcher,
            inputs={"tool_calls": [{"name": "analyze_results"}]},
            start_time_ns=p2_s + 3 * sub_dur,
        )
        ar = mlflow.start_span_no_context(
            name="analyze_results", span_type=SpanType.TOOL, parent_span=rt2,
            inputs={"documents": ["Google Willow...", "IBM Heron...", "Qiskit 2.0..."]},
            start_time_ns=p2_s + 3 * sub_dur + 500,
        )
        ar.set_outputs({
            "analysis": {
                "themes": ["error correction", "quantum advantage", "open-source tooling"],
                "key_facts": 8,
                "confidence": 0.92,
            }
        })
        ar.end(end_time_ns=p2_s + 4 * sub_dur - 500)
        rt2.set_outputs({"results": "analysis complete"})
        rt2.end(end_time_ns=p2_s + 4 * sub_dur)

        # researcher: agent → final summary
        ra3 = mlflow.start_span_no_context(
            name="agent", span_type=SpanType.CHAIN, parent_span=researcher,
            inputs={"messages": [{"role": "tool", "content": "analysis complete"}]},
            start_time_ns=p2_s + 4 * sub_dur,
        )
        _llm_span(ra3, "ChatOpenAI", [{"role": "tool", "content": "analysis..."}],
                   p2_s + 4 * sub_dur + 500, p2_e - 1000, 350, 180)
        ra3.set_outputs({"response": "Research findings compiled."})
        ra3.end(end_time_ns=p2_e)

        researcher.set_outputs({"research_summary": "3 key breakthroughs identified"})
        researcher.end(end_time_ns=p2_e)

        # --- Phase 3: Supervisor re-evaluates, delegates to writer ---
        p3_s, p3_e = phase_boundaries[2]
        sup2 = mlflow.start_span_no_context(
            name="supervisor", span_type=SpanType.CHAIN, parent_span=root,
            inputs={"messages": [{"role": "tool", "content": "research complete"}]},
            start_time_ns=p3_s,
        )
        _llm_span(sup2, "ChatOpenAI",
                   [{"role": "tool", "content": "Research findings compiled."}],
                   p3_s + 1000, p3_e - 1000, 200, 10)
        sup2.set_outputs({"next": "writer"})
        sup2.end(end_time_ns=p3_e)

        # --- Phase 4: Writer sub-agent (agent→tool→agent loop) ---
        p4_s, p4_e = phase_boundaries[3]
        writer = mlflow.start_span_no_context(
            name="writer", span_type=SpanType.CHAIN, parent_span=root,
            inputs={"task": "write blog post from research findings"},
            start_time_ns=p4_s,
        )
        w_sub = (p4_e - p4_s) // 3

        # writer: agent → decides to format
        wa1 = mlflow.start_span_no_context(
            name="agent", span_type=SpanType.CHAIN, parent_span=writer,
            inputs={"messages": [{"role": "user", "content": "Write blog post"}]},
            start_time_ns=p4_s,
        )
        _llm_span(wa1, "ChatOpenAI", [{"role": "user", "content": "Write blog post..."}],
                   p4_s + 500, p4_s + w_sub - 500, 400, 350)
        wa1.set_outputs({"tool_calls": [{"name": "format_markdown"}]})
        wa1.end(end_time_ns=p4_s + w_sub)

        # writer: tools → format_markdown
        wt1 = mlflow.start_span_no_context(
            name="tools", span_type=SpanType.CHAIN, parent_span=writer,
            inputs={"tool_calls": [{"name": "format_markdown"}]},
            start_time_ns=p4_s + w_sub,
        )
        fm = mlflow.start_span_no_context(
            name="format_markdown", span_type=SpanType.TOOL, parent_span=wt1,
            inputs={"content": "draft blog post text...", "style": "technical-blog"},
            start_time_ns=p4_s + w_sub + 500,
        )
        fm.set_outputs({"formatted": "## Quantum Computing in 2025..."})
        fm.end(end_time_ns=p4_s + 2 * w_sub - 500)
        wt1.set_outputs({"results": "formatted"})
        wt1.end(end_time_ns=p4_s + 2 * w_sub)

        # writer: agent → final output
        wa2 = mlflow.start_span_no_context(
            name="agent", span_type=SpanType.CHAIN, parent_span=writer,
            inputs={"messages": [{"role": "tool", "content": "formatted markdown"}]},
            start_time_ns=p4_s + 2 * w_sub,
        )
        _llm_span(wa2, "ChatOpenAI", [{"role": "tool", "content": "formatted..."}],
                   p4_s + 2 * w_sub + 500, p4_e - 1000, 500, 280)
        wa2.set_outputs({"response": response})
        wa2.end(end_time_ns=p4_e)

        writer.set_outputs({"blog_post": response})
        writer.end(end_time_ns=p4_e)

        # --- Phase 5: Supervisor finishes ---
        p5_s, p5_e = phase_boundaries[4]
        sup3 = mlflow.start_span_no_context(
            name="supervisor", span_type=SpanType.CHAIN, parent_span=root,
            inputs={"messages": [{"role": "tool", "content": "blog post written"}]},
            start_time_ns=p5_s,
        )
        _llm_span(sup3, "ChatOpenAI",
                   [{"role": "tool", "content": "Blog post complete."}],
                   p5_s + 1000, p5_e - 1000, 150, 8)
        sup3.set_outputs({"next": "FINISH"})
        sup3.end(end_time_ns=p5_e)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)
        return root.trace_id

    def _create_triage_handoff_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create an OpenAI Agents SDK-style triage + handoff trace.

        A triage agent routes to a billing specialist that uses tools in a loop.

        Span tree (~12 spans, depth 2-3):
            AgentRunner.run (AGENT)
              ├── input_guardrail (CHAIN)
              ├── Triage Agent (AGENT)
              │     ├── Response (CHAT_MODEL)
              │     └── Handoff (CHAIN)
              └── Billing Agent (AGENT)
                    ├── Response (CHAT_MODEL) — calls get_subscription
                    ├── get_subscription (TOOL)
                    ├── Response (CHAT_MODEL) — calls process_cancellation
                    ├── process_cancellation (TOOL)
                    ├── Response (CHAT_MODEL) — calls process_refund
                    ├── process_refund (TOOL)
                    └── Response (CHAT_MODEL) — final answer
        """
        response = self._get_response(trace_def, version)
        total = end_ns - start_ns
        model = CLAUDE_SONNET_4_5

        def _response_span(parent, messages, start, end, prompt_tok, comp_tok, outputs=None):
            s = mlflow.start_span_no_context(
                name="Response", span_type=SpanType.CHAT_MODEL, parent_span=parent,
                inputs={"messages": messages},
                attributes={
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": prompt_tok,
                        "output_tokens": comp_tok,
                        "total_tokens": prompt_tok + comp_tok,
                    },
                    SpanAttributeKey.MODEL: model.name,
                    SpanAttributeKey.MODEL_PROVIDER: model.provider,
                    SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tok, comp_tok),
                },
                start_time_ns=start,
            )
            s.set_outputs(outputs or {"message": {"role": "assistant", "content": "..."}})
            s.end(end_time_ns=end)

        # Root: AgentRunner.run
        root = mlflow.start_span_no_context(
            name="AgentRunner.run", span_type=SpanType.AGENT,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "multi_agent"},
            start_time_ns=start_ns,
        )

        # Phase boundaries: guardrail(3%), triage(10%), billing(82%), pad(5%)
        guard_end = start_ns + int(total * 0.03)
        triage_end = guard_end + int(total * 0.10)
        billing_end = end_ns - int(total * 0.05)

        # --- Input guardrail ---
        guardrail = mlflow.start_span_no_context(
            name="content_policy_guardrail", span_type=SpanType.CHAIN, parent_span=root,
            inputs={"text": trace_def.query},
            start_time_ns=start_ns + 500,
        )
        guardrail.set_outputs({"triggered": False, "reason": "content acceptable"})
        guardrail.end(end_time_ns=guard_end)

        # --- Triage Agent ---
        triage = mlflow.start_span_no_context(
            name="Triage Agent", span_type=SpanType.AGENT, parent_span=root,
            inputs={"messages": [{"role": "user", "content": trace_def.query}]},
            start_time_ns=guard_end + 500,
        )
        _response_span(
            triage,
            [{"role": "system", "content": "Route to billing, support, or sales."},
             {"role": "user", "content": trace_def.query}],
            guard_end + 1000, triage_end - int(total * 0.02), 90, 15,
            outputs={"tool_calls": [{"name": "transfer_to_billing_agent"}]},
        )
        handoff = mlflow.start_span_no_context(
            name="Handoff", span_type=SpanType.CHAIN, parent_span=triage,
            inputs={"from_agent": "Triage Agent"},
            start_time_ns=triage_end - int(total * 0.02) + 500,
        )
        handoff.set_outputs({"to_agent": "Billing Agent"})
        handoff.end(end_time_ns=triage_end)
        triage.set_outputs({"handoff": "Billing Agent"})
        triage.end(end_time_ns=triage_end)

        # --- Billing Agent (Response/Tool loop) ---
        billing = mlflow.start_span_no_context(
            name="Billing Agent", span_type=SpanType.AGENT, parent_span=root,
            inputs={
                "messages": [{"role": "user", "content": trace_def.query}],
                "handoff_from": "Triage Agent",
            },
            start_time_ns=triage_end + 500,
        )
        billing_dur = billing_end - triage_end
        step = billing_dur // 7  # 7 child spans

        t = triage_end + 1000

        # Response 1: decides to call get_subscription
        _response_span(billing,
                        [{"role": "user", "content": trace_def.query}],
                        t, t + step - 500, 120, 20,
                        outputs={"tool_calls": [{"name": "get_subscription"}]})
        t += step

        # get_subscription tool
        gs = mlflow.start_span_no_context(
            name="get_subscription", span_type=SpanType.TOOL, parent_span=billing,
            inputs={"user_id": "usr_2847", "include_history": True},
            start_time_ns=t,
        )
        gs.set_outputs({
            "plan": "Premium", "price": 29.99, "status": "active",
            "billing_date": "2025-03-15", "next_billing": "2025-04-15",
        })
        gs.end(end_time_ns=t + step - 500)
        t += step

        # Response 2: decides to cancel
        _response_span(billing,
                        [{"role": "tool", "content": "subscription details..."}],
                        t, t + step - 500, 200, 18,
                        outputs={"tool_calls": [{"name": "process_cancellation"}]})
        t += step

        # process_cancellation tool
        pc = mlflow.start_span_no_context(
            name="process_cancellation", span_type=SpanType.TOOL, parent_span=billing,
            inputs={"user_id": "usr_2847", "effective": "immediate", "retain_access_until": "2025-04-15"},
            start_time_ns=t,
        )
        pc.set_outputs({"status": "cancelled", "confirmation_id": "CAN-2025-4421"})
        pc.end(end_time_ns=t + step - 500)
        t += step

        # Response 3: decides to refund
        _response_span(billing,
                        [{"role": "tool", "content": "cancellation confirmed"}],
                        t, t + step - 500, 250, 15,
                        outputs={"tool_calls": [{"name": "process_refund"}]})
        t += step

        # process_refund tool
        pr = mlflow.start_span_no_context(
            name="process_refund", span_type=SpanType.TOOL, parent_span=billing,
            inputs={"user_id": "usr_2847", "charge_date": "2025-03-15", "amount": 29.99},
            start_time_ns=t,
        )
        pr.set_outputs({"refund_id": "REF-2025-8847", "amount": 29.99, "eta_days": "3-5"})
        pr.end(end_time_ns=t + step - 500)
        t += step

        # Response 4: final answer
        _response_span(billing,
                        [{"role": "tool", "content": "refund processed"}],
                        t, billing_end - 500, 300, _estimate_tokens(response),
                        outputs={"message": {"role": "assistant", "content": response}})

        billing.set_outputs({"response": response})
        billing.end(end_time_ns=billing_end)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)
        return root.trace_id

    def _create_hierarchical_multi_agent_trace(
        self,
        trace_def: DemoTrace,
        version: Literal["v1", "v2"],
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a hierarchical multi-agent trace (supervisor of supervisors).

        Top supervisor → research_team (team_supervisor → web_researcher + data_analyst)
        → report_writer.

        Span tree (~28 spans, depth 4-5):
            LangGraph (CHAIN)
              ├── top_supervisor (CHAIN) → ChatOpenAI
              ├── research_team (CHAIN)
              │     ├── team_supervisor (CHAIN) → ChatOpenAI
              │     ├── web_researcher (CHAIN)
              │     │     ├── agent (CHAIN) → ChatOpenAI
              │     │     ├── tools (CHAIN) → search_competitors (TOOL)
              │     │     └── agent (CHAIN) → ChatOpenAI
              │     ├── team_supervisor (CHAIN) → ChatOpenAI
              │     ├── data_analyst (CHAIN)
              │     │     ├── agent (CHAIN) → ChatOpenAI
              │     │     ├── tools (CHAIN) → query_sales_db (TOOL)
              │     │     ├── agent (CHAIN) → ChatOpenAI
              │     │     ├── tools (CHAIN) → calculate_metrics (TOOL)
              │     │     └── agent (CHAIN) → ChatOpenAI
              │     └── team_supervisor (CHAIN) → ChatOpenAI
              ├── top_supervisor (CHAIN) → ChatOpenAI
              ├── report_writer (CHAIN)
              │     ├── agent (CHAIN) → ChatOpenAI
              │     └── agent (CHAIN) → ChatOpenAI
              └── top_supervisor (CHAIN) → ChatOpenAI
        """
        response = self._get_response(trace_def, version)
        total = end_ns - start_ns
        model = GEMINI_3_PRO

        def _llm(parent, start, end, prompt_tok, comp_tok, messages=None):
            s = mlflow.start_span_no_context(
                name="ChatOpenAI", span_type=SpanType.CHAT_MODEL, parent_span=parent,
                inputs={"messages": messages or []},
                attributes={
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": prompt_tok, "output_tokens": comp_tok,
                        "total_tokens": prompt_tok + comp_tok,
                    },
                    SpanAttributeKey.MODEL: model.name,
                    SpanAttributeKey.MODEL_PROVIDER: model.provider,
                    SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tok, comp_tok),
                },
                start_time_ns=start,
            )
            s.set_outputs({"message": {"role": "assistant", "content": "..."}})
            s.end(end_time_ns=end)

        def _node(name, parent, start, end, inputs=None, outputs=None):
            s = mlflow.start_span_no_context(
                name=name, span_type=SpanType.CHAIN, parent_span=parent,
                inputs=inputs or {}, start_time_ns=start,
            )
            if outputs:
                s.set_outputs(outputs)
            return s

        # Phase allocation: top_sup1(4%), research_team(42%), top_sup2(4%),
        #                    report_writer(42%), top_sup3(4%), padding(4%)
        def _ts(pct):
            return start_ns + int(total * pct)

        root = mlflow.start_span_no_context(
            name="LangGraph", span_type=SpanType.CHAIN,
            inputs={"query": trace_def.query},
            metadata={DEMO_VERSION_TAG: version, DEMO_TRACE_TYPE_TAG: "multi_agent"},
            start_time_ns=start_ns,
        )

        # --- Top supervisor 1: delegate to research_team ---
        ts1 = _node("top_supervisor", root, _ts(0), _ts(0.04))
        _llm(ts1, _ts(0.005), _ts(0.035), 100, 12,
             [{"role": "user", "content": trace_def.query}])
        ts1.set_outputs({"next": "research_team"})
        ts1.end(end_time_ns=_ts(0.04))

        # --- Research team subgraph ---
        rt = _node("research_team", root, _ts(0.04), _ts(0.46),
                    inputs={"task": "analyze Q1 sales and competitor pricing"})

        # team_supervisor 1: delegate to web_researcher
        tsup1 = _node("team_supervisor", rt, _ts(0.04), _ts(0.07))
        _llm(tsup1, _ts(0.045), _ts(0.065), 90, 10)
        tsup1.set_outputs({"next": "web_researcher"})
        tsup1.end(end_time_ns=_ts(0.07))

        # web_researcher subgraph
        wr = _node("web_researcher", rt, _ts(0.07), _ts(0.20),
                    inputs={"task": "research competitor pricing"})

        wr_a1 = _node("agent", wr, _ts(0.07), _ts(0.10))
        _llm(wr_a1, _ts(0.075), _ts(0.095), 110, 20)
        wr_a1.set_outputs({"tool_calls": [{"name": "search_competitors"}]})
        wr_a1.end(end_time_ns=_ts(0.10))

        wr_t1 = _node("tools", wr, _ts(0.10), _ts(0.15))
        sc = mlflow.start_span_no_context(
            name="search_competitors", span_type=SpanType.TOOL, parent_span=wr_t1,
            inputs={"industry": "SaaS", "segment": "mid-market"},
            start_time_ns=_ts(0.105),
        )
        sc.set_outputs({
            "competitors": [
                {"name": "CompetitorA", "price": "$24/mo", "change": "-20% last quarter"},
                {"name": "CompetitorB", "price": "freemium", "change": "new tier launched"},
            ]
        })
        sc.end(end_time_ns=_ts(0.145))
        wr_t1.set_outputs({"results": "2 competitors found"})
        wr_t1.end(end_time_ns=_ts(0.15))

        wr_a2 = _node("agent", wr, _ts(0.15), _ts(0.20))
        _llm(wr_a2, _ts(0.155), _ts(0.195), 250, 120)
        wr_a2.set_outputs({"response": "Competitor analysis complete"})
        wr_a2.end(end_time_ns=_ts(0.20))

        wr.set_outputs({"analysis": "CompetitorA undercut by 20%, CompetitorB launched freemium"})
        wr.end(end_time_ns=_ts(0.20))

        # team_supervisor 2: delegate to data_analyst
        tsup2 = _node("team_supervisor", rt, _ts(0.20), _ts(0.23))
        _llm(tsup2, _ts(0.205), _ts(0.225), 180, 10)
        tsup2.set_outputs({"next": "data_analyst"})
        tsup2.end(end_time_ns=_ts(0.23))

        # data_analyst subgraph (deeper: agent→tool→agent→tool→agent)
        da = _node("data_analyst", rt, _ts(0.23), _ts(0.42),
                    inputs={"task": "analyze Q1 sales data"})

        da_a1 = _node("agent", da, _ts(0.23), _ts(0.27))
        _llm(da_a1, _ts(0.235), _ts(0.265), 100, 18)
        da_a1.set_outputs({"tool_calls": [{"name": "query_sales_db"}]})
        da_a1.end(end_time_ns=_ts(0.27))

        da_t1 = _node("tools", da, _ts(0.27), _ts(0.32))
        qdb = mlflow.start_span_no_context(
            name="query_sales_db", span_type=SpanType.TOOL, parent_span=da_t1,
            inputs={"query": "SELECT segment, SUM(revenue) FROM sales WHERE quarter='Q1-2025' GROUP BY segment"},
            start_time_ns=_ts(0.275),
        )
        qdb.set_outputs({
            "rows": [
                {"segment": "Enterprise", "revenue": 2800000, "yoy_change": "+22%"},
                {"segment": "Mid-market", "revenue": 1100000, "yoy_change": "-8%"},
                {"segment": "SMB", "revenue": 300000, "yoy_change": "+5%"},
            ]
        })
        qdb.end(end_time_ns=_ts(0.315))
        da_t1.set_outputs({"results": "3 segments returned"})
        da_t1.end(end_time_ns=_ts(0.32))

        da_a2 = _node("agent", da, _ts(0.32), _ts(0.35))
        _llm(da_a2, _ts(0.325), _ts(0.345), 220, 22)
        da_a2.set_outputs({"tool_calls": [{"name": "calculate_metrics"}]})
        da_a2.end(end_time_ns=_ts(0.35))

        da_t2 = _node("tools", da, _ts(0.35), _ts(0.39))
        cm = mlflow.start_span_no_context(
            name="calculate_metrics", span_type=SpanType.TOOL, parent_span=da_t2,
            inputs={"data": "Q1 sales by segment", "metrics": ["yoy_growth", "target_gap", "churn_rate"]},
            start_time_ns=_ts(0.355),
        )
        cm.set_outputs({
            "total_revenue": 4200000, "yoy_growth": "12%", "target": "18%",
            "target_gap": "-6%", "mid_market_churn": "12 accounts lost",
        })
        cm.end(end_time_ns=_ts(0.385))
        da_t2.set_outputs({"results": "metrics calculated"})
        da_t2.end(end_time_ns=_ts(0.39))

        da_a3 = _node("agent", da, _ts(0.39), _ts(0.42))
        _llm(da_a3, _ts(0.395), _ts(0.415), 300, 150)
        da_a3.set_outputs({"response": "Q1 analysis complete: $4.2M revenue, 12% YoY"})
        da_a3.end(end_time_ns=_ts(0.42))

        da.set_outputs({"analysis": "Q1 $4.2M (+12%), mid-market declining"})
        da.end(end_time_ns=_ts(0.42))

        # team_supervisor 3: finish research
        tsup3 = _node("team_supervisor", rt, _ts(0.42), _ts(0.46))
        _llm(tsup3, _ts(0.425), _ts(0.455), 400, 15)
        tsup3.set_outputs({"next": "FINISH"})
        tsup3.end(end_time_ns=_ts(0.46))

        rt.set_outputs({"research": "competitive + sales analysis complete"})
        rt.end(end_time_ns=_ts(0.46))

        # --- Top supervisor 2: delegate to report_writer ---
        ts2 = _node("top_supervisor", root, _ts(0.46), _ts(0.50))
        _llm(ts2, _ts(0.465), _ts(0.495), 450, 12)
        ts2.set_outputs({"next": "report_writer"})
        ts2.end(end_time_ns=_ts(0.50))

        # --- Report writer ---
        rw = _node("report_writer", root, _ts(0.50), _ts(0.92),
                    inputs={"task": "create executive summary with recommendations"})

        rw_a1 = _node("agent", rw, _ts(0.50), _ts(0.70))
        _llm(rw_a1, _ts(0.505), _ts(0.695), 600, 450,
             [{"role": "user", "content": "Write executive summary from research findings"}])
        rw_a1.set_outputs({"draft": "Executive summary draft..."})
        rw_a1.end(end_time_ns=_ts(0.70))

        rw_a2 = _node("agent", rw, _ts(0.70), _ts(0.92))
        _llm(rw_a2, _ts(0.705), _ts(0.915), 700, 320,
             [{"role": "user", "content": "Refine and finalize the report"}])
        rw_a2.set_outputs({"response": response})
        rw_a2.end(end_time_ns=_ts(0.92))

        rw.set_outputs({"report": response})
        rw.end(end_time_ns=_ts(0.92))

        # --- Top supervisor 3: finish ---
        ts3 = _node("top_supervisor", root, _ts(0.92), _ts(0.96))
        _llm(ts3, _ts(0.925), _ts(0.955), 200, 8)
        ts3.set_outputs({"next": "FINISH"})
        ts3.end(end_time_ns=_ts(0.96))

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)
        return root.trace_id

    def _create_session_traces(
        self, version: Literal["v1", "v2"], start_index: int
    ) -> _TraceSetResult:
        """Create multi-turn conversation session traces."""
        trace_ids = []
        current_session = None
        turn_counter = 0
        trace_index = start_index
        min_start_ns = float("inf")
        max_end_ns = 0

        for trace_def in SESSION_TRACES:
            if trace_def.session_id != current_session:
                current_session = trace_def.session_id
                turn_counter = 0

            turn_counter += 1
            versioned_session_id = f"{trace_def.session_id}-{version}"

            start_ns, end_ns = _get_trace_timestamps(trace_index, version)
            min_start_ns = min(min_start_ns, start_ns)
            max_end_ns = max(max_end_ns, end_ns)
            if trace_id := self._create_session_turn_trace(
                trace_def, turn_counter, version, versioned_session_id, start_ns, end_ns
            ):
                trace_ids.append(trace_id)
            trace_index += 1

        return _TraceSetResult(
            trace_ids=trace_ids,
            start_time_ns=int(min_start_ns),
            end_time_ns=int(max_end_ns),
        )

    def _create_session_turn_trace(
        self,
        trace_def: DemoTrace,
        turn: int,
        version: Literal["v1", "v2"],
        versioned_session_id: str,
        start_ns: int,
        end_ns: int,
    ) -> str | None:
        """Create a single turn in a conversation session."""
        response = self._get_response(trace_def, version)
        prompt_tokens = _estimate_tokens(trace_def.query) + 80
        completion_tokens = _estimate_tokens(response)

        total_duration = end_ns - start_ns
        tool_end = start_ns + int(total_duration * 0.3)
        llm_start = tool_end + 1000

        root = mlflow.start_span_no_context(
            name="chat_agent",
            span_type=SpanType.AGENT,
            inputs={"message": trace_def.query, "turn": turn},
            metadata={
                TraceMetadataKey.TRACE_SESSION: versioned_session_id,
                TraceMetadataKey.TRACE_USER: trace_def.session_user or "user",
                DEMO_VERSION_TAG: version,
                DEMO_TRACE_TYPE_TAG: "session",
            },
            start_time_ns=start_ns,
        )

        tool_start = start_ns + 5000
        for tool in trace_def.tools:
            tool_span = mlflow.start_span_no_context(
                name=tool.name,
                span_type=SpanType.TOOL,
                parent_span=root,
                inputs=tool.input,
                start_time_ns=tool_start,
            )
            tool_span.set_outputs(tool.output)
            tool_span.end(end_time_ns=tool_end)
            tool_start = tool_end + 1000

        model = _DEMO_MODELS[turn % len(_DEMO_MODELS)]
        llm = mlflow.start_span_no_context(
            name="generate_response",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={
                "messages": [
                    {"role": "system", "content": "You are an MLflow assistant."},
                    {"role": "user", "content": trace_def.query},
                ],
                "model": model.name,
            },
            attributes={
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                SpanAttributeKey.MODEL: model.name,
                SpanAttributeKey.MODEL_PROVIDER: model.provider,
                SpanAttributeKey.LLM_COST: _compute_cost(model, prompt_tokens, completion_tokens),
            },
            start_time_ns=llm_start,
        )
        llm.set_outputs({"role": "assistant", "content": response})
        llm.end(end_time_ns=end_ns - 5000)

        root.set_outputs({"response": response})
        root.end(end_time_ns=end_ns)

        return root.trace_id
