# Trace Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TraceView as a first-class entity enabling span filtering, JSONPath extraction, natural language summarization, and assistant-driven view creation for MLflow traces.

**Architecture:** TraceView follows the Assessment entity pattern — new dataclass, DB table, protobuf, REST endpoints, store CRUD, and client methods. Span filter + JSONPath utilities are ported from project-0xfffff. The assistant creates views via Python API and signals the frontend with a structured marker. The judge infrastructure powers `trace.summarize()` and `trace.analyze()`.

**Tech Stack:** Python (dataclasses, SQLAlchemy, protobuf, alembic), `jsonpath-ng` (Python), React/TypeScript, `jsonpath-plus` (JS), Dubois design system

**Spec:** `docs/plans/2026-03-25-trace-views-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `mlflow/entities/trace_view.py` | `TraceView` and `SpanFilter` dataclasses with dict/JSON serialization |
| `mlflow/tracing/utils/view_utils.py` | `find_first_matching_span`, `apply_span_filter`, `apply_jsonpath`, `validate_jsonpath`, `apply_view` |
| `mlflow/store/db_migrations/versions/xxx_add_trace_views_table.py` | Alembic migration |
| `tests/tracing/utils/test_view_utils.py` | Unit tests for span filter + JSONPath utilities |
| `tests/store/tracking/test_trace_views.py` | Store CRUD tests |
| `mlflow/server/js/src/shared/web-shared/model-trace-explorer/TraceViewSelector.tsx` | View selector dropdown component |
| `mlflow/server/js/src/shared/web-shared/model-trace-explorer/hooks/useTraceViews.ts` | React Query hook for fetching/managing views |

### Modified Files

| File | Change |
|------|--------|
| `mlflow/store/tracking/dbmodels/models.py` (after line ~1123) | Add `SqlTraceView` ORM model |
| `mlflow/store/tracking/abstract_store.py` (after line ~603) | Add abstract view CRUD methods |
| `mlflow/store/tracking/sqlalchemy_store.py` (after line ~4200) | Implement view CRUD |
| `mlflow/store/tracking/rest_store.py` (after line ~837) | REST proxy for view CRUD |
| `mlflow/server/handlers.py` (after line ~4083, and ~6711) | View endpoint handlers + registration |
| `mlflow/tracking/client.py` | Add `create_trace_view`, `list_trace_views`, etc. |
| `mlflow/tracking/_tracking_service/client.py` | Add service-layer methods |
| `mlflow/entities/trace.py` (after line ~296) | Add `create_view`, `views`, `summarize`, `analyze` convenience methods |
| `pyproject.toml` | Add `jsonpath-ng` as optional dependency |
| `mlflow/server/js/src/shared/web-shared/model-trace-explorer/ModelTraceExplorerContent.tsx` | Integrate TraceViewSelector |
| `mlflow/server/js/src/shared/web-shared/model-trace-explorer/ModelTraceExplorerDetailView.tsx` | Apply view filtering to span tree |
| `mlflow/server/js/src/shared/web-shared/model-trace-explorer/right-pane/ModelTraceExplorerDefaultSpanView.tsx` | Apply JSONPath to inputs/outputs |
| `mlflow/server/js/src/assistant/AssistantContext.tsx` | Parse `[trace_view_created]` markers |
| `mlflow/server/js/package.json` | Add `jsonpath-plus` dependency |

---

## Task 1: SpanFilter and TraceView Entity Classes

**Files:**
- Create: `mlflow/entities/trace_view.py`
- Test: `tests/entities/test_trace_view.py`

- [ ] **Step 1: Write tests for SpanFilter and TraceView dataclasses**

```python
# tests/entities/test_trace_view.py
import json
from dataclasses import asdict

import pytest

from mlflow.entities.trace_view import SpanFilter, TraceView
from mlflow.exceptions import MlflowException


class TestSpanFilter:
    def test_create_with_all_fields(self):
        sf = SpanFilter(
            span_name="ChatOpenAI",
            span_type="CHAT_MODEL",
            attribute_key="model",
            attribute_value="gpt-4",
        )
        assert sf.span_name == "ChatOpenAI"
        assert sf.span_type == "CHAT_MODEL"
        assert sf.attribute_key == "model"
        assert sf.attribute_value == "gpt-4"

    def test_create_minimal(self):
        sf = SpanFilter(span_type="TOOL")
        assert sf.span_name is None
        assert sf.span_type == "TOOL"

    def test_to_dict(self):
        sf = SpanFilter(span_type="TOOL")
        d = sf.to_dict()
        assert d == {"span_name": None, "span_type": "TOOL", "attribute_key": None, "attribute_value": None}

    def test_from_dict(self):
        sf = SpanFilter.from_dict({"span_type": "TOOL", "span_name": "search"})
        assert sf.span_type == "TOOL"
        assert sf.span_name == "search"

    def test_to_json_and_back(self):
        sf = SpanFilter(span_name="Retriever", span_type="RETRIEVER")
        json_str = sf.to_json()
        sf2 = SpanFilter.from_json(json_str)
        assert sf == sf2


class TestTraceView:
    def test_create_trace_scoped(self):
        view = TraceView(
            name="Tool Calls",
            trace_id="tr-abc123",
            span_filter=SpanFilter(span_type="TOOL"),
            input_path="$.query",
            output_path="$.result",
            created_by="forrest",
        )
        assert view.name == "Tool Calls"
        assert view.trace_id == "tr-abc123"
        assert view.experiment_id is None
        assert view.span_filter.span_type == "TOOL"

    def test_create_experiment_scoped(self):
        view = TraceView(
            name="SME Review",
            experiment_id="123",
            span_filter=SpanFilter(span_type="CHAT_MODEL"),
        )
        assert view.experiment_id == "123"
        assert view.trace_id is None

    def test_scope_validation_rejects_both(self):
        view = TraceView(name="bad", trace_id="tr-1", experiment_id="1")
        with pytest.raises(MlflowException, match="exactly one"):
            view.validate_scope()

    def test_scope_validation_rejects_neither(self):
        view = TraceView(name="bad")
        with pytest.raises(MlflowException, match="exactly one"):
            view.validate_scope()

    def test_to_dict_round_trip(self):
        view = TraceView(
            name="Test",
            trace_id="tr-abc",
            span_filter=SpanFilter(span_type="TOOL"),
            input_path="$.query",
            view_id="tv-123",
            created_by="user",
        )
        d = view.to_dict()
        view2 = TraceView.from_dict(d)
        assert view2.name == view.name
        assert view2.span_filter.span_type == "TOOL"
        assert view2.input_path == "$.query"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/entities/test_trace_view.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlflow.entities.trace_view'`

- [ ] **Step 3: Implement SpanFilter and TraceView**

```python
# mlflow/entities/trace_view.py
from __future__ import annotations

import json
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class SpanFilter:
    span_name: str | None = None
    span_type: str | None = None
    attribute_key: str | None = None
    attribute_value: str | None = None

    def to_dict(self) -> dict:
        return {
            "span_name": self.span_name,
            "span_type": self.span_type,
            "attribute_key": self.attribute_key,
            "attribute_value": self.attribute_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpanFilter:
        return cls(
            span_name=d.get("span_name"),
            span_type=d.get("span_type"),
            attribute_key=d.get("attribute_key"),
            attribute_value=d.get("attribute_value"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> SpanFilter:
        return cls.from_dict(json.loads(s))


@dataclass
class TraceView:
    name: str
    trace_id: str | None = None
    experiment_id: str | None = None
    span_filter: SpanFilter | None = None
    input_path: str | None = None
    output_path: str | None = None
    created_by: str | None = None
    description: str | None = None
    view_id: str | None = None
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None

    def validate_scope(self):
        """Validate that exactly one scope (trace or experiment) is set.
        Called explicitly before persistence, not in __post_init__,
        to allow intermediate construction (e.g., from_dict, ORM hydration)."""
        has_trace = self.trace_id is not None
        has_experiment = self.experiment_id is not None
        if has_trace == has_experiment:
            raise MlflowException(
                "TraceView must have exactly one of trace_id or experiment_id set.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @property
    def scope(self) -> str:
        return "trace" if self.trace_id is not None else "experiment"

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "experiment_id": self.experiment_id,
            "span_filter": self.span_filter.to_dict() if self.span_filter else None,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "created_by": self.created_by,
            "description": self.description,
            "create_time_ms": self.create_time_ms,
            "last_update_time_ms": self.last_update_time_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceView:
        span_filter = None
        if d.get("span_filter"):
            span_filter = SpanFilter.from_dict(d["span_filter"])
        return cls(
            view_id=d.get("view_id"),
            name=d["name"],
            trace_id=d.get("trace_id"),
            experiment_id=d.get("experiment_id"),
            span_filter=span_filter,
            input_path=d.get("input_path"),
            output_path=d.get("output_path"),
            created_by=d.get("created_by"),
            description=d.get("description"),
            create_time_ms=d.get("create_time_ms"),
            last_update_time_ms=d.get("last_update_time_ms"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/entities/test_trace_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlflow/entities/trace_view.py tests/entities/test_trace_view.py
git commit -s -m "feat: add TraceView and SpanFilter entity classes

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Span Filter & JSONPath Utilities

**Files:**
- Create: `mlflow/tracing/utils/view_utils.py`
- Test: `tests/tracing/utils/test_view_utils.py`
- Modify: `pyproject.toml` (add `jsonpath-ng` optional dep)
- Reference: `/Users/forrest.murray/Documents/project-0xfffff/server/utils/span_filter_utils.py`
- Reference: `/Users/forrest.murray/Documents/project-0xfffff/server/utils/jsonpath_utils.py`

- [ ] **Step 1: Add `jsonpath-ng` as optional dependency**

In `pyproject.toml`, add `jsonpath-ng` to the optional dependencies section (likely under `[project.optional-dependencies]`). Find the appropriate group or create a `genai` group entry.

- [ ] **Step 2: Write tests for span filter utilities**

```python
# tests/tracing/utils/test_view_utils.py
import json

import pytest

from mlflow.entities.trace_view import SpanFilter


# --- Sample data ---

SAMPLE_SPANS_DICTS = [
    {
        "name": "AgentExecutor",
        "span_type": "CHAIN",
        "inputs": {"question": "What is the weather?"},
        "outputs": {"answer": "It is sunny."},
        "attributes": {"framework": "langchain"},
    },
    {
        "name": "ChatOpenAI",
        "span_type": "CHAT_MODEL",
        "inputs": {"messages": [{"role": "user", "content": "What is the weather?"}]},
        "outputs": {"choices": [{"message": {"content": "It is sunny."}}]},
        "attributes": {"model": "gpt-4"},
    },
    {
        "name": "Retriever",
        "span_type": "RETRIEVER",
        "inputs": {"query": "weather"},
        "outputs": {"documents": ["doc1", "doc2"]},
        "attributes": {},
    },
]


class TestFindFirstMatchingSpan:
    def test_match_by_span_name(self):
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        result = find_first_matching_span(
            SAMPLE_SPANS_DICTS, SpanFilter(span_name="ChatOpenAI")
        )
        assert result is not None
        assert result["name"] == "ChatOpenAI"

    def test_match_by_span_type(self):
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        result = find_first_matching_span(
            SAMPLE_SPANS_DICTS, SpanFilter(span_type="RETRIEVER")
        )
        assert result is not None
        assert result["name"] == "Retriever"

    def test_match_by_attribute_key_value(self):
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        result = find_first_matching_span(
            SAMPLE_SPANS_DICTS,
            SpanFilter(attribute_key="model", attribute_value="gpt-4"),
        )
        assert result is not None
        assert result["name"] == "ChatOpenAI"

    def test_combined_filter(self):
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        result = find_first_matching_span(
            SAMPLE_SPANS_DICTS,
            SpanFilter(span_type="CHAT_MODEL", attribute_key="model"),
        )
        assert result is not None
        assert result["name"] == "ChatOpenAI"

    def test_no_match(self):
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        result = find_first_matching_span(
            SAMPLE_SPANS_DICTS, SpanFilter(span_name="NonExistent")
        )
        assert result is None

    def test_empty_spans(self):
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        result = find_first_matching_span([], SpanFilter(span_type="TOOL"))
        assert result is None

    def test_match_by_attribute_in_mlflow_wire_format(self):
        """MLflow raw wire format stores span_type in attributes as a JSON string."""
        from mlflow.tracing.utils.view_utils import find_first_matching_span

        spans = [
            {
                "name": "MyTool",
                "attributes": {"mlflow.spanType": '"TOOL"'},
                "inputs": {},
                "outputs": {},
            }
        ]
        result = find_first_matching_span(spans, SpanFilter(span_type="TOOL"))
        assert result is not None
        assert result["name"] == "MyTool"


class TestApplyJsonpath:
    def test_simple_extraction(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath('{"message": "hello"}', "$.message")
        assert success is True
        assert result == "hello"

    def test_nested_extraction(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath(
            '{"response": {"text": "answer"}}', "$.response.text"
        )
        assert success is True
        assert result == "answer"

    def test_array_wildcard(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        data = '{"messages": [{"content": "a"}, {"content": "b"}]}'
        result, success = apply_jsonpath(data, "$.messages[*].content")
        assert success is True
        assert result == "a\nb"

    def test_no_match(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath('{"foo": "bar"}', "$.missing")
        assert success is False
        assert result is None

    def test_invalid_json(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath("not json", "$.anything")
        assert success is False

    def test_empty_expr(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath('{"x": 1}', "")
        assert success is False

    def test_none_expr(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath('{"x": 1}', None)
        assert success is False

    def test_null_result(self):
        from mlflow.tracing.utils.view_utils import apply_jsonpath

        result, success = apply_jsonpath('{"value": null}', "$.value")
        assert success is False

    def test_library_not_installed(self):
        """apply_jsonpath should return (None, False) if jsonpath-ng is not installed."""
        from unittest import mock

        import mlflow.tracing.utils.view_utils as vu

        with mock.patch.object(vu, "_HAS_JSONPATH", False):
            result, success = vu.apply_jsonpath('{"x": 1}', "$.x")
            assert success is False
            assert result is None


class TestValidateJsonpath:
    def test_valid_expression(self):
        from mlflow.tracing.utils.view_utils import validate_jsonpath

        is_valid, error = validate_jsonpath("$.messages[0].content")
        assert is_valid is True
        assert error is None

    def test_empty_is_valid(self):
        from mlflow.tracing.utils.view_utils import validate_jsonpath

        is_valid, error = validate_jsonpath("")
        assert is_valid is True

    def test_invalid_syntax(self):
        from mlflow.tracing.utils.view_utils import validate_jsonpath

        is_valid, error = validate_jsonpath("$[invalid[")
        assert is_valid is False
        assert error is not None


class TestApplyView:
    def test_full_pipeline(self):
        """Test span filter + JSONPath pipeline end-to-end."""
        from mlflow.entities.trace_view import SpanFilter, TraceView
        from mlflow.tracing.utils.view_utils import apply_view

        # Minimal trace-like structure with span dicts
        # apply_view needs a Trace object; this test will be updated
        # once we know the exact Trace construction for tests
        pass
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/tracing/utils/test_view_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlflow.tracing.utils.view_utils'`

- [ ] **Step 4: Implement view_utils.py**

Port from project-0xfffff, adapting to use MLflow's `Span` entity and `SpanFilter` dataclass:

```python
# mlflow/tracing/utils/view_utils.py
from __future__ import annotations

import json
import logging
from typing import Any

from mlflow.entities.trace_view import SpanFilter, TraceView

_logger = logging.getLogger(__name__)

try:
    from jsonpath_ng import parse as jsonpath_parse
    from jsonpath_ng.exceptions import JsonPathParserError

    _HAS_JSONPATH = True
except ImportError:
    _HAS_JSONPATH = False


def find_first_matching_span(
    spans: list[dict[str, Any]], filter_config: SpanFilter
) -> dict[str, Any] | None:
    if not spans:
        return None
    for span in spans:
        if _span_matches(span, filter_config):
            return span
    return None


def _span_matches(span: dict[str, Any], f: SpanFilter) -> bool:
    if f.span_name is not None and span.get("name") != f.span_name:
        return False
    if f.span_type is not None:
        span_type = span.get("span_type")
        if span_type is None:
            span_type = _unwrap_json_str(
                span.get("attributes", {}).get("mlflow.spanType")
            )
        if span_type != f.span_type:
            return False
    if f.attribute_key is not None:
        attributes = span.get("attributes", {})
        if not isinstance(attributes, dict):
            return False
        actual = attributes.get(f.attribute_key)
        if actual is None:
            return False
        if f.attribute_value is not None and str(actual) != str(f.attribute_value):
            return False
    return True


def _unwrap_json_str(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def apply_span_filter(
    spans_data: list[dict[str, Any]], filter_config: SpanFilter | None
) -> tuple[str | None, str | None]:
    if not filter_config or not spans_data:
        return None, None
    matched = find_first_matching_span(spans_data, filter_config)
    if not matched:
        return None, None
    inputs = matched.get("inputs")
    if inputs is None:
        inputs = _unwrap_json_str(
            matched.get("attributes", {}).get("mlflow.spanInputs")
        )
    outputs = matched.get("outputs")
    if outputs is None:
        outputs = _unwrap_json_str(
            matched.get("attributes", {}).get("mlflow.spanOutputs")
        )
    return _to_json_string(inputs), _to_json_string(outputs)


def _to_json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def apply_jsonpath(data: str, jsonpath_expr: str | None) -> tuple[str | None, bool]:
    if not jsonpath_expr or not jsonpath_expr.strip():
        return None, False
    if not _HAS_JSONPATH:
        _logger.warning("jsonpath-ng is not installed. Install it with: pip install jsonpath-ng")
        return None, False
    try:
        parsed_data = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None, False
    try:
        expr = jsonpath_parse(jsonpath_expr.strip())
        matches = [match.value for match in expr.find(parsed_data)]
    except Exception:
        return None, False
    if not matches:
        return None, False
    string_matches = []
    for match in matches:
        if match is None:
            continue
        if isinstance(match, str):
            if match:
                string_matches.append(match)
        else:
            str_val = str(match)
            if str_val and str_val not in ("None", "null"):
                string_matches.append(str_val)
    if not string_matches:
        return None, False
    result = "\n".join(string_matches)
    if not result or result.strip() == "":
        return None, False
    return result, True


def validate_jsonpath(expr: str) -> tuple[bool, str | None]:
    if not expr or not expr.strip():
        return True, None
    if not _HAS_JSONPATH:
        return False, "jsonpath-ng is not installed"
    try:
        jsonpath_parse(expr.strip())
        return True, None
    except Exception as e:
        return False, f"Invalid JSONPath syntax: {e!s}"


def apply_view(trace_spans_data: list[dict[str, Any]], view: TraceView, fallback_input: str | None = None, fallback_output: str | None = None) -> tuple[str, str]:
    base_input = fallback_input
    base_output = fallback_output
    if view.span_filter:
        filtered_input, filtered_output = apply_span_filter(
            trace_spans_data, view.span_filter
        )
        if filtered_input is not None:
            base_input = filtered_input
        if filtered_output is not None:
            base_output = filtered_output
    display_input = base_input or ""
    display_output = base_output or ""
    if view.input_path and base_input:
        extracted, success = apply_jsonpath(base_input, view.input_path)
        if success:
            display_input = extracted
    if view.output_path and base_output:
        extracted, success = apply_jsonpath(base_output, view.output_path)
        if success:
            display_output = extracted
    return display_input, display_output
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/tracing/utils/test_view_utils.py -v`
Expected: PASS (install jsonpath-ng first if needed: `uv pip install jsonpath-ng`)

- [ ] **Step 6: Commit**

```bash
git add mlflow/tracing/utils/view_utils.py tests/tracing/utils/test_view_utils.py pyproject.toml
git commit -s -m "feat: add span filter and JSONPath utilities for trace views

Ported from project-0xfffff prototype. Includes find_first_matching_span,
apply_span_filter, apply_jsonpath, validate_jsonpath, and apply_view pipeline.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Database Migration & ORM Model

**Files:**
- Create: `mlflow/store/db_migrations/versions/xxx_add_trace_views_table.py`
- Modify: `mlflow/store/tracking/dbmodels/models.py` (after line ~1123)

- [ ] **Step 1: Generate alembic revision ID**

Run: `python -c "import uuid; print(uuid.uuid4().hex[:12])"` to generate a unique revision ID.

- [ ] **Step 2: Create alembic migration**

```python
# mlflow/store/db_migrations/versions/{revision_id}_add_trace_views_table.py
"""add trace_views table

Create Date: 2026-03-26
"""

import sqlalchemy as sa
from alembic import op

revision = "{generated_revision_id}"
down_revision = "c3d6457b6d8a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "trace_views",
        sa.Column("view_id", sa.String(50), primary_key=True, nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("trace_id", sa.String(50), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("span_filter", sa.Text(), nullable=True),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(256), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["trace_id"], ["trace_info.request_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"], ["experiments.experiment_id"]
        ),
        sa.CheckConstraint(
            "(trace_id IS NOT NULL AND experiment_id IS NULL) "
            "OR (trace_id IS NULL AND experiment_id IS NOT NULL)",
            name="ck_trace_views_scope",
        ),
    )

    with op.batch_alter_table("trace_views") as batch_op:
        batch_op.create_index(
            "index_trace_views_trace_id_created",
            ["trace_id", "created_timestamp"],
        )
        batch_op.create_index(
            "index_trace_views_experiment_id_created",
            ["experiment_id", "created_timestamp"],
        )
        batch_op.create_index(
            "index_trace_views_last_updated",
            ["last_updated_timestamp"],
        )


def downgrade():
    op.drop_table("trace_views")
```

- [ ] **Step 3: Add SqlTraceView ORM model**

Add after `SqlAssessments` class in `mlflow/store/tracking/dbmodels/models.py` (after line ~1123):

```python
class SqlTraceView(Base):
    __tablename__ = "trace_views"

    view_id = Column(String(50), primary_key=True, nullable=False)
    name = Column(String(256), nullable=False)
    trace_id = Column(
        String(50), ForeignKey("trace_info.request_id", ondelete="CASCADE"), nullable=True
    )
    experiment_id = Column(Integer, ForeignKey("experiments.experiment_id"), nullable=True)
    span_filter = Column(Text, nullable=True)
    input_path = Column(Text, nullable=True)
    output_path = Column(Text, nullable=True)
    created_by = Column(String(256), nullable=True)
    description = Column(Text, nullable=True)
    created_timestamp = Column(BigInteger, nullable=False)
    last_updated_timestamp = Column(BigInteger, nullable=False)

    def to_mlflow_entity(self):
        from mlflow.entities.trace_view import SpanFilter, TraceView

        span_filter = None
        if self.span_filter:
            import json
            span_filter = SpanFilter.from_dict(json.loads(self.span_filter))

        return TraceView(
            view_id=self.view_id,
            name=self.name,
            trace_id=self.trace_id,
            experiment_id=str(self.experiment_id) if self.experiment_id else None,
            span_filter=span_filter,
            input_path=self.input_path,
            output_path=self.output_path,
            created_by=self.created_by,
            description=self.description,
            create_time_ms=self.created_timestamp,
            last_update_time_ms=self.last_updated_timestamp,
        )

    @classmethod
    def from_mlflow_entity(cls, view):
        import json

        return cls(
            view_id=view.view_id or f"tv-{uuid.uuid4().hex[:12]}",
            name=view.name,
            trace_id=view.trace_id,
            experiment_id=int(view.experiment_id) if view.experiment_id else None,
            span_filter=view.span_filter.to_json() if view.span_filter else None,
            input_path=view.input_path,
            output_path=view.output_path,
            created_by=view.created_by,
            description=view.description,
            created_timestamp=view.create_time_ms or int(time.time() * 1000),
            last_updated_timestamp=view.last_update_time_ms or int(time.time() * 1000),
        )
```

Add required imports at the top of the file: `import uuid`, `import time`.

- [ ] **Step 4: Verify migration applies cleanly**

Run: `uv run mlflow db upgrade sqlite:///test_migration.db && rm test_migration.db`
Expected: Migration applies without errors.

- [ ] **Step 5: Commit**

```bash
git add mlflow/store/db_migrations/versions/*_add_trace_views_table.py mlflow/store/tracking/dbmodels/models.py
git commit -s -m "feat: add trace_views database table and ORM model

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Abstract Store & SQLAlchemy Store CRUD

**Files:**
- Modify: `mlflow/store/tracking/abstract_store.py` (after line ~603)
- Modify: `mlflow/store/tracking/sqlalchemy_store.py` (after line ~4200)
- Test: `tests/store/tracking/test_trace_views.py`

- [ ] **Step 1: Write store CRUD tests**

```python
# tests/store/tracking/test_trace_views.py
import pytest

from mlflow.entities.trace_view import SpanFilter, TraceView
from mlflow.exceptions import MlflowException


class TestTraceViewStoreCRUD:
    """Tests for TraceView CRUD in the SQLAlchemy store.

    These tests require a store fixture. Follow the pattern in
    tests/store/tracking/test_sqlalchemy_store.py for setup.
    """

    def test_create_trace_view(self, store, trace_id):
        view = TraceView(
            name="Tool Calls",
            trace_id=trace_id,
            span_filter=SpanFilter(span_type="TOOL"),
            input_path="$.query",
            created_by="test_user",
        )
        created = store.create_trace_view(view)
        assert created.view_id is not None
        assert created.view_id.startswith("tv-")
        assert created.name == "Tool Calls"
        assert created.span_filter.span_type == "TOOL"
        assert created.create_time_ms is not None

    def test_get_trace_view(self, store, trace_id):
        view = TraceView(
            name="Test", trace_id=trace_id, span_filter=SpanFilter(span_type="LLM")
        )
        created = store.create_trace_view(view)
        fetched = store.get_trace_view(trace_id, created.view_id)
        assert fetched.view_id == created.view_id
        assert fetched.name == "Test"

    def test_list_trace_views(self, store, trace_id):
        store.create_trace_view(
            TraceView(name="View 1", trace_id=trace_id)
        )
        store.create_trace_view(
            TraceView(name="View 2", trace_id=trace_id)
        )
        views = store.list_trace_views(trace_id=trace_id)
        assert len(views) >= 2
        names = {v.name for v in views}
        assert "View 1" in names
        assert "View 2" in names

    def test_update_trace_view(self, store, trace_id):
        view = TraceView(name="Original", trace_id=trace_id)
        created = store.create_trace_view(view)
        updated = store.update_trace_view(
            trace_id=trace_id,
            view_id=created.view_id,
            name="Updated",
            input_path="$.new_path",
        )
        assert updated.name == "Updated"
        assert updated.input_path == "$.new_path"

    def test_delete_trace_view(self, store, trace_id):
        view = TraceView(name="ToDelete", trace_id=trace_id)
        created = store.create_trace_view(view)
        store.delete_trace_view(trace_id, created.view_id)
        with pytest.raises(MlflowException):
            store.get_trace_view(trace_id, created.view_id)

    def test_create_experiment_view(self, store, experiment_id):
        view = TraceView(
            name="Experiment Template",
            experiment_id=experiment_id,
            span_filter=SpanFilter(span_type="CHAT_MODEL"),
        )
        created = store.create_trace_view(view)
        assert created.experiment_id == experiment_id

    def test_list_includes_experiment_views(self, store, trace_id, experiment_id):
        """Listing views for a trace should include experiment-scoped views."""
        store.create_trace_view(
            TraceView(name="Trace View", trace_id=trace_id)
        )
        store.create_trace_view(
            TraceView(name="Experiment View", experiment_id=experiment_id)
        )
        views = store.list_trace_views(trace_id=trace_id)
        names = {v.name for v in views}
        assert "Trace View" in names
        assert "Experiment View" in names

    def test_get_nonexistent_view_raises(self, store, trace_id):
        with pytest.raises(MlflowException):
            store.get_trace_view(trace_id, "tv-nonexistent")
```

These tests need fixtures. Add a conftest or inline them at the top of the file:

```python
import pytest
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

@pytest.fixture()
def store(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'test.db'}"
    s = SqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))
    return s

@pytest.fixture()
def experiment_id(store):
    return str(store.create_experiment("test_experiment"))

@pytest.fixture()
def trace_id(store, experiment_id):
    # Create a minimal trace so we have a valid trace_id to attach views to.
    # Follow the pattern in tests/store/tracking/test_sqlalchemy_store.py
    # for creating traces via store._start_trace_v3 / store._end_trace_v3.
    # The exact method depends on the store's public API — check the test file.
    from mlflow.entities import TraceInfo, TraceState, TraceLocation, MlflowExperimentLocation
    import time
    trace_info = store.start_trace(
        experiment_id=experiment_id,
        timestamp_ms=int(time.time() * 1000),
        request_metadata={},
        tags={},
    )
    return trace_info.trace_id
```

Note: The exact `start_trace` API may differ — check `tests/store/tracking/test_sqlalchemy_store.py` for the current pattern. The key is having a valid `trace_id` that exists in `trace_info` so the FK constraint is satisfied.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/store/tracking/test_trace_views.py -v`
Expected: FAIL — methods don't exist yet.

- [ ] **Step 3: Add abstract methods to `abstract_store.py`**

Add after `delete_assessment` (line ~603) in `mlflow/store/tracking/abstract_store.py`:

```python
def create_trace_view(self, view: "TraceView") -> "TraceView":
    raise NotImplementedError

def get_trace_view(self, trace_id: str, view_id: str) -> "TraceView":
    raise NotImplementedError

def list_trace_views(
    self,
    trace_id: str | None = None,
    experiment_id: str | None = None,
) -> list["TraceView"]:
    raise NotImplementedError

def update_trace_view(
    self,
    trace_id: str | None = None,
    experiment_id: str | None = None,
    view_id: str = "",
    name: str | None = None,
    span_filter: "SpanFilter | None" = None,
    input_path: str | None = None,
    output_path: str | None = None,
    description: str | None = None,
) -> "TraceView":
    raise NotImplementedError

def delete_trace_view(
    self,
    trace_id: str | None = None,
    experiment_id: str | None = None,
    view_id: str = "",
) -> None:
    raise NotImplementedError
```

- [ ] **Step 4: Implement CRUD in `sqlalchemy_store.py`**

Add after the assessment methods section (after line ~4200) in `mlflow/store/tracking/sqlalchemy_store.py`. Follow the assessment CRUD pattern:
- Use `self.ManagedSessionMaker()` context manager
- Call `self._validate_trace_accessible(session, trace_id)` for trace-scoped ops
- Generate `view_id` with `f"tv-{uuid.uuid4().hex[:12]}"`
- Set timestamps with `int(time.time() * 1000)`
- Use `SqlTraceView.from_mlflow_entity()` and `.to_mlflow_entity()`
- For `list_trace_views(trace_id=...)`, query both trace-scoped views AND experiment-scoped views (by looking up the trace's experiment_id)
- Raise `RESOURCE_DOES_NOT_EXIST` when view not found

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/store/tracking/test_trace_views.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mlflow/store/tracking/abstract_store.py mlflow/store/tracking/sqlalchemy_store.py tests/store/tracking/test_trace_views.py
git commit -s -m "feat: add TraceView CRUD to abstract and SQLAlchemy stores

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: REST Endpoint Handlers

**Files:**
- Modify: `mlflow/server/handlers.py` (after line ~4083 for handlers, ~6711 for registration)
- Reference: Assessment handler pattern at lines 3983-4083

- [ ] **Step 1: Add JSON-based endpoint handler functions**

Since this is a prototype, use JSON request/response directly rather than protobuf. This avoids proto generation and keeps the prototype lean. Add after the assessment handlers (line ~4083) in `mlflow/server/handlers.py`. Also add URL route registrations — MLflow uses Flask blueprints, so register these as new routes alongside the existing trace/assessment routes.

```python
import json as json_module
from mlflow.entities.trace_view import SpanFilter, TraceView

# TraceView API handlers — JSON-based (no protobuf)

@catch_mlflow_exception
@_disable_if_artifacts_only
def _create_trace_view(trace_id=None, experiment_id=None):
    request_data = _get_request_json()
    view = TraceView(
        name=request_data["name"],
        trace_id=trace_id,
        experiment_id=experiment_id,
        span_filter=SpanFilter.from_dict(request_data["span_filter"]) if request_data.get("span_filter") else None,
        input_path=request_data.get("input_path"),
        output_path=request_data.get("output_path"),
        created_by=request_data.get("created_by"),
        description=request_data.get("description"),
    )
    created = _get_tracking_store().create_trace_view(view)
    return Response(
        response=json_module.dumps({"trace_view": created.to_dict()}),
        status=200,
        mimetype="application/json",
    )


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_trace_view(trace_id, view_id):
    view = _get_tracking_store().get_trace_view(trace_id, view_id)
    return Response(
        response=json_module.dumps({"trace_view": view.to_dict()}),
        status=200,
        mimetype="application/json",
    )


@catch_mlflow_exception
@_disable_if_artifacts_only
def _list_trace_views(trace_id=None, experiment_id=None):
    views = _get_tracking_store().list_trace_views(
        trace_id=trace_id, experiment_id=experiment_id
    )
    return Response(
        response=json_module.dumps({"trace_views": [v.to_dict() for v in views]}),
        status=200,
        mimetype="application/json",
    )


@catch_mlflow_exception
@_disable_if_artifacts_only
def _update_trace_view(trace_id=None, experiment_id=None, view_id=None):
    request_data = _get_request_json()
    updated = _get_tracking_store().update_trace_view(
        trace_id=trace_id,
        experiment_id=experiment_id,
        view_id=view_id,
        name=request_data.get("name"),
        span_filter=SpanFilter.from_dict(request_data["span_filter"]) if request_data.get("span_filter") else None,
        input_path=request_data.get("input_path"),
        output_path=request_data.get("output_path"),
        description=request_data.get("description"),
    )
    return Response(
        response=json_module.dumps({"trace_view": updated.to_dict()}),
        status=200,
        mimetype="application/json",
    )


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_trace_view(trace_id=None, experiment_id=None, view_id=None):
    _get_tracking_store().delete_trace_view(
        trace_id=trace_id, experiment_id=experiment_id, view_id=view_id
    )
    return Response(response="{}", status=200, mimetype="application/json")
```

- [ ] **Step 2: Register Flask routes**

Find where trace and assessment routes are registered (look for `app.add_url_rule` or Flask blueprint registrations in `mlflow/server/__init__.py` or `handlers.py`). Add:

```python
# Trace-scoped view routes
app.add_url_rule(
    "/ajax-api/2.0/mlflow/traces/<trace_id>/views",
    view_func=_create_trace_view,
    methods=["POST"],
    defaults={"experiment_id": None},
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/traces/<trace_id>/views",
    view_func=_list_trace_views,
    methods=["GET"],
    defaults={"experiment_id": None},
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/traces/<trace_id>/views/<view_id>",
    view_func=_get_trace_view,
    methods=["GET"],
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/traces/<trace_id>/views/<view_id>",
    view_func=_update_trace_view,
    methods=["PATCH"],
    defaults={"experiment_id": None},
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/traces/<trace_id>/views/<view_id>",
    view_func=_delete_trace_view,
    methods=["DELETE"],
    defaults={"experiment_id": None},
)
# Experiment-scoped view routes
app.add_url_rule(
    "/ajax-api/2.0/mlflow/experiments/<experiment_id>/views",
    view_func=_create_trace_view,
    methods=["POST"],
    defaults={"trace_id": None},
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/experiments/<experiment_id>/views",
    view_func=_list_trace_views,
    methods=["GET"],
    defaults={"trace_id": None},
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/experiments/<experiment_id>/views/<view_id>",
    view_func=_update_trace_view,
    methods=["PATCH"],
    defaults={"trace_id": None},
)
app.add_url_rule(
    "/ajax-api/2.0/mlflow/experiments/<experiment_id>/views/<view_id>",
    view_func=_delete_trace_view,
    methods=["DELETE"],
    defaults={"trace_id": None},
)
```

Note: Check how existing handlers get JSON bodies (`flask.request.get_json()`) and adapt `_get_request_json()` accordingly. This may already exist as a helper.

- [ ] **Step 5: Test endpoints manually**

Start dev server: `nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &`

Test with curl:
```bash
# Create a trace-scoped view
curl -X POST http://localhost:5000/mlflow/traces/TEST_TRACE_ID/views \
  -H "Content-Type: application/json" \
  -d '{"trace_view": {"name": "Test View"}}'
```

- [ ] **Step 6: Commit**

```bash
git add mlflow/server/handlers.py mlflow/server/__init__.py
git commit -s -m "feat: add JSON REST endpoint handlers for TraceView CRUD

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: REST Store Proxy & Tracking Client

**Files:**
- Modify: `mlflow/store/tracking/rest_store.py` (after line ~837)
- Modify: `mlflow/tracking/client.py`
- Modify: `mlflow/tracking/_tracking_service/client.py`

- [ ] **Step 1: Add REST store proxy methods**

Follow the assessment pattern in `rest_store.py` (lines 731-837). Each method serializes the request to protobuf, calls `_call_endpoint()`, and deserializes the response.

- [ ] **Step 2: Add service client methods**

In `mlflow/tracking/_tracking_service/client.py`, add methods that delegate to the store:

```python
def create_trace_view(self, view):
    return self.store.create_trace_view(view)

def get_trace_view(self, trace_id, view_id):
    return self.store.get_trace_view(trace_id, view_id)

def list_trace_views(self, trace_id=None, experiment_id=None):
    return self.store.list_trace_views(trace_id=trace_id, experiment_id=experiment_id)

def update_trace_view(self, trace_id=None, experiment_id=None, view_id="", name=None, span_filter=None, input_path=None, output_path=None, description=None):
    return self.store.update_trace_view(trace_id=trace_id, experiment_id=experiment_id, view_id=view_id, name=name, span_filter=span_filter, input_path=input_path, output_path=output_path, description=description)

def delete_trace_view(self, trace_id=None, experiment_id=None, view_id=""):
    return self.store.delete_trace_view(trace_id=trace_id, experiment_id=experiment_id, view_id=view_id)
```

- [ ] **Step 3: Add MlflowClient methods**

In `mlflow/tracking/client.py`, add the public-facing client methods:

```python
def create_trace_view(
    self,
    trace_id: str | None = None,
    experiment_id: str | None = None,
    name: str = "",
    span_filter: SpanFilter | None = None,
    input_path: str | None = None,
    output_path: str | None = None,
    created_by: str | None = None,
    description: str | None = None,
) -> TraceView:
    view = TraceView(
        name=name,
        trace_id=trace_id,
        experiment_id=experiment_id,
        span_filter=span_filter,
        input_path=input_path,
        output_path=output_path,
        created_by=created_by,
        description=description,
    )
    return self._tracking_client.create_trace_view(view)

def list_trace_views(self, trace_id: str | None = None, experiment_id: str | None = None) -> list[TraceView]:
    return self._tracking_client.list_trace_views(trace_id=trace_id, experiment_id=experiment_id)

def get_trace_view(self, trace_id: str, view_id: str) -> TraceView:
    return self._tracking_client.get_trace_view(trace_id, view_id)

def update_trace_view(self, trace_id: str | None = None, experiment_id: str | None = None, view_id: str = "", name: str | None = None, span_filter: SpanFilter | None = None, input_path: str | None = None, output_path: str | None = None, description: str | None = None) -> TraceView:
    return self._tracking_client.update_trace_view(trace_id=trace_id, experiment_id=experiment_id, view_id=view_id, name=name, span_filter=span_filter, input_path=input_path, output_path=output_path, description=description)

def delete_trace_view(self, trace_id: str | None = None, experiment_id: str | None = None, view_id: str = "") -> None:
    return self._tracking_client.delete_trace_view(trace_id=trace_id, experiment_id=experiment_id, view_id=view_id)
```

- [ ] **Step 4: Commit**

```bash
git add mlflow/store/tracking/rest_store.py mlflow/tracking/client.py mlflow/tracking/_tracking_service/client.py
git commit -s -m "feat: add TraceView REST store proxy and MlflowClient methods

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Trace Convenience Methods (create_view, views, summarize, analyze)

**Files:**
- Modify: `mlflow/entities/trace.py` (after line ~296)
- Test: `tests/entities/test_trace_view_methods.py`

- [ ] **Step 1: Write tests for convenience methods**

```python
# tests/entities/test_trace_view_methods.py
from unittest import mock

import pytest

from mlflow.entities.trace_view import SpanFilter


class TestTraceCreateView:
    def test_create_view_delegates_to_client(self, sample_trace):
        with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.create_trace_view.return_value = mock.MagicMock()
            result = sample_trace.create_view(
                name="Test", span_filter=SpanFilter(span_type="TOOL")
            )
            mock_client.create_trace_view.assert_called_once()


class TestTraceViews:
    def test_views_delegates_to_client(self, sample_trace):
        with mock.patch("mlflow.tracking.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.list_trace_views.return_value = []
            result = sample_trace.views
            mock_client.list_trace_views.assert_called_once_with(
                trace_id=sample_trace.info.trace_id
            )


class TestTraceSummarize:
    def test_summarize_calls_invoke_judge_model(self, sample_trace):
        with mock.patch(
            "mlflow.genai.judges.utils.invocation_utils.invoke_judge_model",
            return_value=mock.MagicMock(value="This trace shows..."),
        ) as mock_invoke:
            result = sample_trace.summarize(model="openai:/gpt-4o-mini")
            assert result == "This trace shows..."
            mock_invoke.assert_called_once()


class TestTraceAnalyze:
    def test_analyze_calls_invoke_judge_model(self, sample_trace):
        with mock.patch(
            "mlflow.genai.judges.utils.invocation_utils.invoke_judge_model",
            return_value=mock.MagicMock(value="3 tool calls were made"),
        ) as mock_invoke:
            result = sample_trace.analyze(
                "how many tools were used?", model="openai:/gpt-4o-mini"
            )
            assert result == "3 tool calls were made"
            mock_invoke.assert_called_once()
```

Note: A `sample_trace` fixture is needed — construct a minimal `Trace` with `TraceInfo` and `TraceData`. Check existing test files for the pattern.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/entities/test_trace_view_methods.py -v`
Expected: FAIL — methods don't exist on Trace yet.

- [ ] **Step 3: Add methods to Trace class**

Add after `search_assessments` (line ~296) in `mlflow/entities/trace.py`:

```python
def create_view(
    self,
    name: str,
    span_filter: "SpanFilter | None" = None,
    input_path: str | None = None,
    output_path: str | None = None,
    created_by: str | None = None,
    description: str | None = None,
) -> "TraceView":
    from mlflow.tracking import MlflowClient

    return MlflowClient().create_trace_view(
        trace_id=self.info.trace_id,
        name=name,
        span_filter=span_filter,
        input_path=input_path,
        output_path=output_path,
        created_by=created_by,
        description=description,
    )

@property
def views(self) -> list["TraceView"]:
    from mlflow.tracking import MlflowClient

    return MlflowClient().list_trace_views(trace_id=self.info.trace_id)

def delete_view(self, view_id: str) -> None:
    from mlflow.tracking import MlflowClient

    MlflowClient().delete_trace_view(trace_id=self.info.trace_id, view_id=view_id)

def summarize(
    self,
    model: str = "openai:/gpt-4o-mini",
    view: "TraceView | None" = None,
) -> str:
    from mlflow.genai.judges.utils.invocation_utils import invoke_judge_model

    prompt = "Summarize this trace concisely. Focus on what the agent did, key decisions, and the outcome."
    if view:
        from mlflow.tracing.utils.view_utils import apply_view
        spans_data = [s.to_dict() for s in self.data.spans]
        filtered_in, filtered_out = apply_view(
            spans_data, view,
            fallback_input=self.data.request,
            fallback_output=self.data.response,
        )
        prompt += f"\n\nFiltered input:\n{filtered_in}\n\nFiltered output:\n{filtered_out}"
    feedback = invoke_judge_model(
        model_uri=model,
        prompt=prompt,
        assessment_name="trace_summary",
        trace=self,
    )
    return feedback.value

def analyze(
    self,
    question: str,
    model: str = "openai:/gpt-4o-mini",
    view: "TraceView | None" = None,
) -> str:
    from mlflow.genai.judges.utils.invocation_utils import invoke_judge_model

    prompt = question
    if view:
        from mlflow.tracing.utils.view_utils import apply_view
        spans_data = [s.to_dict() for s in self.data.spans]
        filtered_in, filtered_out = apply_view(
            spans_data, view,
            fallback_input=self.data.request,
            fallback_output=self.data.response,
        )
        prompt += f"\n\nFiltered input:\n{filtered_in}\n\nFiltered output:\n{filtered_out}"
    feedback = invoke_judge_model(
        model_uri=model,
        prompt=prompt,
        assessment_name="trace_analysis",
        trace=self,
    )
    return feedback.value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/entities/test_trace_view_methods.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlflow/entities/trace.py tests/entities/test_trace_view_methods.py
git commit -s -m "feat: add create_view, views, summarize, analyze to Trace

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Frontend — View Selector Component & Hook

**Files:**
- Create: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/hooks/useTraceViews.ts`
- Create: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/TraceViewSelector.tsx`
- Modify: `mlflow/server/js/package.json` (add `jsonpath-plus`)

- [ ] **Step 1: Add `jsonpath-plus` dependency**

Run: `cd mlflow/server/js && yarn add jsonpath-plus`

- [ ] **Step 2: Create useTraceViews hook**

```typescript
// mlflow/server/js/src/shared/web-shared/model-trace-explorer/hooks/useTraceViews.ts
import { useQuery } from '@tanstack/react-query';

export interface SpanFilter {
  span_name?: string | null;
  span_type?: string | null;
  attribute_key?: string | null;
  attribute_value?: string | null;
}

export interface TraceView {
  view_id: string;
  name: string;
  trace_id?: string | null;
  experiment_id?: string | null;
  span_filter?: SpanFilter | null;
  input_path?: string | null;
  output_path?: string | null;
  created_by?: string | null;
  description?: string | null;
  create_time_ms?: number | null;
  last_update_time_ms?: number | null;
}

export function useTraceViews(traceId: string | null) {
  return useQuery({
    queryKey: ['traceViews', traceId],
    queryFn: async (): Promise<TraceView[]> => {
      if (!traceId) return [];
      const response = await fetch(
        `/ajax-api/2.0/mlflow/traces/${encodeURIComponent(traceId)}/views`
      );
      if (!response.ok) return [];
      const data = await response.json();
      return data.trace_views ?? [];
    },
    enabled: !!traceId,
  });
}
```

- [ ] **Step 3: Create TraceViewSelector component**

```typescript
// mlflow/server/js/src/shared/web-shared/model-trace-explorer/TraceViewSelector.tsx
import { Select, Typography } from '@databricks/design-system';
import { useTraceViews, type TraceView } from './hooks/useTraceViews';

interface TraceViewSelectorProps {
  traceId: string | null;
  activeViewId: string | null;
  onViewChange: (view: TraceView | null) => void;
}

export function TraceViewSelector({
  traceId,
  activeViewId,
  onViewChange,
}: TraceViewSelectorProps) {
  const { data: views = [], isLoading } = useTraceViews(traceId);

  if (isLoading || views.length === 0) return null;

  const traceViews = views.filter((v) => v.trace_id);
  const experimentViews = views.filter((v) => v.experiment_id);

  return (
    <Select
      value={activeViewId ?? '__raw__'}
      onChange={(value) => {
        if (value === '__raw__') {
          onViewChange(null);
        } else {
          const view = views.find((v) => v.view_id === value);
          onViewChange(view ?? null);
        }
      }}
    >
      <Select.Option value="__raw__">Raw Trace</Select.Option>
      {traceViews.length > 0 && (
        <Select.OptGroup label="Trace Views">
          {traceViews.map((v) => (
            <Select.Option key={v.view_id} value={v.view_id}>
              {v.name}
            </Select.Option>
          ))}
        </Select.OptGroup>
      )}
      {experimentViews.length > 0 && (
        <Select.OptGroup label="Experiment Views">
          {experimentViews.map((v) => (
            <Select.Option key={v.view_id} value={v.view_id}>
              {v.name}
            </Select.Option>
          ))}
        </Select.OptGroup>
      )}
    </Select>
  );
}
```

- [ ] **Step 4: Commit**

```bash
cd mlflow/server/js
git add package.json yarn.lock src/shared/web-shared/model-trace-explorer/hooks/useTraceViews.ts src/shared/web-shared/model-trace-explorer/TraceViewSelector.tsx
git commit -s -m "feat: add TraceViewSelector component and useTraceViews hook

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Frontend — Integrate View Selector into Trace Explorer

**Files:**
- Modify: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/ModelTraceExplorerContent.tsx`
- Modify: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/ModelTraceExplorerDetailView.tsx`
- Modify: `mlflow/server/js/src/shared/web-shared/model-trace-explorer/right-pane/ModelTraceExplorerDefaultSpanView.tsx`

- [ ] **Step 1: Add view state to ModelTraceExplorerContent**

Read `ModelTraceExplorerContent.tsx` and add:
- `useState` for `activeView: TraceView | null`
- Render `<TraceViewSelector>` in the header area
- Pass `activeView` down to `ModelTraceExplorerDetailView`

- [ ] **Step 2: Apply span filter dimming in detail view**

In `ModelTraceExplorerDetailView.tsx`, when `activeView` has a `span_filter`:
- Compare each span against the filter
- Add a CSS class (e.g., `opacity-30`) to non-matching spans in the timeline tree
- Auto-expand/select the first matching span

- [ ] **Step 3: Apply JSONPath extraction in right pane**

In `ModelTraceExplorerDefaultSpanView.tsx`, when `activeView` has `input_path` or `output_path`:
- Use `jsonpath-plus` to extract values from span inputs/outputs
- Display extracted values instead of full JSON
- Show a banner: "Filtered by: {view.name}" with clear button

- [ ] **Step 4: Add visual indicator banner**

When a view is active, show a dismissible info banner at the top of the trace explorer:
```
Viewing: "Tool Calls Only" — TOOL spans, output $.result  [Clear]
```

- [ ] **Step 5: Test manually**

Start dev server, create a view via Python API, verify:
- View selector appears when views exist
- Selecting a view dims non-matching spans
- JSONPath extraction works on inputs/outputs
- "Clear" returns to raw trace

- [ ] **Step 6: Commit**

```bash
cd mlflow/server/js
git add src/shared/web-shared/model-trace-explorer/
git commit -s -m "feat: integrate TraceViewSelector into trace explorer with filtering

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Assistant Integration — Marker Parsing

**Files:**
- Modify: `mlflow/server/js/src/assistant/AssistantContext.tsx`

- [ ] **Step 1: Add marker parsing to message handler**

In `AssistantContext.tsx`, find where streamed messages are finalized (look for `finalizeStreamingMessage` or similar). Add a post-processing step:

```typescript
const TRACE_VIEW_MARKER_REGEX = /\[trace_view_created:\s*(\{.*?\})\]/g;

function parseTraceViewMarkers(content: string): {
  cleanContent: string;
  viewUpdates: Array<{ view_id: string; trace_id: string }>;
} {
  const viewUpdates: Array<{ view_id: string; trace_id: string }> = [];
  const cleanContent = content.replace(TRACE_VIEW_MARKER_REGEX, (match, jsonStr) => {
    try {
      const parsed = JSON.parse(jsonStr);
      if (parsed.view_id && parsed.trace_id) {
        viewUpdates.push(parsed);
        return ''; // Strip marker from displayed message
      }
    } catch {
      // Malformed marker — leave it visible
    }
    return match;
  });
  return { cleanContent: cleanContent.trim(), viewUpdates };
}
```

- [ ] **Step 2: Handle view updates**

When `viewUpdates` is non-empty:
- Check if the current page context has a matching `traceId`
- If so, invalidate the `traceViews` React Query cache to trigger a refetch
- Show a toast: "Assistant applied view: {view_name}"
- Optionally set the active view in the trace explorer

- [ ] **Step 3: Test manually with assistant**

With the dev server running and assistant configured:
1. Navigate to a trace detail page
2. Ask the assistant "show me only the tool calls"
3. Verify the assistant runs Python to create a view
4. Verify the marker is parsed and the view selector updates
5. Verify the marker text is hidden from the chat

- [ ] **Step 4: Commit**

```bash
cd mlflow/server/js
git add src/assistant/AssistantContext.tsx
git commit -s -m "feat: parse trace_view_created markers from assistant messages

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Skill Update (External Repo)

**Files:**
- Modify: `analyze-mlflow-trace/SKILL.md` in https://github.com/mlflow/skills

- [ ] **Step 1: Clone the skills repo**

```bash
git clone https://github.com/mlflow/skills /tmp/mlflow-skills
cd /tmp/mlflow-skills
```

- [ ] **Step 2: Add view creation section to the skill**

Add a new section to `analyze-mlflow-trace/SKILL.md`:

```markdown
## Creating Trace Views

When the user asks to filter, focus, or customize what they see in a trace
(e.g., "show me only the tool calls", "focus on the LLM spans"), create a
TraceView using the Python API.

### Steps

1. Determine the appropriate SpanFilter from the user's request
2. Create the view using the MLflow Python API
3. Output the marker so the UI can apply it

### Example

```python
import mlflow
from mlflow.entities.trace_view import SpanFilter

trace = mlflow.get_trace("TRACE_ID_FROM_CONTEXT")
view = trace.create_view(
    name="Tool Calls",
    span_filter=SpanFilter(span_type="TOOL"),
    output_path="$.result",
    created_by="assistant",
)
print(f'[trace_view_created: {{"view_id": "{view.view_id}", "trace_id": "{trace.info.trace_id}"}}]')
```

### Trigger phrases
- "Show me only the tool calls"
- "Focus on the LLM spans"
- "Filter to just the retriever"
- "Show key decision points"
- "Create a view for..."
- "What did the tools do?" (analyze first, then offer to create a view)

### SpanFilter mapping
| User request | SpanFilter |
|---|---|
| "tool calls" | `SpanFilter(span_type="TOOL")` |
| "LLM spans" / "model calls" | `SpanFilter(span_type="LLM")` or `SpanFilter(span_type="CHAT_MODEL")` |
| "retriever" / "search" | `SpanFilter(span_type="RETRIEVER")` |
| Specific span name | `SpanFilter(span_name="ExactName")` |
| By attribute | `SpanFilter(attribute_key="key", attribute_value="value")` |
```

- [ ] **Step 3: Commit and push**

```bash
cd /tmp/mlflow-skills
git add analyze-mlflow-trace/SKILL.md
git commit -s -m "feat: add trace view creation instructions to analyze-mlflow-trace skill

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 4: Update the submodule in the MLflow repo**

```bash
cd /Users/forrest.murray/Documents/mlflow
git submodule update --remote mlflow/assistant/skills
git add mlflow/assistant/skills
git commit -s -m "chore: update skills submodule with trace view instructions

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 12: Workspace-Aware Store & Final Testing

**Files:**
- Modify: `mlflow/store/tracking/sqlalchemy_workspace_store.py`
- Test: `tests/store/tracking/test_sqlalchemy_store_workspace.py`

- [ ] **Step 1: Add workspace-aware overrides**

In `sqlalchemy_workspace_store.py`, override the view CRUD methods to filter by workspace, following the assessment pattern (lines ~357-387 of the workspace store). The key change is using `_trace_query()` to scope queries.

- [ ] **Step 2: Write workspace variant tests**

Add trace view CRUD tests to `tests/store/tracking/test_sqlalchemy_store_workspace.py` following the existing workspace test patterns.

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest tests/store/tracking/test_trace_views.py tests/entities/test_trace_view.py tests/tracing/utils/test_view_utils.py tests/entities/test_trace_view_methods.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Run linting**

```bash
uv run ruff check mlflow/entities/trace_view.py mlflow/tracing/utils/view_utils.py mlflow/store/tracking/dbmodels/models.py --fix
uv run ruff format mlflow/entities/trace_view.py mlflow/tracing/utils/view_utils.py mlflow/store/tracking/dbmodels/models.py
```

- [ ] **Step 5: Commit**

```bash
git add mlflow/store/tracking/sqlalchemy_workspace_store.py tests/store/tracking/test_sqlalchemy_store_workspace.py
git commit -s -m "feat: add workspace-aware TraceView store and tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```
