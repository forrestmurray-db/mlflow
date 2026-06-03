from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Milestone:
    label: str
    description: str


@dataclass
class TraceSummary:
    trace_id: str
    summary: str
    milestones: list[Milestone] = field(default_factory=list)
    model: str = ""

    def create_view(self, model: str | None = None, name: str | None = None) -> "TraceView":
        from mlflow.genai.agents.trace_view_agent import create_view_from_summary

        return create_view_from_summary(
            trace_id=self.trace_id,
            summary=self,
            model=model or self.model,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "summary": self.summary,
            "milestones": [{"label": m.label, "description": m.description} for m in self.milestones],
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceSummary:
        return cls(
            trace_id=d["trace_id"],
            summary=d["summary"],
            milestones=[
                Milestone(label=m["label"], description=m["description"])
                for m in d.get("milestones", [])
            ],
            model=d.get("model", ""),
        )
