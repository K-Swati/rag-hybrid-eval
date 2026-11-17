"""Long-lived objects the routes share, built once at start-up."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_eval.api.jobs import EvalJob
from rag_eval.core.metrics import normalize
from rag_eval.wiring import Components


@dataclass
class AppState:
    c: Components
    job: EvalJob
    _gold: dict[str, dict] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gold = {normalize(q["question"]): q for q in self.c.questions}

    def gold_for(self, question: str) -> dict | None:
        """The gold item when ``question`` is one of the benchmark questions, so answers can be checked live."""
        return self._gold.get(normalize(question))
