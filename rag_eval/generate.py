"""Answer generation: an offline extractive baseline and an OpenAI-backed generator."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from rag_eval.retrieval import Hit, content_terms

SYSTEM_PROMPT = (
    "You answer questions about the Nimbus product using ONLY the numbered context passages. "
    "Cite passages like [1]. If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the documentation.\" Be concise."
)
NO_ANSWER = "I don't know based on the documentation."
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Answer:
    text: str
    generator: str


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Answer: ...


class ExtractiveGenerator:
    """Returns the context sentence that best overlaps the question. No LLM, no API key."""

    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        terms = set(content_terms(question))
        best, best_score = "", 0.0
        for rank, hit in enumerate(hits):
            for sentence in _SENTENCE.split(hit.chunk.text):
                sentence = sentence.strip()
                if not sentence or sentence.startswith("#"):
                    continue
                overlap = len(terms & set(content_terms(sentence)))
                score = overlap + 0.01 / (1 + rank)  # tie-break towards higher-ranked chunks
                if score > best_score:
                    best, best_score = sentence, score
        return Answer(best or NO_ANSWER, self.name)


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str | None = None, client=None) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    @staticmethod
    def build_messages(question: str, hits: list[Hit]) -> list[dict]:
        context = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        resp = self._client.chat.completions.create(
            model=self.model, messages=self.build_messages(question, hits), temperature=0
        )
        return Answer((resp.choices[0].message.content or "").strip(), self.name)


def get_generator(name: str | None = None) -> Generator:
    name = (name or os.environ.get("GENERATOR", "extractive")).lower()
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "openai":
        return OpenAIGenerator()
    raise ValueError(f"unknown generator {name!r}; choose 'extractive' or 'openai'")
