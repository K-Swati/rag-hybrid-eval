"""Runs the benchmark in a background thread so the HTTP request returns at once.

With OpenAI models a full run makes hundreds of API calls, far longer than a
request should block. One run at a time; the UI polls ``GET /eval/status``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from rag_eval.evaluate import run_benchmark, write_reports
from rag_eval.wiring import Components


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


class EvalJob:
    def __init__(self, components: Components) -> None:
        self.c = components
        self._lock = threading.Lock()
        self._status: dict = {"state": "idle", "done": 0, "total": 0}
        self._thread: threading.Thread | None = None

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start(self, k: int, generation: bool) -> bool:
        """Start a run. Returns False if one is already running."""
        with self._lock:
            if self._status["state"] == "running":
                return False
            self._status = {"state": "running", "done": 0, "total": 0, "k": k, "generation": generation,
                            "started_at": _now()}
        self._thread = threading.Thread(target=self._run, args=(k, generation), daemon=True)
        self._thread.start()
        return True

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self._status.update(done=done, total=total)

    def _run(self, k: int, generation: bool) -> None:
        c = self.c
        try:
            report = run_benchmark(
                [d.to_document() for d in c.kb], c.questions, c.router, c.writer, c.warehouse, c.embedder,
                c.generator if generation else None, k=k, default_strategy=c.registry.default_strategy,
                progress=self._progress,
            )
            write_reports(report, c.settings.reports_dir)
            final = {"state": "done"}
        except Exception as exc:  # surfaced to the UI through /eval/status
            final = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self._status.update(final, finished_at=_now())
