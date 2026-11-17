"""Configuration, read from environment variables in one place.

Only the entry points (``rag_eval.api`` and the CLI) call ``Settings.from_env``.
Everything below them receives plain arguments, so the core, the warehouse and
the services can be built and tested without touching the environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATABASE_URL = "postgresql://rag_reader:rag_reader@localhost:5432/icecream"


@dataclass(frozen=True)
class Settings:
    database_url: str = LOCAL_DATABASE_URL  # the read-only role; see db/init/03_reader_role.sql
    questions_path: Path = ROOT / "data" / "eval" / "questions.json"
    reports_dir: Path = ROOT / "reports"
    llm: str = "offline"  # offline | openai: the default for the three settings below
    router: str = "keywords"  # keywords | openai
    sql_writer: str = "rules"  # rules | openai
    generator: str = "extractive"  # extractive | openai
    embeddings: str = "hashing"  # hashing (offline) | openai
    chunking: str = "heading"  # heading | recursive | fixed
    qdrant_url: str | None = None  # None = in-memory Qdrant

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        d = cls()
        llm = (env.get("LLM") or d.llm).lower()
        if llm not in ("offline", "openai"):
            raise ValueError(f"LLM must be 'offline' or 'openai', not {llm!r}")
        online = llm == "openai"
        return cls(
            database_url=env.get("DATABASE_URL") or d.database_url,
            questions_path=Path(env.get("QUESTIONS_PATH") or d.questions_path),
            reports_dir=Path(env.get("REPORTS_DIR") or d.reports_dir),
            llm=llm,
            router=(env.get("ROUTER") or ("openai" if online else d.router)).lower(),
            sql_writer=(env.get("SQL_WRITER") or ("openai" if online else d.sql_writer)).lower(),
            generator=(env.get("GENERATOR") or ("openai" if online else d.generator)).lower(),
            embeddings=(env.get("EMBEDDINGS") or d.embeddings).lower(),
            chunking=(env.get("CHUNKING") or d.chunking).lower(),
            qdrant_url=env.get("QDRANT_URL") or None,
        )
