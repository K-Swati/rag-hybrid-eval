"""A local Postgres for running the app without Docker.

``pgserver`` ships real Postgres binaries as a pip package. The data lives in
``.pgdata/`` (git-ignored), so the database is created and seeded once and kept
between runs. The server stops when the Python process that started it exits.
Only the CLI entry point calls this; nothing below it knows where Postgres came from.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval.warehouse.db import _with_database, init_database

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / ".pgdata"


def start_local_postgres(data_dir: Path = DEFAULT_DATA_DIR, reset: bool = False) -> str:
    """Start (or reuse) the local server, seed ``icecream`` if missing, return the read-only URL."""
    try:
        import pgserver
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("pgserver is not installed: pip install -r requirements.txt") from exc
    import psycopg

    data_dir.mkdir(parents=True, exist_ok=True)
    admin_url = pgserver.get_server(data_dir, cleanup_mode="stop").get_uri()
    with psycopg.connect(admin_url) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'icecream'").fetchone()
    db_url = init_database(admin_url, reset=True) if (reset or not exists) else _with_database(admin_url, "icecream")
    return as_reader(db_url)


def as_reader(url: str) -> str:
    """Same server and database, logged in as the read-only ``rag_reader`` role."""
    scheme, rest = url.split("://", 1)
    return f"{scheme}://rag_reader:rag_reader@{rest.rsplit('@', 1)[-1]}"
