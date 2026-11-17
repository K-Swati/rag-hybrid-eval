"""Shared fixtures. Tests that need Postgres get a fresh ``icecream`` database.

Where Postgres comes from, in order:
1. ``TEST_ADMIN_URL``: an admin URL for a server you run (CI uses a service container).
2. ``pgserver`` (in requirements.txt): a throwaway local Postgres, no Docker needed.
Without either, database tests are skipped and the rest still run.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from rag_eval.warehouse.db import init_database


def reader_url(url: str) -> str:
    """Same server and database, logged in as the read-only role."""
    scheme, rest = url.split("://", 1)
    host = rest.rsplit("@", 1)[-1]
    return f"{scheme}://rag_reader:rag_reader@{host}"


@pytest.fixture(scope="session")
def admin_url():
    url = os.environ.get("TEST_ADMIN_URL")
    if url:
        yield url
        return
    try:
        import pgserver
    except ImportError:
        pytest.skip("no Postgres: set TEST_ADMIN_URL or pip install pgserver")
    with tempfile.TemporaryDirectory() as tmp:
        srv = pgserver.get_server(tmp, cleanup_mode="stop")
        yield srv.get_uri()


@pytest.fixture(scope="session")
def db_url(admin_url) -> str:
    """URL of a freshly seeded ``icecream`` database, as the read-only ``rag_reader`` role."""
    return reader_url(init_database(admin_url, reset=True))
