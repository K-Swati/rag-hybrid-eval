"""Postgres access: the data dictionary, table browsing, export rows and read-only queries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rag_eval.core.sql import QueryResult, check_select, plain

INIT_DIR = Path(__file__).resolve().parent.parent.parent / "db" / "init"
EXPORT_TABLES = ("regions", "plants", "suppliers", "products", "product_ingredients", "distributors",
                 "production_batches", "promotions", "quality_incidents", "documents")


class WarehouseUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str
    primary_key: bool
    references: str | None  # "table.column" for foreign keys


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    rows: int
    columns: list[Column]


_CATALOG_SQL = """
SELECT c.relname, coalesce(obj_description(c.oid, 'pg_class'), ''),
       a.attname, format_type(a.atttypid, a.atttypmod), coalesce(col_description(c.oid, a.attnum), ''),
       EXISTS (SELECT 1 FROM pg_constraint k WHERE k.conrelid = c.oid AND k.contype = 'p' AND a.attnum = ANY (k.conkey)),
       (SELECT f.relname || '.' || fa.attname FROM pg_constraint k
          JOIN pg_class f ON f.oid = k.confrelid
          JOIN pg_attribute fa ON fa.attrelid = k.confrelid AND fa.attnum = k.confkey[1]
         WHERE k.conrelid = c.oid AND k.contype = 'f' AND k.conkey[1] = a.attnum LIMIT 1)
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
  JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
 WHERE c.relkind = 'r'
 ORDER BY c.relname, a.attnum
"""


class Warehouse:
    """One short-lived connection per call: simple, thread-safe and fine at this scale."""

    def __init__(self, url: str, timeout_ms: int = 5000, max_rows: int = 200) -> None:
        self.url, self.timeout_ms, self.max_rows = url, timeout_ms, max_rows
        self._catalog: list[Table] | None = None

    def _connect(self):
        import psycopg

        try:
            return psycopg.connect(self.url, connect_timeout=5)
        except psycopg.OperationalError as exc:
            raise WarehouseUnavailable(
                f"cannot reach Postgres at {self.safe_url}: {str(exc).strip().splitlines()[0]}\n"
                "Start it with `docker compose up -d postgres`, or set DATABASE_URL."
            ) from exc

    @property
    def safe_url(self) -> str:
        """The URL without its password, for logs and error messages."""
        if "@" in self.url and "://" in self.url:
            scheme, rest = self.url.split("://", 1)
            creds, host = rest.rsplit("@", 1)
            return f"{scheme}://{creds.split(':', 1)[0]}:***@{host}"
        return self.url

    def ping(self) -> str:
        with self._connect() as conn:
            return conn.execute("SELECT version()").fetchone()[0]

    # ---------- data dictionary ----------

    def catalog(self, refresh: bool = False) -> list[Table]:
        if self._catalog is not None and not refresh:
            return self._catalog
        with self._connect() as conn:
            raw = conn.execute(_CATALOG_SQL).fetchall()
            tables: dict[str, tuple[str, list[Column]]] = {}
            for tname, tdesc, cname, ctype, cdesc, pk, ref in raw:
                tables.setdefault(tname, (tdesc, []))[1].append(Column(cname, ctype, cdesc, pk, ref))
            counts = {t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables}
        self._catalog = [Table(t, d, counts[t], cols) for t, (d, cols) in tables.items()]
        return self._catalog

    def table(self, name: str) -> Table:
        for t in self.catalog():
            if t.name == name:
                return t
        raise KeyError(name)

    def schema_text(self) -> str:
        """The data dictionary as compact text, for the SQL-writing prompt."""
        lines = []
        for t in self.catalog():
            lines.append(f"TABLE {t.name} -- {t.description} ({t.rows} rows)")
            for c in t.columns:
                extra = " PK" if c.primary_key else ""
                extra += f" -> {c.references}" if c.references else ""
                desc = f" -- {c.description}" if c.description else ""
                lines.append(f"  {c.name} {c.type}{extra}{desc}")
        return "\n".join(lines)

    # ---------- reads ----------

    def rows(self, table: str, limit: int = 50, offset: int = 0, order: str | None = None) -> QueryResult:
        t = self.table(table)  # validates the name against the catalog
        key = order if order in {c.name for c in t.columns} else t.columns[0].name
        return self.query(f'SELECT * FROM "{t.name}" ORDER BY "{key}" LIMIT {int(limit)} OFFSET {int(offset)}',
                          max_rows=limit)

    def fetch_all(self, tables: tuple[str, ...] = EXPORT_TABLES) -> dict[str, list[dict]]:
        """Whole tables as row dicts, for the knowledge-base export."""
        out = {}
        with self._connect() as conn:
            for t in tables:
                cur = conn.execute(f'SELECT * FROM "{t}"')
                cols = [d.name for d in cur.description]
                out[t] = [{cols[i]: plain(v) for i, v in enumerate(row)} for row in cur.fetchall()]
        return out

    def query(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Run one SELECT in a read-only transaction with a timeout. Raises ``UnsafeSQL`` or the driver's error."""
        sql = check_select(sql)
        limit = max_rows or self.max_rows
        t0 = time.perf_counter()
        with self._connect() as conn:
            conn.read_only = True
            with conn.transaction():
                conn.execute(f"SET LOCAL statement_timeout = {int(self.timeout_ms)}")
                cur = conn.execute(sql)
                cols = [d.name for d in cur.description or []]
                rows = cur.fetchmany(limit + 1)
        truncated = len(rows) > limit
        return QueryResult(cols, [[plain(v) for v in r] for r in rows[:limit]], truncated,
                           round((time.perf_counter() - t0) * 1000, 1))


def init_database(admin_url: str, name: str = "icecream", reset: bool = False) -> str:
    """Create database ``name`` from ``db/init/*.sql`` using an admin connection. Returns its URL.

    Docker runs the same files on first start, so this is for a Postgres you run yourself.
    """
    import psycopg
    from psycopg import sql as pgsql

    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if exists and reset:
            conn.execute(pgsql.SQL("DROP DATABASE {} WITH (FORCE)").format(pgsql.Identifier(name)))
            exists = None
        if exists:
            raise RuntimeError(f"database {name!r} already exists; pass reset=True (--reset) to recreate it")
        conn.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(name)))
    db_url = _with_database(admin_url, name)
    with psycopg.connect(db_url, autocommit=True) as conn:
        for path in sorted(INIT_DIR.glob("*.sql")):
            conn.execute(path.read_text(encoding="utf-8"))
    return db_url


def _with_database(url: str, name: str) -> str:
    """Swap the database name in a postgresql:// URL, keeping host, credentials and query string."""
    head, sep, query = url.partition("?")
    scheme, rest = head.split("://", 1)
    authority = rest.split("/", 1)[0]
    return f"{scheme}://{authority}/{name}{sep}{query}"
