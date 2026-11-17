"""SQL safety and result handling, independent of any database driver.

``check_select`` is the first of four guards on generated SQL. The others are
the read-only transaction and statement timeout in ``Warehouse.query`` and the
read-only ``rag_reader`` database role (``db/init/03_reader_role.sql``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|copy|vacuum|analyze|cluster|reindex|"
    r"call|do|execute|prepare|deallocate|listen|notify|unlisten|lock|set|reset|comment|refresh|security|discard|"
    r"pg_sleep|pg_read_file|pg_read_binary_file|pg_ls_dir|pg_terminate_backend|pg_cancel_backend|dblink|lo_import|"
    r"lo_export|into)\b",
    re.IGNORECASE,
)


class UnsafeSQL(ValueError):
    """Raised when generated SQL is not a single read-only SELECT."""


def _strip_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return re.sub(r'"(?:[^"]|"")*"', '""', sql)


def check_select(sql: str) -> str:
    """Return ``sql`` without a trailing semicolon if it is one read-only SELECT, else raise ``UnsafeSQL``."""
    cleaned = sql.strip().rstrip(";").strip()
    bare = _strip_literals_and_comments(cleaned)
    if not cleaned:
        raise UnsafeSQL("empty query")
    if ";" in bare:
        raise UnsafeSQL("only one statement is allowed")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeSQL("only SELECT queries are allowed")
    bad = _FORBIDDEN.search(bare)
    if bad:
        raise UnsafeSQL(f"keyword {bad.group(1).upper()!r} is not allowed")
    return cleaned


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool = False
    ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def scalar(self):
        return self.rows[0][0] if len(self.rows) == 1 and len(self.columns) == 1 else None


def plain(value):
    """Driver values -> JSON-friendly values (Decimal -> float, date -> ISO string)."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _norm(value) -> str:
    value = plain(value)
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{round(float(value), 1):.1f}"
    return str(value).strip().lower()


def same_result(expected: QueryResult, got: QueryResult) -> bool:
    """True when ``got`` answers like ``expected``: same row count and every expected column appears in ``got``.

    Columns are matched by their values, not their names, and rows as a multiset, so
    aliases, column order and extra columns (such as an added SKU) do not count as errors.
    """
    if len(expected.rows) != len(got.rows):
        return False
    got_cols = [sorted(_norm(r[i]) for r in got.rows) for i in range(len(got.columns))]
    return all(
        sorted(_norm(r[i]) for r in expected.rows) in got_cols for i in range(len(expected.columns))
    )


def indian_group(n: int) -> str:
    """12345678 -> '1,23,45,678' (Indian digit grouping)."""
    s = str(abs(int(n)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join(re.findall(r"\d{1,2}", head[::-1]))[::-1]
        s = f"{head},{tail}"
    return ("-" if n < 0 else "") + s


def format_inr(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} crore"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} lakh"
    return f"₹{indian_group(round(x))}"


def format_value(value, column: str = "") -> str:
    value = plain(value)
    col = column.lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if col in ("year", "yr"):
            return str(int(value))
        if any(w in col for w in ("inr", "revenue", "discount", "margin", "price", "cost", "mrp")):
            return format_inr(value)
        if "pct" in col or "percent" in col or "share" in col:
            return f"{value:.1f}%"
        if float(value).is_integer():
            return indian_group(int(value))
        return f"{value:,.2f}"
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-01", value) and "month" in col:
        return datetime.fromisoformat(value).strftime("%b %Y")
    return str(value)
