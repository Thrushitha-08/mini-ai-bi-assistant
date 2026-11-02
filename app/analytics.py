"""The five analytics functions. Every number comes from a SQL query.

Business rules (documented assumptions)
  * Revenue = Quantity x Unit_price - Discount_amount  (per transaction, computed IN SQL).
  * Only transactions with status 'Completed' count. 'Returned' ones are excluded entirely
    (the CSV does not say how returns should be refunded, so the simplest rule is used:
    a returned sale is not revenue). Their count/value is still reported for transparency.
  * Rows that failed cleaning (zero quantity, missing price, ...) are not in the table at all.
  * Amounts are assumed to be INR and formatted with Indian digit grouping (Rs 14,39,200).

Safety: user text NEVER becomes SQL. The only user-derived values that reach a query are
month strings that are validated against ^YYYY-MM$ and passed as bound parameters.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional

from app import config

COMPLETED = "Completed"
RETURNED = "Returned"

# Constant SQL fragment (not user input). Revenue of ONE transaction.
REVENUE_EXPR = "(quantity * unit_price - discount_amount)"

# Columns allowed in a "group by" ranking. Whitelisted, never taken from the user.
_RANKING_COLUMNS = {"product", "region"}

_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class AnalyticsError(Exception):
    """The request cannot be answered (bad parameters, month not in data, ...). Maps to HTTP 400."""


# --------------------------------------------------------------------------- #
# Formatting helpers (one consistent approach for the whole app)
# --------------------------------------------------------------------------- #
def tidy_number(value) -> float:
    """Whole numbers become ints (1439200 instead of 1439200.0); others keep 2 decimals."""
    value = round(float(value), 2)
    return int(value) if value == int(value) else value


def format_currency(amount) -> str:
    """Indian digit grouping: 1439200 -> '₹14,39,200', 1234.5 -> '₹1,234.50'."""
    amount = round(float(amount), 2)
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(amount):.2f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    grouped = ",".join(groups + [tail]) if groups else tail
    suffix = "" if frac == "00" else f".{frac}"
    return f"{sign}{config.CURRENCY_SYMBOL}{grouped}{suffix}"


def format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def month_label(month: str) -> str:
    """'2026-01' -> 'January 2026'."""
    year, mm = month.split("-")
    return f"{_MONTH_NAMES[int(mm) - 1]} {year}"


def validate_month(month: str) -> str:
    if not isinstance(month, str) or not _MONTH_RE.match(month):
        raise AnalyticsError(f"'{month}' is not a valid month (expected YYYY-MM, e.g. 2026-01).")
    return month


# --------------------------------------------------------------------------- #
# Helpers that read the data
# --------------------------------------------------------------------------- #
def available_months(conn: sqlite3.Connection) -> List[str]:
    """Months (YYYY-MM) that have at least one completed sale, oldest first."""
    rows = conn.execute(
        "SELECT DISTINCT order_month FROM sales WHERE status = ? ORDER BY order_month", (COMPLETED,)
    ).fetchall()
    return [r["order_month"] for r in rows]


def _share(part: float, whole: float) -> Optional[float]:
    return round(part / whole * 100, 1) if whole else None


def _percent_change(old: float, new: float) -> Optional[float]:
    """% change from old to new; None when old is 0 (undefined)."""
    return round((new - old) / old * 100, 1) if old else None


# --------------------------------------------------------------------------- #
# 1. Total revenue
# --------------------------------------------------------------------------- #
def total_revenue(conn: sqlite3.Connection) -> Dict[str, object]:
    sql = (
        f"SELECT COALESCE(SUM({REVENUE_EXPR}), 0) AS revenue, COUNT(*) AS transactions, "
        "COALESCE(SUM(quantity), 0) AS units, COALESCE(SUM(discount_amount), 0) AS discount "
        "FROM sales WHERE status = ?"
    )
    done = conn.execute(sql, (COMPLETED,)).fetchone()
    returned = conn.execute(sql, (RETURNED,)).fetchone()
    return {
        "total_revenue": tidy_number(done["revenue"]),
        "completed_transactions": done["transactions"],
        "units_sold": done["units"],
        "total_discount": tidy_number(done["discount"]),
        "returned_transactions_excluded": returned["transactions"],
        "returned_value_excluded": tidy_number(returned["revenue"]),
    }


# --------------------------------------------------------------------------- #
# 2 & 4. Rankings (product, region) share one safe implementation
# --------------------------------------------------------------------------- #
def _revenue_ranking(conn: sqlite3.Connection, column: str) -> Dict[str, object]:
    if column not in _RANKING_COLUMNS:  # defence in depth: column name is never user input
        raise ValueError(f"unsupported ranking column: {column}")
    sql = (
        f"SELECT {column} AS name, SUM({REVENUE_EXPR}) AS revenue, "
        "SUM(quantity) AS units, COUNT(*) AS transactions "
        f"FROM sales WHERE status = ? GROUP BY {column} "
        "ORDER BY revenue DESC, name ASC"
    )
    rows = conn.execute(sql, (COMPLETED,)).fetchall()
    if not rows:
        raise AnalyticsError("There are no completed sales in the database.")

    total = sum(r["revenue"] for r in rows)
    ranking = [
        {
            "rank": i,
            "name": r["name"],
            "revenue": tidy_number(r["revenue"]),
            "units_sold": r["units"],
            "transactions": r["transactions"],
            "share_percent": _share(r["revenue"], total),
        }
        for i, r in enumerate(rows, start=1)
    ]
    top_revenue = ranking[0]["revenue"]
    return {
        "ranking": ranking,
        "top_names": [r["name"] for r in ranking if r["revenue"] == top_revenue],  # handles ties
        "top_revenue": top_revenue,
        "total_revenue": tidy_number(total),
    }


def product_revenue_ranking(conn: sqlite3.Connection) -> Dict[str, object]:
    return _revenue_ranking(conn, "product")


def region_revenue_ranking(conn: sqlite3.Connection) -> Dict[str, object]:
    return _revenue_ranking(conn, "region")


# --------------------------------------------------------------------------- #
# 3. Monthly trend
# --------------------------------------------------------------------------- #
def monthly_revenue_trend(conn: sqlite3.Connection) -> Dict[str, object]:
    sql = (
        f"SELECT order_month AS month, SUM({REVENUE_EXPR}) AS revenue, COUNT(*) AS transactions "
        "FROM sales WHERE status = ? GROUP BY order_month ORDER BY order_month"
    )
    rows = conn.execute(sql, (COMPLETED,)).fetchall()
    if not rows:
        raise AnalyticsError("There are no completed sales in the database.")

    months: List[dict] = []
    previous: Optional[float] = None
    for r in rows:
        revenue = float(r["revenue"])
        months.append({
            "month": r["month"],
            "month_label": month_label(r["month"]),
            "revenue": tidy_number(revenue),
            "transactions": r["transactions"],
            "change_vs_previous": None if previous is None else tidy_number(revenue - previous),
            "percent_change_vs_previous": None if previous is None else _percent_change(previous, revenue),
        })
        previous = revenue

    highest = max(months, key=lambda m: m["revenue"])
    lowest = min(months, key=lambda m: m["revenue"])
    return {"months": months, "highest": highest, "lowest": lowest, "latest": months[-1]}


# --------------------------------------------------------------------------- #
# 5. Compare two months
# --------------------------------------------------------------------------- #
def _month_revenue(conn: sqlite3.Connection, month: str) -> Dict[str, object]:
    row = conn.execute(
        f"SELECT COALESCE(SUM({REVENUE_EXPR}), 0) AS revenue, COUNT(*) AS transactions "
        "FROM sales WHERE status = ? AND order_month = ?",
        (COMPLETED, month),
    ).fetchone()
    return {"revenue": float(row["revenue"]), "transactions": row["transactions"]}


def compare_months(conn: sqlite3.Connection, month_a: str, month_b: str) -> Dict[str, object]:
    """Revenue change going FROM month_a TO month_b (both 'YYYY-MM')."""
    validate_month(month_a)
    validate_month(month_b)
    if month_a == month_b:
        raise AnalyticsError("Please give two different months to compare.")

    known = available_months(conn)
    for month in (month_a, month_b):
        if month not in known:
            span = f"{month_label(known[0])} to {month_label(known[-1])}" if known else "none"
            raise AnalyticsError(
                f"There are no completed sales for {month_label(month)}. Data is available for: {span}."
            )

    a, b = _month_revenue(conn, month_a), _month_revenue(conn, month_b)
    change = b["revenue"] - a["revenue"]
    return {
        "month_a": month_a,
        "month_a_label": month_label(month_a),
        "revenue_a": tidy_number(a["revenue"]),
        "transactions_a": a["transactions"],
        "month_b": month_b,
        "month_b_label": month_label(month_b),
        "revenue_b": tidy_number(b["revenue"]),
        "transactions_b": b["transactions"],
        "absolute_change": tidy_number(change),
        "percent_change": _percent_change(a["revenue"], b["revenue"]),
        "direction": "increase" if change > 0 else "decrease" if change < 0 else "no change",
    }
