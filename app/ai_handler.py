"""Question understanding + answer formatting.

Architecture (the rule that keeps answers correct; the LLM is Google Gemini, called over plain HTTP):

    question --> [LLM or keyword fallback] --> intent (+ months)   <- the ONLY thing AI decides
             --> Python validates the intent against a fixed whitelist
             --> analytics.py runs an approved, parameterised SQL query
             --> Python formats the answer text from the SQL result

The LLM never sees the data, never writes SQL and never produces a number.
If the LLM is not configured or fails, a deterministic keyword matcher is used and
the response says so (`interpreter: "keyword_fallback"`).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from app import analytics
from app.analytics import AnalyticsError, format_currency, format_percent, month_label
from app.config import LLMSettings, get_llm_settings
from app.database import get_load_summary, open_readonly

logger = logging.getLogger(__name__)

# The complete list of things this assistant can do. Anything else is "unsupported".
INTENTS = ("total_revenue", "highest_revenue_product", "monthly_revenue_trend", "highest_sales_region", "revenue_change_between_months")
UNSUPPORTED = "unsupported"

SUPPORTED_QUESTIONS = [
    "What is the total revenue?",
    "Which product generated the highest revenue?",
    "What is the monthly revenue trend?",
    "Which region generated the highest sales?",
    "How did revenue change between January and February 2026?",
]


class InterpretationError(Exception):
    """The LLM answered, but not with a usable intent."""


@dataclass
class Interpretation:
    intent: str                                   # one of INTENTS or UNSUPPORTED
    params: Dict[str, str] = field(default_factory=dict)
    interpreter: str = "keyword_fallback"         # "llm" | "keyword_fallback"
    note: Optional[str] = None                    # e.g. why the LLM was not used
    message: Optional[str] = None                 # explanation when intent is unsupported


# =========================================================================== #
# 1. Safety guard: refuse SQL / destructive commands before anything else
# =========================================================================== #
_SQL_RE = re.compile(
    r"\bselect\b.+\bfrom\b|\binsert\s+into\b|\bupdate\s+\w+\s+set\b|\bdelete\s+from\b"
    r"|\bdrop\s+(table|database|index|view)\b|\balter\s+table\b|\btruncate\s+table\b"
    r"|;\s*(select|insert|update|delete|drop|alter|truncate|pragma|attach)\b"
    r"|\bpragma\s+\w+|\battach\s+database\b",
    re.IGNORECASE | re.DOTALL,
)
_DESTRUCTIVE_RE = re.compile(
    r"\b(delete|drop|truncate|insert|update|alter|remove|erase|wipe|modify)\s+"
    r"(all|every|everything|the|this|that|my|table|tables|database|rows?|records?|data|sales|from|into)\b",
    re.IGNORECASE,
)


def is_blocked_request(question: str) -> bool:
    """True if the text looks like SQL or a command to change/delete data."""
    return bool(_SQL_RE.search(question) or _DESTRUCTIVE_RE.search(question))


BLOCKED_MESSAGE = (
    "This assistant is read-only. It cannot run SQL or modify, delete or insert data. "
    "Please ask a business question about the sales data instead."
)


# =========================================================================== #
# 2. Month extraction ("January and February 2026" -> ["2026-01", "2026-02"])
# =========================================================================== #
_MONTH_NUMBERS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True)) + r")\b\.?(?:\s*,?\s*(\d{4})\b)?",
    re.IGNORECASE,
)
_ISO_RE = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b")
_SLASH_RE = re.compile(r"\b(0?[1-9]|1[0-2])/(\d{4})\b")
# "may" is also an ordinary verb ("How may I ...?"). If one of these words follows, it is the verb.
_MODAL_FOLLOWERS = {"i", "we", "you", "he", "she", "it", "they", "be", "have", "not", "also",
                    "need", "want", "like", "ask", "see", "know", "help", "show", "tell"}


def extract_months(question: str, default_year: Optional[int]) -> List[str]:
    """Return months mentioned in the question as 'YYYY-MM', in the order they appear.

    A month without a year takes the year of the nearest following year-bearing month in
    the question ("January and February 2026" -> both 2026), else the previous one, else
    `default_year` (the dataset's year when it spans a single year).
    """
    found: List[Tuple[int, int, Optional[int]]] = []  # (position, month_number, year or None)

    for m in _NAME_RE.finditer(question):
        word = m.group(1).lower()
        if word == "may":
            following = re.match(r"\s*([a-z]+)", question[m.end():], re.IGNORECASE)
            if following and following.group(1).lower() in _MODAL_FOLLOWERS and not m.group(2):
                continue
        found.append((m.start(), _MONTH_NUMBERS[word], int(m.group(2)) if m.group(2) else None))
    for m in _ISO_RE.finditer(question):
        found.append((m.start(), int(m.group(2)), int(m.group(1))))
    for m in _SLASH_RE.finditer(question):
        found.append((m.start(), int(m.group(1)), int(m.group(2))))
    found.sort()

    months: List[str] = []
    for index, (_, number, year) in enumerate(found):
        if year is None:
            later = [y for (_, _, y) in found[index + 1:] if y]
            earlier = [y for (_, _, y) in found[:index] if y]
            year = later[0] if later else earlier[-1] if earlier else default_year
        if year is None:
            raise AnalyticsError("Please include the year (for example 'January 2026') so I know which month you mean.")
        months.append(f"{year:04d}-{number:02d}")
    return months


def _default_year(available: List[str]) -> Optional[int]:
    years = {int(m[:4]) for m in available}
    return years.pop() if len(years) == 1 else None


# =========================================================================== #
# 3. Keyword / intent fallback (works with no LLM and no internet)
# =========================================================================== #
_TREND_RE = re.compile(r"\b(monthly|month[- ]?wise|month[- ]by[- ]month|per month|each month|by month|trend|trends|over time)\b", re.I)
_COMPARE_RE = re.compile(
    r"\b(compare|compared|comparison|change|changed|difference|differ|vs|versus|growth|grew|"
    r"increase|increased|decrease|decreased|rise|rose|fall|fell|dropped|between)\b", re.I)
_REGION_RE = re.compile(r"\b(region|regions|regional|zone|zones|territory|territories)\b", re.I)
_PRODUCT_RE = re.compile(r"\b(product|products|item|items|sku|skus)\b", re.I)
_MONEY_RE = re.compile(r"\b(revenue|sales|income|turnover|earnings)\b", re.I)
_OTHER_DIMENSION_RE = re.compile(
    r"\b(category|categories|channel|channels|online|retail|customer|customers|discount|discounts|"
    r"quantity|units|profit|margin|cost)\b", re.I)

UNSUPPORTED_MESSAGE = (
    "I can't answer that yet. I can answer: total revenue, the top product by revenue, the monthly "
    "revenue trend, the top region, and how revenue changed between two months."
)
MONTH_FILTER_MESSAGE = (
    "Filtering a single month is not supported yet. Try asking about the overall figure, or compare two "
    "months, e.g. 'How did revenue change between January and February 2026?'"
)


def interpret_with_keywords(question: str, available: List[str]) -> Interpretation:
    """Deterministic intent detection used when no LLM is available."""
    months = extract_months(question, _default_year(available))

    if len(months) >= 2:
        intent = "revenue_change_between_months"
    elif _OTHER_DIMENSION_RE.search(question):
        return Interpretation(UNSUPPORTED, message=UNSUPPORTED_MESSAGE)
    elif _TREND_RE.search(question):
        intent = "monthly_revenue_trend"
    elif _COMPARE_RE.search(question):
        intent = "revenue_change_between_months"       # asked to compare but named < 2 months -> ask for them
    elif _REGION_RE.search(question):
        intent = "highest_sales_region"
    elif _PRODUCT_RE.search(question):
        intent = "highest_revenue_product"
    elif _MONEY_RE.search(question):
        intent = "total_revenue"
    else:
        return Interpretation(UNSUPPORTED, message=UNSUPPORTED_MESSAGE)
    return Interpretation(intent)


# =========================================================================== #
# 4. LLM interpretation (Google Gemini). Only used when configured; always validated.
# =========================================================================== #
# NOTE: literal JSON braces are doubled ({{ }}) because we fill in the month list with str.format().
SYSTEM_PROMPT = """You are an intent classifier for a sales-analytics assistant.
You must NOT answer the question, calculate anything, or write SQL. Only classify it.

Choose exactly one intent:
- total_revenue: the overall / total revenue or sales
- highest_revenue_product: which product earned the most revenue, or product-wise revenue
- monthly_revenue_trend: revenue month by month / monthly trend
- highest_sales_region: which region has the highest sales or revenue, or region-wise revenue
- revenue_change_between_months: how revenue changed between two specific months
- unsupported: anything else, or anything you are unsure about (other metrics, filters, chit-chat,
  requests to change data, ambiguous questions)

For revenue_change_between_months also return month_a (first month mentioned) and month_b (second month
mentioned) as "YYYY-MM". The dataset contains these months: {available_months}.
If the user gives no year, assume the dataset year. For every other intent use null months.

Reply with ONLY a JSON object, no explanation and no markdown, exactly like:
{{"intent": "total_revenue", "month_a": null, "month_b": null}}"""


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _call_llm(question: str, available_months: List[str], settings: LLMSettings) -> str:
    """Ask Gemini to classify the question and return its raw text reply.

    Only the question text and the list of available months are sent. No data, no SQL.
    The API key goes in a request HEADER (never in the URL), so it cannot appear in
    error messages or logs.
    """
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT.format(
            available_months=", ".join(available_months) or "none")}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    response = requests.post(
        GEMINI_URL.format(model=settings.model),
        headers={"x-goog-api-key": settings.api_key, "Content-Type": "application/json"},
        json=body,
        timeout=settings.timeout,
    )
    if response.status_code != 200:
        # Report only the status code: the response body is never copied into our errors.
        raise InterpretationError(f"Gemini returned HTTP {response.status_code}")
    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise InterpretationError("Gemini reply had an unexpected structure") from exc


def _parse_llm_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise InterpretationError("LLM reply did not contain a JSON object")
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise InterpretationError("LLM reply was not valid JSON") from exc
    if not isinstance(data, dict):
        raise InterpretationError("LLM reply JSON was not an object")
    return data


def interpret_with_llm(question: str, available: List[str], settings: LLMSettings) -> Interpretation:
    data = _parse_llm_json(_call_llm(question, available, settings))
    intent = data.get("intent")
    if intent not in INTENTS and intent != UNSUPPORTED:     # whitelist check: LLM output is untrusted
        raise InterpretationError("LLM returned an intent that is not on the approved list")

    if intent == UNSUPPORTED:
        return Interpretation(UNSUPPORTED, interpreter="llm", message=UNSUPPORTED_MESSAGE)

    params: Dict[str, str] = {}
    if intent == "revenue_change_between_months":
        month_a, month_b = data.get("month_a"), data.get("month_b")
        try:
            params = {"month_a": analytics.validate_month(month_a), "month_b": analytics.validate_month(month_b)}
        except AnalyticsError:
            params = {}  # answer_question() will re-extract months from the question text
    return Interpretation(intent, params, interpreter="llm")


def interpret(question: str, available: List[str]) -> Interpretation:
    """Use the LLM if configured and healthy; otherwise (or on any failure) use the keyword fallback."""
    settings = get_llm_settings()
    note: Optional[str]
    if settings.enabled:
        try:
            return interpret_with_llm(question, available, settings)
        except Exception as exc:  # network, auth, timeout, bad JSON, bad intent ... all end up here
            logger.warning("LLM interpretation failed (%s); using keyword fallback", type(exc).__name__)
            # InterpretationError messages are written by us (safe to show); for anything else show only the type.
            reason = str(exc) if isinstance(exc, InterpretationError) else type(exc).__name__
            note = f"The LLM call failed ({reason}); the keyword/intent fallback was used instead."
    else:
        note = settings.unavailable_reason
    fallback = interpret_with_keywords(question, available)
    fallback.note = note
    return fallback


# =========================================================================== #
# 5. Run the approved analytics function and format the answer
# =========================================================================== #
_REASON_TEXT = {
    "MISSING_TRANSACTION_ID": "missing transaction id", "DUPLICATE_TRANSACTION_ID": "duplicate transaction id",
    "INVALID_DATE": "invalid date", "MISSING_TEXT_FIELD": "blank text field", "INVALID_STATUS": "unknown status",
    "INVALID_QUANTITY": "invalid quantity", "NON_POSITIVE_QUANTITY": "zero or negative quantity",
    "MISSING_UNIT_PRICE": "missing unit price", "INVALID_UNIT_PRICE": "invalid unit price",
    "INVALID_DISCOUNT": "invalid discount", "DISCOUNT_EXCEEDS_GROSS": "discount larger than the sale",
}


def _data_quality_note(conn: sqlite3.Connection) -> str:
    """One sentence telling the user how many CSV records were left out of every figure, and why."""
    summary = get_load_summary(conn)
    if not summary:
        return ""
    if not summary["rejected_records"]:
        return f" Data quality: all {summary['total_records']} CSV records passed cleaning."
    why = ", ".join(
        f"{_REASON_TEXT.get(code, code.lower())} x{count}" for code, count in summary["rejection_reasons"].items()
    )
    return (f" Data quality: {summary['rejected_records']} of {summary['total_records']} CSV records were "
            f"rejected during cleaning ({why}) and are not counted.")


_BASE_EXPLANATION = (
    "Revenue = quantity x unit price - discount, summed over Completed transactions only; "
    "Returned transactions are excluded."
)


def _join_names(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _format_total(r: dict) -> Tuple[str, dict, list, str]:
    answer = (f"Total revenue from completed sales is {format_currency(r['total_revenue'])} across "
              f"{r['completed_transactions']} transactions ({r['units_sold']} units sold).")
    data = {**r, "total_revenue_formatted": format_currency(r["total_revenue"])}
    table = [
        {"metric": "Total revenue (completed sales)", "value": format_currency(r["total_revenue"])},
        {"metric": "Completed transactions", "value": r["completed_transactions"]},
        {"metric": "Units sold", "value": r["units_sold"]},
        {"metric": "Discounts given", "value": format_currency(r["total_discount"])},
        {"metric": "Returned transactions excluded", "value": r["returned_transactions_excluded"]},
        {"metric": "Value of returned transactions excluded", "value": format_currency(r["returned_value_excluded"])},
    ]
    explanation = _BASE_EXPLANATION
    return answer, data, table, explanation


def _format_ranking(r: dict, noun: str, key: str) -> Tuple[str, dict, list, str]:
    names = r["top_names"]
    top_row = r["ranking"][0]
    revenue = format_currency(r["top_revenue"])
    if len(names) == 1:
        answer = (f"{names[0]} generated the highest revenue: {revenue} "
                  f"({format_percent(top_row['share_percent'])} of total completed-sales revenue).")
    else:
        answer = f"{_join_names(names)} are tied for the highest revenue at {revenue} each."
    data = {key: names[0], "revenue": r["top_revenue"], "revenue_formatted": revenue,
            "share_percent": top_row["share_percent"], "tied": names if len(names) > 1 else []}
    table = [
        {"rank": x["rank"], noun: x["name"], "revenue": x["revenue"], "revenue_display": format_currency(x["revenue"]),
         "share_percent": x["share_percent"], "units_sold": x["units_sold"], "transactions": x["transactions"]}
        for x in r["ranking"]
    ]
    explanation = f"{_BASE_EXPLANATION} Ranked by revenue across {len(r['ranking'])} {noun}s."
    return answer, data, table, explanation


def _format_trend(r: dict) -> Tuple[str, dict, list, str]:
    hi, lo, last = r["highest"], r["lowest"], r["latest"]
    answer = (f"Revenue was highest in {hi['month_label']} ({format_currency(hi['revenue'])}) and lowest in "
              f"{lo['month_label']} ({format_currency(lo['revenue'])}).")
    if last["percent_change_vs_previous"] is not None:
        direction = "up" if last["change_vs_previous"] > 0 else "down" if last["change_vs_previous"] < 0 else "flat"
        answer += (f" The latest month, {last['month_label']}, was {format_currency(last['revenue'])}, "
                   f"{direction} {format_percent(abs(last['percent_change_vs_previous']))} on the previous month.")
    data = {
        "months_covered": len(r["months"]),
        "highest_month": hi["month_label"], "highest_month_revenue": hi["revenue"],
        "highest_month_revenue_formatted": format_currency(hi["revenue"]),
        "lowest_month": lo["month_label"], "lowest_month_revenue": lo["revenue"],
        "lowest_month_revenue_formatted": format_currency(lo["revenue"]),
    }
    table = [
        {"month": m["month_label"], "month_key": m["month"], "revenue": m["revenue"],
         "revenue_display": format_currency(m["revenue"]), "transactions": m["transactions"],
         "change_vs_previous": "-" if m["change_vs_previous"] is None else format_currency(m["change_vs_previous"]),
         "percent_change": None if m["percent_change_vs_previous"] is None else m["percent_change_vs_previous"]}
        for m in r["months"]
    ]
    explanation = (f"{_BASE_EXPLANATION} Orders are grouped by order month; months with no completed "
                   f"sales do not appear.")
    return answer, data, table, explanation


def _format_comparison(r: dict) -> Tuple[str, dict, list, str]:
    change = format_currency(abs(r["absolute_change"]))
    if r["direction"] == "no change":
        answer = (f"Revenue was unchanged between {r['month_a_label']} and {r['month_b_label']} "
                  f"({format_currency(r['revenue_a'])}).")
    else:
        verb = "increased" if r["direction"] == "increase" else "decreased"
        pct = f" ({format_percent(abs(r['percent_change']))})" if r["percent_change"] is not None else ""
        answer = (f"Revenue {verb} by {change}{pct} from {r['month_a_label']} "
                  f"({format_currency(r['revenue_a'])}) to {r['month_b_label']} ({format_currency(r['revenue_b'])}).")
    data = {**r, "revenue_a_formatted": format_currency(r["revenue_a"]),
            "revenue_b_formatted": format_currency(r["revenue_b"]),
            "absolute_change_formatted": format_currency(r["absolute_change"])}
    table = [
        {"month": r["month_a_label"], "revenue": r["revenue_a"], "revenue_display": format_currency(r["revenue_a"]),
         "transactions": r["transactions_a"]},
        {"month": r["month_b_label"], "revenue": r["revenue_b"], "revenue_display": format_currency(r["revenue_b"]),
         "transactions": r["transactions_b"]},
    ]
    explanation = (f"{_BASE_EXPLANATION} Change = {r['month_b_label']} revenue minus {r['month_a_label']} revenue; "
                   f"percentage is relative to {r['month_a_label']}.")
    return answer, data, table, explanation


def run_intent(conn: sqlite3.Connection, intent: str, params: Dict[str, str]) -> Tuple[str, dict, list, str]:
    """Dispatch to ONE approved analytics function. Unknown intents cannot get past this table."""
    if intent == "total_revenue":
        return _format_total(analytics.total_revenue(conn))
    if intent == "highest_revenue_product":
        return _format_ranking(analytics.product_revenue_ranking(conn), "product", "product")
    if intent == "highest_sales_region":
        return _format_ranking(analytics.region_revenue_ranking(conn), "region", "region")
    if intent == "monthly_revenue_trend":
        return _format_trend(analytics.monthly_revenue_trend(conn))
    if intent == "revenue_change_between_months":
        if "month_a" not in params or "month_b" not in params:
            raise AnalyticsError("Please name the two months to compare, for example: "
                                 "'How did revenue change between January and February 2026?'")
        return _format_comparison(analytics.compare_months(conn, params["month_a"], params["month_b"]))
    raise AnalyticsError(UNSUPPORTED_MESSAGE)


# =========================================================================== #
# 6. Public entry points
# =========================================================================== #
def _failure(question: str, status: int, message: str, interp: Optional[Interpretation] = None,
             data: Optional[dict] = None) -> Tuple[int, dict]:
    return status, {
        "question": question, "answer": message, "data": data or {}, "supporting_table": [],
        "explanation": "The question was not answered, so no figures were calculated.",
        "success": False,
        "intent": interp.intent if interp else None,
        "interpreter": interp.interpreter if interp else None,
        "interpreter_note": interp.note if interp else None,
    }


def answer_question(question: str, conn: sqlite3.Connection) -> Tuple[int, dict]:
    """Answer one question. Returns (http_status, json_payload).

    sqlite3.Error is deliberately NOT caught here; the caller maps it to HTTP 503.
    """
    question = (question or "").strip()
    if not question:
        return _failure(question, 400, "Please type a question.")
    if is_blocked_request(question):
        return _failure(question, 400, BLOCKED_MESSAGE)

    available = analytics.available_months(conn)
    interp: Optional[Interpretation] = None
    try:
        interp = interpret(question, available)
        if interp.intent == UNSUPPORTED:
            return _failure(question, 400, interp.message or UNSUPPORTED_MESSAGE, interp,
                            {"supported_questions": SUPPORTED_QUESTIONS})

        mentioned = extract_months(question, _default_year(available))
        if interp.intent == "revenue_change_between_months":
            if not interp.params:                       # keyword path, or LLM gave no valid months
                if len(mentioned) == 2:
                    interp.params = {"month_a": mentioned[0], "month_b": mentioned[1]}
                elif len(mentioned) > 2:
                    raise AnalyticsError("Please name exactly two months to compare.")
        elif mentioned:                                 # e.g. "total revenue in March": would silently ignore the month
            return _failure(question, 400, MONTH_FILTER_MESSAGE, interp,
                            {"supported_questions": SUPPORTED_QUESTIONS})

        answer, data, table, explanation = run_intent(conn, interp.intent, interp.params)
    except AnalyticsError as exc:
        return _failure(question, 400, str(exc), interp,
                        {"supported_questions": SUPPORTED_QUESTIONS})

    return 200, {
        "question": question, "answer": answer, "data": data, "supporting_table": table,
        "explanation": explanation + _data_quality_note(conn),
        "success": True, "intent": interp.intent, "interpreter": interp.interpreter,
        "interpreter_note": interp.note,
    }


def answer_question_from_db(question: str, db_path) -> Tuple[int, dict]:
    """Open the database read-only, answer, and turn database/unexpected errors into 503/500."""
    try:
        with open_readonly(db_path) as conn:
            return answer_question(question, conn)
    except sqlite3.Error:
        logger.exception("Database error while answering a question")
        return _failure(question, 503,
                        "The analytics database is unavailable. Build it with `python -m app.database` and try again.")
    except Exception:  # last-resort safety net; details are logged, not leaked to the client
        logger.exception("Unexpected error while answering a question")
        return _failure(question, 500, "Something went wrong while answering that question.")
