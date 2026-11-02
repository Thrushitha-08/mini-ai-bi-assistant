"""Question understanding: keyword fallback, LLM path (faked), and safety guards."""
import json

import pytest

from app import ai_handler
from app.ai_handler import answer_question, extract_months, is_blocked_request
from app.analytics import AnalyticsError
from app.config import get_llm_settings


def ask(conn, question):
    return answer_question(question, conn)


# ---------------------------------------------------------------- keyword fallback (no LLM)
@pytest.mark.parametrize("question, intent", [
    ("What is the total revenue?", "total_revenue"),
    ("how much sales did we make overall", "total_revenue"),
    ("Which product generated the highest revenue?", "highest_revenue_product"),
    ("What is the monthly revenue trend?", "monthly_revenue_trend"),
    ("show revenue month by month", "monthly_revenue_trend"),
    ("Which region generated the highest sales?", "highest_sales_region"),
    ("How did revenue change between January and February 2026?", "revenue_change_between_months"),
    ("compare march vs april", "revenue_change_between_months"),
])
def test_supported_questions_are_understood_without_an_llm(conn, question, intent):
    status, payload = ask(conn, question)
    assert status == 200 and payload["success"] is True
    assert payload["intent"] == intent
    assert payload["interpreter"] == "keyword_fallback"          # honest: no LLM was called
    assert "fallback" in payload["interpreter_note"]


@pytest.mark.parametrize("question", [
    "What is the weather today?", "Write me a poem", "revenue by category", "profit margin by product",
])
def test_unsupported_questions_are_refused_with_guidance(conn, question):
    status, payload = ask(conn, question)
    assert status == 400 and payload["success"] is False
    assert payload["data"]["supported_questions"]
    assert payload["supporting_table"] == []


def test_comparison_without_months_asks_for_them_instead_of_guessing(conn):
    status, payload = ask(conn, "How did revenue change between two months?")
    assert status == 400 and payload["success"] is False
    assert "name the two months" in payload["answer"]


def test_month_filter_is_declined_rather_than_silently_ignored(conn):
    status, payload = ask(conn, "What is the total revenue in March 2026?")
    assert status == 400 and "single month" in payload["answer"]


def test_month_outside_the_data_gets_a_clear_message(conn):
    status, payload = ask(conn, "Compare January 2025 and February 2025")
    assert status == 400 and "no completed sales for January 2025" in payload["answer"]


def test_answers_are_formatted_from_database_numbers(conn):
    _, payload = ask(conn, "Which product generated the highest revenue?")
    top = payload["supporting_table"][0]
    assert payload["data"]["product"] == top["product"]
    assert payload["data"]["revenue"] == top["revenue"]
    assert top["product"] in payload["answer"] and top["revenue_display"] in payload["answer"]


# ---------------------------------------------------------------- safety guards
@pytest.mark.parametrize("text", [
    "DROP TABLE sales;", "delete all records", "SELECT * FROM sales", "'; DROP TABLE sales; --",
    "update sales set unit_price = 0", "please delete the data", "INSERT INTO sales VALUES (1)",
    "total revenue; DELETE FROM sales", "alter table sales add column x",
])
def test_sql_and_destructive_requests_are_blocked_and_data_is_untouched(conn, text):
    before = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    status, payload = ask(conn, text)
    assert status == 400 and payload["success"] is False and "read-only" in payload["answer"]
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == before


@pytest.mark.parametrize("text", [
    "Why did revenue drop between January and February 2026?", "Which product had the biggest drop in revenue?",
    "What is the total revenue?",
])
def test_normal_business_wording_is_not_blocked(text):
    assert is_blocked_request(text) is False


# ---------------------------------------------------------------- month extraction
@pytest.mark.parametrize("text, expected", [
    ("January and February 2026", ["2026-01", "2026-02"]),
    ("between Jan 2026 and Mar 2026", ["2026-01", "2026-03"]),
    ("compare 2026-02 with 2026-04", ["2026-02", "2026-04"]),
    ("march vs april", ["2026-03", "2026-04"]),                  # year defaults to the dataset's year
    ("December 2025 to January 2026", ["2025-12", "2026-01"]),
    ("Revenue in May and June", ["2026-05", "2026-06"]),
    ("How may I see the revenue?", []),                           # "may" the verb is not a month
    ("What is the total revenue?", []),
])
def test_extract_months(text, expected):
    assert extract_months(text, default_year=2026) == expected


def test_extract_months_needs_a_year_when_the_data_spans_several():
    with pytest.raises(AnalyticsError, match="include the year"):
        extract_months("compare january and february", default_year=None)


# ---------------------------------------------------------------- LLM path (the Gemini call is faked)
def enable_llm(monkeypatch, reply):
    """Pretend an LLM is configured; `reply` is a string, or an Exception to raise."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    def fake(question, available_months, settings):
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(ai_handler, "_call_llm", fake)


def test_llm_only_picks_the_intent_python_and_sql_produce_the_numbers(conn, monkeypatch):
    _, from_keywords = ask(conn, "What is the total revenue?")
    enable_llm(monkeypatch, '{"intent": "total_revenue", "month_a": null, "month_b": null}')
    status, payload = ask(conn, "give me the big picture on income")      # no keyword would catch this
    assert status == 200 and payload["interpreter"] == "llm" and payload["interpreter_note"] is None
    assert payload["data"]["total_revenue"] == from_keywords["data"]["total_revenue"]   # same SQL number


def test_llm_reply_wrapped_in_markdown_fences_is_accepted(conn, monkeypatch):
    enable_llm(monkeypatch, '```json\n{"intent": "highest_sales_region", "month_a": null, "month_b": null}\n```')
    status, payload = ask(conn, "who is winning geographically?")
    assert status == 200 and payload["intent"] == "highest_sales_region" and payload["interpreter"] == "llm"


def test_llm_month_comparison_is_validated_and_used(conn, monkeypatch):
    enable_llm(monkeypatch, '{"intent": "revenue_change_between_months", "month_a": "2026-02", "month_b": "2026-03"}')
    status, payload = ask(conn, "how did the second month go versus the third")
    assert status == 200 and payload["data"]["month_a"] == "2026-02" and payload["data"]["month_b"] == "2026-03"


def test_llm_month_comparison_with_bad_months_is_rejected_by_python(conn, monkeypatch):
    enable_llm(monkeypatch, '{"intent": "revenue_change_between_months", "month_a": "2026-01\'; DROP TABLE sales;--", "month_b": "2026-02"}')
    status, payload = ask(conn, "compare things")
    assert status == 400 and payload["success"] is False
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] > 0


@pytest.mark.parametrize("bad_reply", [
    '{"intent": "DROP TABLE sales"}',                    # not on the approved list
    '{"intent": "run_sql", "sql": "DELETE FROM sales"}',
    "I think the answer is 42",                          # no JSON at all
    "{not json}",
    "[1, 2, 3]",
])
def test_untrusted_llm_output_falls_back_safely(conn, monkeypatch, bad_reply):
    enable_llm(monkeypatch, bad_reply)
    status, payload = ask(conn, "What is the total revenue?")
    assert status == 200 and payload["interpreter"] == "keyword_fallback"
    assert "LLM call failed" in payload["interpreter_note"]


def test_llm_outage_falls_back_and_does_not_leak_details(conn, monkeypatch):
    enable_llm(monkeypatch, ConnectionError("401 invalid key test-key-not-real"))
    status, payload = ask(conn, "Which product generated the highest revenue?")
    assert status == 200 and payload["interpreter"] == "keyword_fallback"
    assert "ConnectionError" in payload["interpreter_note"]
    assert "test-key" not in json.dumps(payload)           # the key / error text never reaches the client


def test_llm_saying_unsupported_is_respected(conn, monkeypatch):
    enable_llm(monkeypatch, '{"intent": "unsupported", "month_a": null, "month_b": null}')
    status, payload = ask(conn, "What is the total revenue?")
    assert status == 400 and payload["interpreter"] == "llm"


def test_python_still_blocks_month_filters_even_if_the_llm_ignores_them(conn, monkeypatch):
    enable_llm(monkeypatch, '{"intent": "total_revenue", "month_a": null, "month_b": null}')
    status, payload = ask(conn, "total revenue in March 2026")
    assert status == 400 and "single month" in payload["answer"]


# ---------------------------------------------------------------- LLM unavailable
def test_llm_is_not_called_when_nothing_is_configured(conn, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called when it is not configured")
    monkeypatch.setattr(ai_handler, "_call_llm", must_not_be_called)
    status, payload = ask(conn, "What is the total revenue?")
    assert status == 200 and payload["interpreter"] == "keyword_fallback"
    assert "No LLM configured" in payload["interpreter_note"]


def test_blank_api_key_uses_fallback_and_says_why(conn, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")               # set but empty
    status, payload = ask(conn, "What is the total revenue?")
    assert status == 200 and payload["interpreter"] == "keyword_fallback"
    assert "GEMINI_API_KEY is not set" in payload["interpreter_note"]


# ---------------------------------------------------------------- config / prompt hygiene
def test_api_key_is_not_exposed_by_settings_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-secret-value")
    settings = get_llm_settings()
    assert settings.enabled and "sk-secret-value" not in repr(settings)


def test_system_prompt_template_braces_are_escaped():
    rendered = ai_handler.SYSTEM_PROMPT.format(available_months="2026-01, 2026-02")   # the month list is filled in with str.format()
    assert "2026-01, 2026-02" in rendered and '{"intent"' in rendered
