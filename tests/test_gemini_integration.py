"""Gemini integration tests. Only the network call (requests.post) is faked, so the REAL
_call_llm / interpret / validation / SQL code runs. No API key or internet is needed."""
import json

import pytest

from app import ai_handler
from app.ai_handler import answer_question, answer_question_from_db
from app.analytics import AnalyticsError
from app.database import connect_readonly

KEY = "AIza-FAKE-KEY-for-tests"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def gemini_ok(reply_text):
    """The structure Gemini really returns: candidates[0].content.parts[0].text"""
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": reply_text}]}}]})


@pytest.fixture
def gemini(monkeypatch):
    """Configure a (fake) key and capture every outgoing HTTP call. Set `calls.reply` per test."""
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    class Calls(list):
        reply = None   # FakeResponse to return, or an Exception instance to raise

    calls = Calls()

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
        if isinstance(calls.reply, Exception):
            raise calls.reply
        return calls.reply

    monkeypatch.setattr(ai_handler.requests, "post", fake_post)
    return calls


# ---------------------------------------------------------------- 1. valid LLM interpretation
def test_valid_gemini_intent_is_used(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "highest_revenue_product", "month_a": null, "month_b": null}')
    status, payload = answer_question("which item makes us the most money?", conn)
    assert status == 200 and payload["success"] is True
    assert payload["interpreter"] == "llm" and payload["interpreter_note"] is None
    assert payload["intent"] == "highest_revenue_product"
    assert len(gemini) == 1                                   # exactly one Gemini call


def test_gemini_month_parameters_are_passed_through_python_validation(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "revenue_change_between_months", "month_a": "2026-03", "month_b": "2026-04"}')
    status, payload = answer_question("how did April go compared to March?", conn)
    assert status == 200 and payload["interpreter"] == "llm"
    assert (payload["data"]["month_a"], payload["data"]["month_b"]) == ("2026-03", "2026-04")


@pytest.mark.parametrize("intent", ["total_revenue", "highest_revenue_product", "monthly_revenue_trend",
                                    "highest_sales_region"])
def test_each_allowed_intent_works_through_the_llm_path(conn, gemini, intent):
    gemini.reply = gemini_ok(json.dumps({"intent": intent, "month_a": None, "month_b": None}))
    status, payload = answer_question("tell me something about the business", conn)
    assert status == 200 and payload["intent"] == intent and payload["interpreter"] == "llm"


# ---------------------------------------------------------------- what is sent to Gemini
def test_request_contains_only_the_question_and_the_key_is_in_a_header_not_the_url(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "total_revenue", "month_a": null, "month_b": null}')
    answer_question("What is the total revenue?", conn)
    call = gemini[0]
    assert call["url"].startswith("https://generativelanguage.googleapis.com/")
    assert KEY not in call["url"]
    assert call["headers"]["x-goog-api-key"] == KEY
    assert call["timeout"] and call["timeout"] > 0                          # a timeout is always set
    sent = json.dumps(call["body"])
    assert "What is the total revenue?" in sent
    assert "1439200" not in sent and "Monitor" not in sent                  # no business data is sent
    assert call["body"]["generationConfig"]["responseMimeType"] == "application/json"


# ---------------------------------------------------------------- 2. invalid LLM response -> fallback
@pytest.mark.parametrize("reply_text", [
    "Sure! The total revenue is 5 million.",                     # prose, no JSON
    '{"intent": "revenue_is_5_million"}',                        # not an allowed intent
    '{"intent": "total_revenue"',                                # broken JSON
    "[]",
    "",
])
def test_invalid_gemini_reply_falls_back_to_keywords(conn, gemini, reply_text):
    gemini.reply = gemini_ok(reply_text)
    status, payload = answer_question("What is the total revenue?", conn)
    assert status == 200 and payload["interpreter"] == "keyword_fallback"
    assert "LLM call failed" in payload["interpreter_note"]


@pytest.mark.parametrize("structure", [{}, {"candidates": []}, {"error": {"message": "x"}},
                                       {"candidates": [{"finishReason": "SAFETY"}]}, "not a dict"])
def test_unexpected_gemini_response_structure_falls_back(conn, gemini, structure):
    gemini.reply = FakeResponse(200, structure)
    status, payload = answer_question("What is the total revenue?", conn)
    assert status == 200 and payload["interpreter"] == "keyword_fallback"


# ---------------------------------------------------------------- 3. unsupported / ambiguous question
def test_gemini_unsupported_answer_gives_a_safe_response_and_no_figures(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "unsupported", "month_a": null, "month_b": null}')
    status, payload = answer_question("What will our revenue be next year?", conn)
    assert status == 400 and payload["success"] is False and payload["interpreter"] == "llm"
    assert payload["supporting_table"] == [] and "total_revenue" not in payload["data"]


def test_ambiguous_comparison_asks_for_months_instead_of_inventing_them(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "revenue_change_between_months", "month_a": null, "month_b": null}')
    status, payload = answer_question("How did revenue change between two months?", conn)
    assert status == 400 and "name the two months" in payload["answer"]


def test_unsupported_question_without_llm_is_still_refused(conn):
    status, payload = answer_question("What is the weather today?", conn)
    assert status == 400 and payload["success"] is False


# ---------------------------------------------------------------- 4. missing key -> keyword fallback
def test_missing_key_never_calls_gemini(conn, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("requests.post must not be called without an API key")
    monkeypatch.setattr(ai_handler.requests, "post", boom)
    status, payload = answer_question("Which region generated the highest sales?", conn)
    assert status == 200 and payload["intent"] == "highest_sales_region"
    assert payload["interpreter"] == "keyword_fallback"
    assert "GEMINI_API_KEY is not set" in payload["interpreter_note"]


# ---------------------------------------------------------------- 5. API failure -> fallback
@pytest.mark.parametrize("failure", [
    pytest.importorskip("requests").exceptions.Timeout("timed out"),
    pytest.importorskip("requests").exceptions.ConnectionError("no internet"),
    FakeResponse(400, {"error": {"message": "API key not valid"}}, text="bad key " + KEY),
    FakeResponse(403, {}),
    FakeResponse(429, {}),
    FakeResponse(500, {}, text="server error"),
    FakeResponse(503, None),
])
def test_gemini_failures_fall_back_without_leaking_secrets(conn, gemini, failure):
    gemini.reply = failure
    status, payload = answer_question("What is the monthly revenue trend?", conn)
    assert status == 200 and payload["success"] is True
    assert payload["intent"] == "monthly_revenue_trend"                    # the fallback still answers
    assert payload["interpreter"] == "keyword_fallback" and "LLM call failed" in payload["interpreter_note"]
    assert KEY not in json.dumps(payload)


def test_http_error_note_shows_only_the_status_code(conn, gemini):
    gemini.reply = FakeResponse(429, {}, text="quota exceeded for key " + KEY)
    _, payload = answer_question("What is the total revenue?", conn)
    assert "HTTP 429" in payload["interpreter_note"] and "quota" not in payload["interpreter_note"]


# ---------------------------------------------------------------- 6. numbers still come from SQLite
def test_numbers_come_from_sqlite_even_if_the_llm_tries_to_supply_them(small_conn, gemini):
    gemini.reply = gemini_ok(json.dumps({
        "intent": "total_revenue", "month_a": None, "month_b": None,
        "total_revenue": 999999999, "answer": "Revenue is 999999999", "result": {"revenue": 42},
    }))
    status, payload = answer_question("What is the total revenue?", small_conn)
    assert status == 200 and payload["interpreter"] == "llm"
    assert payload["data"]["total_revenue"] == 1140                  # hand-calculated value in the small DB
    assert "999999999" not in json.dumps(payload) and "42" not in payload["answer"]


def test_llm_and_keyword_paths_give_identical_figures(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "highest_sales_region", "month_a": null, "month_b": null}')
    _, via_llm = answer_question("who sells the most, geographically?", conn)
    gemini.reply = FakeResponse(500, {})
    _, via_keywords = answer_question("Which region generated the highest sales?", conn)
    assert via_llm["interpreter"] == "llm" and via_keywords["interpreter"] == "keyword_fallback"
    assert via_llm["data"] == via_keywords["data"] and via_llm["supporting_table"] == via_keywords["supporting_table"]


def test_figures_change_when_the_database_changes(small_db, gemini):
    """Proof the answer is a live SQL result: a different database gives a different number."""
    gemini.reply = gemini_ok('{"intent": "total_revenue", "month_a": null, "month_b": null}')
    status, payload = answer_question_from_db("What is the total revenue?", small_db)
    assert payload["data"]["total_revenue"] == 1140 and status == 200


# ---------------------------------------------------------------- 7. destructive SQL cannot run
@pytest.mark.parametrize("reply_text", [
    '{"intent": "total_revenue", "sql": "DROP TABLE sales"}',                    # extra "sql" field
    '{"intent": "DROP TABLE sales; --"}',
    '{"intent": "run_sql", "query": "DELETE FROM sales"}',
    '{"intent": "revenue_change_between_months", "month_a": "2026-01\'; DROP TABLE sales;--", "month_b": "2026-02"}',
])
def test_llm_supplied_sql_is_never_executed(small_db, gemini, reply_text):
    gemini.reply = gemini_ok(reply_text)
    answer_question_from_db("What is the total revenue?", small_db)
    answer_question_from_db("compare things please", small_db)
    check = connect_readonly(small_db)
    try:
        assert check.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 5      # nothing was deleted
    finally:
        check.close()


def test_user_sent_sql_is_blocked_before_gemini_is_even_called(conn, gemini):
    gemini.reply = gemini_ok('{"intent": "total_revenue", "month_a": null, "month_b": null}')
    status, payload = answer_question("DROP TABLE sales; DELETE FROM sales", conn)
    assert status == 400 and "read-only" in payload["answer"]
    assert len(gemini) == 0                                          # the text never left the server


def test_the_database_connection_is_read_only(conn):
    import sqlite3
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM sales")


def test_only_the_five_allowed_intents_exist():
    assert set(ai_handler.INTENTS) == {"total_revenue", "highest_revenue_product", "monthly_revenue_trend",
                                       "highest_sales_region", "revenue_change_between_months"}
