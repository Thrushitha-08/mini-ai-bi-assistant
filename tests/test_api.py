"""HTTP-level tests with FastAPI's TestClient (needs `fastapi` and `httpx`; skipped if missing)."""
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture
def client(real_db):
    return TestClient(create_app(db_path=real_db))


def post(client, question):
    return client.post("/ask", json={"question": question})


# ---------------------------------------------------------------- valid questions
def test_total_revenue(client, conn):
    expected = conn.execute(
        "SELECT SUM(quantity * unit_price - discount_amount) FROM sales WHERE status = 'Completed'").fetchone()[0]
    response = post(client, "What is the total revenue?")
    body = response.json()
    assert response.status_code == 200 and body["success"] is True
    assert body["data"]["total_revenue"] == expected
    assert body["interpreter"] == "keyword_fallback"        # no LLM key in tests -> fallback, reported honestly
    assert body["explanation"] and body["supporting_table"]


def test_highest_revenue_product(client, conn):
    expected = conn.execute(
        "SELECT product, SUM(quantity * unit_price - discount_amount) r FROM sales "
        "WHERE status = 'Completed' GROUP BY product ORDER BY r DESC LIMIT 1").fetchone()
    body = post(client, "Which product generated the highest revenue?").json()
    assert body["data"]["product"] == expected["product"] and body["data"]["revenue"] == expected["r"]
    assert expected["product"] in body["answer"]


def test_monthly_revenue_trend(client, conn):
    n_months = conn.execute("SELECT COUNT(DISTINCT order_month) FROM sales WHERE status = 'Completed'").fetchone()[0]
    body = post(client, "What is the monthly revenue trend?").json()
    assert body["success"] and len(body["supporting_table"]) == n_months
    assert [r["month_key"] for r in body["supporting_table"]] == sorted(r["month_key"] for r in body["supporting_table"])


def test_top_region_and_month_comparison(client):
    region = post(client, "Which region generated the highest sales?").json()
    assert region["success"] and region["intent"] == "highest_sales_region"
    comparison = post(client, "How did revenue change between January and February 2026?").json()
    assert comparison["success"] and comparison["data"]["month_a"] == "2026-01" and comparison["data"]["month_b"] == "2026-02"


# ---------------------------------------------------------------- invalid questions / requests
def test_unsupported_question_returns_400_with_guidance(client):
    response = post(client, "What is the weather in Paris?")
    body = response.json()
    assert response.status_code == 400 and body["success"] is False
    assert body["data"]["supported_questions"]


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": "   "}, {"question": "hi"}, {"question": "x" * 501},
    {"question": 123}, {"wrong_field": "What is the total revenue?"}, {"question": None},
])
def test_invalid_request_bodies_return_422_in_the_standard_shape(client, payload):
    response = client.post("/ask", json=payload)
    body = response.json()
    assert response.status_code == 422 and body["success"] is False
    assert body["data"]["errors"] and "question" in body["answer"].lower()


def test_non_json_body_returns_422(client):
    response = client.post("/ask", content="this is not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["success"] is False


def test_wrong_http_method_is_rejected(client):
    assert client.get("/ask").status_code == 405


def test_destructive_question_is_refused_and_data_survives(client, real_db):
    response = post(client, "DROP TABLE sales; DELETE FROM sales")
    assert response.status_code == 400 and response.json()["success"] is False
    check = sqlite3.connect(real_db)
    try:
        assert check.execute("SELECT COUNT(*) FROM sales").fetchone()[0] > 0
    finally:
        check.close()


# ---------------------------------------------------------------- database errors
def test_empty_database_file_gives_503_not_a_crash(tmp_path):
    broken = tmp_path / "broken.db"
    sqlite3.connect(broken).close()                          # a valid SQLite file with no tables
    response = TestClient(create_app(db_path=broken)).post("/ask", json={"question": "What is the total revenue?"})
    body = response.json()
    assert response.status_code == 503 and body["success"] is False
    assert "no such table" not in body["answer"]            # internal details are not leaked


def test_missing_database_file_gives_503(tmp_path):
    response = TestClient(create_app(db_path=tmp_path / "missing.db")).post(          # no lifespan -> not built
        "/ask", json={"question": "What is the total revenue?"})
    assert response.status_code == 503 and response.json()["success"] is False


def test_app_builds_the_database_on_startup_if_missing(tmp_path, real_csv):
    db = tmp_path / "auto.db"
    with TestClient(create_app(db_path=db, csv_path=real_csv)) as started:    # `with` runs the lifespan
        assert db.exists()
        assert started.post("/ask", json={"question": "What is the total revenue?"}).status_code == 200


# ---------------------------------------------------------------- helper endpoints
def test_health_reports_database_and_interpreter(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database_ready"] is True
    assert body["interpreter"] == "keyword_fallback" and body["llm_provider"] is None


def test_dataset_summary_exposes_cleaning_statistics(client):
    body = client.get("/dataset/summary").json()
    summary = body["summary"]
    assert summary["accepted_records"] + summary["rejected_records"] == summary["total_records"]
    assert len(body["rejected_records"]) == summary["rejected_records"]
    assert all(r["reason"] for r in body["rejected_records"])


def test_health_with_a_key_reports_gemini_without_exposing_the_key(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-123")
    response = client.get("/health")
    body = response.json()
    assert body["interpreter"] == "llm" and body["llm_provider"] == "gemini"
    assert "secret-key-123" not in response.text
