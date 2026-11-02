"""Shared fixtures. Every test gets its own temporary database; the real CSV is only read."""
from pathlib import Path

import pytest

from app.database import connect_readonly, initialize_database

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = PROJECT_ROOT / "data" / "ai_bi_assistant_sales_data - GET.csv"
CSV_HEADER = "Transaction_Id,Order_date,Product,Category,Region,Sales_channel,Quantity,Unit_price,Discount_amount,Status"

# A tiny dataset whose answers were worked out BY HAND (see tests/test_analytics.py):
#   T4 is Returned - if it were wrongly counted, Beta (500 + 1500) would beat Alpha (190 + 400).
SMALL_ROWS = [
    "T1,01-01-2026,Alpha,Cat,North,Online,2,100,10,Completed",   # 2*100-10 = 190  (Jan)
    "T2,15-01-2026,Beta,Cat,South,Retail,1,500,0,Completed",      # 500             (Jan)
    "T3,03-02-2026,Alpha,Cat,North,Online,4,100,0,Completed",     # 400             (Feb)
    "T4,05-02-2026,Beta,Cat,South,Retail,3,500,0,Returned",       # 1500, EXCLUDED
    "T5,07-02-2026,Gamma,Cat,East,Online,1,50,0,Completed",       # 50              (Feb)
]


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """No test may call a real LLM, even if the developer has keys in their environment/.env."""
    for var in ("GEMINI_API_KEY", "GEMINI_MODEL", "LLM_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def real_csv():
    return REAL_CSV


@pytest.fixture
def make_csv(tmp_path):
    """Return a function that writes a CSV with the standard header and the given rows."""
    def _make(rows, name="data.csv"):
        path = tmp_path / name
        path.write_text(CSV_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path
    return _make


@pytest.fixture
def real_db(tmp_path):
    path = tmp_path / "real.db"
    initialize_database(REAL_CSV, path)
    return path


@pytest.fixture
def conn(real_db):
    connection = connect_readonly(real_db)
    yield connection
    connection.close()


@pytest.fixture
def small_db(tmp_path, make_csv):
    path = tmp_path / "small.db"
    initialize_database(make_csv(SMALL_ROWS), path)
    return path


@pytest.fixture
def small_conn(small_db):
    connection = connect_readonly(small_db)
    yield connection
    connection.close()
