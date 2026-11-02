"""CSV loading / cleaning: invalid data must be rejected with a reason, never silently fixed."""
import hashlib

import pandas as pd
import pytest

from app.data_loader import DataLoadError, clean_sales_csv
from app.database import connect_readonly, get_load_summary, get_rejected_records, initialize_database


def _clean(make_csv, rows):
    return clean_sales_csv(make_csv(rows))


def test_real_csv_counts_match_independent_pandas_check(real_csv):
    raw = pd.read_csv(real_csv)  # plain pandas, independent of our cleaning code
    expected_bad = raw[(raw["Quantity"] <= 0) | raw["Unit_price"].isna() | (raw["Unit_price"] <= 0)]

    result = clean_sales_csv(real_csv)

    assert result.stats["total_records"] == len(raw)
    assert result.stats["rejected_records"] == len(expected_bad)
    assert {r["transaction_id"] for r in result.rejected} == set(expected_bad["Transaction_Id"])
    assert result.stats["accepted_records"] == len(raw) - len(expected_bad)
    assert result.stats["accepted_records"] + result.stats["rejected_records"] == result.stats["total_records"]


def test_every_rejected_record_explains_why(real_csv):
    reasons = {r["transaction_id"]: r["reason"] for r in clean_sales_csv(real_csv).rejected}
    assert any("quantity" in text for text in reasons.values())
    assert any("unit price is missing" in text for text in reasons.values())
    assert all(text.strip() for text in reasons.values())


def test_returned_transactions_are_kept_but_labelled(real_csv):
    result = clean_sales_csv(real_csv)
    returned = result.accepted[result.accepted["status"] == "Returned"]
    assert len(returned) == result.stats["returned_records"] == 1


def test_dates_are_read_as_day_month_year(make_csv):
    result = _clean(make_csv, ["A,03-01-2026,P,C,N,Online,1,10,0,Completed"])
    row = result.accepted.iloc[0]
    assert row["order_date"] == "2026-01-03"     # 3 January, NOT 1 March
    assert row["order_month"] == "2026-01"


def test_impossible_date_is_rejected(make_csv):
    result = _clean(make_csv, ["A,31-02-2026,P,C,N,Online,1,10,0,Completed",
                               "B,not-a-date,P,C,N,Online,1,10,0,Completed",
                               "C,,P,C,N,Online,1,10,0,Completed"])
    assert result.stats["accepted_records"] == 0
    assert all("order date" in r["reason"] for r in result.rejected)


def test_duplicate_transaction_id_keeps_first_only(make_csv):
    result = _clean(make_csv, ["A,01-01-2026,P,C,N,Online,1,10,0,Completed",
                               "A,02-01-2026,P,C,N,Online,9,99,0,Completed"])
    assert list(result.accepted["transaction_id"]) == ["A"]
    assert result.accepted.iloc[0]["quantity"] == 1                    # the FIRST row won
    assert result.stats["rejection_reasons"] == {"DUPLICATE_TRANSACTION_ID": 1}


@pytest.mark.parametrize("quantity", ["0", "-3", "abc", "", "2.5"])
def test_bad_quantities_are_rejected(make_csv, quantity):
    result = _clean(make_csv, [f"A,01-01-2026,P,C,N,Online,{quantity},10,0,Completed"])
    assert result.stats["accepted_records"] == 0 and result.stats["rejected_records"] == 1


@pytest.mark.parametrize("price", ["", "0", "-5", "free"])
def test_bad_unit_prices_are_rejected_not_invented(make_csv, price):
    result = _clean(make_csv, [f"A,01-01-2026,P,C,N,Online,2,{price},0,Completed"])
    assert result.stats["accepted_records"] == 0
    assert "unit price" in result.rejected[0]["reason"]


def test_bad_discounts_and_statuses_are_rejected(make_csv):
    result = _clean(make_csv, [
        "A,01-01-2026,P,C,N,Online,1,10,-1,Completed",       # negative discount
        "B,01-01-2026,P,C,N,Online,1,10,50,Completed",       # discount bigger than the sale
        "C,01-01-2026,P,C,N,Online,1,10,,Completed",         # blank discount (not assumed to be 0)
        "D,01-01-2026,P,C,N,Online,1,10,0,Cancelled",        # unknown status
    ])
    assert result.stats["accepted_records"] == 0
    assert result.stats["rejected_records"] == 4


def test_status_is_normalised_but_values_are_not_invented(make_csv):
    result = _clean(make_csv, ["A,01-01-2026,  Widget ,C,N,Online,1,10,0,  completed "])
    row = result.accepted.iloc[0]
    assert row["status"] == "Completed" and row["product"] == "Widget"


def test_all_problems_in_a_row_are_reported(make_csv):
    result = _clean(make_csv, ["A,bad-date,P,C,N,Online,0,,0,Completed"])
    reason = result.rejected[0]["reason"]
    assert "order date" in reason and "quantity" in reason and "unit price" in reason


def test_original_csv_is_never_modified(real_csv, tmp_path):
    before = hashlib.sha256(real_csv.read_bytes()).hexdigest()
    initialize_database(real_csv, tmp_path / "x.db")
    assert hashlib.sha256(real_csv.read_bytes()).hexdigest() == before


def test_missing_file_and_missing_columns_raise_clear_errors(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        clean_sales_csv(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("Transaction_Id,Product\nA,B\n")
    with pytest.raises(DataLoadError, match="missing required column"):
        clean_sales_csv(bad)


def test_database_stores_accepted_rejected_and_summary(real_db, real_csv):
    connection = connect_readonly(real_db)
    try:
        stats = clean_sales_csv(real_csv).stats
        summary = get_load_summary(connection)
        assert summary["accepted_records"] == stats["accepted_records"]
        assert summary["rejected_records"] == stats["rejected_records"]
        assert connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == stats["accepted_records"]
        assert len(get_rejected_records(connection)) == stats["rejected_records"]
    finally:
        connection.close()


def test_rebuilding_the_database_is_repeatable(real_csv, tmp_path):
    path = tmp_path / "again.db"
    first = initialize_database(real_csv, path)
    second = initialize_database(real_csv, path)       # must replace, not append
    connection = connect_readonly(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == first["accepted_records"]
        assert second["accepted_records"] == first["accepted_records"]
    finally:
        connection.close()
