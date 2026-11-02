"""SQL analytics: results are checked against (a) an independent pandas calculation and
(b) a tiny dataset whose answers were worked out by hand."""
import sqlite3

import pandas as pd
import pytest

from app import analytics
from app.analytics import AnalyticsError, format_currency
from app.database import connect_readonly, initialize_database


def expected_frame(real_csv):
    """Independent re-implementation of the business rules using plain pandas."""
    df = pd.read_csv(real_csv)
    df = df[(df["Status"] == "Completed") & (df["Quantity"] > 0) & (df["Unit_price"] > 0)].copy()
    df["revenue"] = df["Quantity"] * df["Unit_price"] - df["Discount_amount"]
    df["month"] = pd.to_datetime(df["Order_date"], format="%d-%m-%Y").dt.strftime("%Y-%m")
    return df


# ---------------------------------------------------------------- real CSV vs pandas
def test_total_revenue_matches_pandas(conn, real_csv):
    df = expected_frame(real_csv)
    result = analytics.total_revenue(conn)
    assert result["total_revenue"] == df["revenue"].sum()
    assert result["completed_transactions"] == len(df)
    assert result["units_sold"] == df["Quantity"].sum()


def test_product_ranking_matches_pandas(conn, real_csv):
    expected = expected_frame(real_csv).groupby("Product")["revenue"].sum().sort_values(ascending=False)
    result = analytics.product_revenue_ranking(conn)
    assert result["top_names"] == [expected.index[0]]
    assert result["top_revenue"] == expected.iloc[0]
    assert {r["name"]: r["revenue"] for r in result["ranking"]} == expected.to_dict()
    assert [r["revenue"] for r in result["ranking"]] == sorted((r["revenue"] for r in result["ranking"]), reverse=True)


def test_region_ranking_matches_pandas(conn, real_csv):
    expected = expected_frame(real_csv).groupby("Region")["revenue"].sum()
    result = analytics.region_revenue_ranking(conn)
    assert result["top_names"] == [expected.idxmax()]
    assert {r["name"]: r["revenue"] for r in result["ranking"]} == expected.to_dict()


def test_monthly_trend_matches_pandas_and_is_chronological(conn, real_csv):
    expected = expected_frame(real_csv).groupby("month")["revenue"].sum()
    months = analytics.monthly_revenue_trend(conn)["months"]
    assert [m["month"] for m in months] == sorted(expected.index)
    assert {m["month"]: m["revenue"] for m in months} == expected.to_dict()
    assert months[0]["change_vs_previous"] is None
    assert months[1]["change_vs_previous"] == months[1]["revenue"] - months[0]["revenue"]


def test_compare_months_matches_pandas(conn, real_csv):
    by_month = expected_frame(real_csv).groupby("month")["revenue"].sum()
    result = analytics.compare_months(conn, "2026-01", "2026-02")
    jan, feb = by_month["2026-01"], by_month["2026-02"]
    assert result["revenue_a"] == jan and result["revenue_b"] == feb
    assert result["absolute_change"] == feb - jan
    assert result["percent_change"] == round((feb - jan) / jan * 100, 1)


def test_rankings_and_trend_add_up_to_the_total(conn):
    total = analytics.total_revenue(conn)["total_revenue"]
    assert sum(r["revenue"] for r in analytics.product_revenue_ranking(conn)["ranking"]) == total
    assert sum(r["revenue"] for r in analytics.region_revenue_ranking(conn)["ranking"]) == total
    assert sum(m["revenue"] for m in analytics.monthly_revenue_trend(conn)["months"]) == total


# ---------------------------------------------------------------- hand-checked small data
def test_returned_transactions_are_excluded_from_every_metric(small_conn):
    # Completed: 190 + 500 + 400 + 50 = 1140.  Returned T4 (1500) must NOT be included.
    total = analytics.total_revenue(small_conn)
    assert total["total_revenue"] == 1140
    assert total["completed_transactions"] == 4
    assert total["returned_transactions_excluded"] == 1 and total["returned_value_excluded"] == 1500

    products = analytics.product_revenue_ranking(small_conn)
    assert products["top_names"] == ["Alpha"] and products["top_revenue"] == 590   # Beta would win (2000) if returns counted
    assert {r["name"]: r["revenue"] for r in products["ranking"]} == {"Alpha": 590, "Beta": 500, "Gamma": 50}

    regions = analytics.region_revenue_ranking(small_conn)
    assert {r["name"]: r["revenue"] for r in regions["ranking"]} == {"North": 590, "South": 500, "East": 50}


def test_monthly_trend_and_comparison_by_hand(small_conn):
    months = analytics.monthly_revenue_trend(small_conn)["months"]
    assert [(m["month"], m["revenue"]) for m in months] == [("2026-01", 690), ("2026-02", 450)]
    assert months[1]["change_vs_previous"] == -240 and months[1]["percent_change_vs_previous"] == -34.8

    comparison = analytics.compare_months(small_conn, "2026-01", "2026-02")
    assert comparison["direction"] == "decrease"
    assert comparison["absolute_change"] == -240 and comparison["percent_change"] == -34.8


def test_ranking_ties_are_reported(tmp_path, make_csv):
    path = tmp_path / "tie.db"
    initialize_database(make_csv(["A,01-01-2026,P1,C,N,Online,1,100,0,Completed",
                                  "B,02-01-2026,P2,C,S,Online,2,50,0,Completed"]), path)
    connection = connect_readonly(path)
    try:
        assert analytics.product_revenue_ranking(connection)["top_names"] == ["P1", "P2"]
    finally:
        connection.close()


# ---------------------------------------------------------------- edge cases & safety
def test_month_with_no_sales_gives_a_helpful_error(conn):
    with pytest.raises(AnalyticsError, match="no completed sales for March 2025"):
        analytics.compare_months(conn, "2025-03", "2026-01")


def test_comparing_a_month_with_itself_is_rejected(conn):
    with pytest.raises(AnalyticsError, match="two different months"):
        analytics.compare_months(conn, "2026-01", "2026-01")


@pytest.mark.parametrize("bad", ["2026-13", "Jan 2026", "2026-1", "", "2026-01'; DROP TABLE sales;--"])
def test_malformed_month_is_rejected_before_any_sql_runs(conn, bad):
    with pytest.raises(AnalyticsError, match="not a valid month"):
        analytics.compare_months(conn, bad, "2026-02")
    assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] > 0   # table still intact


def test_database_connection_is_read_only(conn):
    for statement in ("DELETE FROM sales", "DROP TABLE sales", "UPDATE sales SET unit_price = 0",
                      "INSERT INTO sales SELECT * FROM sales"):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute(statement)


def test_ranking_column_is_whitelisted(conn):
    with pytest.raises(ValueError):
        analytics._revenue_ranking(conn, "status; DROP TABLE sales")


def test_database_with_no_valid_sales_edge_case(tmp_path, make_csv):
    # Every row is invalid -> empty `sales` table. Total is 0; rankings explain that there is no data.
    path = tmp_path / "empty.db"
    summary = initialize_database(make_csv(["A,01-01-2026,P,C,N,Online,0,10,0,Completed",
                                            "B,02-01-2026,P,C,N,Online,1,,0,Completed"]), path)
    assert summary["accepted_records"] == 0 and summary["rejected_records"] == 2
    connection = connect_readonly(path)
    try:
        assert analytics.total_revenue(connection)["total_revenue"] == 0
        for function in (analytics.product_revenue_ranking, analytics.region_revenue_ranking,
                         analytics.monthly_revenue_trend):
            with pytest.raises(AnalyticsError, match="no completed sales"):
                function(connection)
    finally:
        connection.close()


# ---------------------------------------------------------------- currency formatting
@pytest.mark.parametrize("amount, text", [
    (0, "₹0"), (999, "₹999"), (1000, "₹1,000"), (100000, "₹1,00,000"), (1439200, "₹14,39,200"),
    (1234.5, "₹1,234.50"), (-5000, "-₹5,000"), (12345678.9, "₹1,23,45,678.90"),
])
def test_currency_uses_indian_grouping(amount, text):
    assert format_currency(amount) == text


def test_month_label():
    assert analytics.month_label("2026-01") == "January 2026"
