"""SQLite storage: schema, loading cleaned data, and read-only connections.

How the data is stored
  * `sales`            - one row per ACCEPTED transaction (typed columns + CHECK constraints)
  * `rejected_records` - one row per REJECTED CSV row, with the reason and the raw values
  * `load_summary`     - a single row with record counts from the last load

How it is queried
  * The API only ever opens the database in READ-ONLY mode (`mode=ro`), so even a bug
    could not modify data. Only `initialize_database` (a command-line step) writes.
  * All queries live in analytics.py and use bound parameters.

Run `python -m app.database` to (re)build the database from the CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List

from app import config
from app.data_loader import SALES_COLUMNS, clean_sales_csv

SCHEMA = """
CREATE TABLE sales (
    transaction_id  TEXT    PRIMARY KEY,
    order_date      TEXT    NOT NULL,                      -- ISO date, YYYY-MM-DD
    order_month     TEXT    NOT NULL,                      -- YYYY-MM (used for monthly reports)
    product         TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    sales_channel   TEXT    NOT NULL,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      REAL    NOT NULL CHECK (unit_price > 0),
    discount_amount REAL    NOT NULL CHECK (discount_amount >= 0),
    status          TEXT    NOT NULL CHECK (status IN ('Completed', 'Returned'))
);
CREATE INDEX idx_sales_month   ON sales (order_month);
CREATE INDEX idx_sales_product ON sales (product);
CREATE INDEX idx_sales_region  ON sales (region);

CREATE TABLE rejected_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row     INTEGER NOT NULL,                        -- line number in the CSV (header = 1)
    transaction_id TEXT,
    reason         TEXT    NOT NULL,
    raw_record     TEXT    NOT NULL                         -- original values as JSON
);

CREATE TABLE load_summary (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    source_file       TEXT    NOT NULL,
    loaded_at         TEXT    NOT NULL,
    total_records     INTEGER NOT NULL,
    accepted_records  INTEGER NOT NULL,
    rejected_records  INTEGER NOT NULL,
    completed_records INTEGER NOT NULL,
    returned_records  INTEGER NOT NULL,
    rejection_reasons TEXT    NOT NULL                      -- JSON {reason_code: count}
);
"""

_INSERT_SALE = (
    f"INSERT INTO sales ({', '.join(SALES_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in SALES_COLUMNS)})"
)


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #
def connect_readonly(db_path) -> sqlite3.Connection:
    """Open the database read-only. Any INSERT/UPDATE/DELETE/DROP will raise an error."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def open_readonly(db_path) -> Iterator[sqlite3.Connection]:
    conn = connect_readonly(db_path)
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Building the database
# --------------------------------------------------------------------------- #
def initialize_database(csv_path=None, db_path=None) -> Dict[str, object]:
    """Clean the CSV and (re)create the SQLite database. Returns the load summary.

    The database is built in a temporary file and swapped in at the end, so a
    failed load never leaves a half-written database behind.
    """
    csv_path = Path(csv_path or config.get_csv_path())
    db_path = Path(db_path or config.get_db_path())

    result = clean_sales_csv(csv_path)  # raises DataLoadError if the CSV is unusable

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    summary = dict(result.stats)
    summary["source_file"] = csv_path.name
    summary["loaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(SCHEMA)
        with conn:  # one transaction: everything is inserted, or nothing is
            rows = [
                tuple(record[col] for col in SALES_COLUMNS)
                for record in result.accepted.to_dict("records")
            ]
            conn.executemany(_INSERT_SALE, rows)
            conn.executemany(
                "INSERT INTO rejected_records (source_row, transaction_id, reason, raw_record) "
                "VALUES (?, ?, ?, ?)",
                [
                    (r["source_row"], r["transaction_id"], r["reason"], json.dumps(r["raw_record"]))
                    for r in result.rejected
                ],
            )
            conn.execute(
                "INSERT INTO load_summary (id, source_file, loaded_at, total_records, accepted_records, "
                "rejected_records, completed_records, returned_records, rejection_reasons) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    summary["source_file"], summary["loaded_at"], summary["total_records"],
                    summary["accepted_records"], summary["rejected_records"],
                    summary["completed_records"], summary["returned_records"],
                    json.dumps(summary["rejection_reasons"]),
                ),
            )
    finally:
        conn.close()

    os.replace(tmp_path, db_path)
    return summary


def ensure_database(csv_path=None, db_path=None) -> None:
    """Build the database only if it does not exist yet (used when the API starts)."""
    db_path = Path(db_path or config.get_db_path())
    if not db_path.exists():
        initialize_database(csv_path, db_path)


# --------------------------------------------------------------------------- #
# Reading metadata
# --------------------------------------------------------------------------- #
def get_load_summary(conn: sqlite3.Connection) -> Dict[str, object]:
    row = conn.execute("SELECT * FROM load_summary WHERE id = 1").fetchone()
    if row is None:
        return {}
    summary = dict(row)
    summary["rejection_reasons"] = json.loads(summary["rejection_reasons"])
    return summary


def get_rejected_records(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute(
        "SELECT source_row, transaction_id, reason, raw_record FROM rejected_records ORDER BY source_row"
    ).fetchall()
    return [
        {
            "source_row": r["source_row"],
            "transaction_id": r["transaction_id"],
            "reason": r["reason"],
            "raw_record": json.loads(r["raw_record"]),
        }
        for r in rows
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SQLite database from the sales CSV.")
    parser.add_argument("--csv", default=None, help="path to the CSV (default: data/ai_bi_assistant_sales_data - GET.csv)")
    parser.add_argument("--db", default=None, help="path of the SQLite file to create (default: data/sales.db)")
    args = parser.parse_args()

    info = initialize_database(args.csv, args.db)
    print(f"Database built at: {Path(args.db or config.get_db_path()).resolve()}")
    print(f"  Source file      : {info['source_file']}")
    print(f"  Total records    : {info['total_records']}")
    print(f"  Accepted         : {info['accepted_records']} "
          f"({info['completed_records']} completed, {info['returned_records']} returned)")
    print(f"  Rejected         : {info['rejected_records']}  {info['rejection_reasons']}")
    with open_readonly(args.db or config.get_db_path()) as c:
        for r in get_rejected_records(c):
            print(f"    - CSV row {r['source_row']} ({r['transaction_id']}): {r['reason']}")
