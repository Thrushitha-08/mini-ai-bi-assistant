"""CSV loading and cleaning (Pandas).

The original CSV is only ever READ, never modified.

Cleaning rules (every rejected row gets a reason; nothing is silently "fixed"):
  * Transaction_Id must be present and unique (first valid occurrence wins).
  * Order_date must parse. The file uses DD-MM-YYYY (e.g. 03-01-2026 = 3 Jan 2026).
  * Product / Category / Region / Sales_channel must not be blank.
  * Status must be "Completed" or "Returned" (case-insensitive).
  * Quantity must be a whole number > 0          (zero / negative -> rejected + flagged).
  * Unit_price must be present and > 0           (missing / invalid -> rejected + flagged).
  * Discount_amount must be present, >= 0 and not larger than Quantity x Unit_price.

Returned transactions are VALID records: they are kept in the database with
status 'Returned' but are excluded from every completed-sales revenue metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = [
    "Transaction_Id", "Order_date", "Product", "Category", "Region",
    "Sales_channel", "Quantity", "Unit_price", "Discount_amount", "Status",
]
SALES_COLUMNS = [
    "transaction_id", "order_date", "order_month", "product", "category", "region",
    "sales_channel", "quantity", "unit_price", "discount_amount", "status",
]
# The supplied file uses DD-MM-YYYY. ISO and DD/MM/YYYY are also accepted;
# anything else is rejected rather than guessed.
DATE_FORMATS = ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y")
STATUS_LOOKUP = {"completed": "Completed", "returned": "Returned"}

Reason = Tuple[str, str]  # (machine-readable code, human-readable message)


class DataLoadError(Exception):
    """The CSV file is missing, unreadable, or does not have the expected columns."""


@dataclass
class CleaningResult:
    accepted: pd.DataFrame      # valid rows, ready for the `sales` table
    rejected: List[dict]        # {source_row, transaction_id, reason, raw_record}
    stats: Dict[str, object]    # counts shown in the UI / README


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_raw_csv(csv_path) -> pd.DataFrame:
    """Read the CSV as plain strings (no automatic NaN/number conversion)."""
    path = Path(csv_path)
    if not path.is_file():
        raise DataLoadError(f"CSV file not found: {path}")
    try:
        # dtype=str + keep_default_na=False => blanks stay "" and we decide what they mean.
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Could not read CSV ({type(exc).__name__}): {exc}") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataLoadError(f"CSV is missing required column(s): {', '.join(missing)}")
    for col in REQUIRED_COLUMNS:
        raw[col] = raw[col].str.strip()
    return raw


# --------------------------------------------------------------------------- #
# Row-level helpers
# --------------------------------------------------------------------------- #
def _parse_number(text) -> Optional[float]:
    """Return a finite float, or None if blank / not a number."""
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "")
    if cleaned == "":
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _parse_date(text: str) -> Optional[datetime]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def validate_row(raw: dict, accepted_ids: Dict[str, int]) -> Tuple[Optional[dict], List[Reason]]:
    """Validate one CSV row. Returns (clean_row, []) or (None, [reasons...]).

    ALL problems of a row are collected, not just the first one.
    `accepted_ids` maps already-accepted transaction ids to their CSV row number.
    """
    reasons: List[Reason] = []

    txn_id = (raw.get("Transaction_Id") or "").strip()
    if not txn_id:
        reasons.append(("MISSING_TRANSACTION_ID", "transaction id is missing"))
    elif txn_id in accepted_ids:
        reasons.append(("DUPLICATE_TRANSACTION_ID",
                        f"duplicate transaction id (already accepted from CSV row {accepted_ids[txn_id]})"))

    date_text = (raw.get("Order_date") or "").strip()
    parsed_date = _parse_date(date_text) if date_text else None
    if parsed_date is None:
        reasons.append(("INVALID_DATE", f"order date '{date_text}' is missing or not DD-MM-YYYY"))

    text_fields = {}
    for column in ("Product", "Category", "Region", "Sales_channel"):
        value = " ".join((raw.get(column) or "").split())  # trim + collapse inner spaces
        if not value:
            reasons.append(("MISSING_TEXT_FIELD", f"{column} is blank"))
        text_fields[column] = value

    status_text = (raw.get("Status") or "").strip()
    status = STATUS_LOOKUP.get(status_text.lower())
    if status is None:
        reasons.append(("INVALID_STATUS", f"status '{status_text}' is not Completed or Returned"))

    # --- quantity -----------------------------------------------------------
    quantity = None
    qty_value = _parse_number(raw.get("Quantity"))
    if qty_value is None:
        reasons.append(("INVALID_QUANTITY", "quantity is missing or not a number"))
    elif not float(qty_value).is_integer():
        reasons.append(("INVALID_QUANTITY", f"quantity {qty_value:g} is not a whole number"))
    elif qty_value <= 0:
        reasons.append(("NON_POSITIVE_QUANTITY", f"quantity is {int(qty_value)} (must be greater than 0)"))
    else:
        quantity = int(qty_value)

    # --- unit price ---------------------------------------------------------
    unit_price = None
    price_text = (raw.get("Unit_price") or "").strip()
    price_value = _parse_number(price_text)
    if price_text == "":
        reasons.append(("MISSING_UNIT_PRICE", "unit price is missing"))
    elif price_value is None:
        reasons.append(("INVALID_UNIT_PRICE", f"unit price '{price_text}' is not a number"))
    elif price_value <= 0:
        reasons.append(("INVALID_UNIT_PRICE", f"unit price {price_value:g} must be greater than 0"))
    else:
        unit_price = price_value

    # --- discount -----------------------------------------------------------
    discount = None
    discount_value = _parse_number(raw.get("Discount_amount"))
    if discount_value is None:
        reasons.append(("INVALID_DISCOUNT", "discount amount is missing or not a number"))
    elif discount_value < 0:
        reasons.append(("INVALID_DISCOUNT", f"discount {discount_value:g} is negative"))
    else:
        discount = discount_value
    if quantity is not None and unit_price is not None and discount is not None:
        if discount > quantity * unit_price:
            reasons.append(("DISCOUNT_EXCEEDS_GROSS", "discount is larger than quantity x unit price"))

    if reasons:
        return None, reasons

    clean = {
        "transaction_id": txn_id,
        "order_date": parsed_date.strftime("%Y-%m-%d"),
        "order_month": parsed_date.strftime("%Y-%m"),
        "product": text_fields["Product"],
        "category": text_fields["Category"],
        "region": text_fields["Region"],
        "sales_channel": text_fields["Sales_channel"],
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_amount": discount,
        "status": status,
    }
    return clean, []


# --------------------------------------------------------------------------- #
# Whole-file cleaning
# --------------------------------------------------------------------------- #
def clean_sales_dataframe(raw: pd.DataFrame) -> CleaningResult:
    accepted_rows: List[dict] = []
    rejected: List[dict] = []
    accepted_ids: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}

    for position, raw_row in enumerate(raw.to_dict("records")):
        source_row = position + 2  # row 1 of the file is the header
        clean, reasons = validate_row(raw_row, accepted_ids)
        if clean is not None:
            accepted_rows.append(clean)
            accepted_ids[clean["transaction_id"]] = source_row
        else:
            for code, _ in reasons:
                reason_counts[code] = reason_counts.get(code, 0) + 1
            rejected.append({
                "source_row": source_row,
                "transaction_id": (raw_row.get("Transaction_Id") or "").strip(),
                "reason": "; ".join(message for _, message in reasons),
                "raw_record": {k: raw_row.get(k, "") for k in REQUIRED_COLUMNS},
            })

    accepted = pd.DataFrame(accepted_rows, columns=SALES_COLUMNS)
    stats = {
        "total_records": int(len(raw)),
        "accepted_records": int(len(accepted)),
        "rejected_records": int(len(rejected)),
        "completed_records": int((accepted["status"] == "Completed").sum()),
        "returned_records": int((accepted["status"] == "Returned").sum()),
        "rejection_reasons": reason_counts,
    }
    return CleaningResult(accepted=accepted, rejected=rejected, stats=stats)


def clean_sales_csv(csv_path) -> CleaningResult:
    """Read + clean the CSV at `csv_path`."""
    return clean_sales_dataframe(read_raw_csv(csv_path))
