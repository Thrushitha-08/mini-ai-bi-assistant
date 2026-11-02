"""Central configuration.

Everything is read from environment variables (optionally from a local `.env`
file). Nothing secret is ever hard-coded in the source code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # does NOT override variables that are already set

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "ai_bi_assistant_sales_data - GET.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sales.db"

# The CSV has no currency column. Assumption: amounts are Indian Rupees (INR).
# Change CURRENCY_SYMBOL in .env if that is wrong; no code change is needed.
CURRENCY_SYMBOL = os.getenv("CURRENCY_SYMBOL", "₹")


def get_csv_path() -> Path:
    return Path(os.getenv("CSV_PATH") or DEFAULT_CSV_PATH)


def get_db_path() -> Path:
    return Path(os.getenv("DB_PATH") or DEFAULT_DB_PATH)


# --------------------------------------------------------------------------- #
# LLM settings (Google Gemini). The key only ever comes from the environment.
# --------------------------------------------------------------------------- #
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # override with GEMINI_MODEL if Google retires it


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(repr=False)  # repr=False so the key can never leak into logs
    model: str = DEFAULT_GEMINI_MODEL
    timeout: float = 15.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        """Human-readable reason why the LLM is not used (shown in API responses)."""
        if not self.api_key:
            return "No LLM configured (GEMINI_API_KEY is not set); the keyword/intent fallback was used."
        return ""


def get_llm_settings() -> LLMSettings:
    """Read LLM settings from the environment *at call time* (so tests can change them)."""
    return LLMSettings(
        api_key=(os.getenv("GEMINI_API_KEY") or "").strip(),
        model=(os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip(),
        timeout=float(os.getenv("LLM_TIMEOUT") or 15),
    )
