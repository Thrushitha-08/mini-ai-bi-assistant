"""FastAPI backend.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import config
from app.ai_handler import answer_question_from_db
from app.database import ensure_database, get_load_summary, get_rejected_records, open_readonly
from app.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

ERROR_RESPONSES = {
    400: {"model": AskResponse, "description": "Unsupported or unanswerable question"},
    422: {"model": AskResponse, "description": "Invalid request body"},
    503: {"model": AskResponse, "description": "Database unavailable"},
}


def create_app(db_path=None, csv_path=None) -> FastAPI:
    """App factory. Tests pass a temporary `db_path`; normal runs use the defaults from .env."""
    resolved_db = Path(db_path or config.get_db_path())
    resolved_csv = Path(csv_path or config.get_csv_path())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ensure_database(resolved_csv, resolved_db)  # builds the DB on first start only
        yield

    app = FastAPI(
        title="Mini AI Business Intelligence Assistant",
        description="Ask questions about sales data in plain English. Answers come from SQL, not from the LLM.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.db_path = resolved_db

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc: RequestValidationError):
        problems = [
            {"field": ".".join(str(part) for part in err["loc"] if part != "body"), "message": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={
            "question": None,
            "answer": 'Invalid request. Send JSON like {"question": "What is the total revenue?"} '
                      "with a question of 3-500 characters.",
            "data": {"errors": problems}, "supporting_table": [],
            "explanation": "The request body failed validation, so nothing was calculated.",
            "success": False, "intent": None, "interpreter": None, "interpreter_note": None,
        })

    @app.post("/ask", response_model=AskResponse, responses=ERROR_RESPONSES, tags=["assistant"])
    def ask(request: AskRequest):
        """Answer a natural-language question using approved, read-only SQL queries."""
        status, payload = answer_question_from_db(request.question, app.state.db_path)
        return JSONResponse(status_code=status, content=payload)

    @app.get("/health", tags=["system"])
    def health():
        """Is the API up, is the database readable, and which interpreter will be used?"""
        settings = config.get_llm_settings()
        try:
            with open_readonly(app.state.db_path) as conn:
                conn.execute("SELECT 1 FROM sales LIMIT 1")
            database_ready = True
        except sqlite3.Error:
            database_ready = False
        return {
            "status": "ok" if database_ready else "degraded",
            "database_ready": database_ready,
            "interpreter": "llm" if settings.enabled else "keyword_fallback",
            "llm_provider": "gemini" if settings.enabled else None,
            "llm_note": None if settings.enabled else settings.unavailable_reason,
        }

    @app.get("/dataset/summary", tags=["system"])
    def dataset_summary():
        """Record counts and the list of rejected CSV rows (for the UI overview)."""
        try:
            with open_readonly(app.state.db_path) as conn:
                return {"summary": get_load_summary(conn), "rejected_records": get_rejected_records(conn)}
        except sqlite3.Error:
            logger.exception("Database error while reading the dataset summary")
            return JSONResponse(status_code=503, content={
                "detail": "The analytics database is unavailable. Build it with `python -m app.database`."})

    return app


app = create_app()
