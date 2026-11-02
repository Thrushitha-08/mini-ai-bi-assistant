"""Request / response models for the API (Pydantic v2)."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["Which product generated the highest revenue?"],
        description="A business question about the sales data (3-500 characters).",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return value


class AskResponse(BaseModel):
    question: Optional[str] = None
    answer: str
    data: Dict[str, Any] = Field(default_factory=dict, description="Key figures behind the answer")
    supporting_table: List[Dict[str, Any]] = Field(default_factory=list, description="Rows used to produce the answer")
    explanation: str
    success: bool
    intent: Optional[str] = Field(None, description="Which approved analytics operation was run")
    interpreter: Optional[str] = Field(None, description="'llm' or 'keyword_fallback' - how the question was understood")
    interpreter_note: Optional[str] = Field(None, description="Why the fallback was used, if it was")
