"""Pydantic models for API request/response."""

from pydantic import BaseModel


class DetectRequest(BaseModel):
    question: str  # English question


class MultilingualResponse(BaseModel):
    lang: str
    question: str  # translated question
    response: str  # LLM response in that language


class PairSimilarity(BaseModel):
    lang_a: str
    lang_b: str
    similarity: float


class DetectResponse(BaseModel):
    original_question: str
    original_answer: str  # Stage 0 response
    label: str  # "Hallucinated" | "Not Hallucinated" | "Needs Verification"
    confidence: float
    consistency_score: float
    multilingual_responses: list[MultilingualResponse]
    pair_similarities: list[PairSimilarity]
    stage_reached: int  # 2 = stopped at consistency, 3 = went to RAG


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
