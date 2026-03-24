"""MLADE Backend - FastAPI entry point."""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from schemas.api_models import (
    DetectRequest,
    DetectResponse,
    MultilingualResponse,
    PairSimilarity,
    HealthResponse,
)
from pipeline import stage0_generate, stage1_multilingual, stage2_consistency
import config

app = FastAPI(
    title="MLADE",
    description="Multilingual Language-Aware Detection for Hallucination via Adaptive RAG",
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        models_loaded={
            "google_translate_api": bool(config.GOOGLE_TRANSLATE_API_KEY),
            "voyage_ai_api": bool(config.VOYAGE_API_KEY),
            "jina_api": bool(config.JINA_API_KEY),
            "dashscope_api": bool(config.DASHSCOPE_API_KEY),
        },
    )


@app.post("/detect", response_model=DetectResponse)
def detect_hallucination(req: DetectRequest):
    """Full pipeline: Stage 0 -> Stage 1 -> Stage 2 (-> Stage 3 TODO)."""

    # Stage 0: Generate original answer
    original_answer = stage0_generate.run(req.question)

    # Stage 1: Multilingual parallel generation
    multilingual = stage1_multilingual.run(req.question)

    # Stage 2: Consistency check
    stage2_result = stage2_consistency.run(multilingual)

    # Build response
    ml_responses = [
        MultilingualResponse(
            lang=lc,
            question=data["question"],
            response=data["response"],
        )
        for lc, data in multilingual.items()
    ]

    pair_sims = [
        PairSimilarity(
            lang_a=p["lang_a"],
            lang_b=p["lang_b"],
            similarity=round(p["similarity"], 4),
        )
        for p in stage2_result["pair_similarities"]
    ]

    label = stage2_result["label"]
    stage_reached = 2

    # If inconsistent, Stage 3 would run here (TODO)
    if not stage2_result["is_consistent"]:
        # For now, mark as needing verification
        label = "Needs Verification"
        stage_reached = 2  # will become 3 when RAG is implemented

    return DetectResponse(
        original_question=req.question,
        original_answer=original_answer,
        label=label,
        confidence=round(stage2_result["confidence"], 4),
        consistency_score=round(stage2_result["consistency_score"], 4),
        multilingual_responses=ml_responses,
        pair_similarities=pair_sims,
        stage_reached=stage_reached,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
