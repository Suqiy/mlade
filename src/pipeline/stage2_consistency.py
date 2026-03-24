"""Stage 2: Cross-lingual Consistency Check.

Encode all multilingual responses with Voyage AI (voyage-multilingual-2),
compute pairwise cosine similarity, and make routing decision based on
consistency threshold τ.
"""

from models import encoder
import config


def run(multilingual_responses: dict[str, dict]) -> dict:
    """Evaluate cross-lingual consistency.

    Args:
        multilingual_responses: Output from Stage 1.
            Dict mapping lang_code -> {"question": str, "response": str}.

    Returns:
        {
            "consistency_score": float,
            "is_consistent": bool,
            "pair_similarities": [{"lang_a", "lang_b", "similarity"}],
            "label": str,        # routing decision
            "confidence": float,
        }
    """
    lang_codes = list(multilingual_responses.keys())
    responses = [multilingual_responses[lc]["response"] for lc in lang_codes]

    # Compute consistency score across all pairs
    score, pair_details = encoder.consistency_score(responses)

    # Map pair indices back to language codes
    pair_similarities = []
    for p in pair_details:
        i, j = p["pair"]
        pair_similarities.append({
            "lang_a": lang_codes[i],
            "lang_b": lang_codes[j],
            "similarity": p["similarity"],
        })

    is_consistent = score >= config.CONSISTENCY_THRESHOLD

    if is_consistent:
        label = "Not Hallucinated"
        confidence = min(score, 1.0)
    else:
        # Inconsistent -> needs Stage 3 RAG verification
        label = "Needs Verification"
        confidence = 1.0 - score  # higher inconsistency = higher suspicion

    return {
        "consistency_score": score,
        "is_consistent": is_consistent,
        "pair_similarities": pair_similarities,
        "label": label,
        "confidence": confidence,
    }
