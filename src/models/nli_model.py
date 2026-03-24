"""DeBERTa-v3-large NLI model - Natural Language Inference for claim verification.

Model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  - Trained on MNLI + FEVER-NLI + ANLI + LingNLI + WANLI (885K pairs)
  - MNLI accuracy: 91.2% (vs 85.7% for old mDeBERTa multilingual)
  - ANLI accuracy: 70.2% (vs 53.7%)
  - English-only, no multilingual tax

Proper NLI: premise (evidence) + hypothesis (claim) -> entailment/contradiction/neutral.
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

_model = None
_tokenizer = None
# DeBERTa-v3-large label mapping: 0=entailment, 1=neutral, 2=contradiction
_label_names = ["Entailment", "Neutral", "Contradiction"]


def load():
    """Load DeBERTa-v3-large NLI model."""
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"Loading NLI model from {config.NLI_MODEL}...")
    _tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
    _model.eval()
    print("NLI model loaded.")


def verify_claim(claim: str, evidence: str) -> dict:
    """Check if evidence supports, contradicts, or is neutral to the claim.

    NLI format: premise = evidence, hypothesis = claim.
    "Does the evidence support/contradict/say nothing about the claim?"

    Args:
        claim: An atomic factual claim (e.g., "Einstein was born in 1879").
        evidence: Wikipedia passage text.

    Returns:
        {"label": "Entailment"|"Contradiction"|"Neutral", "scores": dict}
    """
    load()

    # Truncate evidence to avoid exceeding model max length
    evidence_truncated = evidence[:1500]

    inputs = _tokenizer(
        evidence_truncated,  # premise
        claim,               # hypothesis
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    scores = {}
    for i, label_name in enumerate(_label_names):
        scores[label_name] = round(probs[i].item(), 4)

    top_idx = probs.argmax().item()
    top_label = _label_names[top_idx]

    return {"label": top_label, "scores": scores}


def verify_claim_with_evidence(claim: str, evidences: list[dict]) -> dict:
    """Verify a claim against multiple evidence passages.

    Logic:
    - If ANY evidence entails the claim -> Entailment (supported)
    - If ANY evidence contradicts AND no entailment -> Contradiction
    - Otherwise -> Neutral

    Args:
        claim: An atomic factual claim.
        evidences: List of {"title": str, "text": str, "score": float}.

    Returns:
        {
            "label": "Entailment"|"Contradiction"|"Neutral",
            "best_evidence": str,
            "best_evidence_title": str,
            "scores": dict,
        }
    """
    best_entailment = None
    best_entailment_score = 0
    best_contradiction = None
    best_contradiction_score = 0
    best_neutral = None

    for ev in evidences:
        result = verify_claim(claim, ev["text"])

        if result["scores"]["Entailment"] > best_entailment_score:
            best_entailment_score = result["scores"]["Entailment"]
            best_entailment = {"result": result, "evidence": ev}

        if result["scores"]["Contradiction"] > best_contradiction_score:
            best_contradiction_score = result["scores"]["Contradiction"]
            best_contradiction = {"result": result, "evidence": ev}

        if best_neutral is None:
            best_neutral = {"result": result, "evidence": ev}

    # Entailment takes priority: if evidence supports the claim, it's not hallucinated
    if best_entailment and best_entailment_score > 0.5:
        ev = best_entailment["evidence"]
        return {
            "label": "Entailment",
            "best_evidence": ev["text"][:200],
            "best_evidence_title": ev["title"],
            "scores": best_entailment["result"]["scores"],
        }

    # Contradiction: evidence actively refutes the claim
    if best_contradiction and best_contradiction_score > 0.5:
        ev = best_contradiction["evidence"]
        return {
            "label": "Contradiction",
            "best_evidence": ev["text"][:200],
            "best_evidence_title": ev["title"],
            "scores": best_contradiction["result"]["scores"],
        }

    # Neutral: evidence doesn't say enough
    ev = best_neutral["evidence"] if best_neutral else evidences[0]
    return {
        "label": "Neutral",
        "best_evidence": ev["text"][:200],
        "best_evidence_title": ev["title"],
        "scores": best_neutral["result"]["scores"] if best_neutral else {"Entailment": 0, "Contradiction": 0, "Neutral": 1},
    }
