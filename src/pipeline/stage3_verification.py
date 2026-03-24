"""Stage 3: Evidence-based RAG Verification.

Flow:
  1. Retrieve evidence from Wikipedia (coarse retrieval via Wikipedia API)
  2. Claim decomposition: Qwen breaks English answer into atomic claims
  3. Fine retrieval: Jina V3 embedding finds most relevant passages per claim
  4. NLI verification: DeBERTa-v3-large checks each claim against its best evidence
  5. If hallucinated: Qwen generates correct answer based on evidence
"""

import time
from models import target_llm, nli_model
import config


def _get_retriever():
    """Get the Wikipedia retriever."""
    from retrieval import wikipedia_retriever
    return wikipedia_retriever


def decompose_claims(answer: str) -> list[str]:
    """Decompose an LLM answer into atomic factual claims using Qwen."""
    prompt = f"""Break the following answer into independent atomic factual claims.
Each claim should contain exactly one fact that can be independently verified.
Return ONLY the claims, one per line, with no numbering or bullet points.

Answer: {answer}

Atomic claims:"""

    response = target_llm.generate(
        prompt,
        system_prompt="You extract atomic factual claims from text. Output only the claims, one per line.",
    )

    claims = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            for prefix in ["- ", "• ", "· "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
            if line[0].isdigit() and len(line) > 2 and (line[1] == "." or (line[1].isdigit() and line[2] == ".")):
                line = line.split(".", 1)[-1].strip()
            claims.append(line)

    return claims if claims else [answer]


def generate_corrected_answer(question: str, claim_results: list[dict]) -> str:
    """Use Qwen to generate a corrected answer based on evidence."""
    evidence_context = ""
    for c in claim_results:
        evidence_context += f"Claim: {c['claim']}\n"
        evidence_context += f"  Verdict: {c['nli_result']}\n"
        if c["evidence"]:
            evidence_context += f"  Evidence ({c['evidence_title']}): {c['evidence']}\n"
        evidence_context += "\n"

    prompt = f"""Based on the following evidence, provide a correct and concise answer to the question.
Only use information supported by the evidence. Do not make up facts.

Question: {question}

Evidence and claim verification results:
{evidence_context}

Correct answer:"""

    response = target_llm.generate(
        prompt,
        system_prompt="You are a factual assistant. Answer based ONLY on the provided evidence. Be concise.",
    )

    return response.strip()


def run(original_answer: str, question: str) -> dict:
    """Run Stage 3: RAG-based verification of the English answer.

    Args:
        original_answer: The LLM's original ENGLISH answer from Stage 0.
        question: The original ENGLISH question (for retrieval).
    """
    retriever = _get_retriever()

    total_start = time.time()

    # Step 1: Retrieve evidence
    print(f"\n  [Stage 3.1] Retrieving evidence from Wikipedia...")
    retrieval_start = time.time()
    passages = retriever.search(question)
    retrieval_time = time.time() - retrieval_start

    if not passages:
        return {
            "label": "Not Hallucinated",
            "confidence": 0.5,
            "claims": [],
            "corrected_answer": None,
            "retrieval_time": retrieval_time,
            "total_time": time.time() - total_start,
        }

    # Step 2: Claim decomposition
    print("  [Stage 3.2] Decomposing English answer into atomic claims...")
    claims = decompose_claims(original_answer)
    print(f"    Found {len(claims)} claims:")
    for c in claims:
        print(f"      - {c}")

    # Step 3 & 4: For each claim, fine-retrieve (Jina V3) + NLI verify
    print("  [Stage 3.3] Fine retrieval (Jina V3) + NLI verification per claim...")
    claim_results = []
    has_contradiction = False
    has_neutral = False

    for claim in claims:
        relevant_passages = retriever.find_relevant_passages(claim, passages, top_k=3)

        result = nli_model.verify_claim_with_evidence(claim, relevant_passages)

        claim_results.append({
            "claim": claim,
            "nli_result": result["label"],
            "evidence_title": result["best_evidence_title"],
            "evidence": result["best_evidence"],
            "scores": result["scores"],
        })

        if result["label"] == "Contradiction":
            has_contradiction = True
        elif result["label"] == "Neutral":
            has_neutral = True

        print(f"    [{result['label']}] {claim}")
        print(f"      -> Evidence from: {result['best_evidence_title']}")

    # Final decision
    corrected_answer = None
    if has_contradiction:
        label = "Hallucinated"
        n_contradictions = sum(1 for c in claim_results if c["nli_result"] == "Contradiction")
        confidence = min(0.6 + 0.4 * (n_contradictions / len(claims)), 1.0)

        print("\n  [Stage 3.4] Generating corrected answer based on evidence...")
        corrected_answer = generate_corrected_answer(question, claim_results)
        print(f"    Corrected: {corrected_answer}")
    elif has_neutral:
        label = "Not Hallucinated"
        confidence = 0.6
    else:
        label = "Not Hallucinated"
        confidence = 0.9

    total_time = time.time() - total_start

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "claims": claim_results,
        "corrected_answer": corrected_answer,
        "retrieval_time": round(retrieval_time, 3),
        "total_time": round(total_time, 3),
    }
