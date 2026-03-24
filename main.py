"""MLADE Main Program - Run the full hallucination detection pipeline.

Usage:
    python main.py "When was quantum field theory developed?"
    python main.py  (interactive mode)
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import stage0_generate, stage1_multilingual, stage2_consistency, stage3_verification
import config


def run_pipeline(question: str, gold_answer: str = None) -> dict:
    """Run full pipeline. Always runs RAG for evaluation purposes.

    Args:
        question: English question.
        gold_answer: Ground truth answer from dataset (optional, for evaluation).
    """
    pipeline_start = time.time()

    print("=" * 60)
    print(f"Question: {question}")
    if gold_answer:
        print(f"Gold Answer: {gold_answer}")
    print("=" * 60)

    # ---- Stage 0: Original Response ----
    print("\n[Stage 0] Generating original answer...")
    original_answer = stage0_generate.run(question)
    print(f"  Answer: {original_answer}")

    # ---- Stage 1: Multilingual Generation ----
    print("\n[Stage 1] Translating question & generating multilingual responses...")
    multilingual = stage1_multilingual.run(question)

    for lang, data in multilingual.items():
        print(f"  [{lang}] Q: {data['question']}")
        print(f"       A: {data['response']}")

    # ---- Stage 2: Consistency Check (Gate) ----
    print("\n[Stage 2] Computing cross-lingual consistency...")
    stage2_result = stage2_consistency.run(multilingual)
    stage2_time = time.time() - pipeline_start

    print(f"\n  Consistency Score: {stage2_result['consistency_score']:.4f}")
    print(f"  Threshold (tau):  {config.CONSISTENCY_THRESHOLD}")
    print(f"  Gate Decision:    {'CONSISTENT -> Skip RAG' if stage2_result['is_consistent'] else 'INCONSISTENT -> Trigger RAG'}")
    print()

    for p in stage2_result["pair_similarities"]:
        print(f"  {p['lang_a']} <-> {p['lang_b']}: {p['similarity']:.4f}")

    # ---- Stage 3: ALWAYS run RAG (for evaluation) ----
    print("\n[Stage 3] Running RAG verification (always, for evaluation)...")
    stage3_result = stage3_verification.run(original_answer, question)

    total_time = time.time() - pipeline_start

    # ---- Final Output ----
    print("\n" + "=" * 60)
    print("  EVALUATION RESULT")
    print("=" * 60)

    print(f"\n  --- Answers ---")
    print(f"  LLM Answer:       {original_answer}")
    if gold_answer:
        print(f"  Gold Answer:       {gold_answer}")
    if stage3_result.get("corrected_answer"):
        print(f"  Corrected Answer:  {stage3_result['corrected_answer']}")

    print(f"\n  --- Gate (Stage 2) ---")
    print(f"  Consistency Score: {stage2_result['consistency_score']:.4f}")
    print(f"  Gate Decision:     {'Skip RAG (consistent)' if stage2_result['is_consistent'] else 'Trigger RAG (inconsistent)'}")

    print(f"\n  --- RAG (Stage 3) ---")
    print(f"  RAG Label:         {stage3_result['label']}")
    print(f"  RAG Confidence:    {stage3_result['confidence']:.4f}")

    gate_says_no_hallucination = stage2_result["is_consistent"]
    rag_says_no_hallucination = stage3_result["label"] == "Not Hallucinated"

    if gate_says_no_hallucination and rag_says_no_hallucination:
        agreement = "AGREE - Both say no hallucination (Gate saved RAG time)"
    elif not gate_says_no_hallucination and not rag_says_no_hallucination:
        agreement = "AGREE - Both say hallucination (Gate correctly triggered RAG)"
    elif gate_says_no_hallucination and not rag_says_no_hallucination:
        agreement = "DISAGREE - Gate missed hallucination! (False Negative)"
    else:
        agreement = "DISAGREE - Gate false alarm, no hallucination (False Positive, harmless)"

    print(f"\n  --- Gate vs RAG ---")
    print(f"  Agreement:         {agreement}")

    print(f"\n  --- Time ---")
    print(f"  Without RAG:       {stage2_time:.1f}s (Stage 0+1+2)")
    print(f"  With RAG:          {total_time:.1f}s (Stage 0+1+2+3)")
    print(f"  RAG Extra Cost:    {total_time - stage2_time:.1f}s")
    print(f"  Retrieval Time:    {stage3_result['retrieval_time']:.3f}s")

    if stage3_result["claims"]:
        print(f"\n  --- Evidence Chain ---")
        for c in stage3_result["claims"]:
            print(f"    [{c['nli_result']}] {c['claim']}")
            if c["evidence"]:
                print(f"      Evidence ({c['evidence_title']}): {c['evidence'][:100]}...")

    print("=" * 60)

    return {
        "question": question,
        "gold_answer": gold_answer,
        "original_answer": original_answer,
        "corrected_answer": stage3_result.get("corrected_answer"),
        "consistency_score": stage2_result["consistency_score"],
        "gate_decision": "skip_rag" if stage2_result["is_consistent"] else "trigger_rag",
        "rag_label": stage3_result["label"],
        "rag_confidence": stage3_result["confidence"],
        "gate_rag_agree": gate_says_no_hallucination == rag_says_no_hallucination,
        "time_without_rag": round(stage2_time, 3),
        "time_with_rag": round(total_time, 3),
        "rag_extra_cost": round(total_time - stage2_time, 3),
        "retrieval_time": stage3_result["retrieval_time"],
        "multilingual_responses": multilingual,
        "pair_similarities": stage2_result["pair_similarities"],
        "claims": stage3_result["claims"],
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_pipeline(question)
    else:
        print("MLADE - Hallucination Detection Pipeline (Evaluation Mode)")
        print("All questions run both Gate + RAG for comparison.")
        print("Type a question and press Enter. Type 'quit' to exit.\n")
        while True:
            question = input("Enter question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            run_pipeline(question)
            print()
