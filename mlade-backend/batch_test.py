"""Batch test: run multiple questions through the pipeline and print a summary table."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline

# Questions: (question, expected_label, note)
# expected_label: "Not Hallucinated" or "Hallucinated"
QUESTIONS = [
    # --- Should NOT hallucinate (ground truth checks) ---
    ("What is the capital of China?",                          "Not Hallucinated", "Easy factual"),
    ("What is the chemical formula for water?",                "Not Hallucinated", "Easy science"),
    ("Who wrote Romeo and Juliet?",                            "Not Hallucinated", "Famous author"),

    # --- Likely to hallucinate ---
    ("What is the capital of Australia?",                      "Not Hallucinated", "LLMs often say Sydney"),
    ("In what year did the Titanic sink?",                     "Not Hallucinated", "1912"),
    ("Who was the first person to reach the South Pole?",      "Not Hallucinated", "Amundsen 1911"),
    ("How many moons does Mars have?",                         "Not Hallucinated", "2 moons"),
    ("What year did Einstein win the Nobel Prize?",            "Not Hallucinated", "1921 (not 1905)"),
    ("Who invented the telephone?",                            "Not Hallucinated", "Disputed: Bell vs Gray"),
    ("What is the longest river in the world?",                "Not Hallucinated", "Amazon vs Nile dispute"),
]


def main():
    results = []
    for question, expected, note in QUESTIONS:
        print(f"\n{'='*70}")
        print(f"TESTING: {question}")
        print(f"{'='*70}")
        try:
            r = run_pipeline(question)
            results.append({
                "question": question,
                "note": note,
                "expected": expected,
                "llm_answer": r["original_answer"],
                "gate": r["gate_decision"],
                "consistency": r["consistency_score"],
                "rag_label": r["rag_label"],
                "rag_conf": r["rag_confidence"],
                "agree": r["gate_rag_agree"],
                "corrected": r.get("corrected_answer"),
                "time_no_rag": r["time_without_rag"],
                "time_rag": r["time_with_rag"],
            })
        except Exception as e:
            results.append({
                "question": question,
                "note": note,
                "expected": expected,
                "error": str(e),
            })

    # Print summary table
    print("\n\n")
    print("=" * 100)
    print("BATCH TEST SUMMARY")
    print("=" * 100)
    print(f"{'#':<3} {'Question':<48} {'Gate':<12} {'Score':<7} {'RAG':<18} {'Agree':<7} {'Note'}")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"{i:<3} {r['question'][:47]:<48} ERROR: {r['error'][:40]}")
            continue
        gate = "Skip" if r["gate"] == "skip_rag" else "Trigger"
        score = f"{r['consistency']:.3f}"
        rag = r["rag_label"]
        agree = "✓" if r["agree"] else "✗"
        note = r["note"]
        print(f"{i:<3} {r['question'][:47]:<48} {gate:<12} {score:<7} {rag:<18} {agree:<7} {note}")

    print("-" * 100)

    # Count stats
    total = len([r for r in results if "error" not in r])
    gate_triggers = sum(1 for r in results if "error" not in r and r["gate"] == "trigger_rag")
    rag_hallucinated = sum(1 for r in results if "error" not in r and r["rag_label"] == "Hallucinated")
    agrees = sum(1 for r in results if "error" not in r and r["agree"])

    print(f"\nStats: {total} questions | Gate triggered: {gate_triggers}/{total} | "
          f"RAG flagged hallucination: {rag_hallucinated}/{total} | Gate-RAG agree: {agrees}/{total}")

    print("\nHallucinated answers (with corrections):")
    for r in results:
        if "error" not in r and r["rag_label"] == "Hallucinated":
            print(f"  Q: {r['question']}")
            print(f"  LLM: {r['llm_answer']}")
            print(f"  Corrected: {r['corrected']}")
            print()


if __name__ == "__main__":
    main()
