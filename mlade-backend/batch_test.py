"""Batch test: run multiple questions through the full pipeline, save detailed CSV."""

import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline

QUESTIONS = [
    ("What is the capital of China?",     "Not Hallucinated", "Easy factual"),
    ("What is the capital of Australia?", "Not Hallucinated", "LLMs often say Sydney"),
]


def main():
    results = []

    for question, expected, note in QUESTIONS:
        print(f"\n{'='*70}")
        print(f"TESTING: {question}")
        print(f"{'='*70}")
        try:
            r = run_pipeline(question)

            ml = r.get("multilingual_responses", {})
            pair_sims = r.get("pair_similarities", [])
            claims = r.get("claims", [])

            row = {
                # 基本信息
                "question":           question,
                "note":               note,
                "expected":           expected,
                "llm_answer":         r["original_answer"],
                "corrected_answer":   r.get("corrected_answer", ""),

                # Stage 2
                "consistency_score":  r["consistency_score"],
                "gate_decision":      r["gate_decision"],

                # Stage 3
                "rag_label":          r["rag_label"],
                "rag_confidence":     r["rag_confidence"],
                "gate_rag_agree":     r["gate_rag_agree"],

                # 时间
                "time_without_rag_s": r["time_without_rag"],
                "time_with_rag_s":    r["time_with_rag"],
                "rag_extra_cost_s":   r["rag_extra_cost"],
                "retrieval_time_s":   r["retrieval_time"],

                # 多语言回答
                "en_question":  ml.get("en", {}).get("question", ""),
                "en_answer":    ml.get("en", {}).get("response", ""),
                "ar_question":  ml.get("ar", {}).get("question", ""),
                "ar_answer":    ml.get("ar", {}).get("response", ""),
                "ja_question":  ml.get("ja", {}).get("question", ""),
                "ja_answer":    ml.get("ja", {}).get("response", ""),
                "ru_question":  ml.get("ru", {}).get("question", ""),
                "ru_answer":    ml.get("ru", {}).get("response", ""),

                # 语言对相似度
                **{f"sim_{p['lang_a']}_{p['lang_b']}": round(p["similarity"], 4) for p in pair_sims},

                # 证据链（最多3条claim）
                **{f"claim_{i+1}": c["claim"] for i, c in enumerate(claims[:3])},
                **{f"claim_{i+1}_nli": c["nli_result"] for i, c in enumerate(claims[:3])},
                **{f"claim_{i+1}_evidence": c["evidence"][:120] if c["evidence"] else "" for i, c in enumerate(claims[:3])},

                "error": "",
            }
            results.append(row)

        except Exception as e:
            results.append({
                "question": question, "note": note,
                "expected": expected, "error": str(e),
            })
            print(f"  -> ERROR: {e}")

    # 收集所有出现过的字段
    all_keys = []
    seen = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # 保存 CSV
    csv_file = "results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in all_keys})

    print(f"\n\n结果已保存到 {csv_file}")

    # 打印汇总表
    print("\n" + "="*100)
    print(f"{'#':<3} {'Question':<45} {'Gate':<12} {'Score':<7} {'RAG':<18} {'Agree':<6} {'Time(no RAG)':<13} {'Time(RAG)'}")
    print("-"*100)
    for i, r in enumerate(results, 1):
        if r.get("error"):
            print(f"{i:<3} {r['question'][:44]:<45} ERROR: {r['error'][:40]}")
            continue
        gate = "Skip" if r["gate_decision"] == "skip_rag" else "Trigger"
        agree = "✓" if r["gate_rag_agree"] else "✗"
        print(f"{i:<3} {r['question'][:44]:<45} {gate:<12} {r['consistency_score']:<7} "
              f"{r['rag_label']:<18} {agree:<6} {r['time_without_rag_s']}s{'':<8} {r['time_with_rag_s']}s")
    print("="*100)


if __name__ == "__main__":
    main()