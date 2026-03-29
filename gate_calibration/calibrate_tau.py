"""Gate Threshold (τ) Calibration.

Step 1: Use GPT-4o-mini as independent judge to label gold vs qwen match.
Step 2: Batch-encode all responses via Voyage AI, compute consistency scores locally.
Step 3: Two-phase search (coarse + fine) to find optimal τ via Youden's J statistic.

Usage:
    python gate_calibration/calibrate_tau.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from itertools import combinations
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config.env"))

import config

# Fill in your OpenAI API key here
OPENAI_API_KEY = ""  # Fill in your OpenAI API key here


# ============================================================
# Step 1: GPT-4o-mini as judge for ground-truth labeling
# ============================================================

def get_judge_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def judge_single(client: OpenAI, question: str, gold: str, qwen: str) -> bool:
    """Use GPT-4o-mini to judge whether qwen_answer matches gold_answer.

    Returns True if hallucinated (qwen does NOT match gold).
    """
    prompt = f"""You are an answer-equivalence judge. Given a factual question, a gold (correct) answer, and a model's answer, determine whether the model's answer conveys the same core fact as the gold answer.

Question: {question}
Gold answer: {gold}
Model answer: {qwen}

Rules:
- Focus on factual correctness, not phrasing.
- If the model answer contains the correct fact (even with extra info), it is CORRECT.
- If the model answer states a wrong fact, it is WRONG.
- If the model answer is vague or incomplete but not contradictory, it is CORRECT.

Respond with ONLY one word: CORRECT or WRONG"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "WRONG" in verdict


def label_all(dev: pd.DataFrame) -> pd.Series:
    """Label all samples using GPT-4o-mini as judge."""
    client = get_judge_client()
    labels = []
    total = len(dev)

    for i, (_, row) in enumerate(dev.iterrows()):
        try:
            is_hall = judge_single(
                client,
                str(row["question"]),
                str(row["gold_answer"]),
                str(row["qwen_answer"]),
            )
        except Exception as e:
            print(f"  Error at id={row['id']}: {e}")
            is_hall = None
        labels.append(is_hall)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{total}] labeled")

    return pd.Series(labels, index=dev.index)


# ============================================================
# Step 2: Batch encode + local consistency scoring
# ============================================================

def voyage_batch_encode(texts: list[str], cache_path: str, batch_size: int = 100, max_retries: int = 8) -> np.ndarray:
    """Encode texts via Voyage AI in batches with retry and checkpoint.

    Saves progress after each batch so it can resume on restart.
    """
    import requests
    import json

    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Load checkpoint if exists
    all_embeddings = []
    start_batch = 0
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        all_embeddings = cached["embeddings"]
        start_batch = cached["next_batch"]
        print(f"  Resuming from batch {start_batch + 1}/{total_batches} ({len(all_embeddings)} texts already encoded)")

    for batch_idx in range(start_batch, total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.voyageai.com/v1/embeddings",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
                    },
                    json={
                        "model": config.VOYAGE_EMBEDDING_MODEL,
                        "input": batch_texts,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(embeddings)
                print(f"  Batch {batch_idx + 1}/{total_batches}: encoded {len(batch_texts)} texts")

                # Save checkpoint
                with open(cache_path, "w") as f:
                    json.dump({"embeddings": all_embeddings, "next_batch": batch_idx + 1}, f)

                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = min(30, 5 * (attempt + 1))  # 5s, 10s, 15s, ..., 30s
                    print(f"  Batch {batch_idx + 1}: rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    # Save checkpoint before crashing
                    with open(cache_path, "w") as f:
                        json.dump({"embeddings": all_embeddings, "next_batch": batch_idx}, f)
                    print(f"  Saved checkpoint at batch {batch_idx + 1}. Re-run to resume.")
                    raise

        time.sleep(25)  # 25s between batches to stay under rate limit

    return np.array(all_embeddings, dtype=np.float32)


def compute_all_consistency_scores(dev: pd.DataFrame, cache_path: str) -> np.ndarray:
    """Batch-encode all 4-language responses, compute consistency scores locally."""
    n = len(dev)
    langs = ["qwen_answer", "ar_answer", "ja_answer", "ru_answer"]

    all_texts = []
    for _, row in dev.iterrows():
        for lang in langs:
            all_texts.append(str(row[lang]))

    print(f"  Total texts to encode: {len(all_texts)} ({n} samples × 4 languages)")

    all_embeddings = voyage_batch_encode(all_texts, cache_path)

    emb_dim = all_embeddings.shape[1]
    embeddings = all_embeddings.reshape(n, 4, emb_dim)

    scores = []
    for i in range(n):
        pair_sims = []
        for a, b in combinations(range(4), 2):
            vec_a = embeddings[i, a]
            vec_b = embeddings[i, b]
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            pair_sims.append(sim)
        scores.append(np.mean(pair_sims))

    return np.array(scores)


# ============================================================
# Step 3: Grid search for optimal τ
# ============================================================

def grid_search(scores: np.ndarray, labels: np.ndarray, tau_candidates: np.ndarray):
    """Find τ using Youden's J statistic (optimal ROC cutpoint).

    Youden's J = Sensitivity + Specificity - 1
    This balances both types of errors:
      - Sensitivity: % of hallucinations caught by gate
      - Specificity: % of correct answers that skip RAG

    Prediction rule: ConsistencyScore < τ  →  sent to RAG (suspected hallucination)
    """
    results = []
    for tau in tau_candidates:
        pred = (scores < tau).astype(int)

        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for hallucinated
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for correct
        youden_j = sensitivity + specificity - 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        accuracy = (tp + tn) / len(labels)
        rag_pct = (tp + fp) / len(labels) * 100

        results.append({
            "tau": round(tau, 3),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "youden_j": round(youden_j, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "rag_pct": round(rag_pct, 1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev_path = os.path.join(script_dir, "dev_1000.csv")
    labeled_path = os.path.join(script_dir, "dev_1000_labeled.csv")
    scored_path = os.path.join(script_dir, "dev_1000_scored.csv")
    embed_cache_path = os.path.join(script_dir, "embeddings_cache.json")

    # ---- Step 1: Ground-truth labels via GPT-4o-mini ----
    if os.path.exists(labeled_path):
        print("Step 1: Loading existing GPT labels...")
        dev = pd.read_csv(labeled_path)
        valid = dev.dropna(subset=["is_hallucinated"])
        valid["is_hallucinated"] = valid["is_hallucinated"].astype(bool)
    else:
        print("Loading dev set...")
        dev = pd.read_csv(dev_path)
        print(f"  {len(dev)} samples loaded.")

        print("\nStep 1: Labeling ground truth via GPT-4o-mini judge...")
        dev["is_hallucinated"] = label_all(dev)
        dev.to_csv(labeled_path, index=False)
        print(f"  Saved labels to {labeled_path}")

        valid = dev.dropna(subset=["is_hallucinated"])
        valid["is_hallucinated"] = valid["is_hallucinated"].astype(bool)

    n_hall = valid["is_hallucinated"].sum()
    print(f"  {n_hall} hallucinated, {len(valid) - n_hall} correct, {len(dev) - len(valid)} failed")

    # ---- Step 2: Batch consistency scores ----
    if os.path.exists(scored_path):
        print("\nStep 2: Loading existing scored data (skipping Voyage AI)...")
        valid = pd.read_csv(scored_path)
        valid["is_hallucinated"] = valid["is_hallucinated"].astype(bool)
        print(f"  Loaded {len(valid)} scored samples")
    else:
        print("\nStep 2: Computing consistency scores (batch mode)...")
        valid = valid.copy().reset_index(drop=True)
        scores = compute_all_consistency_scores(valid, embed_cache_path)
        valid["consistency_score"] = scores

        # Save
        valid.to_csv(scored_path, index=False)
        print(f"  Saved {len(valid)} scored samples to {scored_path}")

    # ---- Step 3: Find optimal τ via Youden's J ----
    # Phase 1: Coarse search (step=0.01)
    print("\nStep 3: Finding optimal τ via Youden's J statistic...")
    print("  Phase 1: Coarse search (step=0.01)...")
    tau_coarse = np.arange(0.10, 0.90, 0.01)
    labels_arr = valid["is_hallucinated"].astype(int).values
    score_arr = valid["consistency_score"].values

    coarse_results = grid_search(score_arr, labels_arr, tau_coarse)
    coarse_best = max(coarse_results, key=lambda r: r["youden_j"])
    print(f"  Coarse best: τ={coarse_best['tau']:.2f}, J={coarse_best['youden_j']:.4f}")

    # Phase 2: Fine search around coarse best (step=0.001)
    print(f"  Phase 2: Fine search around τ={coarse_best['tau']:.2f} ±0.05 (step=0.001)...")
    tau_fine = np.arange(coarse_best["tau"] - 0.05, coarse_best["tau"] + 0.051, 0.001)
    fine_results = grid_search(score_arr, labels_arr, tau_fine)
    best = max(fine_results, key=lambda r: r["youden_j"])

    # Print fine results
    print(f"\n  {'τ':<8} {'Sens':<8} {'Spec':<8} {'J':<8} {'Prec':<8} {'→RAG%':<7} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
    print("  " + "-" * 78)
    for r in fine_results:
        marker = " <<<" if r["tau"] == best["tau"] else ""
        print(f"  {r['tau']:<8.3f} {r['sensitivity']:<8.4f} {r['specificity']:<8.4f} {r['youden_j']:<8.4f} {r['precision']:<8.4f} {r['rag_pct']:<7.1f} {r['tp']:<6} {r['fp']:<6} {r['fn']:<6} {r['tn']:<6}{marker}")

    # Save all results
    all_results = coarse_results + fine_results
    grid_path = os.path.join(script_dir, "grid_search_results.csv")
    pd.DataFrame(all_results).to_csv(grid_path, index=False)
    print(f"\n  Saved grid search results to {grid_path}")

    skip_pct = 100 - best["rag_pct"]
    print(f"\n{'='*60}")
    print(f"  RESULT: τ = {best['tau']}")
    print(f"  Youden's J:   {best['youden_j']:.4f}")
    print(f"  Sensitivity:  {best['sensitivity']:.4f} (catches {best['tp']} of {best['tp']+best['fn']} hallucinations)")
    print(f"  Specificity:  {best['specificity']:.4f} (skips {best['tn']} of {best['tn']+best['fp']} correct answers)")
    print(f"  → RAG:        {best['rag_pct']}% of samples sent to RAG")
    print(f"  → Skipped:    {skip_pct:.1f}% of samples skip RAG")
    print(f"\n  Set CONSISTENCY_THRESHOLD={best['tau']} in config.env")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
