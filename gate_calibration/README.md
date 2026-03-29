# Gate Threshold (τ) Calibration

This folder contains the code and data for calibrating the consistency gate threshold τ in Stage 2.

## How to run

```bash
python gate_calibration/calibrate_tau.py
```

The script has three steps, each with checkpoint files. If interrupted, re-running will skip completed steps automatically.

## Files

| File | Description |
|------|-------------|
| `calibrate_tau.py` | Main calibration script. Requires OpenAI API key (hardcoded in file). |
| `dev_1000.csv` | Development set: 1,000 samples randomly drawn from the full NQ dataset (excluding test set). All samples have complete 4-language answers with no errors. |
| `dev_1000_labeled.csv` | Step 1 output: `dev_1000.csv` + `is_hallucinated` column, labeled by GPT-4o-mini as independent judge. |
| `dev_1000_scored.csv` | Step 2 output: `dev_1000_labeled.csv` + `consistency_score` column, computed via Voyage AI batch encoding. |
| `grid_search_results.csv` | Step 3 output: all candidate τ values with their Sensitivity, Specificity, Youden's J, Precision, F1, etc. |
| `embeddings_cache.json` | Voyage AI embedding checkpoint (can be deleted after calibration is complete). |

## Method

1. **Step 1 (GPT-4o-mini judge):** For each sample, GPT-4o-mini compares `gold_answer` vs `qwen_answer` and labels it as hallucinated or correct.
2. **Step 2 (Voyage AI encoding):** All 4,000 texts (1,000 samples x 4 languages) are batch-encoded via Voyage AI (`voyage-multilingual-2`), then consistency scores (mean pairwise cosine similarity) are computed locally.
3. **Step 3 (Two-phase search):** A coarse sweep (step=0.01) finds the approximate best τ, then a fine sweep (step=0.001) refines it. The selection criterion is Youden's J statistic (Sensitivity + Specificity - 1), which balances the gate's ability to catch hallucinations against correctly skipping RAG for consistent answers.

## Result

| Metric | Value |
|--------|-------|
| τ | 0.274 |
| Youden's J | 0.2399 |
| Sensitivity | 40.2% (catches 258/642 hallucinations) |
| Specificity | 83.8% (correctly skips 300/358 correct answers) |
| Samples sent to RAG | 31.6% |
| Samples skipped | 68.4% |
