"""对 test.csv 的每条问题跑 RAG 验证 + Gate 一致性检验 + gold vs qwen 对比。"""

import sys
import os
import csv
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import stage3_verification
from models import nli_model, encoder
import config

# ---- 配置 ----
INPUT_CSV  = "test.csv"
OUTPUT_CSV = "test_RAG.csv"
BATCH_SIZE = 20
N          = 100   # 先测10条，改成 None 跑全量

FIELDNAMES = [
    "id", "question", "gold_answer", "qwen_answer",
    "consistency_score", "is_consistent",
    "sim_en_ar", "sim_en_ja", "sim_en_ru",
    "sim_ar_ja", "sim_ar_ru", "sim_ja_ru",
    "gold_qwen_similarity", "gold_qwen_consistent",
    "rag_label", "rag_confidence", "corrected_answer",
    "retrieval_time_s", "rag_total_time_s",
    "claim_1", "claim_1_nli", "claim_1_evidence",
    "claim_2", "claim_2_nli", "claim_2_evidence",
    "claim_3", "claim_3_nli", "claim_3_evidence",
    "error",
]

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def compute_all_similarities(texts):
    """一次 API 调用编码所有文本，计算所有pair的相似度。"""
    embeddings = encoder.encode(texts)
    sims = {}
    for i, j in combinations(range(len(texts)), 2):
        sims[(i, j)] = cosine_sim(embeddings[i], embeddings[j])
    return sims

# 预加载 NLI 模型
print("预加载 NLI 模型...")
nli_model.load()
print("NLI 模型加载完成\n")

# 读取输入 CSV
print(f"读取 {INPUT_CSV}...")
source_rows = []
with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row.get("id"):
            source_rows.append(row)
        if N and len(source_rows) >= N:
            break
print(f"共读取 {len(source_rows)} 条\n")

# 断点续跑
done_ids = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("id"):
                done_ids.add(int(row["id"]))
    print(f"检测到已完成 {len(done_ids)} 条，跳过继续...\n")

pending = [r for r in source_rows if int(r["id"]) not in done_ids]
print(f"剩余待处理: {len(pending)} 条\n")

write_header = len(done_ids) == 0
f_out = open(OUTPUT_CSV, "a" if not write_header else "w", newline="", encoding="utf-8-sig")
writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
if write_header:
    writer.writeheader()

batch  = []
errors = 0

for i, item in enumerate(pending):
    rid         = int(item["id"])
    question    = item["question"]
    gold_answer = item.get("gold_answer", "")
    qwen_answer = item.get("qwen_answer", "")
    ar_answer   = item.get("ar_answer", "")
    ja_answer   = item.get("ja_answer", "")
    ru_answer   = item.get("ru_answer", "")

    row = {k: "" for k in FIELDNAMES}
    row.update({"id": rid, "question": question,
                "gold_answer": gold_answer, "qwen_answer": qwen_answer})

    try:
        # ---- 一次 Voyage API 调用：编码 en/ar/ja/ru + gold，共5个文本 ----
        # 索引: 0=en(qwen), 1=ar, 2=ja, 3=ru, 4=gold
        all_texts = [qwen_answer, ar_answer, ja_answer, ru_answer, gold_answer]
        sims = compute_all_similarities(all_texts)

        # Gate 一致性（前4个语言 pair）
        lang_pair_keys = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        lang_pair_names = ["en_ar","en_ja","en_ru","ar_ja","ar_ru","ja_ru"]
        lang_sims = [sims[k] for k in lang_pair_keys]
        consistency_score = float(np.mean(lang_sims))
        is_consistent = consistency_score >= config.CONSISTENCY_THRESHOLD

        row["consistency_score"] = round(consistency_score, 4)
        row["is_consistent"]     = is_consistent
        for name, val in zip(lang_pair_names, lang_sims):
            row[f"sim_{name}"] = round(val, 4)

        # gold vs qwen（索引 0 和 4）
        gold_qwen_sim = sims[(0, 4)]
        row["gold_qwen_similarity"] = round(gold_qwen_sim, 4)
        row["gold_qwen_consistent"] = gold_qwen_sim >= config.CONSISTENCY_THRESHOLD

        # ---- RAG 验证 ----
        result = stage3_verification.run(qwen_answer, question)
        claims = result.get("claims", [])

        row["rag_label"]        = result["label"]
        row["rag_confidence"]   = result["confidence"]
        row["corrected_answer"] = result.get("corrected_answer", "")
        row["retrieval_time_s"] = result["retrieval_time"]
        row["rag_total_time_s"] = result["total_time"]
        row["error"]            = ""

        for j in range(3):
            if j < len(claims):
                row[f"claim_{j+1}"]          = claims[j]["claim"]
                row[f"claim_{j+1}_nli"]      = claims[j]["nli_result"]
                row[f"claim_{j+1}_evidence"] = claims[j]["evidence"][:150] if claims[j]["evidence"] else ""
            else:
                row[f"claim_{j+1}"]          = ""
                row[f"claim_{j+1}_nli"]      = ""
                row[f"claim_{j+1}_evidence"] = ""

    except Exception as e:
        errors += 1
        row["error"] = str(e)

    batch.append(row)

    if len(batch) >= BATCH_SIZE:
        writer.writerows(batch)
        f_out.flush()
        batch = []
        done  = len(done_ids) + i + 1
        total = len(done_ids) + len(pending)
        print(f"进度: {done}/{total} ({done/total*100:.1f}%) | 错误: {errors}")

if batch:
    writer.writerows(batch)
    f_out.flush()

f_out.close()
print(f"\n✅ 完成！结果保存在 {OUTPUT_CSV}，共错误 {errors} 条")