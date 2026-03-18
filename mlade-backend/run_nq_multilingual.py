"""多线程生成 NQ 问题的多语言回答（ar/ja/ru），保存到单独 CSV。"""

import sys
import os
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import target_llm, translator
import config

# ---- 配置 ----
SOURCE_CSV  = "nq_qwen_answers.csv"   # 读取问题来源
OUTPUT_CSV  = "nq_multilingual.csv"
BATCH_SIZE  = 100
MAX_WORKERS = 20    # 翻译+Qwen共4次调用，线程别太多
LANGS       = ["ar", "ja", "ru"]

FIELDNAMES = (
    ["id", "question"] +
    [f"{l}_question" for l in LANGS] +
    [f"{l}_answer"   for l in LANGS] +
    ["error"]
)

# 读取源 CSV
print(f"读取 {SOURCE_CSV}...")
source_rows = []
with open(SOURCE_CSV, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row.get("id") and row.get("question"):
            source_rows.append({"id": int(row["id"]), "question": row["question"]})
print(f"共 {len(source_rows)} 条问题\n")

# 断点续跑：读取已完成的 id
done_ids = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("id"):
                done_ids.add(int(row["id"]))
    print(f"检测到已完成 {len(done_ids)} 条，跳过继续...\n")

pending = [r for r in source_rows if r["id"] not in done_ids]
print(f"剩余待处理: {len(pending)} 条，使用 {MAX_WORKERS} 线程\n")

# 初始化 CSV
write_header = len(done_ids) == 0
f_out = open(OUTPUT_CSV, "a" if not write_header else "w", newline="", encoding="utf-8-sig")
writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
if write_header:
    writer.writeheader()

# 按顺序输出的缓冲区
lock          = threading.Lock()
result_buffer = {}
pending_ids   = [r["id"] for r in pending]
next_to_write_idx = 0   # pending_ids 里的指针
completed     = 0
errors        = 0

def flush_in_order():
    global next_to_write_idx
    rows_to_write = []
    while next_to_write_idx < len(pending_ids):
        pid = pending_ids[next_to_write_idx]
        if pid not in result_buffer:
            break
        rows_to_write.append(result_buffer.pop(pid))
        next_to_write_idx += 1
    if rows_to_write:
        writer.writerows(rows_to_write)
        f_out.flush()

def process_one(item):
    global completed, errors
    rid      = item["id"]
    question = item["question"]

    try:
        translated = {}
        answers    = {}
        for lang in LANGS:
            q_translated = translator.translate(question, src_lang="en", tgt_lang=lang)
            translated[lang] = q_translated
            system_prompt = config.SYSTEM_PROMPTS[lang]
            answers[lang] = target_llm.generate(q_translated, system_prompt=system_prompt)

        row = {"id": rid, "question": question, "error": ""}
        for lang in LANGS:
            row[f"{lang}_question"] = translated[lang]
            row[f"{lang}_answer"]   = answers[lang]

    except Exception as e:
        row = {"id": rid, "question": question, "error": str(e)}
        for lang in LANGS:
            row[f"{lang}_question"] = ""
            row[f"{lang}_answer"]   = ""

    with lock:
        result_buffer[rid] = row
        completed += 1
        if row["error"]:
            errors += 1
        flush_in_order()
        if completed % BATCH_SIZE == 0:
            total = len(pending)
            print(f"进度: {len(done_ids)+completed}/{len(done_ids)+total} "
                  f"({(len(done_ids)+completed)/(len(done_ids)+total)*100:.1f}%) | 错误: {errors}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_one, item): item for item in pending}
    for future in as_completed(futures):
        future.result()

with lock:
    flush_in_order()

f_out.close()
print(f"\n✅ 完成！结果保存在 {OUTPUT_CSV}，共错误 {errors} 条")
