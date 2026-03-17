"""多线程获取 Qwen 对 NQ train 全量问题的原始回答，按顺序保存到 CSV。"""
#你加载的是 google-research-datasets/nq_open 的 train split，总共 87,925 条。
import sys
import os
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from pipeline import stage0_generate

# ---- 配置 ----
SPLIT       = "train"
CSV_FILE    = "nq_qwen_answers.csv"
BATCH_SIZE  = 100
MAX_WORKERS = 8

print("加载 NQ train 数据集...")
ds = load_dataset("google-research-datasets/nq_open", split=SPLIT)
total = len(ds)
print(f"共 {total} 条\n")

FIELDNAMES = ["id", "question", "gold_answer", "qwen_answer", "error"]

# 读取已完成的 id，断点续跑
done_ids = set()
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("id"):
                done_ids.add(int(row["id"]))
    print(f"检测到已完成 {len(done_ids)} 条，跳过继续...\n")

pending = [i for i in range(total) if (i + 1) not in done_ids]
print(f"剩余待处理: {len(pending)} 条，使用 {MAX_WORKERS} 线程\n")

# 追加写入 CSV
write_header = len(done_ids) == 0
f_out = open(CSV_FILE, "a" if not write_header else "w", newline="", encoding="utf-8-sig")
writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
if write_header:
    writer.writeheader()

# 按顺序输出的缓冲区
lock           = threading.Lock()
result_buffer  = {}   # {i: row}
next_to_write  = min(pending) if pending else 0  # 下一个要写的 i
completed      = 0
errors         = 0
write_count    = 0
pending_set    = set(pending)

def flush_in_order():
    """把 buffer 里连续完成的部分按顺序写入 CSV。"""
    global next_to_write, write_count
    rows_to_write = []
    while next_to_write in result_buffer:
        rows_to_write.append(result_buffer.pop(next_to_write))
        next_to_write += 1
        # 跳过已完成的 id（断点续跑时 next_to_write 可能跳过已有数据）
        while next_to_write < total and (next_to_write + 1) in done_ids:
            next_to_write += 1

    if rows_to_write:
        writer.writerows(rows_to_write)
        f_out.flush()
        write_count += len(rows_to_write)

def process_one(i):
    global completed, errors
    item        = ds[i]
    question    = item["question"]
    gold_answer = item["answer"][0] if item["answer"] else ""

    try:
        qwen_answer = stage0_generate.run(question)
        row = {"id": i+1, "question": question, "gold_answer": gold_answer,
               "qwen_answer": qwen_answer, "error": ""}
    except Exception as e:
        row = {"id": i+1, "question": question, "gold_answer": gold_answer,
               "qwen_answer": "", "error": str(e)}

    with lock:
        result_buffer[i] = row
        completed += 1
        if row["error"]:
            errors += 1
        flush_in_order()
        if completed % BATCH_SIZE == 0:
            print(f"进度: {len(done_ids) + completed}/{total} "
                  f"({(len(done_ids)+completed)/total*100:.1f}%) | 错误: {errors}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_one, i): i for i in pending}
    for future in as_completed(futures):
        future.result()

# 写入剩余 buffer
with lock:
    flush_in_order()

f_out.close()
print(f"\n✅ 完成！结果保存在 {CSV_FILE}，共错误 {errors} 条") 
