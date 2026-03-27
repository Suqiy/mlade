# import json
# import time
# import pandas as pd
# from google import genai
# from google.genai import types
# from tqdm import tqdm
# import os

# # ── 配置 ─────────────────────────────────────────────
# GEMINI_KEY  = "AIzaSyBFboHA0cdZP2z3za-rKD9JcF009nTfHSk"
# INPUT_FILE  = "nq_qwen_answers.csv"
# OUTPUT_FILE = "nq_labeled_full.csv"

# client = genai.Client(api_key=GEMINI_KEY)

# JUDGE_PROMPT = """You are a strict factual answer checker.

# [Reference Answer - treat as the only source of truth]
# {golden}

# [Answer to Evaluate]
# {qwen}

# Think step by step:
# 1. List all factual claims in the answer to evaluate
# 2. Check each claim against the reference answer
# 3. Give final judgment

# Output strict JSON only, no other text:
# {{"label": "CONSISTENT" or "HALLUCINATION" or "PARTIAL", "reason": "one sentence"}}"""


# def parse_response(text: str) -> dict:
#     text = text.strip()
#     text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
#     start = text.find("{")
#     end   = text.rfind("}") + 1
#     if start == -1 or end == 0:
#         raise ValueError(f"No JSON found: {text[:80]}")
#     return json.loads(text[start:end])


# def call_gemini(golden: str, qwen: str) -> dict:
#     for attempt in range(5):
#         try:
#             resp = client.models.generate_content(
#                 model="gemini-2.5-flash",
#                 contents=JUDGE_PROMPT.format(golden=golden, qwen=qwen),
#                 config=types.GenerateContentConfig(temperature=0)
#             )
#             return parse_response(resp.text)
#         except Exception as e:
#             err = str(e)
#             if "429" in err:
#                 wait = 10 * (attempt + 1)
#                 tqdm.write(f"  限速，等待 {wait}s...")
#                 time.sleep(wait)
#             else:
#                 tqdm.write(f"  错误: {err[:80]}")
#                 if attempt == 4:
#                     return {"label": "ERROR", "reason": err}
#                 time.sleep(3)


# # ── 断点续跑 ─────────────────────────────────────────
# df = pd.read_csv(INPUT_FILE)

# if os.path.exists(OUTPUT_FILE):
#     done_df  = pd.read_csv(OUTPUT_FILE)
#     done_ids = set(done_df["id"].tolist())
#     df       = df[~df["id"].isin(done_ids)]
#     print(f"断点续跑：已完成 {len(done_ids)} 条，剩余 {len(df)} 条")
# else:
#     done_df = pd.DataFrame()
#     print(f"全新开始，共 {len(df)} 条")

# rows    = df.to_dict("records")
# results = []
# SAVE_EVERY = 100  # 每100条自动保存一次

# for i, row in enumerate(tqdm(rows, desc="Judging")):
#     result = call_gemini(str(row["gold_answer"]), str(row["qwen_answer"]))
#     results.append({"id": row["id"], **result})
#     tqdm.write(f"  [{row['id']}] {result['label']} — {result.get('reason','')[:70]}")

#     # 每100条保存一次
#     if (i + 1) % SAVE_EVERY == 0:
#         tmp = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
#         tmp.to_csv(OUTPUT_FILE, index=False)
#         tqdm.write(f"  ✅ 已保存 {len(tmp)} 条")

#     time.sleep(1)

# # ── 最终保存 ─────────────────────────────────────────
# final_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
# final_df.to_csv(OUTPUT_FILE, index=False)

# # ── 统计 ─────────────────────────────────────────────
# print(f"\n{'='*40}")
# print(f"总计: {len(final_df)} 条")
# print(f"\n标签分布:")
# print(final_df["label"].value_counts())
# print(f"\n错误数: {(final_df['label'] == 'ERROR').sum()}")

import json
import time
import threading
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# ── 配置 ─────────────────────────────────────────────
GEMINI_KEY   = "API_KEY"
INPUT_FILE   = "nq_qwen_answers.csv"
OUTPUT_FILE  = "nq_labeled_full_test.csv"
MAX_WORKERS  = 8      # 并发线程数，429 频繁就调低
SAVE_EVERY   = 100    # 每处理多少条保存一次
MAX_RETRIES  = 6      # 最大重试次数

client = genai.Client(api_key=GEMINI_KEY)

JUDGE_PROMPT = """You are a factual answer checker.
Reference: {golden}
Answer: {qwen}
Does the Answer contain hallucinations compared to the Reference?
Respond with JSON only: {{"label": "CONSISTENT" or "HALLUCINATION" or "PARTIAL"}}"""

# ── 线程锁（保护共享状态）────────────────────────────
_results_lock = threading.Lock()
_save_lock    = threading.Lock()
_rate_event   = threading.Event()   # 限速时让所有线程暂停
_rate_event.set()                   # 初始：不限速，直接通过


def parse_response(text: str) -> dict:
    text  = text.strip()
    text  = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found: {text[:80]}")
    return json.loads(text[start:end])


def call_gemini(golden: str, qwen: str) -> dict:
    for attempt in range(MAX_RETRIES):
        # 如果触发了全局限速，等到恢复再继续
        _rate_event.wait()

        try:
            resp = client.models.generate_content(
                #model="gemini-2.5-flash",
                model="gemini-2.5-flash-lite",
                contents=JUDGE_PROMPT.format(golden=golden, qwen=qwen),
                config=types.GenerateContentConfig(temperature=0)
            )
            return parse_response(resp.text)

        except Exception as e:
            err = str(e)

            if "429" in err:
                # 触发全局暂停，避免所有线程同时轰炸
                if _rate_event.is_set():
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s...
                    tqdm.write(f"  ⚠️  限速(429)，全局暂停 {wait}s ...")
                    _rate_event.clear()
                    time.sleep(wait)
                    _rate_event.set()
                else:
                    # 其他线程已在处理限速，等它恢复
                    _rate_event.wait()

            elif "503" in err or "UNAVAILABLE" in err:
                wait = 3 * (attempt + 1)   # 3s, 6s, 9s...
                tqdm.write(f"  ⚠️  503 过载，{wait}s 后重试 (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)

            else:
                tqdm.write(f"  ❌ 未知错误: {err[:100]}")
                if attempt == MAX_RETRIES - 1:
                    return {"label": "ERROR"}
                time.sleep(2)

    return {"label": "ERROR"}


def process_row(row: dict) -> dict:
    """单条处理，供线程池调用"""
    result = call_gemini(str(row["gold_answer"]), str(row["qwen_answer"]))
    return {"id": row["id"], **result}


# ── 断点续跑 ─────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

if os.path.exists(OUTPUT_FILE):
    done_df  = pd.read_csv(OUTPUT_FILE)
    done_ids = set(done_df["id"].tolist())
    df       = df[~df["id"].isin(done_ids)]
    print(f"断点续跑：已完成 {len(done_ids)} 条，剩余 {len(df)} 条")
else:
    done_df = pd.DataFrame()
    print(f"全新开始，共 {len(df)} 条")

rows    = df.to_dict("records")
results = []


def save_checkpoint():
    with _save_lock:
        tmp = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
        tmp.to_csv(OUTPUT_FILE, index=False)
        tqdm.write(f"  ✅ 已保存 {len(tmp)} 条")


# ── 并发执行 ─────────────────────────────────────────
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_row, row): row for row in rows}

    with tqdm(total=len(rows), desc="Judging") as pbar:
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
            except Exception as e:
                row    = futures[future]
                result = {"id": row["id"], "label": "ERROR"}
                tqdm.write(f"  ❌ [{row['id']}] 线程异常: {e}")

            with _results_lock:
                results.append(result)
                count = len(results)

            tqdm.write(f"  [{result['id']}] {result['label']}")
            pbar.update(1)

            # 定期保存（线程安全）
            if count % SAVE_EVERY == 0:
                save_checkpoint()

            time.sleep(0.5)   # 轻微节流，防止瞬间并发爆炸

# ── 最终保存 ─────────────────────────────────────────
final_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n最终保存完成：{len(final_df)} 条")

# ── 统计 ─────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"总计: {len(final_df)} 条")
print(f"\n标签分布:")
print(final_df["label"].value_counts())
print(f"\n错误数: {(final_df['label'] == 'ERROR').sum()}")