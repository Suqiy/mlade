import pandas as pd
from tqdm import tqdm
import time
import concurrent.futures
import random

# 从主程序中导入 app
from RAG_hallucination_detection import app

# ==========================================
# 核心配置
# ==========================================
# 强烈建议保持在 5 到 10 之间。太高会被维基百科封 IP 或触发 OpenAI 限流！
MAX_WORKERS = 15

# 处理单行数据，包含独立的容错和防封控延迟
def process_single_row(index, query, answer):
    inputs = {
        "query": str(query),
        "answer": str(answer)
    }
    
    # 加入一小段随机延迟 (0.1 ~ 0.5秒)，打散并发请求，降低被 Wikipedia 封禁的概率
    time.sleep(random.uniform(0.1, 0.5))
    
    try:
        # 运行 LangGraph
        result = app.invoke(inputs)
        verification = result.get("verification", {})
        
        return {
            "index": index, # 返回 index 用于后续恢复数据顺序
            "claim": result.get("claim", ""),
            "label": verification.get("label", "ERROR"),
            "score": verification.get("factuality_score", 0.0),
            "reasoning": verification.get("reasoning", "")
        }
    except Exception as e:
        # 捕获异常，确保单个线程的崩溃不会影响全局
        return {
            "index": index,
            "claim": "ERROR",
            "label": "ERROR",
            "score": 0.0,
            "reasoning": f"Exception: {str(e)}"
        }


def batch_evaluate_csv_concurrent(input_csv_path, output_csv_path):
    print(f"正在加载数据: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)

    except Exception as e:
        print(f"读取 CSV 文件失败: {e}")
        return

    if 'question' not in df.columns or 'qwen_answer' not in df.columns:
        print("CSV 文件中必须包含 'question' 和 'qwen_answer' 列！")
        return

    total_rows = len(df)
    print(f"共加载了 {total_rows} 条数据。开启多线程并发检测 (最大线程数: {MAX_WORKERS})...")

    # 预先分配列表，长度与 df 相同，保证多线程乱序返回后能按原顺序插回
    results_list = [None] * total_rows

    # 记录开始时间以统计整体加速效果
    start_time = time.time()

    # 启动多线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务到线程池，并保留 future 到原始行号的映射
        # 注意：这里我们使用 df 的 enumerate，以确保我们有绝对的行号 index
        future_to_index = {
            executor.submit(process_single_row, idx, row['question'], row['qwen_answer']): idx 
            for idx, row in df.iterrows()
        }

        # 使用 tqdm 监控完成进度
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=total_rows, desc="并发评估进度"):
            res = future.result()
            # 将结果放置在预分配列表中对应的位置
            results_list[res["index"]] = res

    # 解包结果到不同的列
    df['reconstructed_claim'] = [r["claim"] for r in results_list]
    df['hallucination_label'] = [r["label"] for r in results_list]
    df['factuality_score'] = [r["score"] for r in results_list]
    df['verification_reasoning'] = [r["reasoning"] for r in results_list]

    # 保存文件
    print(f"\n正在保存结果到: {output_csv_path}")
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    # 打印耗时
    elapsed_time = time.time() - start_time
    print(f"批量评估完成！总耗时: {elapsed_time:.2f} 秒 (约 {elapsed_time/60:.2f} 分钟)")

if __name__ == "__main__":
    INPUT_FILE = "nq_qwen_answers.csv"   # 替换为你的输入 CSV 文件名
    OUTPUT_FILE = "evaluated_results.csv" # 替换为你想要保存的输出 CSV 文件名
    
    batch_evaluate_csv_concurrent(INPUT_FILE, OUTPUT_FILE)