import pandas as pd

df_qwen = pd.read_csv("nq_qwen_answers.csv")
df_multi = pd.read_csv("nq_multilingual.csv")

print("nq_qwen_answers 行数:", len(df_qwen))
print("nq_multilingual 行数:", len(df_multi))

# 按 id 和 question 合并
merged = pd.merge(df_qwen, df_multi, on=["id", "question"], how="inner", suffixes=("_qwen", "_multi"))

print("合并后行数:", len(merged))
print("列名:", merged.columns.tolist())
print(merged.head(3))

merged.to_csv("nq_answers.csv", index=False)
print("已保存到 nq_answers.csv")