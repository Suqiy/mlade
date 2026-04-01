import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM



# ==========================================
# 0. 全局加载本地模型
# ==========================================
# 将模型加载放在全局，在整个图的运行周期内只需要加载一次
print("Loading local NLI model...")
nli_classifier = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")

print("Loading local FLAN-T5 model...")
# 使用 Google 的 FLAN-T5-small 模型
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# ==========================================
# 1. 定义图的全局状态 (State)
# ==========================================
class GraphState(TypedDict):
    query: str                    # 用户的原始问题
    answer: str                   # 模型生成的简短回答
    claim: str                    # 重构后的完整声明 (Node 1 的输出)
    documents: List[str]          # 用于存储从 Wikipedia 检索到的文本 (Node 2 的输出)
    verification: Dict[str, any]  # 最终的验证结果 (Node 3 的输出)



# ==========================================
# 2. 定义节点 (Nodes)
# ==========================================
# Node 1：将短问答重构成完整的 Claim
def reconstruct_claim_node(state: GraphState):
    query = state["query"]
    answer = state["answer"]
    
    # print(f"\n[Node: Reconstruct] Refactoring Claim using the local FLAN-T5 model...")
    
    try:
        input_text = f"Turn this question and answer into a single declarative sentence.\nQuestion: {query}\nAnswer: {answer}"
        
        # 1. 将文本转换为模型能看懂的张量 (Tokens)
        inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # 2. 模型生成输出 (限制最大长度 128)
        outputs = t5_model.generate(**inputs, max_length=128)
        
        # 3. 将输出的张量解码回人类可读的文本，并去掉特殊字符
        claim = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        if not claim:
            claim = f"{query} - {answer}"
            
    except Exception as e:
        print(f"[Node: Reconstruct] Reconstruction error: {e}")
        # 出错时回退到强拼接策略，保证图能继续往下走
        claim = f"{query} - {answer}"
        
    # print(f"[Node: Reconstruct] The final generated Claim: {claim}")
    
    return {"claim": claim}


# 实例化 Wikipedia Retriever
# 设置最多返回 2 个结果，每个结果最多 1000 个字符
wiki_retriever = WikipediaRetriever(top_k_results=2, doc_content_chars_max=1000)

# Node 2：使用重构后的 Claim 从 Wikipedia 中检索证据
def retrieve_evidence_node(state: GraphState):
    claim = state["claim"]
    # print(f"\n[Node: Retrieve] Searching Wikipedia for: {claim}")
    
    # 使用 claim 进行检索
    docs = wiki_retriever.invoke(claim)
    
    # 提取 Document 对象中的纯文本内容
    doc_contents = [doc.page_content for doc in docs]
    
    # print(f"[Node: Retrieve] Found {len(doc_contents)} documents.")
    
    # 更新状态字典中的 documents 字段
    return {"documents": doc_contents}


# Node 3：使用本地 NLI 模型对比 claim 和 evidence，输出幻觉得分
def verify_claim_node_nli(state: GraphState):
    claim = state["claim"]
    documents = state["documents"]
    
    # 如果没有检索到任何证据，直接返回中立结果
    if not documents:
        print("[Node: Verify] No evidence found, skip the reasoning")
        return {
            "verification": {
                "label": "NEUTRAL", 
                "factuality_score": 0.5, 
                "reasoning": "No evidence retrieved from Wikipedia."
            }
        }
        
    # 将检索到的多个文档拼接成一段完整的前提 (Premise)
    evidence = "\n".join(documents)
    
    # print(f"\n[Node: Verify] Evaluating factual information using a local NLI model...")
    
    try:
        # 进行 NLI 推理
        # 输入格式：text 为前提(Evidence)，text_pair 为假设(Claim)
        # 加入 truncation=True 和 max_length=512，防止维基百科文本过长导致模型崩溃
        # 加入 top_k=None，强制模型返回 [entailment, neutral, contradiction] 三个维度的所有概率
        result = nli_classifier(
            {"text": evidence, "text_pair": claim}, 
            truncation=True, 
            max_length=512,
            top_k=None
        )
        
        # result 是一个包含 3 个字典的列表：
        # [{'label': 'entailment', 'score': 0.45}, {'label': 'neutral', 'score': 0.35}, {'label': 'contradiction', 'score': 0.20}]
        # print(f"[Debug] NLI All Scores: {result}")
        
        # 解析三个标签的得分
        scores = {item['label'].lower(): item['score'] for item in result}

        # 兼容不同模型的标签命名 (有的是 entailment, 有的是 label_1)
        entailment_score = scores.get('entailment', scores.get('label_1', 0.0))
        contradiction_score = scores.get('contradiction', scores.get('label_0', 0.0))
        neutral_score = scores.get('neutral', scores.get('label_2', 0.0))

        # 事实性得分 = 支持的概率 - 反驳的概率。
        # 如果反驳概率很高，得分会变成负数；如果纯中立，得分趋近于 0。
        # 我们将其归一化到 0 ~ 1 之间： (score + 1) / 2
        raw_score = entailment_score - contradiction_score
        factuality_score = (raw_score + 1) / 2
        
        # 根据最高分定性 Label
        if entailment_score > max(contradiction_score, neutral_score):
            final_label = "SUPPORTS"
        elif contradiction_score > max(entailment_score, neutral_score):
            final_label = "REFUTES"
        else:
            final_label = "NEUTRAL"

        reasoning = f"Entailment: {entailment_score:.2f}, Neutral: {neutral_score:.2f}, Contradiction: {contradiction_score:.2f}"

    except Exception as e:
        print(f"[Node: Verify] Inference error: {e}")
        final_label = "NEUTRAL"
        factuality_score = 0.5
        reasoning = f"Error during local NLI verification: {str(e)}"

    response = {
        "label": final_label,
        "factuality_score": round(factuality_score, 4),
        "reasoning": reasoning
    }
    
    # print(f"[Node: Verify] Score: {response['factuality_score']} ({response['label']})")
    return {"verification": response}



# ==========================================
# 3. 构建与编译图 (Graph)
# ==========================================
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("reconstruct", reconstruct_claim_node)
workflow.add_node("retrieve", retrieve_evidence_node)
workflow.add_node("verify", verify_claim_node_nli)

# 设置数据流向
workflow.add_edge(START, "reconstruct")
workflow.add_edge("reconstruct", "retrieve")
workflow.add_edge("retrieve", "verify")
workflow.add_edge("verify", END)

app = workflow.compile()



if __name__ == "__main__":
    # 测试：正确的回答
    inputs_true = {
        "query": "Where is the capital of France?",
        "answer": "Paris."
    }
    print(">>> 测试真实的回答：")
    result_true = app.invoke(inputs_true)
    print("\n--- 最终验证结果 (True) ---")
    print(json.dumps(result_true["verification"], indent=2, ensure_ascii=False))

    print("\n" + "="*50 + "\n")

    # 测试：幻觉回答
    inputs_hallucination = {
        "query": "Where is the capital of France?",
        "answer": "London."  # 故意给出错误答案
    }
    print(">>> 测试包含幻觉的回答：")
    result_hallucination = app.invoke(inputs_hallucination)
    print("\n--- 最终验证结果 (Hallucination) ---")
    print(json.dumps(result_hallucination["verification"], indent=2, ensure_ascii=False))