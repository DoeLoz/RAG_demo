import json
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import ollama
from datasets import load_dataset


def load_processed_pdfs(input_path):
    with open(input_path, 'r', encoding='utf8') as file:
        return json.load(file)


def clean_answer(answer):
    answer = answer.strip()
    if any(phrase in answer for phrase in ["不确定", "无法回答", "无法确定", "无法直接", "抱歉"]):
        return "结合给定的资料，无法回答问题。"
    return answer

def main(processed_pdf_path):
    print("加载预处理的 PDF 文本...")
    pdf_content = load_processed_pdfs(processed_pdf_path)
    pdf_sentences = [x['content'] for x in pdf_content]

    print("构建 BM25 模型...")
    bm25 = BM25Okapi([x.split() for x in pdf_sentences])

    print("加载嵌入模型...")
    embedding_model = SentenceTransformer('moka-ai/m3e-small')
    pdf_embeddings = embedding_model.encode(pdf_sentences, normalize_embeddings=True)

    print("加载 FinanceBench 数据集...")
    dataset = load_dataset("PatronusAI/financebench", split="train")
    questions = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
    print(f"数据集加载完成，共有 {len(questions)} 个问题。")

    for query_idx, question in enumerate(questions):
        print(f"\n正在处理问题 {query_idx + 1}/{len(questions)}: {question['question']}")
        query = question['question']
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

        bm25_scores = bm25.get_scores(query.split())
        bm25_top25_idx = bm25_scores.argsort()[-25:][::-1]


        m3e_scores = query_embedding @ pdf_embeddings.T
        m3e_top25_idx = m3e_scores.argsort()[-25:][::-1]

        fusion_scores = {}
        k = 10
        for idx in bm25_top25_idx:
            fusion_scores[idx] = bm25_scores[idx] / (bm25_top25_idx.tolist().index(idx) + k)
        for idx in m3e_top25_idx:
            fusion_scores[idx] = fusion_scores.get(idx, 0) + m3e_scores[idx] / (m3e_top25_idx.tolist().index(idx) + k)

        sorted_fusion = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [pdf_content[idx] for idx, _ in sorted_fusion[:10]]


        print("重新排序最佳文档...")
        tokenizer = AutoTokenizer.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
        rerank_model = AutoModelForSequenceClassification.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
        rerank_model.cuda()
        rerank_model.eval()

        pairs = [[query, doc['content']] for doc in top_docs]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            inputs = {key: inputs[key].cuda() for key in inputs.keys()}
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        best_doc = top_docs[scores.cpu().numpy().argmax()]


        print("生成答案...")
        messages = [
            {"role": "system", "content": "你是一个金融专家，擅长分析金融问题。你的任务是根据给定的金融问题提供专业的答案。"},
            {"role": "user", "content": f"相关文档：{best_doc['content']}\n问题：{query}"}
        ]

        try:
            response = ollama.chat(
                model='qwen2',
                messages=messages,
                options={
                    'temperature': 0.3,
                    'num_predict': 500,
                    'top_k': 50
                }
            )
            answer = response['message']['content']
        except Exception as e:
            print(f"模型调用失败: {str(e)}")
            answer = "系统暂时无法生成回答"

        questions[query_idx]['generated_answer'] = clean_answer(answer)
        questions[query_idx]['reference'] = f"{best_doc['file']} - Page {best_doc['page']}"

        print(f"问题 {query_idx + 1}: {query}")
        print(f"最佳参考文档：{best_doc['file']} - Page {best_doc['page']}")
        print(f"模型生成答案：{answer}")
        print("-" * 50)

    print("保存结果到 'financebench_results.json'...")
    with open('financebench_results.json', 'w', encoding='utf8') as file:
        json.dump(questions, file, ensure_ascii=False, indent=4)

    print("任务完成！")

if __name__ == "__main__":
    processed_pdf_path = "./financebench/processed_pdfs.json"  # 预处理后的 PDF 数据
    main(processed_pdf_path)
