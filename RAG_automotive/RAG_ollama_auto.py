import jieba
import json
import pdfplumber
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
import ollama


stop_words = set([ ])

def remove_stopwords(text):
    return [word for word in jieba.lcut(text) if word not in stop_words]

def split_text_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def clean_text(text):
    text = re.sub(r'\s*(第.*页|page\s*\d+)\s*', '', text)
    text = re.sub(r'\s*（.*?）\s*', '', text)
    text = re.sub(r'[\n\s]+', ' ', text)
    return text

def clean_answer(answer):
    answer = answer.strip()
    if "不确定" in answer or "无法回答" in answer or "无法确定" in answer or "无法直接" in answer or "抱歉" in answer:
        answer = "结合给定的资料，无法回答问题。"
    return answer

pdf = pdfplumber.open("data_auto.pdf.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    text = pdf.pages[page_idx].extract_text()
    if text:
        cleaned_text = clean_text(text)
        chunked_texts = split_text_with_overlap(cleaned_text, chunk_size=189, overlap_size=52)
        for chunk in chunked_texts:
            pdf_content.append({
                'page': f'page_{page_idx + 1}',
                'content': chunk
            })

with open("questions_auto.json", "r", encoding="utf-8") as file:
    questions = json.load(file)


pdf_content_words = [remove_stopwords(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words, k1=1.2, b=0.75)
m3e_model = SentenceTransformer('moka-ai/m3e-small')

pdf_sentences = [x['content'] for x in pdf_content]
pdf_embeddings = m3e_model.encode(pdf_sentences, normalize_embeddings=True)

fusion_results = []

for query_idx, question in enumerate(questions):
    query = question['question']
    query_words = remove_stopwords(query)

    bm25_scores = bm25.get_scores(query_words)
    bm25_top25_idx = bm25_scores.argsort()[-25:][::-1]

    query_embedding = m3e_model.encode([query], normalize_embeddings=True)[0]
    m3e_scores = query_embedding @ pdf_embeddings.T
    m3e_top25_idx = m3e_scores.argsort()[-25:][::-1]
    fusion_scores = {}
    k = 10
    for idx in bm25_top25_idx:
        fusion_scores[idx] = bm25_scores[idx] / (bm25_top25_idx.tolist().index(idx) + k)
    for idx in m3e_top25_idx:
        fusion_scores[idx] = fusion_scores.get(idx, 0) + m3e_scores[idx] / (m3e_top25_idx.tolist().index(idx) + k)

    sorted_fusion = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [idx for idx, _ in sorted_fusion[:10]]

    tokenizer = AutoTokenizer.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
    rerank_model = AutoModelForSequenceClassification.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
    rerank_model.cuda()
    rerank_model.eval()

    pairs = [[query, pdf_content[idx]['content']] for idx in top_docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    best_doc_idx = top_docs[scores.cpu().numpy().argmax()]

    page_reference = pdf_content[best_doc_idx]['page']
    questions[query_idx]['reference'] = page_reference
    questions[query_idx]['content'] = pdf_content[best_doc_idx]['content']

    print(f"参考资料：\n{pdf_content[best_doc_idx]['content']}")

    messages = [
        {"role": "system",
         "content": "You are 你是一个金融的专家，特别擅长分析金融各类问题。你的任务是根据给定的资料回答用户提出的问题。"},
        {"role": "user", "content": f"资料：{pdf_content[best_doc_idx]['content']}\n问题：{query}"}
    ]
    try:
        response = ollama.chat(
            model='qwen2:latest',
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

    answer = clean_answer(answer)
    questions[query_idx]['answer'] = answer

    print(f"问题 {query_idx + 1}: {query}")
    print(f"参考页面：{page_reference}")
    print(f"答案：{answer}")
    print("-" * 50)

with open('fusion_final.json', 'w', encoding='utf8') as file:
    json.dump(questions, file, ensure_ascii=False, indent=4)