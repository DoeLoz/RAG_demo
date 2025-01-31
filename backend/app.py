from flask import Flask, request, jsonify, render_template
import json
import jieba
import pdfplumber
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
from zhipuai import ZhipuAI

app = Flask(__name__, template_folder="templates", static_folder="static")

stop_words = set(['问题', '请问', '的', '了', '在', '是', '和', '与', '这', '那', '请', '警告', '汽车', '技术', '品牌'])

def remove_stopwords(text):
    return [word for word in jieba.lcut(text) if word not in stop_words]

def clean_text(text):
    text = re.sub(r'\s*(第.*页|page\s*\d+)\s*', '', text)  # 去除页码
    text = re.sub(r'\s*（.*?）\s*', '', text)  # 去除图片标注
    text = re.sub(r'[\n\s]+', ' ', text)  # 去除多余空格和换行符
    return text

def clean_answer(answer):
    answer = answer.strip()
    if "不确定" in answer or "无法回答" in answer or "无法确定" in answer or "无法直接" in answer or "抱歉" in answer:
        answer = "结合给定的资料，无法回答问题。"
    return answer

pdf = pdfplumber.open("trainData.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    text = pdf.pages[page_idx].extract_text()
    if text:
        cleaned_text = clean_text(text)
        pdf_content.append({
            'page': f'page_{page_idx + 1}',
            'content': cleaned_text
        })

pdf_content_words = [remove_stopwords(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words, k1=1.2, b=0.75)

m3e_model = SentenceTransformer('moka-ai/m3e-small')


pdf_sentences = [x['content'] for x in pdf_content]
pdf_embeddings = m3e_model.encode(pdf_sentences, normalize_embeddings=True)


@app.route('/')
def index():
    print("Serving backend.html from templates/")
    return render_template('backend.html')



@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    query = data.get('question')

    if not query:
        return jsonify({'answer': '没有提供问题'}), 400

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

    best_doc_idx = top_docs[0]
    page_reference = pdf_content[best_doc_idx]['page']
    best_doc_content = pdf_content[best_doc_idx]['content']

    zhipu_client = ZhipuAI(api_key="your-api-key")#put your keys here
    messages = [
        {"role": "system", "content": "你是一个汽车领域的专家，特别擅长分析领克汽车的各类问题。你的任务是根据给定的资料回答用户提出的问题。资料可能是不完整的句子或片段，请确保回答简洁、准确。"},
        {"role": "user", "content": f"资料：{best_doc_content}\n问题：{query}"}
    ]
    response = zhipu_client.chat.completions.create(
        model="glm-4-plus",
        messages=messages
    )
    answer = response.choices[0].message.content
    answer = clean_answer(answer)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)


