import jieba
import numpy as np
import json
import pdfplumber
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

stop_words = set(
    ['问题', '请问', '的', '了', '在', '是', '和', '与', '这', '那', '请', '哪', '如何', '怎么', '哪些', '领克',
     'Lynk & Co', '警告'])


def remove_stopwords(text):

    return [word for word in jieba.lcut(text) if word not in stop_words]


def split_text_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

pdf = pdfplumber.open("trainData.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    text = pdf.pages[page_idx].extract_text()
    if text:  
        chunked_texts = split_text_with_overlap(text, chunk_size=60, overlap_size=15)
        for chunk in chunked_texts:
            pdf_content.append({
                'page': f'page_{page_idx + 1}', 
                'content': chunk  
            })


pdf_content_words = [remove_stopwords(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

tokenizer = AutoTokenizer.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
rerank_model = AutoModelForSequenceClassification.from_pretrained('../hugging-face-model/BAAI/bge-reranker-base/')
rerank_model.cuda()
rerank_model.eval()

for query_idx in range(len(questions)):
    query = questions[query_idx]["question"]
    query_words = remove_stopwords(query)

    doc_scores = bm25.get_scores(query_words)
    max_score_page_idxs = doc_scores.argsort()[-5:]  

    pairs = [[query, pdf_content[idx]['content']] for idx in max_score_page_idxs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)

    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    max_score_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
    questions[query_idx]['reference'] = pdf_content[max_score_idx]['page'] 

with open('rerankv2.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
