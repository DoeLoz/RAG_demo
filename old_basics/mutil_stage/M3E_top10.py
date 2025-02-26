import jieba
import numpy as np
import json
import pdfplumber
from sentence_transformers import SentenceTransformer

with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

def split_text_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

pdf = pdfplumber.open("trainData.pdf")
pdf_content = []
print(pdf.pages)

for page_idx in range(len(pdf.pages)):
    text = pdf.pages[page_idx].extract_text()
    print('page_num1:', page_idx)
    for chunk_text in split_text_with_overlap(text, 60, 15):  # chunk_size=60, overlap_size=15
        print('page_num2:', page_idx)
        print(chunk_text[:5]) 
        pdf_content.append({
            'page': f'page_{page_idx + 1}',  
            'content': chunk_text 
        })


model = SentenceTransformer('moka-ai/m3e-small')


question_sentences = [x['question'] for x in questions]
pdf_content_sentences = [x['content'] for x in pdf_content]

question_embeddings = model.encode(question_sentences, normalize_embeddings=True)
pdf_embeddings = model.encode(pdf_content_sentences, normalize_embeddings=True)
for query_idx, feat in enumerate(question_embeddings):
    score = feat @ pdf_embeddings.T  
    top_10_idx = score.argsort()[-10:][::-1]  
    top_10_pages = list(set([pdf_content[idx]['page'] for idx in top_10_idx])) 
    top_10_pages = sorted(top_10_pages, key=lambda x: int(x.split('_')[1])) 

    questions[query_idx]['reference'] = top_10_pages[:10]  

    print(f"问题 {query_idx}: {questions[query_idx]['question']}")
    print(f"前 10 个最佳匹配页面: {top_10_pages[:10]}")


with open('m3e_top10.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)

