import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import json
import pdfplumber

with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

pdf = pdfplumber.open("trainData.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })


question_words = [' '.join(jieba.lcut(x['question'])) for x in questions]
#print(question_words)
pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]
#print(pdf_content_words)
tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)
question_feat = tfidf.transform(question_words)
#print(question_feat)
#(0, 4181) 0.44486360335226943 表示问题0中第4181个词汇的TF-IDF值为0.4448，索引稀疏
pdf_content_feat = tfidf.transform(pdf_content_words)

question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)

for query_idx, feat in enumerate(question_feat):
    score = feat @ pdf_content_feat.T
    score = score.toarray()[0]
    max_score_page_idx = score.argsort()[-1] + 1
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)


with open('tfidf.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)