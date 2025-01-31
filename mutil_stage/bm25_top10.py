import jieba
import json
import pdfplumber
from rank_bm25 import BM25Okapi

stop_words = set(
    ['问题', '请问', '的', '了', '在', '是', '和', '与', '这', '那', '请', '哪', '如何', '怎么', '哪些', '领克',
     'Lynk & Co', '警告'])


def remove_stopwords(text):
    return [word for word in jieba.lcut(text) if word not in stop_words]


with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

pdf = pdfplumber.open("trainData.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

pdf_content_words = [remove_stopwords(x['content']) for x in pdf_content]

bm25 = BM25Okapi(pdf_content_words, k1=1.2, b=0.75)

for query_idx in range(len(questions)):
    query = questions[query_idx]['question']
    query_words = remove_stopwords(query)

    doc_scores = bm25.get_scores(query_words)

    top_10_idx = doc_scores.argsort()[-10:][::-1]  

    top_10_pages = ['page_' + str(idx + 1) for idx in top_10_idx]

    questions[query_idx]['reference'] = top_10_pages

with open('bm25_top10.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)
