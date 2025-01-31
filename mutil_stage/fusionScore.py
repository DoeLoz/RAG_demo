import json


with open('m3e_top10.json', 'r', encoding='utf-8') as file1:
    m3e = json.load(file1)

with open('bm25_top10.json', 'r', encoding='utf-8') as file2:
    bm25 = json.load(file2)

fusion_result = []

k = 40

for q1, q2 in zip(m3e, bm25):
    fusion_score = {}

    for idx, q in enumerate(q1['reference']):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)  
        else:
            fusion_score[q] += 1 / (idx + k)  

    for idx, q in enumerate(q2['reference']):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k) 
        else:
            fusion_score[q] += 1 / (idx + k)  

    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    q1['reference'] = sorted_dict[0][0]

    fusion_result.append(q1)

with open('fusionScore.json', 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)
