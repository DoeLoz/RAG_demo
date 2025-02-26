import json
import pdfplumber

with open("questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)
print(questions[0])

pdf = pdfplumber.open("trainData.pdf")
print(len(pdf.pages))
testText =pdf.pages[124].extract_text()
print(testText)

