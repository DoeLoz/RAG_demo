from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = "BAAI/bge-reranker-base"

local_model_path = "../hugging-face-model/BAAI/bge-reranker-base/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_model_path)

rerank_model = AutoModelForSequenceClassification.from_pretrained(model_name)
rerank_model.save_pretrained(local_model_path)
