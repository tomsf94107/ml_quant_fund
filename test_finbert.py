from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./finbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("✅ FinBERT loaded and ready!")

