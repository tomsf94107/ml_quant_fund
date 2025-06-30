from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/Users/atomnguyen/ml_quant_fund/finbert_model"
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print("✅ FinBERT downloaded and saved.")
