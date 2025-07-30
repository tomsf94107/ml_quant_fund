# test_sentiment.py
from sentiment_utils import get_sentiment_scores

if __name__ == "__main__":
    result = get_sentiment_scores("AAPL", log_to_csv=True)
    print("Sentiment result:", result)

