import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import SENTIMENT_MODEL_DIR, SENTIMENT_LABELS

class SentimentClassifier:
    def __init__(self):
        self.model_dir = SENTIMENT_MODEL_DIR
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = {i: label for i, label in enumerate(SENTIMENT_LABELS)}

    def load(self):
        """Loads the fine-tuned sentiment model and tokenizer."""
        print(f"Loading Sentiment Classifier from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("Sentiment Classifier loaded successfully.")

    def predict_with_scores(self, text: str) -> dict:
        """
        Predicts sentiment and returns scores for all classes.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call load() first.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)

        scores = {self.id2label[i]: prob.item() for i, prob in enumerate(probs)}
        return scores

    def predict(self, text: str) -> str:
        """
        Predicts the sentiment label for the given text.
        """
        scores = self.predict_with_scores(text)
        return max(scores, key=scores.get)
