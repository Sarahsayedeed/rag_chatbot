import joblib
import re
from typing import Tuple
from src.config import LANG_MODEL_PATH, LANG_VECTORIZER_PATH, LANGUAGE_NAMES

class LanguageDetector:
    def __init__(self):
        """Initialize the language detector with paths from config."""
        self.model_path = LANG_MODEL_PATH
        self.vectorizer_path = LANG_VECTORIZER_PATH
        self.model = None
        self.vectorizer = None

    def load(self):
        """Loads the trained model and vectorizer from disk."""
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        return self

    def preprocess(self, text: str) -> str:
        """
        Preprocesses text exactly as done during training:
        - Lowercase
        - Remove URLs and emails
        - Keep unicode chars, remove extra spaces
        """
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text: str) -> str:
        """
        Predicts the language code of the given text.
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model and vectorizer not loaded. Call load() first.")
            
        clean_text = self.preprocess(text)
        if not clean_text:
            return "unknown"
            
        features = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(features)[0]
        return prediction

    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Predicts the language and returns the confidence score.
        For models that support probability (like LogisticRegression, MultinomialNB), 
        it uses predict_proba. For LinearSVC, it uses decision_function.
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model and vectorizer not loaded. Call load() first.")
            
        clean_text = self.preprocess(text)
        if not clean_text:
            return "unknown", 0.0
            
        features = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(features)[0]
        
        # Calculate confidence
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            confidence = float(max(proba))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)[0]
            # Convert decision values to pseudo-probabilities via softmax-like normalization
            import numpy as np
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            confidence = float(np.max(probs))
        else:
            confidence = 1.0
            
        return prediction, confidence
