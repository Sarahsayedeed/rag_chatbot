import os
import string
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

from src.config import INTENT_MODEL_PATH, INTENT_VECTORIZER_PATH

class IntentClassifier:
    def __init__(self):
        self.model_path = INTENT_MODEL_PATH
        self.vectorizer_path = INTENT_VECTORIZER_PATH
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keyword patterns
        self.greeting_pattern = re.compile(r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)\b', re.IGNORECASE)
        self.goodbye_pattern = re.compile(r'\b(bye|goodbye|see you|cya|farewell|later)\b', re.IGNORECASE)
        self.gratitude_pattern = re.compile(r'\b(thanks|thank you|appreciate|gracias|cheers)\b', re.IGNORECASE)

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def _is_greeting(self, text: str) -> bool:
        return bool(self.greeting_pattern.search(text))

    def _is_goodbye(self, text: str) -> bool:
        return bool(self.goodbye_pattern.search(text))

    def _is_gratitude(self, text: str) -> bool:
        return bool(self.gratitude_pattern.search(text))

    def preprocess(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize and remove stopwords, then lemmatize
        tokens = text.split()
        cleaned_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(cleaned_tokens)

    def predict(self, text: str) -> str:
        # First check heuristics
        if self._is_goodbye(text):
            return 'goodbye'
        if self._is_gratitude(text):
            return 'gratitude'
        if self._is_greeting(text):
            return 'greeting'
        
        # Use ML model
        if self.model is None or self.vectorizer is None:
            self.load()
            
        cleaned_text = self.preprocess(text)
        X = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(X)[0]
        
        return prediction
