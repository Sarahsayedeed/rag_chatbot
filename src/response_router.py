"""
Response Router — Orchestrates the 4-stage NLP pipeline.

Pipeline: Language Detection → Sentiment Classification → Intent Classification → (conditional) RAG
"""
import logging

from src.language_detector import LanguageDetector
from src.sentiment_classifier import SentimentClassifier
from src.intent_classifier import IntentClassifier
from src.rag_engine import RAGEngine
from src.config import LANGUAGE_NAMES

logger = logging.getLogger(__name__)

# Intents that need RAG retrieval
RAG_INTENTS = {
    "order_status",
    "order_management",
    "billing_and_refunds",
    "account_management",
    "complaint",
    "out_of_scope",
}

# Intents handled with direct (canned) responses — no RAG needed
DIRECT_RESPONSES = {
    "greeting": "Hello! 👋 Welcome to our customer support. How can I help you today?",
    "goodbye": "Thank you for contacting us! Have a wonderful day. Goodbye! 👋",
    "gratitude": "You're welcome! Is there anything else I can help you with?",
}

# Empathetic prefix for negative sentiment or complaint intent
EMPATHY_PREFIX = (
    "I'm truly sorry you're experiencing this issue. "
    "I understand how frustrating this must be, and I want to help resolve it right away. "
)

ESCALATION_MESSAGE = (
    "I apologize, but I'm unable to fully address your concern automatically. "
    "I've flagged this as a priority issue and recommend speaking with a human agent "
    "who can assist you directly. Please hold on while we connect you, or you can "
    "reach our support team at support@store.com."
)


class ResponseRouter:
    """Orchestrates all four NLP modules into a single pipeline."""

    def __init__(self):
        self.language_detector = LanguageDetector()
        self.sentiment_classifier = SentimentClassifier()
        self.intent_classifier = IntentClassifier()
        self.rag_engine = RAGEngine()
        self._loaded = False

    def load(self):
        """Load all models. Call once at startup."""
        logger.info("Loading all models...")
        self.language_detector.load()
        logger.info("  ✓ Language detector loaded")
        self.sentiment_classifier.load()
        logger.info("  ✓ Sentiment classifier loaded")
        self.intent_classifier.load()
        logger.info("  ✓ Intent classifier loaded")
        self.rag_engine.load()
        logger.info("  ✓ RAG engine loaded")
        self._loaded = True
        logger.info("All models loaded successfully!")

    def process(self, message: str) -> dict:
        """
        Process a customer message through the full pipeline.

        Returns:
            dict with keys:
                - response (str): The final response text
                - language (str): Detected language code
                - language_name (str): Human-readable language name
                - sentiment (str): Detected sentiment
                - intent (str): Classified intent
                - is_priority (bool): Whether this is a priority/escalation case
                - sources (list): Retrieved KB sources (if RAG was used)
                - pipeline_details (dict): Debug info about each stage
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # ── Stage 1: Language Detection ────────────────────────────────────
        language = self.language_detector.predict(message)
        lang_name = LANGUAGE_NAMES.get(language, language)
        logger.info(f"Language detected: {language} ({lang_name})")

        # ── Stage 2: Sentiment Classification ──────────────────────────────
        sentiment = self.sentiment_classifier.predict(message)
        logger.info(f"Sentiment detected: {sentiment}")

        # ── Stage 3: Intent Classification ─────────────────────────────────
        intent = self.intent_classifier.predict(message)
        logger.info(f"Intent classified: {intent}")

        # ── Stage 4: Response Generation (conditional) ─────────────────────
        is_priority = False
        sources = []
        response = ""

        # Check if this is a priority case
        if sentiment == "negative" or intent == "complaint":
            is_priority = True

        # Route based on intent
        if intent in DIRECT_RESPONSES:
            # Greeting / goodbye / gratitude — no RAG needed
            response = DIRECT_RESPONSES[intent]

        elif intent == "out_of_scope" and sentiment != "negative":
            # Out-of-scope, non-negative — suggest escalation politely
            response = (
                "I'm not sure I can help with that specific request. "
                "Let me connect you with a human agent who can assist you better. "
                "Is there anything else related to your orders, account, or payments "
                "I can help with in the meantime?"
            )

        else:
            # Use RAG for all substantive intents
            rag_result = self.rag_engine.answer(
                query=message,
                sentiment=sentiment,
                language=language,
            )
            response = rag_result["response"]
            sources = rag_result.get("sources", [])

            # Prepend empathy for priority cases
            if is_priority and not response.lower().startswith(("i'm sorry", "i apologize", "i understand")):
                response = EMPATHY_PREFIX + response

        # For complaint intent with very negative sentiment, add escalation offer
        if intent == "complaint" and sentiment == "negative":
            if "human agent" not in response.lower() and "escalat" not in response.lower():
                response += (
                    "\n\nIf you'd like to speak with a human agent for further "
                    "assistance, I can arrange that for you right away."
                )

        # Build result
        result = {
            "response": response,
            "language": language,
            "language_name": lang_name,
            "sentiment": sentiment,
            "intent": intent,
            "is_priority": is_priority,
            "sources": [
                {
                    "instruction": s.get("instruction", ""),
                    "score": round(s.get("score", 0.0), 4),
                }
                for s in sources
            ],
            "pipeline_details": {
                "language_detection": {
                    "code": language,
                    "name": lang_name,
                },
                "sentiment_classification": {
                    "label": sentiment,
                },
                "intent_classification": {
                    "label": intent,
                },
                "rag": {
                    "used": intent in RAG_INTENTS,
                    "sources_count": len(sources),
                },
            },
        }

        return result
