"""
Configuration constants and paths for the RAG Chatbot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_base"
FAISS_INDEX_DIR = KB_DIR / "faiss_index"

# ── Model file paths ──────────────────────────────────────────────────────────
LANG_MODEL_PATH = MODELS_DIR / "lang_detect_model.pkl"
LANG_VECTORIZER_PATH = MODELS_DIR / "lang_detect_vectorizer.pkl"
INTENT_MODEL_PATH = MODELS_DIR / "intent_model.pkl"
INTENT_VECTORIZER_PATH = MODELS_DIR / "intent_vectorizer.pkl"
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment_model"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
KB_METADATA_PATH = KB_DIR / "kb_metadata.pkl"

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Free-tier Groq model

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── RAG Settings ──────────────────────────────────────────────────────────────
RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.3

# ── Dataset identifiers (Hugging Face) ────────────────────────────────────────
LANG_DATASET = "papluca/language-identification"
EMOTION_DATASET = "dair-ai/emotion"
BITEXT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# ── Intent mapping (27 fine-grained → 7 condensed) ───────────────────────────
INTENT_MAP = {
    # order_status
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",
    "check_shipping_options": "order_status",
    # order_management
    "cancel_order": "order_management",
    "change_order": "order_management",
    "place_order": "order_management",
    "check_cancellation_fee": "order_management",
    "change_shipping_address": "order_management",
    "set_up_shipping_address": "order_management",
    # billing_and_refunds
    "check_invoice": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "check_refund_policy": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    "check_payment_methods": "billing_and_refunds",
    # account_management
    "create_account": "account_management",
    "edit_account": "account_management",
    "delete_account": "account_management",
    "switch_account": "account_management",
    "recover_password": "account_management",
    # complaint
    "complaint": "complaint",
    "review": "complaint",
    # contact / feedback → out_of_scope-ish but useful
    "contact_customer_service": "complaint",
    "contact_human_agent": "complaint",
    "feedback": "complaint",
    # newsletter
    "newsletter_subscription": "out_of_scope",
}

# ── Sentiment mapping (6 emotions → 3 buckets) ───────────────────────────────
SENTIMENT_MAP = {
    0: "negative",   # sadness
    1: "positive",   # joy
    2: "positive",   # love
    3: "negative",   # anger
    4: "negative",   # fear
    5: "neutral",    # surprise
}

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# ── Prompt template ───────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful, professional customer support assistant \
for an online retailer. Answer the customer's question using ONLY the information \
in the retrieved support responses below. If the customer sounds frustrated \
({detected_sentiment}), acknowledge that before answering. If the retrieved \
context does not cover the question, say so honestly and offer to escalate to a \
human agent rather than guessing. Reply in {detected_language_name}."""

RAG_USER_PROMPT = """Context (retrieved past support responses):
{context}

Customer question: "{user_message}"
"""

# ── Language code → name mapping ──────────────────────────────────────────────
LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}
