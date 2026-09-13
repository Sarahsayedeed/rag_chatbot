# RAG-Based E-commerce Customer Support Chatbot

A full end-to-end NLP pipeline that processes customer messages through **Language Detection → Sentiment Classification → Intent Classification → RAG-based Q&A** to produce grounded, tone-appropriate responses.

## 🏗️ Architecture

```
Customer Message
       │
       ▼
┌─────────────────────┐
│  Language Detection  │  ← TF-IDF (char n-grams) + Logistic Regression
│  (20 languages)      │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Sentiment Analysis  │  ← Fine-tuned DistilBERT (3 classes)
│  neg / neu / pos     │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Intent Classifier   │  ← TF-IDF + LinearSVC (7 condensed intents)
│  + keyword routing   │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Response Router     │  ← Routes to RAG / direct reply / escalation
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  RAG Pipeline        │  ← FAISS + sentence-transformers + Groq LLM
│  retrieve + generate │
└─────────────────────┘
          ▼
    Final Response
```

## 📁 Project Structure

```
rag_chatbot/
├── notebooks/                         # Training notebooks (deliverables)
│   ├── 01_language_detection.ipynb
│   ├── 02_sentiment_classifier.ipynb
│   ├── 03_intent_classifier.ipynb
│   └── 04_rag_pipeline.ipynb
├── models/                            # Saved trained models
│   ├── lang_detect_model.pkl
│   ├── lang_detect_vectorizer.pkl
│   ├── sentiment_model/               # DistilBERT checkpoint
│   ├── intent_model.pkl
│   └── intent_vectorizer.pkl
├── knowledge_base/                    # FAISS index + metadata
│   ├── faiss_index/index.faiss
│   └── kb_metadata.pkl
├── src/                               # Inference modules
│   ├── config.py                      # Central configuration
│   ├── language_detector.py
│   ├── sentiment_classifier.py
│   ├── intent_classifier.py
│   ├── rag_engine.py
│   └── response_router.py            # Pipeline orchestrator
├── templates/chat.html                # Web UI
├── app.py                             # FastAPI deployment
├── requirements.txt
├── .env                               # GROQ_API_KEY
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd rag_chatbot
pip install -r requirements.txt
```

### 2. Set Up Groq API Key

Get a free API key from [console.groq.com](https://console.groq.com/) and add it to `.env`:

```
GROQ_API_KEY=your_key_here
```

### 3. Train All Models (Run Notebooks)

Run each notebook in order to train and save models:

```bash
jupyter notebook notebooks/01_language_detection.ipynb
jupyter notebook notebooks/02_sentiment_classifier.ipynb
jupyter notebook notebooks/03_intent_classifier.ipynb
jupyter notebook notebooks/04_rag_pipeline.ipynb
```

Each notebook will:
- Download the dataset from Hugging Face
- Train and evaluate the model
- Save the trained model to `models/`

### 4. Launch the Chatbot

```bash
python app.py
```

Visit [http://localhost:8000](http://localhost:8000) for the chat UI, or use the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order #12345?"}'
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web chat UI |
| `/chat` | POST | Full pipeline — process message and get response |
| `/health` | GET | Health check |

### POST /chat — Request

```json
{
  "message": "I want a refund for my damaged item"
}
```

### POST /chat — Response

```json
{
  "response": "I understand how frustrating this must be. To process your refund...",
  "language": "en",
  "language_name": "English",
  "sentiment": "negative",
  "intent": "billing_and_refunds",
  "is_priority": true,
  "sources": [
    {"instruction": "I need a refund", "score": 0.8921}
  ],
  "pipeline_details": {
    "language_detection": {"code": "en", "name": "English"},
    "sentiment_classification": {"label": "negative"},
    "intent_classification": {"label": "billing_and_refunds"},
    "rag": {"used": true, "sources_count": 3}
  }
}
```

## 🧠 Module Details

### Module 1: Language Detection
- **Dataset**: papluca/language-identification (90k samples, 20 languages)
- **Approach**: TF-IDF with character n-grams (3–5) + Logistic Regression
- **Enhancement**: Character-level features capture morphological patterns across scripts
- **Expected accuracy**: ~97%+

### Module 2: Sentiment Classifier
- **Dataset**: dair-ai/emotion (20k Twitter messages, 6 emotions → 3 sentiment buckets)
- **Approach**: Fine-tuned DistilBERT with class-weight balancing
- **Known limitation**: Domain shift (Twitter vs customer support) — documented in notebook
- **Mapping**: sadness/anger/fear → negative; joy/love → positive; surprise → neutral

### Module 3: Intent Classifier
- **Dataset**: bitext/Bitext-customer-support-llm-chatbot (26,872 rows, 27 intents)
- **Approach**: TF-IDF (word unigrams+bigrams) + LinearSVC with GridSearchCV
- **Intent condensation**: 27 → 7 groups + 3 keyword-detected small-talk categories
- **Enhancement**: Keyword detection for greetings/goodbye/gratitude (bypasses ML model)

### Module 4: Q&A RAG Pipeline
- **Knowledge base**: bitext dataset (instruction-response pairs)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector store**: FAISS IndexFlatIP (cosine similarity via normalized vectors)
- **LLM**: Groq API (llama-3.3-70b-versatile) with fallback to best-match response

## 🔀 Response Routing Logic

| Condition | Behavior |
|-----------|----------|
| Greeting / goodbye / thanks | Direct response (no RAG) |
| Negative sentiment OR complaint intent | **Priority flag** + empathetic prefix + RAG |
| Standard query (order, billing, account) | Standard RAG response |
| Out-of-scope | Suggest human escalation |

**Design decision**: Complaint + negative sentiment messages get a mandatory empathetic acknowledgment ("I'm truly sorry...") prepended to the RAG response, and are flagged as `is_priority: true` in the API response. This ensures frustrated customers feel heard before receiving the informational answer.

## ⚠️ Known Limitations

1. **Sentiment domain shift**: The emotion model is trained on Twitter data, not customer support text. Formal complaint language may be misclassified.
2. **English-only knowledge base**: The RAG retrieval works best for English queries. Non-English queries are still processed but retrieval quality may be lower.
3. **Groq dependency**: The LLM generation requires a Groq API key. Without it, the system falls back to returning the closest knowledge base response directly.

## 📦 Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) | Language detection training | 90k |
| [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | Sentiment classifier training | 20k |
| [bitext/Bitext-customer-support-llm-chatbot](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | Intent training + RAG knowledge base | 26,872 |
