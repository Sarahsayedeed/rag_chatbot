"""
FastAPI Application — RAG-Based E-commerce Customer Support Chatbot.

Serves the chatbot API and a simple web UI for demo purposes.
"""
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.response_router import ResponseRouter

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global router instance ────────────────────────────────────────────────────
router = ResponseRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup."""
    logger.info("=" * 60)
    logger.info("Starting RAG Customer Support Chatbot...")
    logger.info("=" * 60)
    try:
        router.load()
        logger.info("=" * 60)
        logger.info("Chatbot ready! Visit http://localhost:8000")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error("Make sure you've run all 4 training notebooks first!")
        raise
    yield
    logger.info("Shutting down chatbot...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Customer Support Chatbot",
    description="E-commerce customer support chatbot powered by RAG with language detection, sentiment analysis, and intent classification.",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files and templates
templates_dir = PROJECT_ROOT / "templates"
static_dir = PROJECT_ROOT / "static"

if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Request/Response Models ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Where is my order #12345?"
            }
        }
    }


class ChatResponse(BaseModel):
    response: str
    language: str
    language_name: str
    sentiment: str
    intent: str
    is_priority: bool
    sources: list
    pipeline_details: dict


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the chat web UI."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a customer message through the full NLP pipeline.

    The message flows through 4 stages:
    1. Language Detection — identifies the message language
    2. Sentiment Classification — detects emotional tone (negative/neutral/positive)
    3. Intent Classification — routes to the correct handling path
    4. Q&A RAG — retrieves grounded information and generates a response

    Returns the final response along with all pipeline metadata.
    """
    try:
        result = router.process(request.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "An internal error occurred. Please try again.",
                "detail": str(e),
            },
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if router._loaded else "not_ready",
        "models_loaded": router._loaded,
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
