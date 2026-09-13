import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import logging

try:
    from src.config import (
        FAISS_INDEX_PATH,
        KB_METADATA_PATH,
        EMBEDDING_MODEL_NAME,
        GROQ_API_KEY,
        GROQ_MODEL,
        RAG_TOP_K,
        RAG_SIMILARITY_THRESHOLD,
        RAG_SYSTEM_PROMPT,
        RAG_USER_PROMPT,
        LANGUAGE_NAMES
    )
except ImportError:
    from config import (
        FAISS_INDEX_PATH,
        KB_METADATA_PATH,
        EMBEDDING_MODEL_NAME,
        GROQ_API_KEY,
        GROQ_MODEL,
        RAG_TOP_K,
        RAG_SIMILARITY_THRESHOLD,
        RAG_SYSTEM_PROMPT,
        RAG_USER_PROMPT,
        LANGUAGE_NAMES
    )

class RAGEngine:
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.metadata = None
        self.groq_client = None
        
    def load(self):
        logging.info("Loading RAG Engine components...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Load FAISS index
        if os.path.exists(FAISS_INDEX_PATH):
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            logging.warning(f"FAISS index not found at {FAISS_INDEX_PATH}")
            
        # Load metadata
        if os.path.exists(KB_METADATA_PATH):
            with open(KB_METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            logging.warning(f"KB metadata not found at {KB_METADATA_PATH}")
            
        # Initialize Groq client
        if GROQ_API_KEY:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        else:
            logging.warning("GROQ_API_KEY not found. LLM generation will be disabled.")
            
        logging.info("RAG Engine loaded successfully.")

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedding_model.encode([query])[0]
        # Normalize for inner product search
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        if not self.faiss_index or not self.metadata:
            logging.error("FAISS index or metadata not loaded.")
            return []
            
        if top_k is None:
            top_k = RAG_TOP_K
            
        query_embedding = self.embed_query(query)
        
        # Search index
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for j, i in enumerate(indices[0]):
            score = scores[0][j]
            if i != -1 and score >= RAG_SIMILARITY_THRESHOLD:
                match = self.metadata[i].copy()
                match['score'] = float(score)
                results.append(match)
                
        return results

    def generate_response(self, query: str, retrieved_chunks: list[dict], sentiment: str = 'neutral', language: str = 'en') -> str:
        if not retrieved_chunks:
            return "I'm sorry, I don't have enough information to answer that question."
            
        if not self.groq_client:
            # Fallback to returning best match
            return retrieved_chunks[0]['response']
            
        # Format context
        context_str = "\n\n".join([
            f"Question: {chunk['instruction']}\nAnswer: {chunk['response']}"
            for chunk in retrieved_chunks
        ])
        
        lang_name = LANGUAGE_NAMES.get(language, 'English')
        
        # Format prompt
        system_prompt = RAG_SYSTEM_PROMPT.format(
            detected_language_name=lang_name,
            detected_sentiment=sentiment,
        )
        
        user_prompt = RAG_USER_PROMPT.format(
            user_message=query,
            context=context_str,
        )
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=GROQ_MODEL,
                temperature=0.3,
                max_tokens=512,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return retrieved_chunks[0]['response']

    def answer(self, query: str, sentiment: str = 'neutral', language: str = 'en') -> dict:
        retrieved = self.retrieve(query)
        
        response = self.generate_response(
            query=query,
            retrieved_chunks=retrieved,
            sentiment=sentiment,
            language=language
        )
        
        return {
            'response': response,
            'sources': retrieved,
            'used_llm': self.groq_client is not None
        }
