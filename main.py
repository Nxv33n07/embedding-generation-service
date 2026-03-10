import os
import logging
import httpx
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load env variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ecommerce AI Service", description="AI components for embeddings, vector search, and LLM chat.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-index")

OSS_LLM_URL = os.getenv("OSS_LLM_URL")
OSS_LLM_MODEL = os.getenv("OSS_LLM_MODEL", "gpt-oss:20b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Initialize models and clients
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
llm_client = OpenAI(base_url=OSS_LLM_URL, api_key="dummy") if OSS_LLM_URL else None

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai_service", "model": EMBEDDING_MODEL_NAME}

@app.post("/embed-description")
async def embed_description(
    product_id: str = Body(...), 
    description: str = Body(...)
):
    """
    Generate embedding for a single product description and store in Pinecone.
    Payload: { "product_id": "ASIN123", "description": "High-quality leather bag..." }
    """
    if not product_id or not description:
        raise HTTPException(status_code=400, detail="Missing product_id or description")
    
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # 1. Generate embedding from description string
        embedding = embedder.encode(description).tolist()
        
        # 2. Store in Pinecone with minimal metadata
        index.upsert(vectors=[{
            "id": product_id,
            "values": embedding,
            "metadata": {"id": product_id} # Minimal metadata to keep it valid
        }])
        
        logger.info(f"Successfully embedded description for product: {product_id}")
        return {"message": "Embedding created and stored", "product_id": product_id}
    
    except Exception as e:
        logger.error(f"Error in embed_description: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: str = Body(..., embed=True)):
    """
    Use the GPT OSS model for chat.
    """
    if not llm_client:
        raise HTTPException(status_code=500, detail="OSS LLM not configured")
    
    try:
        response = llm_client.chat.completions.create(
            model=OSS_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for an ecommerce platform."},
                {"role": "user", "content": message}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str = Query(...), top_k: int = Query(default=10)):
    """
    Semantic search: Query -> Embedding -> Pinecone Search.
    """
    try:
        query_vector = embedder.encode(query).tolist()
        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        matches = []
        for match in results.get("matches", []):
            matches.append({
                "product_id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
            
        return {"query": query, "matches": matches}
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
