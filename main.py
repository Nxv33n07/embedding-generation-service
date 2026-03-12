import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load env variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-index")
OSS_LLM_URL         = os.getenv("OSS_LLM_URL")
OSS_LLM_MODEL       = os.getenv("OSS_LLM_MODEL", "gpt-oss:20b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# ── Model & client init (module-level, loaded once) ────────────────────────────
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedder   = SentenceTransformer(EMBEDDING_MODEL_NAME)
pc         = Pinecone(api_key=PINECONE_API_KEY)
llm_client = OpenAI(base_url=OSS_LLM_URL, api_key="dummy") if OSS_LLM_URL else None

# Pinecone index connection — created once in lifespan, reused across all requests
pinecone_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pinecone_index
    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME} ...")
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("✓ Pinecone index connected")
    yield
    logger.info("AI service shutting down")


app = FastAPI(
    title="Ecommerce AI Service",
    description="Embeddings, vector search, and LLM chat. Content-based and collaborative filtering live in the backend service.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai_service", "model": EMBEDDING_MODEL_NAME}


# ══════════════════════════════════════════════════════════════════════════════
# EMBED
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/embed-description")
async def embed_description(
    product_id: str = Body(...),
    description: str = Body(...)
):
    """
    Generate an embedding for a product description and upsert it into Pinecone.
    Called by the backend whenever a product is created or updated.
    """
    if not product_id or not description:
        raise HTTPException(status_code=400, detail="Missing product_id or description")

    try:
        embedding = await asyncio.to_thread(embedder.encode, description)
        await asyncio.to_thread(
            pinecone_index.upsert,
            vectors=[{"id": product_id, "values": embedding.tolist(), "metadata": {"id": product_id}}]
        )
        logger.info(f"Embedded product: {product_id}")
        return {"message": "Embedding created and stored", "product_id": product_id}

    except Exception as e:
        logger.error(f"embed_description error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH  –  query → embedding → Pinecone → product IDs
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search")
async def search(query: str = Query(...), top_k: int = Query(default=10)):
    """
    Semantic search:
      1. Encode the query with the sentence-transformer model.
      2. Query Pinecone for the top-k nearest product vectors.
      3. Return matching product IDs with cosine similarity scores.
    """
    try:
        query_vector = await asyncio.to_thread(embedder.encode, query)
        results = await asyncio.to_thread(
            pinecone_index.query,
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        matches = [
            {"product_id": match.id, "score": match.score}
            for match in results.matches          # ← Pinecone v3 QueryResponse object
        ]
        return {"query": query, "matches": matches}

    except Exception as e:
        logger.error(f"search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHAT  –  OSS LLM
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(message: str = Body(..., embed=True)):
    """Forward a chat message to the configured OSS LLM."""
    if not llm_client:
        raise HTTPException(status_code=500, detail="OSS LLM not configured (OSS_LLM_URL not set)")

    try:
        response = llm_client.chat.completions.create(
            model=OSS_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for an ecommerce platform."},
                {"role": "user", "content": message},
            ]
        )
        return {"reply": response.choices[0].message.content}

    except Exception as e:
        logger.error(f"chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
