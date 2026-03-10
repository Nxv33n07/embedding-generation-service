# 🚀 Ecommerce AI Service

A standalone FastAPI microservice for your ecommerce platform, providing **Semantic Search (Pinecone)** and **AI Chat (GPT-OSS)**.

## 🌟 Features
- **Semantic Embeddings**: Uses `all-mpnet-base-v2` (768-dimensions) for high-accuracy product search.
- **Vector Search**: Fully integrated with **Pinecone** for extremely fast similarity lookups.
- **GPT-OSS Chat**: Routes conversational queries to your high-performance **GPT-OSS:20b** model.
- **On-Demand Sync**: Simple API to create/update embeddings for single product descriptions.

---

## 🛠️ API Reference

### 1. Create Product Embedding
Generate and store an embedding for a product description. Use this when a product is created or updated in your database.

- **Endpoint**: `POST /embed-description`
- **Payload**:
```json
{
  "product_id": "ASIN123",
  "description": "High-quality leather office chair with ergonomic lumbar support."
}
```

### 2. Semantic Search
Find the most relevant products based on a natural language query.

- **Endpoint**: `GET /search`
- **Params**: `query` (string), `top_k` (default: 10)
- **Example**: `/search?query=blue travel suitcase&top_k=5`
- **Returns**: A list of matching `product_id`s and their relevance scores.

### 3. AI Chat
Interact with the custom GPT-OSS LLM server.

- **Endpoint**: `POST /chat`
- **Payload**:
```json
{
  "message": "Which product should I buy for back pain?"
}
```

---

## ⚙️ Configuration (.env)
Create a `.env` file in the `ai_service` directory with the following:

```bash
# Pinecone Config
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=ecommerce-index

# OSS LLM Config
OSS_LLM_URL=http://3.109.63.164/gptoss/v1
OSS_LLM_MODEL=gpt-oss:20b

# Embedding Config
EMBEDDING_MODEL=all-mpnet-base-v2  # 768 Dimensions
```

---

## 🚢 Deployment (Docker)

To deploy this service on your server:

1. **Build the image**:
   ```bash
   docker build -t ecommerce-ai-service .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 8006:8006 --env-file .env ecommerce-ai-service
   ```

---

## 🧪 Local Testing
1. **Initialize Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run Service**:
   ```bash
   python main.py
   ```
3. **Check Connection**:
   ```bash
   python check_llm.py  # Verifies GPT-OSS connectivity
   ```
