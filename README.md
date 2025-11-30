# ⚖️ VN Legal RAG - Vietnamese Legal AI Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-932259.svg)](https://qdrant.tech/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C.svg)](https://langchain.com/)

A specialized Retrieval-Augmented Generation (RAG) system designed for Vietnamese legal document retrieval and question answering. This project utilizes a **Hybrid Search** strategy (Dense + Sparse embeddings) and **Parent-Child Document Retrieval** to provide accurate, context-aware legal advice powered by Google Gemini.

## 🌟 Key Features

* **Hybrid Search Architecture**: Combines the semantic understanding of dense embeddings with the keyword precision of sparse vectors (BM25).
    * *Dense*: `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`
    * *Sparse*: `Qdrant/bm25` (via FastEmbed)
* **Parent-Child Retrieval**: Splits documents into small chunks for precise vector search (Child) while retrieving the full context (Parent) for the LLM generation. Includes a custom `ScorePreservingRetriever` to maintain ranking quality.
* **Vietnamese Optimized**: Specifically tuned for the Vietnamese language and legal terminology using the Zalo AI corpus.
* **Microservices Architecture**: 
    * **Backend**: Async FastAPI server handling ingestion and chat logic.
    * **Frontend**: Interactive Streamlit UI for easy user interaction.
* **Conversational Memory**: Maintains context across multiple turns of conversation.
* **Parallel Ingestion**: Multi-threaded pipeline to process and index legal documents efficiently.

## 🏗️ Architecture

The system follows a client-server pattern utilizing Qdrant as the vector knowledge base.

```mermaid
graph TD
    User[User] -->|HTTP| FE[Streamlit Frontend]
    FE -->|JSON| API[FastAPI Backend]
    
    subgraph "RAG Pipeline"
        API --> Chain[LangChain RAG]
        Chain -->|1. Contextualize| LLM[Google Gemini]
        Chain -->|2. Hybrid Search| Qdrant[(Qdrant Vector DB)]
        Qdrant -->|Dense + Sparse| Retriever[Score Preserving Retriever]
        Retriever -->|3. Fetch Parent Docs| DocStore[[Local DocStore]]
        DocStore -->|4. Context| LLM
    end
    
    LLM -->|5. Answer| API
````

## 🛠️ Tech Stack

  - **LLM**: Google Gemini 2.5 Flash Lite
  - **Orchestration**: LangChain
  - **Vector Database**: Qdrant (Hybrid mode enabled)
  - **Backend**: FastAPI, Uvicorn
  - **Frontend**: Streamlit
  - **Package Management**: `uv` (Astral)
  - **Data**: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn)

## 🚀 Getting Started

### Prerequisites

  - Python 3.10 or higher
  - [uv](https://github.com/astral-sh/uv) installed (recommended) or pip
  - Docker (for running Qdrant local)
  - Google API Key (for Gemini)

### 1\. Installation

Clone the repository:

```bash
git clone [https://github.com/duongtruongbinh/legal-rag.git](https://github.com/duongtruongbinh/legal-rag.git)
cd legal-rag
```

Install dependencies using `uv`:

```bash
uv sync
```

### 2\. Configuration

Create a `.env` file in the root directory:

```ini
# .env
GOOGLE_API_KEY=your_google_api_key_here

# Qdrant (Local Docker or Cloud)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_hybrid_v2

# Model Settings
LLM_MODEL=gemini-2.5-flash-lite
LLM_TEMPERATURE=0.1
EMBEDDING_DEVICE=cuda  # Use 'cpu' if you don't have a GPU
```

### 3\. Start Infrastructure

Start Qdrant using Docker:

```bash
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 4\. Run the Application

**Step 1: Start the Backend API**

```bash
uvicorn src.api.server:app --reload --port 8000
```

*The API will be available at `http://localhost:8000`.*

**Step 2: Start the Frontend UI**
Open a new terminal:

```bash
streamlit run frontend/ui.py
```

*The UI will open at `http://localhost:8501`.*

## 📚 Data Ingestion

Before asking questions, you need to populate the vector database with legal documents.

**Option 1: Via UI**

1.  Go to the Streamlit interface.
2.  (Optional) Implement an ingestion button in the sidebar (or use the API directly).

**Option 2: Via API (Recommended)**
Trigger the ingestion process using `curl` or Postman:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 100, "collection_name": "legal_hybrid_v2"}'
```

**Option 3: Via Python Script**
Directly run the ingestion module:

```bash
python -m src.rag.ingestion
```

## 🔌 API Endpoints

Documentation is available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check API health status |
| `POST` | `/chat` | Send a query to the RAG system |
| `POST` | `/ingest` | Trigger background document ingestion |
| `GET` | `/ingest/status` | Check progress of ingestion |

## 📂 Project Structure

```
legal-rag/
├── src/
│   ├── api/              # FastAPI app & routers
│   ├── core/             # Configuration & DB connections
│   └── rag/              # Main RAG logic
│       ├── chain.py      # LLM Chain definition
│       ├── retriever.py  # Custom Hybrid Retriever
│       └── ingestion.py  # Data processing pipeline
├── frontend/
│   └── ui.py             # Streamlit application
├── data/                 # Local storage for parent documents
├── pyproject.toml        # Dependencies
└── README.md
```


## 🙏 Acknowledgments

  - Special thanks to **GreenNode** for the Vietnamese Embedding models and Legal Corpus.
  - Built with **LangChain** and **Qdrant**.

