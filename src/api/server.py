"""FastAPI server entry point."""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.api.schemas import HealthResponse
from src.api.routers import chat_router, ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    print("🚀 Starting Legal RAG API...")
    print(f"📍 Qdrant: {settings.qdrant_url}")
    print(f"📦 Collection: {settings.qdrant_collection}")
    yield
    print("👋 Shutting down Legal RAG API...")


app = FastAPI(
    title="Legal RAG API",
    description="Vietnamese Legal Assistant with Hybrid RAG, Reranking & Streaming",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(ingest_router)


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.3.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Alias for health check."""
    return HealthResponse(status="healthy", version="0.3.0")


def main():
    """Run the server."""
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()

