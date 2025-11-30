"""Pydantic models for API request/response validation."""
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Chat endpoint request body."""
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceDocument(BaseModel):
    """Retrieved source document."""
    content: str
    title: str | None = None
    doc_id: str | None = None
    law_id: str | None = None
    relevance_score: float | None = None


class ChatResponse(BaseModel):
    """Chat endpoint response body."""
    answer: str
    sources: list[SourceDocument]


class IngestRequest(BaseModel):
    """Ingest endpoint request body."""
    collection_name: str | None = None
    batch_size: int = Field(default=50, ge=1, le=200)


class IngestResponse(BaseModel):
    """Ingest endpoint response body."""
    status: str
    total_documents: int
    ingested: int
    collection: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str

