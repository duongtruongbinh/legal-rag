"""Pydantic models for API request/response validation."""
import re
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


class StreamingChatRequest(BaseModel):
    """Streaming chat endpoint request body."""
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceDocument(BaseModel):
    """Retrieved source document with legal reference info."""
    content: str = Field(description="Excerpt from the legal document")
    title: str = Field(description="Document/Law title")
    article_ref: str = Field(default="", description="Article reference (e.g., Điều 5, Khoản 2)")
    law_id: str = Field(default="", description="Law identifier")
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Relevance score 0-1")


class ChatResponse(BaseModel):
    """Chat endpoint response body."""
    answer: str
    sources: list[SourceDocument]


class StreamChunk(BaseModel):
    """Single streaming chunk."""
    type: str = Field(..., pattern="^(sources|token|done)$")
    data: str | list[SourceDocument] | None = None


class IngestRequest(BaseModel):
    """Ingest endpoint request body."""
    collection_name: str | None = None
    batch_size: int = Field(default=50, ge=1, le=200)
    max_workers: int = Field(default=4, ge=1, le=16)


class IngestResponse(BaseModel):
    """Ingest endpoint response body."""
    status: str
    total_raw_documents: int
    total_child_documents: int
    ingested: int
    collection: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


def extract_article_reference(text: str) -> str:
    """Extract Điều/Khoản reference from text."""
    patterns = [
        r"(Điều\s+\d+[a-zA-Z]?)",
        r"(Khoản\s+\d+)",
        r"(Điểm\s+[a-zA-Z])",
    ]
    
    refs = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            refs.append(match.group(1))
    
    return ", ".join(refs) if refs else ""


def smart_truncate(text: str, max_length: int = 500) -> str:
    """Truncate text at sentence boundary."""
    if len(text) <= max_length:
        return text
    
    # Find last sentence end before max_length
    truncated = text[:max_length]
    last_period = max(truncated.rfind('.'), truncated.rfind('。'), truncated.rfind('\n'))
    
    if last_period > max_length * 0.5:
        return truncated[:last_period + 1]
    
    # Fallback: truncate at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.7:
        return truncated[:last_space] + "..."
    
    return truncated + "..."
