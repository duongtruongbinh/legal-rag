"""Chat API router with streaming support."""
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from src.api.schemas import (
    ChatRequest, ChatResponse, SourceDocument, StreamingChatRequest,
    extract_article_reference, smart_truncate
)
from src.rag.chain import get_rag_chain, get_streaming_rag_chain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _convert_history(history: list) -> list[HumanMessage | AIMessage]:
    """Convert API history to LangChain messages."""
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in history
    ]


def _extract_sources(documents: list) -> list[SourceDocument]:
    """Extract unique, well-formatted source documents."""
    seen, sources = set(), []
    
    for doc in documents:
        meta = doc.metadata
        parent_id = meta.get("parent_id") or meta.get("_id") or ""
        
        # Deduplicate
        if parent_id in seen:
            continue
        seen.add(parent_id)
        
        # Extract content and article reference
        content = doc.page_content
        article_ref = extract_article_reference(content)
        
        # Get title - clean it up
        title = meta.get("title", "")
        if not title or title == "Unknown":
            # Try to extract from first line of content
            first_line = content.split('\n')[0].strip()
            if len(first_line) < 200:
                title = first_line
            else:
                title = "Văn bản pháp luật"
        
        # Get law_id and format nicely
        law_id = meta.get("law_id", "")
        if law_id:
            # Format: "luat-123" -> "Luật 123"
            law_id = law_id.replace("-", " ").replace("_", " ").title()
        
        # Get relevance score (already normalized 0-1 from ViRanker)
        score = meta.get("relevance_score", 0.5)
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))
        else:
            score = 0.5
        
        sources.append(SourceDocument(
            content=smart_truncate(content, 400),
            title=title[:150] if title else "Văn bản pháp luật",
            article_ref=article_ref,
            law_id=law_id,
            relevance_score=score,
        ))
    
    # Sort by relevance score
    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return sources


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a legal question with RAG."""
    try:
        chain = get_rag_chain(request.temperature)
        result = await chain.ainvoke({
            "input": request.query,
            "chat_history": _convert_history(request.history),
        })
        
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=_extract_sources(result.get("context", [])),
        )
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response(
    query: str, chat_history: list, temperature: float | None
) -> AsyncGenerator[str, None]:
    """Generate SSE streaming response."""
    chain = get_streaming_rag_chain(temperature)
    sources_sent = False
    
    async for event in chain.astream_events(
        {"input": query, "chat_history": chat_history}, version="v2"
    ):
        kind = event.get("event")
        
        if kind == "on_retriever_end" and not sources_sent:
            docs = event.get("data", {}).get("output", [])
            sources = [s.model_dump() for s in _extract_sources(docs)]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
            sources_sent = True
        
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and getattr(chunk, "content", None):
                yield f"data: {json.dumps({'type': 'token', 'data': chunk.content}, ensure_ascii=False)}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/stream")
async def chat_stream(request: StreamingChatRequest) -> StreamingResponse:
    """Stream legal question response with SSE."""
    try:
        return StreamingResponse(
            _stream_response(
                request.query,
                _convert_history(request.history),
                request.temperature,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        logger.exception("Streaming error")
        raise HTTPException(status_code=500, detail=str(e))
