"""Chat API router."""
import logging
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.api.schemas import ChatRequest, ChatResponse, SourceDocument
from src.rag.chain import get_rag_chain
from src.core.vector_db import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def convert_history(history: list) -> list[HumanMessage | AIMessage]:
    """Convert API history format to LangChain messages."""
    messages = []
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    return messages


def _extract_score_from_metadata(metadata: dict) -> float | None:
    """Extract relevance score from metadata."""
    score_keys = [
        "relevance_score",
        "score",
        "similarity_score",
        "distance",
        "relevance",
    ]
    
    for key in score_keys:
        if key in metadata:
            val = metadata[key]
            if val is None:
                continue
            try:
                score = float(val)
                if key == "distance" and score > 1:
                    return max(0.0, 1.0 / (1.0 + score))
                return max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                continue
    
    return None


def extract_sources(documents: list, query: str = "") -> list[SourceDocument]:
    """
    Extract source information from retrieved documents.
    
    Args:
        documents: List of retrieved documents.
        query: Optional query string for re-scoring.
    
    Returns:
        List of SourceDocument with extracted metadata.
    """
    sources = []
    vector_store = None
    
    if query and any(_extract_score_from_metadata(doc.metadata) is None for doc in documents):
        try:
            vector_store = get_vector_store()
        except Exception as e:
            logger.warning(f"Could not get vector store for rescoring: {e}")
    
    for doc in documents:
        meta = doc.metadata
        score = _extract_score_from_metadata(meta)
        
        if score is None and vector_store and query:
            try:
                similar_docs = vector_store.similarity_search_with_score(query, k=1)
                if similar_docs:
                    raw_score = similar_docs[0][1]
                    score = max(0.0, min(1.0, 1.0 - raw_score)) if raw_score <= 1.0 else 1.0 / (1.0 + raw_score)
            except Exception:
                score = 0.5
        
        if score is None:
            score = 0.5
        
        sources.append(SourceDocument(
            content=doc.page_content[:500],
            title=meta.get("title") or meta.get("source") or "Unknown",
            doc_id=meta.get("_id") or meta.get("id") or "",
            law_id=meta.get("law_id") or "",
            relevance_score=float(score),
        ))
    
    return sources


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a legal question with RAG.
    
    Args:
        request: Chat request with query and optional history.
    
    Returns:
        Answer with source documents.
    """
    try:
        chain = get_rag_chain(temperature=request.temperature)
        chat_history = convert_history(request.history)
        
        result = chain.invoke({
            "input": request.query,
            "chat_history": chat_history,
        })
        
        context_docs = result.get("context", [])
        sources = extract_sources(context_docs, query=request.query)
        
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=sources,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

