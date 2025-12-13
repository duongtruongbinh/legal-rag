"""Hybrid retrieval with ViRanker reranking."""
import asyncio
import math
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from src.core.config import settings
from src.core.vector_db import get_vector_store, get_dense_embedding


def _sigmoid(x: float) -> float:
    """Convert logit to probability."""
    return 1 / (1 + math.exp(-x))


class ViRanker:
    """Vietnamese cross-encoder reranker."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Predict normalized relevance scores (0-1) for query-document pairs."""
        if not pairs:
            return []
        
        queries, documents = zip(*pairs)
        inputs = self.tokenizer(
            list(queries), list(documents),
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
            raw_scores = [logits.item()] if logits.dim() == 0 else logits.tolist()
        
        # Normalize logits to 0-1 using sigmoid
        return [_sigmoid(s) for s in raw_scores]


@lru_cache
def get_reranker() -> ViRanker:
    """Get cached reranker instance."""
    return ViRanker(settings.reranker_model, settings.reranker_device)


def _deduplicate_by_parent(documents: list[Document]) -> list[Document]:
    """Deduplicate by parent_id, keeping highest scored."""
    parent_map: dict[str, Document] = {}
    
    for doc in documents:
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            parent_map[str(id(doc))] = doc
            continue
        
        parent_content = doc.metadata.get("parent_content", doc.page_content)
        score = doc.metadata.get("relevance_score", 0)
        
        if parent_id not in parent_map or score > parent_map[parent_id].metadata.get("relevance_score", 0):
            parent_map[parent_id] = Document(
                page_content=parent_content,
                metadata={k: v for k, v in doc.metadata.items() 
                          if k not in ("parent_content", "chunk_index", "total_chunks")}
            )
    
    return list(parent_map.values())


class HybridRerankerRetriever(BaseRetriever):
    """Hybrid retriever with ViRanker reranking."""
    
    vectorstore: Any
    reranker: ViRanker | None = None
    top_k: int = 30
    top_n: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            candidates = [doc for doc, _ in results]
        except Exception:
            candidates = self.vectorstore.similarity_search(query, k=self.top_k)
        
        if not candidates:
            return []
        
        if self.reranker:
            pairs = [(query, doc.page_content) for doc in candidates]
            for doc, score in zip(candidates, self.reranker.predict(pairs)):
                doc.metadata["relevance_score"] = float(score)
            candidates.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        
        parent_docs = _deduplicate_by_parent(candidates)
        parent_docs.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return parent_docs[:self.top_n]
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        loop = asyncio.get_event_loop()
        async_client = AsyncQdrantClient(url=settings.qdrant_url)
        
        try:
            dense_vector = await loop.run_in_executor(
                None, get_dense_embedding().embed_query, query
            )
            
            results = await async_client.query_points(
                collection_name=settings.qdrant_collection,
                query=dense_vector, using="dense",
                limit=self.top_k, with_payload=True,
            )
            
            candidates = [
                Document(
                    page_content=p.payload.get("page_content", ""),
                    metadata={k: v for k, v in p.payload.items() if k != "page_content"}
                ) for p in results.points if p.payload
            ]
            
            if not candidates:
                return []
            
            if self.reranker:
                candidates = await loop.run_in_executor(
                    None, self._sync_rerank, query, candidates
                )
            
            parent_docs = _deduplicate_by_parent(candidates)
            parent_docs.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
            return parent_docs[:self.top_n]
        finally:
            await async_client.close()
    
    def _sync_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        for doc, score in zip(docs, self.reranker.predict(pairs)):
            doc.metadata["relevance_score"] = float(score)
        docs.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return docs


def get_hybrid_retriever(
    top_k: int | None = None,
    top_n: int | None = None,
    use_reranker: bool = True,
) -> BaseRetriever:
    """Get hybrid retriever with optional ViRanker reranking."""
    return HybridRerankerRetriever(
        vectorstore=get_vector_store(),
        reranker=get_reranker() if use_reranker else None,
        top_k=top_k or settings.retrieval_top_k,
        top_n=top_n or settings.reranker_top_n,
    )
