"""Hybrid retrieval with dense + sparse search."""
from functools import lru_cache
from typing import Any, List

from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from src.core.config import settings
from src.core.vector_db import get_vector_store
from src.rag.ingestion import create_parent_splitter, create_child_splitter, get_docstore


def _normalize_score(score: float) -> float:
    """Normalize score/distance to similarity score (0-1 range)."""
    if score < 0:
        return max(0.0, 1 / (1 + abs(score)))
    if score > 1:
        return max(0.0, 1 / (1 + score))
    return max(0.0, min(1.0, 1 - score))


class ScorePreservingRetriever(BaseRetriever):
    """Retriever wrapper that preserves relevance scores from child chunks to parent docs."""
    
    parent_retriever: ParentDocumentRetriever
    vectorstore: Any
        
    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """
        Retrieve documents and preserve scores from child chunks.
        
        Retrieves child chunks with scores, matches them to parent documents,
        and transfers relevance scores to parent docs.
        """
        k = settings.retrieval_top_k
        
        child_results = []
        try:
            if hasattr(self.vectorstore, 'similarity_search_with_score'):
                child_results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,
                )
            else:
                child_docs = self.vectorstore.similarity_search(query, k=k * 3)
                child_results = [(doc, 0.5) for doc in child_docs]
        except Exception as e:
            child_docs = self.vectorstore.similarity_search(query, k=k)
            child_results = [(doc, 0.5) for doc in child_docs]
        
        parent_docs = self.parent_retriever._get_relevant_documents(query, run_manager=run_manager)
        
        parent_scores_map = {}
        
        for child_doc, raw_score in child_results:
            normalized_score = _normalize_score(raw_score)
            
            best_match_id = None
            best_overlap_ratio = 0.0
            
            child_text = child_doc.page_content[:300].strip().lower()
            
            for parent_doc in parent_docs:
                parent_text = parent_doc.page_content.lower()
                
                if child_text in parent_text:
                    overlap_ratio = len(child_text) / max(len(parent_text[:500]), 1)
                    
                    if overlap_ratio > best_overlap_ratio:
                        best_overlap_ratio = overlap_ratio
                        best_match_id = parent_doc.metadata.get("_id") or str(id(parent_doc))
            
            if best_match_id and best_overlap_ratio > 0.05:
                if best_match_id not in parent_scores_map:
                    parent_scores_map[best_match_id] = []
                parent_scores_map[best_match_id].append(normalized_score)
        
        for doc in parent_docs:
            doc_id = doc.metadata.get("_id") or str(id(doc))
            
            if doc_id in parent_scores_map and parent_scores_map[doc_id]:
                max_score = max(parent_scores_map[doc_id])
                doc.metadata["relevance_score"] = float(max_score)
            else:
                position = parent_docs.index(doc)
                position_score = max(0.5, 1.0 - (position * 0.1))
                doc.metadata["relevance_score"] = float(position_score)
        
        return parent_docs
    
    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """Async version using sync implementation."""
        return self._get_relevant_documents(query, run_manager=run_manager)


def get_hybrid_retriever(
    top_k: int | None = None,
) -> BaseRetriever:
    """
    Get hybrid retriever with ParentDocumentRetriever and score preservation.
    
    Uses Qdrant hybrid search (dense + sparse) for child chunks,
    then returns full parent documents from LocalFileStore with preserved scores.
    
    Args:
        top_k: Number of documents to retrieve.
    
    Returns:
        Configured retriever instance with score preservation.
    """
    k = top_k or settings.retrieval_top_k
    vector_store = get_vector_store()
    docstore = get_docstore()
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        parent_splitter=create_parent_splitter(),
        child_splitter=create_child_splitter(),
        search_kwargs={"k": k},
    )
    
    return ScorePreservingRetriever(parent_retriever=parent_retriever, vectorstore=vector_store)


def get_simple_hybrid_retriever(top_k: int | None = None) -> BaseRetriever:
    """
    Get simple hybrid retriever without parent document strategy.
    
    This version will have scores directly from vector search.
    
    Args:
        top_k: Number of documents to retrieve.
    
    Returns:
        Vector store retriever with hybrid search and scores.
    """
    k = top_k or settings.retrieval_top_k
    vector_store = get_vector_store()
    
    class ScoreWrapper(BaseRetriever):
        """Wrapper that adds scores to retrieved documents."""
        vectorstore: Any
        search_k: int
        
        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
            try:
                results = self.vectorstore.similarity_search_with_score(query, k=self.search_k)
                docs = []
                for doc, score in results:
                    similarity = _normalize_score(score)
                    doc.metadata["relevance_score"] = float(similarity)
                    docs.append(doc)
                return docs
            except Exception:
                docs = self.vectorstore.similarity_search(query, k=self.search_k)
                for doc in docs:
                    doc.metadata["relevance_score"] = 0.5
                return docs
    
    return ScoreWrapper(vectorstore=vector_store, search_k=k)
