"""Qdrant Vector Store with Hybrid Search (Dense + Sparse)."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from .config import settings


@lru_cache
def get_dense_embedding() -> HuggingFaceEmbeddings:
    """Initialize dense embedding model (GreenNode)."""
    return HuggingFaceEmbeddings(
        model_name=settings.dense_model,
        model_kwargs={"device": settings.embedding_device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_sparse_embedding() -> FastEmbedSparse:
    """Initialize sparse embedding model (BM25)."""
    return FastEmbedSparse(model_name=settings.sparse_model)


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection_exists(client: QdrantClient, collection_name: str) -> None:
    """Create collection with hybrid search config if not exists."""
    if client.collection_exists(collection_name):
        return
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        },
    )


def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """
    Get QdrantVectorStore with Hybrid Search enabled.
    
    Args:
        collection_name: Optional collection name, defaults to config value.
    
    Returns:
        QdrantVectorStore configured for hybrid retrieval.
    """
    collection = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    ensure_collection_exists(client, collection)
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=get_dense_embedding(),
        sparse_embedding=get_sparse_embedding(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

