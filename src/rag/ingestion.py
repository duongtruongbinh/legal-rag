import shutil
import pickle
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path
from itertools import islice

from langchain.storage import EncoderBackedStore, LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datasets import load_dataset
from tqdm import tqdm

from src.core.config import settings
from src.core.vector_db import get_vector_store


def create_parent_splitter() -> RecursiveCharacterTextSplitter:
    """Create splitter for parent documents (coarse-grained)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def create_child_splitter() -> RecursiveCharacterTextSplitter:
    """Create splitter for child documents (fine-grained for dense retrieval)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def get_docstore() -> EncoderBackedStore:
    """Get persistent storage for parent documents."""
    settings.docstore_path.mkdir(parents=True, exist_ok=True)
    fs = LocalFileStore(str(settings.docstore_path))
    return EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )


def clear_existing_data():
    """Clear local docstore to ensure synchronization."""
    if settings.docstore_path.exists():
        print(f"🧹 Clearing old docstore at: {settings.docstore_path}")
        shutil.rmtree(settings.docstore_path)
    settings.docstore_path.mkdir(parents=True, exist_ok=True)


def load_legal_corpus() -> List[Document]:
    """Load and format Vietnamese legal corpus."""
    try:
        ds_corpus = load_dataset(
            "GreenNode/zalo-ai-legal-text-retrieval-vn", 
            "corpus", 
            split="corpus"
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return []

    documents = []
    print("🔄 Processing raw dataset...")
    for row in tqdm(ds_corpus, desc="Formatting"):
        parts = row["_id"].split("+")
        law_id = parts[0] if parts else row["_id"]
        
        doc = Document(
            page_content=f"{row['title']}\n{row['text']}",
            metadata={
                "_id": row["_id"],
                "law_id": law_id,
                "title": row["title"],
                "source": "zalo-ai-corpus"
            }
        )
        documents.append(doc)
    return documents


def _ingest_batch(retriever: ParentDocumentRetriever, batch: List[Document]) -> int:
    """Helper function to ingest a single batch."""
    try:
        retriever.add_documents(batch, ids=None)
        return len(batch)
    except Exception as e:
        print(f"⚠️ Batch Error: {e}")
        return 0


def ingest_documents(
    batch_size: int = 100,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Parallel ingestion pipeline using ThreadPoolExecutor.
    
    Args:
        batch_size: Documents per batch.
        max_workers: Number of parallel threads.
    
    Returns:
        Dict with ingestion status and statistics.
    """
    clear_existing_data()
    print(f"🔌 Connecting to Qdrant: {settings.qdrant_collection}")
    
    vector_store = get_vector_store() 
    docstore = get_docstore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        parent_splitter=create_parent_splitter(),
        child_splitter=create_child_splitter(),
    )

    documents = load_legal_corpus()
    if not documents:
        return {"status": "failed"}

    def chunked_iterable(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, size))
            if not chunk:
                break
            yield chunk

    batches = list(chunked_iterable(documents, batch_size))
    total_docs = len(documents)
    
    print(f"🚀 Starting Parallel Ingestion: {len(batches)} batches | {max_workers} threads")
    
    total_ingested = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_ingest_batch, retriever, batch) for batch in batches]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Ingesting"):
            total_ingested += future.result()

    return {
        "status": "success",
        "total_documents": total_docs,
        "ingested": total_ingested,
        "docstore_path": str(settings.docstore_path)
    }


if __name__ == "__main__":
    result = ingest_documents(batch_size=100, max_workers=10)
    print(f"\n🎉 Ingestion complete: {result}")