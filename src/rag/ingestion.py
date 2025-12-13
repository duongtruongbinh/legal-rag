"""Document ingestion with Vietnamese legal text splitting."""
import re
import uuid
import concurrent.futures
from typing import Iterator
from itertools import islice

from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
from datasets import load_dataset
from tqdm import tqdm

from src.core.config import settings
from src.core.vector_db import get_vector_store


class VietnameseLegalTextSplitter(TextSplitter):
    """Vietnamese legal document splitter by Điều/Khoản structure."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._article_re = re.compile(r"(Điều\s+\d+[a-zA-Z]?\.?\s*[^\n]*)", re.IGNORECASE)
        self._clause_re = re.compile(r"(\d+\.\s+)")
        self._chapter_re = re.compile(r"(Chương\s+[IVXLCDM\d]+\.?\s*[^\n]*)", re.IGNORECASE)

    def split_text(self, text: str) -> list[str]:
        chunks = []
        for article in self._split_by_articles(text):
            if len(article) <= self.chunk_size:
                chunks.append(article.strip())
            else:
                chunks.extend(self._split_by_clauses(article))
        
        chunks = [c for c in chunks if c.strip()]
        return self._merge_small_chunks(chunks)

    def _split_by_articles(self, text: str) -> list[str]:
        matches = list(self._article_re.finditer(text))
        if not matches:
            return [text] if text.strip() else []
        
        splits, chapter = [], ""
        ch_match = self._chapter_re.search(text[:matches[0].start()])
        if ch_match:
            chapter = ch_match.group(1).strip() + "\n\n"
        
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article = text[m.start():end].strip()
            if chapter:
                article = chapter + article
            
            ch_match = self._chapter_re.search(text[m.start():end])
            if ch_match:
                chapter = ch_match.group(1).strip() + "\n\n"
            splits.append(article)
        
        return splits

    def _split_by_clauses(self, article: str) -> list[str]:
        lines = article.split("\n")
        header, start = "", 0
        
        for i, line in enumerate(lines):
            if self._article_re.match(line.strip()):
                header, start = line.strip() + "\n", i + 1
                break
        
        remaining = "\n".join(lines[start:])
        clause_matches = list(self._clause_re.finditer(remaining))
        
        if not clause_matches:
            return self._fallback_split(article)
        
        chunks = []
        for i, m in enumerate(clause_matches):
            end = clause_matches[i + 1].start() if i + 1 < len(clause_matches) else len(remaining)
            chunk = header + remaining[m.start():end].strip() if header else remaining[m.start():end].strip()
            
            if len(chunk) <= self.chunk_size:
                chunks.append(chunk)
            else:
                chunks.extend(self._fallback_split(chunk))
        
        return chunks

    def _fallback_split(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        
        for s in sentences:
            if len(current) + len(s) + 1 <= self.chunk_size:
                current += (" " if current else "") + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        
        if current:
            chunks.append(current.strip())
        return chunks

    def _merge_small_chunks(self, chunks: list[str], min_size: int = 100) -> list[str]:
        if not chunks:
            return chunks
        
        merged, buffer = [], ""
        for chunk in chunks:
            if len(chunk) < min_size:
                buffer = buffer + "\n\n" + chunk if buffer else chunk
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append(chunk)
        
        if buffer:
            if merged:
                merged[-1] += "\n\n" + buffer
            else:
                merged.append(buffer)
        return merged


def _create_child_documents(
    documents: list[Document],
    parent_splitter: VietnameseLegalTextSplitter,
    child_splitter: VietnameseLegalTextSplitter,
) -> Iterator[Document]:
    """Create child documents with parent content in metadata."""
    for doc in documents:
        for p_idx, parent in enumerate(parent_splitter.split_text(doc.page_content)):
            parent_id = f"{doc.metadata.get('_id', '')}_{p_idx}"
            
            for c_idx, child in enumerate(child_splitter.split_text(parent)):
                yield Document(
                    page_content=child,
                    metadata={
                        **doc.metadata,
                        "parent_id": parent_id,
                        "parent_content": parent,
                        "chunk_index": c_idx,
                    }
                )


def load_legal_corpus() -> list[Document]:
    """Load Vietnamese legal corpus from HuggingFace."""
    try:
        ds = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", "corpus", split="corpus")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return []
    
    print("🔄 Processing dataset...")
    return [
        Document(
            page_content=f"{row['title']}\n{row['text']}",
            metadata={
                "_id": row["_id"],
                "law_id": row["_id"].split("+")[0],
                "title": row["title"],
            }
        ) for row in tqdm(ds, desc="Formatting")
    ]


def ingest_documents(batch_size: int = 100, max_workers: int = 4) -> dict:
    """Parallel ingestion pipeline."""
    print(f"🔌 Connecting to Qdrant: {settings.qdrant_collection}")
    
    vector_store = get_vector_store()
    parent_splitter = VietnameseLegalTextSplitter(settings.parent_chunk_size, settings.parent_chunk_overlap)
    child_splitter = VietnameseLegalTextSplitter(settings.child_chunk_size, settings.child_chunk_overlap)

    raw_docs = load_legal_corpus()
    if not raw_docs:
        return {"status": "failed", "error": "No documents loaded"}

    print("📝 Splitting with Vietnamese legal structure...")
    child_docs = list(_create_child_documents(raw_docs, parent_splitter, child_splitter))
    print(f"📊 Generated {len(child_docs)} chunks from {len(raw_docs)} documents")

    def batch_ingest(batch):
        try:
            vector_store.add_documents(batch, ids=[str(uuid.uuid4()) for _ in batch])
            return len(batch)
        except Exception as e:
            print(f"⚠️ Batch error: {e}")
            return 0

    batches = list(iter(lambda it=iter(child_docs): list(islice(it, batch_size)), []))
    print(f"🚀 Ingesting {len(batches)} batches with {max_workers} workers")
    
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(batch_ingest, b) for b in batches]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Ingesting"):
            total += f.result()

    return {
        "status": "success",
        "total_raw_documents": len(raw_docs),
        "total_child_documents": len(child_docs),
        "ingested": total,
        "collection": settings.qdrant_collection,
    }


if __name__ == "__main__":
    print(f"🎉 {ingest_documents(batch_size=100, max_workers=10)}")
