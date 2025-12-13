"""Ingestion API router."""
from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.api.schemas import IngestRequest, IngestResponse
from src.rag.ingestion import ingest_documents
from src.core.config import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])
ingestion_status: dict = {"running": False, "result": None}


def _run_ingestion(batch_size: int, max_workers: int) -> None:
    """Background ingestion task."""
    global ingestion_status
    try:
        ingestion_status["result"] = ingest_documents(batch_size, max_workers)
    except Exception as e:
        ingestion_status["result"] = {"status": "error", "message": str(e)}
    finally:
        ingestion_status["running"] = False


@router.post("", response_model=IngestResponse)
async def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Trigger background document ingestion."""
    global ingestion_status
    
    if ingestion_status["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")
    
    ingestion_status = {"running": True, "result": None}
    background_tasks.add_task(_run_ingestion, request.batch_size, request.max_workers)
    
    return IngestResponse(
        status="started",
        total_raw_documents=0,
        total_child_documents=0,
        ingested=0,
        collection=request.collection_name or settings.qdrant_collection,
    )


@router.get("/status")
async def get_ingestion_status() -> dict:
    """Get current ingestion status."""
    return ingestion_status
