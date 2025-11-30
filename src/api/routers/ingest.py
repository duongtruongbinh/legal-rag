"""Ingestion API router."""
from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.api.schemas import IngestRequest, IngestResponse
from src.rag.ingestion import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Track ingestion status in memory
ingestion_status: dict = {"running": False, "result": None}


def run_ingestion(collection_name: str | None, batch_size: int) -> None:
    """Background task for document ingestion."""
    global ingestion_status
    try:
        result = ingest_documents(collection_name)
        ingestion_status["result"] = result
    except Exception as e:
        ingestion_status["result"] = {"status": "error", "message": str(e)}
    finally:
        ingestion_status["running"] = False


@router.post("", response_model=IngestResponse)
async def trigger_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Trigger document ingestion process.
    
    Runs ingestion in background to avoid timeout.
    
    Args:
        request: Ingestion parameters.
        background_tasks: FastAPI background tasks.
    
    Returns:
        Ingestion status.
    """
    global ingestion_status
    
    if ingestion_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress",
        )
    
    ingestion_status["running"] = True
    ingestion_status["result"] = None
    
    background_tasks.add_task(
        run_ingestion,
        request.collection_name,
        request.batch_size,
    )
    
    return IngestResponse(
        status="started",
        total_documents=0,
        ingested=0,
        collection=request.collection_name or "default",
    )


@router.get("/status")
async def get_ingestion_status() -> dict:
    """Get current ingestion status."""
    return {
        "running": ingestion_status["running"],
        "result": ingestion_status["result"],
    }

