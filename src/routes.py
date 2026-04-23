import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from models import QueryRequest, QueryResponse, HealthResponse, IndexRequest
from generator import generate
from indexer import index_all_documents, get_index_stats, get_collection
from retriever import get_available_sources
from database import save_query, get_history, get_stats

load_dotenv()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Estado del servicio y estadísticas del índice."""
    stats   = get_index_stats()
    sources = get_available_sources()
    return HealthResponse(
        status="ok",
        version=os.getenv("APP_VERSION", "1.0.0"),
        total_chunks=stats["total_chunks"],
        documents=sources
    )


@router.get("/sources")
def list_sources():
    """Lista los documentos indexados disponibles."""
    sources = get_available_sources()
    return {"sources": sources, "total": len(sources)}


@router.post("/index")
def index_documents(request: IndexRequest):
    """
    Indexa o reindexta los documentos en data/documents/.
    Útil cuando agregas nuevos PDFs al sistema.
    """
    try:
        if request.force:
            # Limpiar colección existente y reindexar
            collection = get_collection()
            collection.delete(where={"chunk_id": {"$gte": 0}})

        results = index_all_documents()
        total   = sum(r["chunks"] for r in results)
        return {
            "message":      "Indexación completada",
            "documents":    results,
            "total_chunks": total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Consulta el sistema RAG con una pregunta en lenguaje natural.
    Recupera fragmentos relevantes y genera una respuesta con el LLM.
    """
    try:
        result = generate(
            query=request.query,
            top_k=request.top_k,
            source_filter=request.source_filter
        )

        request_id = save_query(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=len(result["chunks"])
        )

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            chunks=[{
                "source":    c["metadata"]["source"],
                "relevance": c["relevance"],
                "text":      c["text"][:200]
            } for c in result["chunks"]],
            request_id=request_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def query_history(limit: int = Query(20, ge=1, le=100)):
    """Historial de consultas realizadas."""
    history = get_history(limit)
    return {"history": history, "total": len(history)}


@router.get("/stats")
def system_stats():
    """Estadísticas de uso del sistema RAG."""
    index_stats = get_index_stats()
    query_stats = get_stats()
    return {
        "index":  index_stats,
        "queries": query_stats
    }