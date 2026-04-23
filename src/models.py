from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Petición de consulta al sistema RAG."""
    query:         str   = Field(..., min_length=5, description="Pregunta sobre los documentos financieros")
    top_k:         int   = Field(3, ge=1, le=10,   description="Número de fragmentos a recuperar")
    source_filter: Optional[str] = Field(None,     description="Filtrar por documento específico")


class ChunkInfo(BaseModel):
    """Información de un fragmento recuperado."""
    source:    str
    relevance: float
    text:      str


class QueryResponse(BaseModel):
    """Respuesta completa del sistema RAG."""
    query:      str
    answer:     str
    sources:    list
    chunks:     list
    request_id: int


class IndexRequest(BaseModel):
    """Request para reindexar documentos."""
    force: bool = Field(False, description="Forzar reindexación aunque ya estén indexados")


class HealthResponse(BaseModel):
    status:       str
    version:      str
    total_chunks: int
    documents:    list