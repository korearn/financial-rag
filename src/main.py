import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent))

from routes import router
from database import init_db
from indexer import index_all_documents, get_index_stats

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Al arrancar: inicializa BD e indexa documentos nuevos.
    El sistema está listo para consultas en segundos.
    """
    print("🚀 Iniciando Financial RAG Assistant...")
    init_db()
    print("✓ Base de datos lista")
    index_all_documents()
    stats = get_index_stats()
    print(f"✓ {stats['total_chunks']} chunks disponibles")
    yield
    print("👋 Servidor apagado")


app = FastAPI(
    title=os.getenv("APP_NAME", "Financial RAG Assistant"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="""
Asistente de análisis financiero con RAG (Retrieval-Augmented Generation).

## ¿Cómo funciona?

1. Los documentos PDF se indexan en ChromaDB como embeddings vectoriales
2. Cada consulta busca los fragmentos más relevantes por similitud semántica
3. El LLM local genera una respuesta basada SOLO en los documentos

## Endpoints

- **/health** — Estado del servicio y documentos indexados
- **/sources** — Lista de documentos disponibles
- **/index** — Indexar nuevos documentos PDF
- **/query** — Consultar el sistema RAG
- **/history** — Historial de consultas
- **/stats** — Estadísticas de uso
    """,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Financial RAG Assistant",
        "docs":    "http://localhost:8000/docs",
        "version": os.getenv("APP_VERSION", "1.0.0")
    }