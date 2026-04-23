import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from indexer import get_collection

load_dotenv()
console = Console()

TOP_K = int(os.getenv("TOP_K_RESULTS", 3))

def retrieve(query: str, top_k: int = TOP_K, source_filter: str = None) -> list:
    """
    Busca los fragmentos más relevantes para una consulta.

    ChromaDB compara el embedding de la query contra todos los
    embeddings almacenados y retorna los TOP_K más similares.

    source_filter permite limitar la búsqueda a un documento específico —
    útil cuando el usuario quiere consultar solo el reporte de Tesla
    o solo el reporte mexicano.
    """
    collection = get_collection()

    # Construir filtro opcional por fuente
    where = None
    if source_filter:
        where = {"source": source_filter}

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # Formatear resultados en estructura clara
    # distances en ChromaDB con cosine: 0 = idéntico, 2 = opuesto
    # Lo convertimos a score de relevancia 0-1
    chunks = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            distance  = results["distances"][0][i]
            relevance = round(1 - (distance / 2), 4)

            chunks.append({
                "id":        doc_id,
                "text":      results["documents"][0][i],
                "metadata":  results["metadatas"][0][i],
                "relevance": relevance
            })

    return chunks


def format_context(chunks: list) -> str:
    """
    Formatea los chunks recuperados en un bloque de contexto
    que se incluirá en el prompt del LLM.
    Cada fragmento incluye su fuente para que el LLM pueda citar.
    """
    if not chunks:
        return "No se encontró información relevante en los documentos."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "desconocido")
        relevance = chunk["relevance"]
        context_parts.append(
            f"[Fragmento {i} — Fuente: {source} | Relevancia: {relevance:.2f}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_available_sources() -> list:
    """Lista los documentos disponibles en ChromaDB."""
    collection = get_collection()
    results    = collection.get(include=["metadatas"])

    sources = set()
    for metadata in results["metadatas"]:
        if "source" in metadata:
            sources.add(metadata["source"])

    return sorted(list(sources))