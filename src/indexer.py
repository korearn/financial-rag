import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "data/documents"))
CHROMA_PATH    = Path(os.getenv("CHROMA_PATH", "data/processed/chroma"))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))


def get_chroma_client():
    """
    Crea el cliente de ChromaDB persistente.
    PersistentClient guarda los embeddings en disco —
    no se pierden al reiniciar el servidor.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection():
    """
    Obtiene o crea la colección de documentos financieros.
    Una colección en ChromaDB es como una tabla — agrupa embeddings relacionados.
    Usamos sentence-transformers localmente para no depender de APIs externas.
    """
    client = get_chroma_client()
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
        # Modelo ligero y rápido — 384 dimensiones
        # Se descarga automáticamente la primera vez
    )
    return client.get_or_create_collection(
        name="financial_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
        # cosine similarity — mide ángulo entre vectores
        # mejor que distancia euclidiana para texto
    )


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extrae todo el texto de un PDF página por página.
    PdfReader maneja PDFs con texto seleccionable —
    para PDFs escaneados necesitarías OCR adicional.
    """
    reader = PdfReader(pdf_path)
    text   = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def chunk_text(text: str, source: str) -> list:
    """
    Divide el texto en fragmentos con overlap.
    
    ¿Por qué overlap? Si una respuesta está en la frontera
    entre dos chunks, el overlap garantiza que alguno
    de los dos tenga el contexto completo.
    
    Cada chunk incluye metadata — el retriever la usa
    para citar la fuente de cada fragmento.
    """
    chunks   = []
    start    = 0
    chunk_id = 0

    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]

        if chunk.strip():
            chunks.append({
                "id":       f"{source}_{chunk_id}",
                "text":     chunk,
                "metadata": {
                    "source":   source,
                    "chunk_id": chunk_id,
                    "start":    start,
                    "end":      end
                }
            })
            chunk_id += 1

        # Avanzar con overlap — retrocedemos CHUNK_OVERLAP caracteres
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def get_document_hash(pdf_path: Path) -> str:
    """
    Calcula el hash MD5 del PDF para detectar si ya fue indexado.
    Evita reindexar el mismo documento si el servidor se reinicia.
    """
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()


def is_indexed(collection, doc_hash: str) -> bool:
    """Verifica si un documento ya está en ChromaDB."""
    results = collection.get(where={"doc_hash": doc_hash})
    return len(results["ids"]) > 0


def index_document(pdf_path: Path) -> dict:
    """
    Indexa un PDF completo en ChromaDB.
    Retorna estadísticas del proceso.
    """
    collection = get_collection()
    doc_hash   = get_document_hash(pdf_path)
    source     = pdf_path.stem  # nombre sin extensión

    if is_indexed(collection, doc_hash):
        console.print(f"[dim]⟳ {pdf_path.name} ya indexado — omitiendo[/dim]")
        return {"source": source, "chunks": 0, "status": "skipped"}

    console.print(f"[bold]Indexando:[/bold] {pdf_path.name}")

    # 1. Extraer texto
    text = extract_text_from_pdf(pdf_path)
    if not text:
        console.print(f"[red]✗[/red] No se pudo extraer texto de {pdf_path.name}")
        return {"source": source, "chunks": 0, "status": "error"}

    console.print(f"[dim]  {len(text):,} caracteres extraídos[/dim]")

    # 2. Dividir en chunks
    chunks = chunk_text(text, source)
    console.print(f"[dim]  {len(chunks)} chunks generados[/dim]")

    # 3. Insertar en ChromaDB con progress bar
    ids       = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        metadatas.append({**chunk["metadata"], "doc_hash": doc_hash})

    # ChromaDB acepta lotes — más eficiente que insertar uno por uno
    batch_size = 100
    for i in track(range(0, len(ids), batch_size),
                   description=f"  Embeddings..."):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    console.print(f"[green]✓[/green] {pdf_path.name} — {len(chunks)} chunks indexados\n")
    return {"source": source, "chunks": len(chunks), "status": "indexed"}


def index_all_documents() -> list:
    """
    Indexa todos los PDFs en la carpeta de documentos.
    Punto de entrada principal del indexer.
    """
    pdf_files = list(DOCUMENTS_PATH.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[yellow]No hay PDFs en {DOCUMENTS_PATH}[/yellow]")
        return []

    console.print(f"\n[bold]Indexando {len(pdf_files)} documento(s)...[/bold]\n")
    results = []

    for pdf_path in pdf_files:
        result = index_document(pdf_path)
        results.append(result)

    total_chunks = sum(r["chunks"] for r in results)
    console.print(f"\n[green]✓[/green] Indexación completa — {total_chunks} chunks totales")
    return results


def get_index_stats() -> dict:
    """Estadísticas de la colección ChromaDB."""
    collection = get_collection()
    count      = collection.count()
    return {
        "total_chunks": count,
        "documents_path": str(DOCUMENTS_PATH),
        "chroma_path": str(CHROMA_PATH)
    }

if __name__ == "__main__":
    index_all_documents()