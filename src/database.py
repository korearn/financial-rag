import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "rag_history.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Historial de consultas RAG.
    Guarda cada pregunta, respuesta y fuentes usadas —
    útil para auditoría y para mejorar el sistema.
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query        TEXT NOT NULL,
            answer       TEXT NOT NULL,
            sources      TEXT,
            chunks_used  INTEGER,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_query(query: str, answer: str,
               sources: list, chunks_used: int) -> int:
    """Guarda una consulta y su respuesta en el historial."""
    conn = get_connection()
    cur  = conn.execute("""
        INSERT INTO query_history (query, answer, sources, chunks_used)
        VALUES (?, ?, ?, ?)
    """, (query, answer, ",".join(sources), chunks_used))
    conn.commit()
    request_id = cur.lastrowid or 0
    conn.close()
    return request_id


def get_history(limit: int = 20) -> list:
    """Retorna el historial de consultas."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """Estadísticas de uso del sistema RAG."""
    conn = get_connection()
    row  = conn.execute("""
        SELECT
            COUNT(*)         AS total_queries,
            AVG(chunks_used) AS avg_chunks_used
        FROM query_history
    """).fetchone()
    conn.close()
    return dict(row) if row else {}