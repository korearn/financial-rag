## Instalación

```bash
git clone https://github.com/korearn/financial-rag
cd financial-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Crea `.env` en la raíz:

```env
APP_NAME=Financial RAG Assistant
APP_VERSION=1.0.0
LMSTUDIO_URL=http://localhost:1234/v1/chat/completions
CHROMA_PATH=data/processed/chroma
DOCUMENTS_PATH=data/documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3
```

> En WSL2 reemplaza `localhost` con la IP de Windows:
> `cat /etc/resolv.conf | grep nameserver | awk '{print $2}'`

## Uso

### 1. Agrega tus PDFs

```bash
cp tus_reportes.pdf data/documents/
```

### 2. Inicia el servidor

```bash
uvicorn src.main:app --reload
```

El servidor indexa automáticamente los PDFs al arrancar.
Documentación interactiva en `http://localhost:8000/docs`

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/v1/health` | Estado y documentos indexados |
| GET | `/api/v1/sources` | Lista de documentos disponibles |
| POST | `/api/v1/index` | Reindexar documentos |
| POST | `/api/v1/query` | Consultar el sistema RAG |
| GET | `/api/v1/history` | Historial de consultas |
| GET | `/api/v1/stats` | Estadísticas de uso |

## Ejemplo de consulta

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Tesla total revenue in Q1 2025?",
    "top_k": 3
  }'
```

```json
{
  "query": "What was Tesla total revenue in Q1 2025?",
  "answer": "Tesla reported total revenues of $19.3B in Q1 2025...",
  "sources": ["TSLA-Q1-2025-Update"],
  "chunks": [...],
  "request_id": 1
}
```

## Filtrar por documento

```json
{
  "query": "¿Cuáles fueron los ingresos del trimestre?",
  "top_k": 3,
  "source_filter": "Esp+PR+4T25+vf"
}
```

## Documentos probados

- **TSLA Q1 2025 Update** — Reporte trimestral de Tesla (inglés)
- **FEMSA 4T25** — Reporte de resultados cuarto trimestre 2025 (español)

## Notas técnicas

- El modelo `all-MiniLM-L6-v2` se descarga automáticamente la primera vez (~90MB)
- ChromaDB es persistente — los embeddings se guardan en disco y no se recalculan
- El hash MD5 de cada PDF evita reindexación innecesaria al reiniciar
- `CHUNK_OVERLAP=50` garantiza que el contexto no se pierda entre fragmentos
- En WSL2 `localhost` no apunta a Windows — usar IP del nameserver