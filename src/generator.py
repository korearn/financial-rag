import requests
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from retriever import retrieve, format_context

load_dotenv()

LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://169.254.83.107:1234/v1/chat/completions")


def build_rag_prompt(query: str, context: str) -> str:
    """
    Construye el prompt RAG con el contexto de los documentos.
    La instrucción clave es 'responde SOLO basándote en el contexto' —
    esto evita que el LLM mezcle su conocimiento general con tus documentos.
    """
    return f"""Eres un analista financiero experto. Tu tarea es responder preguntas
basándote EXCLUSIVAMENTE en los fragmentos de documentos financieros proporcionados.

Si la información no está en los fragmentos, di claramente que no encontraste
esa información en los documentos disponibles. No inventes datos financieros.

FRAGMENTOS DE DOCUMENTOS:
{context}

PREGUNTA DEL USUARIO:
{query}

Responde en el mismo idioma que la pregunta. Si los documentos están en inglés
pero la pregunta es en español, responde en español citando los datos en inglés
cuando sea necesario. Sé preciso con los números y cita la fuente."""


def generate(query: str, top_k: int = 3,
             source_filter: str = None) -> dict:
    """
    Función principal del RAG — recupera contexto y genera respuesta.
    Retorna tanto la respuesta como los chunks usados para transparencia.
    """
    # 1. Recuperar fragmentos relevantes
    chunks  = retrieve(query, top_k=top_k, source_filter=source_filter)
    context = format_context(chunks)

    # 2. Construir prompt
    prompt = build_rag_prompt(query, context)

    # 3. Llamar al LLM
    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "Eres un analista financiero experto. Respondes basándote únicamente en los documentos proporcionados."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens":  800
    }

    try:
        response = requests.post(
            LMSTUDIO_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        answer = (
            f"LMStudio no está disponible. "
            f"Fragmentos relevantes encontrados:\n\n{context[:500]}..."
        )
    except Exception as e:
        answer = f"Error generando respuesta: {e}"

    return {
        "query":    query,
        "answer":   answer,
        "chunks":   chunks,
        "context":  context,
        "sources":  list(set(c["metadata"]["source"] for c in chunks))
    }