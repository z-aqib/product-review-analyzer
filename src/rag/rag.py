# src/rag/rag.py

import os
from typing import Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

from .ingest import (
    RAG_CONFIG,
    parse_price,
    embedder,
    index,
    documents,
    metadatas,
)

# ----------------------------------------
# GEMINI CONFIG
# ----------------------------------------
# Load .env from project root
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

genai.configure(api_key=api_key)
RAG_MODEL = genai.GenerativeModel("gemini-2.5-flash")


# ================================
# PUBLIC API: ask(question, k)
# ================================
def ask(question: str, k: int = 3) -> Dict[str, Any]:
    """
    Main RAG query function.

    Uses global:
    - embedder
    - index
    - documents
    - metadatas
    """

    if embedder is None or index is None:
        raise RuntimeError("RAG system not initialized properly.")

    # 1) Encode query + retrieve
    q_emb = embedder.encode([question], normalize_embeddings=RAG_CONFIG["normalize"])
    array_D, array_I = index.search(q_emb, k)

    products = []
    for score, doc_idx in zip(array_D[0], array_I[0]):
        meta = metadatas[doc_idx]
        products.append(
            {
                "product_id": meta["product_id"],
                "name": meta["name"],
                "price": parse_price(meta.get("price")),
                "rating": (float(meta["rating"]) if meta["rating"] == meta["rating"] else None),
                "retrieval_score": float(score),
                "document": documents[doc_idx],
            }
        )

    # 2) Build context for Gemini
    context = "\n\n".join(p["document"] for p in products)

    prompt = f"""
You are an expert Amazon shopping assistant.

Use ONLY the information provided in the context below.
Do NOT hallucinate extra product features not present in the context.

Context:
{context}

User Question: {question}

Give a clear, helpful answer based ONLY on the above data.
"""

    response = RAG_MODEL.generate_content(prompt)
    rag_answer = response.text.strip()

    # 3) Return structured data
    return {
        "question": question,
        "products": products,
        "rag_answer": rag_answer,
    }
