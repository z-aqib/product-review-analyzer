# src/rag/langchain_rag.py
# ============================================================
# LangChain-based RAG for Amazon Products (Milestone Bonus)
# ============================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List

# ---------------------------------------------
# LANGCHAIN IMPORTS
# ---------------------------------------------
from pydantic import Field

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# ---------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------
from .ingest import (
    RAG_CONFIG,
    _load_dataset,
    _build_documents_from_df,
    RAG_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------
# Globals
# ---------------------------------------------
_lc_vectorstore: FAISS | None = None
_lc_documents: List[Document] = []
_retriever: "RatingAwareRetriever" | None = None


# ============================================================
# CUSTOM RETRIEVER (Pydantic v2 + LangChain 0.2 compliant)
# ============================================================
class RatingAwareRetriever(BaseRetriever):
    """
    Custom retriever that:
      1. Runs FAISS similarity_search()
      2. Filters/reranks by rating >= min_rating
    """

    # Required Pydantic fields
    vectorstore: FAISS = Field(...)
    min_rating: float = Field(default=3.5)
    k: int = Field(default=5)

    # Allow additional attributes
    model_config = {"extra": "allow"}

    def __init__(self, vectorstore: FAISS, min_rating: float = 3.5, k: int = 5, **kwargs):
        super().__init__(
            vectorstore=vectorstore,
            min_rating=float(min_rating),
            k=int(k),
            **kwargs,
        )

    # ---------------------------------------------------------
    # THE NEW OFFICIAL LANGCHAIN ENTRYPOINT (0.2+)
    # ---------------------------------------------------------
    def invoke(self, query: str, *, config=None):
        """Main sync entrypoint for LangChain retrievers."""
        return self._get_relevant_documents(query)

    async def ainvoke(self, query: str, *, config=None):
        """Async version."""
        return await self._aget_relevant_documents(query)

    # ---------------------------------------------------------
    # INTERNAL RETRIEVAL LOGIC
    # ---------------------------------------------------------
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:

        # 1. similarity search
        candidates = self.vectorstore.similarity_search(query, k=self.k * 3)

        # 2. filter by rating metadata
        filtered = []
        for d in candidates:
            meta = d.metadata or {}
            rating_raw = meta.get("rating")

            try:
                rating = float(rating_raw)
            except (TypeError, ValueError):
                rating = None

            if rating is None or rating >= self.min_rating:
                filtered.append(d)

        # return at least k
        if len(filtered) >= self.k:
            return filtered[: self.k]

        return candidates[: self.k]

    # Async fallback
    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query)


# ============================================================
# INITIALIZATION
# ============================================================
def _initialize_langchain_rag() -> None:
    """
    Loads dataset → builds LangChain Documents → embeddings → FAISS → custom retriever
    """
    global _lc_vectorstore, _retriever, _lc_documents

    if _lc_vectorstore is not None and _retriever is not None:
        return

    logger.info("[LangChain RAG] Loading dataset...")
    df = _load_dataset()

    # Build same texts & metadata as classic RAG
    docs_str, metas = _build_documents_from_df(df)

    _lc_documents = [
        Document(page_content=text, metadata=meta) for text, meta in zip(docs_str, metas)
    ]

    # Build embeddings
    model_name = RAG_CONFIG.get("model_name", "BAAI/bge-small-en-v1.5")
    normalize = bool(RAG_CONFIG.get("normalize", True))

    logger.info("[LangChain RAG] Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": normalize},
    )

    # Build FAISS
    logger.info("[LangChain RAG] Building FAISS index...")
    _lc_vectorstore = FAISS.from_documents(_lc_documents, embeddings)

    # Custom Retriever
    _retriever = RatingAwareRetriever(
        vectorstore=_lc_vectorstore,
        min_rating=3.8,
        k=5,
    )

    logger.info("[LangChain RAG] Initialization complete.")


# ============================================================
# PUBLIC API
# ============================================================
def ask_langchain(question: str, k: int = 5) -> Dict[str, Any]:
    """
    Main entrypoint for LangChain RAG.
    """
    if _retriever is None:
        _initialize_langchain_rag()

    # override top-k
    _retriever.k = int(k)

    # NEW: use invoke() (NOT get_relevant_documents)
    docs = _retriever.invoke(question)

    # convert docs → product dicts
    products = []
    for d in docs:
        meta = d.metadata or {}
        products.append(
            {
                "product_id": meta.get("product_id"),
                "name": meta.get("name"),
                "price": meta.get("price"),
                "rating": meta.get("rating"),
                "document": d.page_content,
            }
        )

    # Build context string
    context = "\n\n".join(p["document"] for p in products)

    prompt = f"""
You are an expert Amazon shopping assistant.
Use ONLY the product information below.

Context:
{context}

User Question: {question}

Give a clear, honest, helpful answer.
""".strip()

    # Call Gemini (same as your original RAG)
    response = RAG_MODEL.generate_content(prompt)
    answer_text = getattr(response, "text", str(response)).strip()

    return {
        "question": question,
        "products": products,
        "rag_answer": answer_text,
        "engine": "langchain_rating_aware",
    }


# initialize on import
_initialize_langchain_rag()

# ============================================================
# CLI Test
# ============================================================
if __name__ == "__main__":
    import json

    test_q = "Recommend good wireless headphones under 5000 with strong battery life."
    out = ask_langchain(test_q, k=5)
    print(json.dumps(out, indent=2, ensure_ascii=False))
