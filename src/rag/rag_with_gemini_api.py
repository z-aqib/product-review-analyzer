# ================================
# ONE-CLICK RAG ON AMAZON PRODUCTS
# FAISS + BGE embeddings + GEMINI LLM
# WITH CACHING
# ================================

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm.auto import tqdm
import google.generativeai as genai
import joblib  # for saving/loading python objects
from dotenv import load_dotenv

# ----------------------------------------
# CONFIG
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # project root (../.. from src/rag/)
DATA_PATH = BASE_DIR / "data" / "raw" / "amazon.csv"
EMBED_ROOT = BASE_DIR / "data" / "embeddings"

RAG_CONFIG: Dict[str, Any] = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "normalize": True,
    "chunking": "none",  # if you later change to chunking, change this
}

# Load .env from project root
load_dotenv()  # this reads .env and populates os.environ

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

# Gemini config
genai.configure(api_key=api_key)
RAG_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# Globals used by ask()
df: pd.DataFrame | None = None
documents: List[str] = []
metadatas: List[Dict[str, Any]] = []
embedder: SentenceTransformer | None = None
index: faiss.IndexFlatIP | None = None


# ----------------------------------------
# HELPERS
# ----------------------------------------
def _config_hash() -> str:
    """Create a stable hash for the current RAG_CONFIG."""
    cfg_str = json.dumps(RAG_CONFIG, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:8]


def _get_embed_dir() -> Path:
    """Folder for embeddings & index for this config."""
    cfg_hash = _config_hash()
    return EMBED_ROOT / cfg_hash


def _load_dataset() -> pd.DataFrame:
    df_ = pd.read_csv(DATA_PATH)
    # Ensure IDs are strings if needed
    if "product_id" in df_.columns:
        df_["product_id"] = df_["product_id"].astype(str)
    return df_


def _build_documents_from_df(df_: pd.DataFrame) -> tuple[list[str], list[dict]]:
    docs = []
    metas = []

    for _, row in df_.iterrows():
        text = f"""
Product: {row['product_name']}
Category: {row['category']}
Price: {row['discounted_price']} (was {row['actual_price']}, {row['discount_percentage']} off)
Rating: {row['rating']} ⭐ ({row['rating_count']} reviews)
Description: {row['about_product']}
Reviews: {str(row['review_content'])[:1000]}...
        """.strip()

        docs.append(text)
        metas.append(
            {
                "product_id": row["product_id"],
                "name": row["product_name"],
                "price": parse_price(row["discounted_price"]),
                "rating": row["rating"],
            }
        )

    return docs, metas


def _save_index_and_metadata(
    embed_dir: Path,
    index_obj: faiss.IndexFlatIP,
    docs: List[str],
    metas: List[Dict[str, Any]],
    df_len: int,
):
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index_obj, str(embed_dir / "index.faiss"))

    # Save docs + metadatas
    joblib.dump(docs, embed_dir / "documents.pkl")
    joblib.dump(metas, embed_dir / "metadatas.pkl")

    # Save metadata JSON
    data_path_rel = os.path.relpath(DATA_PATH, BASE_DIR)

    meta = {
        "config": RAG_CONFIG,
        "created_at": datetime.utcnow().isoformat(),
        "num_docs_indexed": len(docs),
        "num_rows_dataset": df_len,
        "data_path": data_path_rel,  # e.g. "data/raw/amazon.csv"
        "faiss_index_type": "IndexFlatIP",
        "embedding_dim": index_obj.d,
    }
    with open(embed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_index_and_metadata(embed_dir: Path):
    idx = faiss.read_index(str(embed_dir / "index.faiss"))
    docs = joblib.load(embed_dir / "documents.pkl")
    metas = joblib.load(embed_dir / "metadatas.pkl")
    with open(embed_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, docs, metas, meta


def parse_price(x):
    """
    Cleans price values like '₹1,399', '$249', '£50', '1,299', '--', ''
    Returns float or None.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    x = str(x).strip()
    if x == "" or x == "--" or x.lower() == "nan":
        return None

    # Remove first character if it's a currency symbol
    # (₹, $, £, €, etc.)
    if not x[0].isdigit():
        x = x[1:]

    # Remove commas
    x = x.replace(",", "")

    try:
        return float(x)
    except Exception as e:
        print(f"Warning: could not parse price '{x}': {e}")
        return None


def _initialize_rag():
    """
    Called on module import to:
    - Load dataset
    - Build documents/metadatas
    - Load or build FAISS index + embeddings with caching
    """
    global df, documents, metadatas, embedder, index

    print("[RAG] Loading dataset...")
    df = _load_dataset()
    n_rows = len(df)
    print(f"[RAG] Loaded {n_rows} rows from {DATA_PATH}")

    # Build documents/metadatas fresh each time (cheap)
    docs, metas = _build_documents_from_df(df)

    # Prepare embedding model
    print("[RAG] Loading embedding model:", RAG_CONFIG["model_name"])
    embedder = SentenceTransformer(RAG_CONFIG["model_name"])
    emb_dim = embedder.get_sentence_embedding_dimension()
    normalize = RAG_CONFIG.get("normalize", True)

    embed_dir = _get_embed_dir()
    meta_path = embed_dir / "metadata.json"

    if embed_dir.exists() and meta_path.exists():
        # Try to load existing index
        print(f"[RAG] Found existing embeddings in {embed_dir}, loading...")
        idx, docs_cached, metas_cached, meta = _load_index_and_metadata(embed_dir)

        # Basic consistency check: config & dim
        if meta["config"] == RAG_CONFIG and meta["embedding_dim"] == emb_dim:
            # Check if we need to add new rows
            n_indexed = meta.get("num_docs_indexed", 0)
            if n_indexed == len(docs):
                print("[RAG] All docs already indexed. Reusing cached index.")
                documents = docs_cached
                metadatas = metas_cached
                index = idx
                return
            elif n_indexed < len(docs):
                print(
                    f"[RAG] Dataset has grown ({len(docs)} docs, "
                    f"{n_indexed} indexed). Embedding only new docs..."
                )

                # Use cached docs/metas for first part, and new ones for tail
                documents = docs_cached
                metadatas = metas_cached

                # New docs are from n_indexed onwards
                new_docs = docs[n_indexed:]
                new_metas = metas[n_indexed:]

                batch_size = 32
                all_new_embs = []
                for i in tqdm(
                    range(0, len(new_docs), batch_size),
                    desc="Embedding new docs",
                ):
                    batch = new_docs[i : i + batch_size]
                    batch_emb = embedder.encode(
                        batch,
                        normalize_embeddings=normalize,
                        show_progress_bar=False,
                    )
                    all_new_embs.append(batch_emb)
                    idx.add(batch_emb)

                # Extend docs/metas and update metadata
                documents.extend(new_docs)
                metadatas.extend(new_metas)

                _save_index_and_metadata(embed_dir, idx, documents, metadatas, n_rows)
                index = idx
                print("[RAG] Incremental embedding complete.")
                return
            else:
                # This means cached index has MORE docs than we built from df
                print(
                    "[RAG] Warning: cached index has more docs than current dataset. "
                    "Rebuilding index from scratch."
                )

        else:
            print(
                "[RAG] Config or embedding dim changed " "(or metadata mismatch). Rebuilding index."
            )

    # If we reach here: either no cache, or we decided to rebuild
    print("[RAG] Creating new embeddings + FAISS index from scratch...")
    index_flat = faiss.IndexFlatIP(emb_dim)

    batch_size = 32
    normalize = RAG_CONFIG.get("normalize", True)

    all_embs = []
    for i in tqdm(
        range(0, len(docs), batch_size),
        desc="Embedding all docs",
    ):
        batch = docs[i : i + batch_size]
        batch_emb = embedder.encode(
            batch,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        all_embs.append(batch_emb)
        index_flat.add(batch_emb)

    _ = np.vstack(all_embs)  # not strictly needed later

    documents = docs
    metadatas = metas
    index = index_flat

    _save_index_and_metadata(embed_dir, index_flat, documents, metadatas, n_rows)
    print("[RAG] Index built and cached at", embed_dir)


# Call initializer when module is imported
_initialize_rag()


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
                "rating": float(meta["rating"]) if meta["rating"] == meta["rating"] else None,
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
