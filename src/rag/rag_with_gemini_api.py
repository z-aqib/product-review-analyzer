# ================================
# ONE-CLICK RAG ON AMAZON PRODUCTS
# FAISS + BGE embeddings + GEMINI LLM
# ================================

# Step 1: Install required packages
# !pip install -q torch transformers sentence-transformers faiss-cpu pandas tqdm google-generativeai

# Step 2: Imports
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm.auto import tqdm
import google.generativeai as genai

# Step 3: Load dataset
df = pd.read_csv("/kaggle/input/mlops-amazon/amazon.csv")  # Adjust path if needed

print(f"Loaded {len(df)} products")

# Step 4: Create documents for retrieval
documents = []
metadatas = []

for _, row in df.iterrows():
    text = f"""
Product: {row['product_name']}
Category: {row['category']}
Price: {row['discounted_price']} (was {row['actual_price']}, {row['discount_percentage']} off)
Rating: {row['rating']} ⭐ ({row['rating_count']} reviews)
Description: {row['about_product']}
Reviews: {row['review_content'][:1000]}...
    """.strip()

    documents.append(text)
    metadatas.append(
        {
            "product_id": row["product_id"],
            "name": row["product_name"],
            "price": row["discounted_price"],
            "rating": row["rating"],
        }
    )

print(f"Created {len(documents)} searchable documents")

# Step 5: Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")  # Fast + accurate

# Step 6: Create embeddings + FAISS index
print("Creating embeddings and FAISS index...")
dimension = 384

index = faiss.IndexFlatIP(dimension)

batch_size = 32
embeddings = []

for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
    batch = documents[i : i + batch_size]
    batch_emb = embedder.encode(batch, normalize_embeddings=True)
    embeddings.append(batch_emb)
    index.add(batch_emb)

embeddings = np.vstack(embeddings)
print(f"Indexed {index.ntotal} products")


# Step 7: Load Gemini LLM
genai.configure(api_key="YOUR_API_KEY")  # <--- replace with your key
model = genai.GenerativeModel("gemini-2.5-flash")

print("RAG System Ready with Gemini!")


# ================================
# Step 8: Query function (GEMINI)
# ================================
def ask(question: str, k: int = 3):
    # 1) Retrieve
    q_emb = embedder.encode([question], normalize_embeddings=True)
    array_D, array_I = index.search(q_emb, k)

    products = []
    for score, doc_idx in zip(array_D[0], array_I[0]):
        meta = metadatas[doc_idx]
        products.append(
            {
                "product_id": meta["product_id"],
                "name": meta["name"],
                "price": float(meta["price"]) if meta["price"] == meta["price"] else None,
                "rating": float(meta["rating"]) if meta["rating"] == meta["rating"] else None,
                "retrieval_score": float(score),
                "document": documents[doc_idx],
            }
        )

    # 2) Build context for Gemini (just like before)
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

    response = model.generate_content(prompt)
    rag_answer = response.text.strip()

    # 3) Return structured data instead of print
    return {
        "question": question,
        "products": products,
        "rag_answer": rag_answer,
    }

    # print("\nAnswer:\n")
    # print(textwrap.fill(answer, width=90))
    # print("\n" + "-"*80)
