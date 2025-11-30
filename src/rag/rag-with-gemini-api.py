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
import textwrap
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
    # Retrieve
    q_emb = embedder.encode([question], normalize_embeddings=True)
    array_D, array_I = index.search(q_emb, k)

    context = "\n\n".join([documents[i] for i in array_I[0]])
    scores = array_D[0]

    print(f"\nTop {k} Retrieved Products:\n")
    for idx, (score, doc_idx) in enumerate(zip(scores, array_I[0])):
        meta = metadatas[doc_idx]
        print(
            f"{idx+1}. {meta['name'][:80]}... ({meta['rating']}⭐, {meta['price']}) "
            f"[Score: {score:.3f}]"
        )

    print("\n" + "=" * 60)
    print("GENERATING ANSWER WITH GEMINI...")
    print("=" * 60)

    # LLM prompt for Gemini
    prompt = f"""
You are an expert Amazon shopping assistant.

Use ONLY the information provided in the context below.
Do NOT hallucinate extra product features not present in the context.

Context:
{context}

User Question: {question}

Give a clear, helpful answer based ONLY on the above data.
"""

    # Generate answer
    response = model.generate_content(prompt)
    answer = response.text.strip()

    print("\nAnswer:\n")
    print(textwrap.fill(answer, width=90))
    print("\n" + "-" * 80)


# ================================
# NOW ASK QUESTIONS USING GEMINI
# ================================

ask("Recommend a good fast charging USB-C cable under 300 rupees")
ask("Which cable has the highest rating and supports 60W charging?")
ask("What is the best iPhone lightning cable in the list?")
ask("Show me durable braided cables from boAt or Ambrane")
ask("Which product has the most reviews and good rating?")
