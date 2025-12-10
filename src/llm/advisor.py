# src/llm/advisor.py
import os
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
import random
import time

# -------------------------------------
# GEMINI CONFIG
# -------------------------------------
# Load .env from project root
load_dotenv()  # this reads .env and populates os.environ

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

genai.configure(api_key=api_key)

# Model for FINAL RESPONSE (advisor)
ADVISOR_MODEL = genai.GenerativeModel("gemini-2.5-flash-lite")


# -------------------------------------
# UTILITY: Merge ML & RAG product info
# -------------------------------------
def merge_ml_and_rag(
    ml_candidates: List[Dict],
    rag_products: List[Dict],
) -> List[Dict]:
    """
    Merge product info from ML and RAG using product_id.
    Produces unified list of product dictionaries.
    """
    rag_map = {str(p["product_id"]): p for p in rag_products}

    merged = []
    for c in ml_candidates:
        pid = str(c["product_id"])
        if pid not in rag_map:
            continue

        r = rag_map[pid]
        merged.append(
            {
                "product_id": pid,
                "product_name": r.get("name") or c.get("product_name") or pid,
                "ml_score": float(c.get("score", 0.0)),
                "price": r.get("price"),
                "rating": r.get("rating"),
                "retrieval_score": float(r.get("retrieval_score", 0.0)),
                "snippet": r.get("document", "")[:800],  # Limit size
            }
        )
    return merged


# -------------------------------------
# BUILD PROMPT
# -------------------------------------

# -------------------------------------
# A/B TESTING PROMPT VARIANTS
# -------------------------------------

PROMPT_VARIANT_A = """
You are a friendly product advisor AI. Provide a detailed and warm explanation.

User Query:
{user_query}

Candidate Products (ML + RAG):
{products_block}

RAG Review Summary:
{rag_answer}

Give a helpful answer with clear reasoning and a friendly tone.
"""

PROMPT_VARIANT_B = """
You are a structured and concise product advisor AI.

User Query:
{user_query}

Top Product Data:
{products_block}

RAG Summary:
{rag_answer}

Provide a short, structured recommendation with bullet points and a final verdict.
"""


# def build_advisor_prompt(
#     user_query: str,
#     merged_products: List[Dict],
#     rag_answer: str,
# ) -> str:
#     """
#     Create the prompt for Gemini that acts as the FINAL ADVISOR.
#     """

#     product_lines = []
#     for p in merged_products:
#         line = f"- {p['product_name']} (ID: {p['product_id']})"
#         details = []

#         if p.get("price") is not None:
#             details.append(f"price ≈ {p['price']}")
#         if p.get("rating") is not None:
#             details.append(f"rating {p['rating']}⭐")

#         details.append(f"ML_score {p['ml_score']:.3f}")
#         details.append(f"retrieval_score {p['retrieval_score']:.3f}")

#         line += " [" + ", ".join(details) + "]"
#         product_lines.append(line)

#     products_block = "\n".join(product_lines)

#     prompt = f"""
# You are a friendly product advisor AI.

# You will receive:
# 1. The user's question.
# 2. Candidate products from an ML recommender.
# 3. Summaries of real Amazon product reviews (RAG output).

# Your job:
# - Understand user needs (budget, brand, purpose, concerns).
# - Compare products using BOTH ML ranking + review evidence.
# - Recommend the best 1–3 products.
# - Explain clearly, in a friendly tone.
# - Do NOT hallucinate facts outside the provided summaries.

# User Query:
# {user_query}

# Candidate Products (ML + RAG):
# {products_block}

# RAG Review Summary:
# {rag_answer}

# Now give a helpful final answer to the user (2–4 short paragraphs). You may give a shorter or longer answer where needed. Dont give options for your responses, pick the best response you think is applicable and return that.
# """.strip()

#     return textwrap.dedent(prompt)


def build_advisor_prompt(
    user_query: str,
    merged_products: List[Dict],
    rag_answer: str,
    prompt_variant: str = "A",  # <── NEW
) -> str:
    """
    Build prompt based on selected A/B prompt variant.
    """

    product_lines = []
    for p in merged_products:
        line = f"- {p['product_name']} (ID: {p['product_id']})"
        details = []

        if p.get("price") is not None:
            details.append(f"price ≈ {p['price']}")
        if p.get("rating") is not None:
            details.append(f"rating {p['rating']}⭐")

        details.append(f"ML_score {p['ml_score']:.3f}")
        details.append(f"retrieval_score {p['retrieval_score']:.3f}")

        line += " [" + ", ".join(details) + "]"
        product_lines.append(line)

    products_block = "\n".join(product_lines)

    # Pick A or B template
    if prompt_variant == "A":
        template = PROMPT_VARIANT_A
    else:
        template = PROMPT_VARIANT_B

    # Fill placeholders
    return template.format(
        user_query=user_query,
        products_block=products_block,
        rag_answer=rag_answer,
    )


# -------------------------------------
# MAIN FUNCTION — CALL GEMINI
# -------------------------------------
# def generate_final_answer(
#     user_query: str,
#     ml_candidates: List[Dict],
#     rag_result: Dict,
# ) -> str:
#     """
#     Entry point:
#     - merges product info
#     - builds prompt
#     - calls Gemini to make final response
#     """

#     merged_products = merge_ml_and_rag(
#         ml_candidates=ml_candidates,
#         rag_products=rag_result.get("products", []),
#     )

#     # If nothing overlaps, fallback to rewriting the RAG answer
#     if not merged_products:
#         fallback_prompt = f"""
# Rewrite the following RAG answer in a friendly tone for the user.

# User Query:
# {user_query}

# RAG Answer:
# {rag_result.get("rag_answer", "")}
# """
#         resp = ADVISOR_MODEL.generate_content(fallback_prompt)
#         return resp.text.strip()

#     # Build advisor prompt
#     final_prompt = build_advisor_prompt(
#         user_query=user_query,
#         merged_products=merged_products,
#         rag_answer=rag_result.get("rag_answer", ""),
#     )

#     # Call Gemini
#     response = ADVISOR_MODEL.generate_content(final_prompt)
#     return response.text.strip()


def generate_final_answer(
    user_query: str,
    ml_candidates: List[Dict],
    rag_result: Dict,
) -> Dict:
    """
    Generates response + logs A/B testing variant used.
    Returns { "variant": "A/B", "answer": "...", "response_time": ... }
    """

    merged_products = merge_ml_and_rag(
        ml_candidates=ml_candidates,
        rag_products=rag_result.get("products", []),
    )

    # Fallback if no products match
    if not merged_products:
        fallback_prompt = f"""
Rewrite the following RAG answer in a friendly tone for the user.

User Query:
{user_query}

RAG Answer:
{rag_result.get("rag_answer", "")}
"""
        resp = ADVISOR_MODEL.generate_content(fallback_prompt)
        return {
            "variant": "FALLBACK",
            "answer": resp.text.strip(),
            "response_time": 0.0,
        }

    # RANDOMLY CHOOSE A or B
    prompt_variant = "A" if random.random() < 0.5 else "B"

    # Build prompt
    final_prompt = build_advisor_prompt(
        user_query=user_query,
        merged_products=merged_products,
        rag_answer=rag_result.get("rag_answer", ""),
        prompt_variant=prompt_variant,
    )

    # Measure response time for metrics
    start = time.time()
    response = ADVISOR_MODEL.generate_content(final_prompt)
    end = time.time()

    return {
        "variant": prompt_variant,
        "answer": response.text.strip(),
        "response_time": round(end - start, 3),
    }
