# src/pipeline.py
"""
End-to-end pipeline:
USER QUERY → ML → RAG → LLM → final text response
"""

from typing import Dict

# ---- Import ML recommender (your API or class-based function)
from .ml.service import get_ml_candidates_for_user

# or if using the FastAPI service, you would switch this to an HTTP request

# ---- Import RAG function (Haaris's ask())
from .rag.rag_service import ask  # you will create this wrapper file from notebook

# ---- Import LLM advisor
from .llm.advisor import generate_final_answer


def run_pipeline(user_id: str, user_query: str) -> Dict:
    """
    Unified pipeline:
    1. Get ML recommendations
    2. Get RAG results
    3. Get final LLM answer
    Returns dict containing all intermediate + final results.
    """

    # 1) ML
    print("Begin ML")
    ml_candidates = get_ml_candidates_for_user(user_id=user_id, k=5)
    print("ML result ", ml_candidates)

    # 2) RAG
    print("Begin RAG")
    rag_result = ask(user_query, k=5)
    print("RAG result ", rag_result)

    # 3) LLM Advisor
    print("Begin LLM Advisor")
    final_answer = generate_final_answer(
        user_query=user_query,
        ml_candidates=ml_candidates,
        rag_result=rag_result,
    )

    return {
        "user_query": user_query,
        "ml_candidates": ml_candidates,
        "rag_result": rag_result,
        "final_answer": final_answer,
    }


# Example usage at the bottom for testing
if __name__ == "__main__":
    # query = "I want a Dell laptop for programming under 150k with good battery."
    query = "I want some good earphones that would last."
    user_id = (
        "AG3D6O4STAQKAY2UVGEUV46KN35Q"  # must exist in your ML matrix, or use a cold-start strategy
    )

    result = run_pipeline(user_id, query)

    print("\n=== FINAL RESPONSE ===\n")
    print(result["final_answer"])
