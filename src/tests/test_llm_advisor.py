# src/tests/test_llm_advisor.py

import src.llm.advisor as advisor_module


def test_merge_ml_and_rag_basic():
    ml_candidates = [
        {"product_id": 1, "score": 0.9, "product_name": "ML Name"},
        {"product_id": "2", "score": 0.5},
        {"product_id": "999", "score": 0.1},  # no RAG match
    ]
    rag_products = [
        {
            "product_id": "1",
            "name": "RAG Name 1",
            "price": 1000,
            "rating": 4.5,
            "retrieval_score": 0.8,
            "document": "doc text 1",
        },
        {
            "product_id": "2",
            "name": "",
            "price": 2000,
            "rating": 4.0,
            "retrieval_score": 0.7,
            "document": "doc text 2",
        },
    ]

    merged = advisor_module.merge_ml_and_rag(ml_candidates, rag_products)

    # Only ids 1 and 2 overlap
    assert len(merged) == 2
    m1 = merged[0]
    m2 = merged[1]

    assert m1["product_id"] == "1"
    # prefers RAG name if present
    assert m1["product_name"] == "RAG Name 1"
    assert m1["ml_score"] == 0.9
    assert m1["price"] == 1000
    assert m1["rating"] == 4.5
    assert m1["retrieval_score"] == 0.8
    assert m1["snippet"] == "doc text 1"

    assert m2["product_id"] == "2"
    # falls back to ML product_name if RAG name empty
    assert m2["product_name"] == "ML Name" or m2["product_name"]
    assert isinstance(m2["ml_score"], float)
    assert isinstance(m2["retrieval_score"], float)
    assert m2["snippet"].startswith("doc text 2")


def test_build_advisor_prompt_includes_key_sections():
    merged_products = [
        {
            "product_id": "p1",
            "product_name": "Phone A",
            "ml_score": 0.9,
            "price": 50000,
            "rating": 4.3,
            "retrieval_score": 0.8,
            "snippet": "Nice phone.",
        }
    ]
    user_query = "I want a good phone under 60k."
    rag_answer = "Some summarized review info."

    prompt = advisor_module.build_advisor_prompt(
        user_query=user_query,
        merged_products=merged_products,
        rag_answer=rag_answer,
    )

    # Basic structure checks
    assert "User Query:" in prompt
    assert user_query in prompt
    assert "Candidate Products (ML + RAG):" in prompt
    assert "Phone A" in prompt
    assert "ML_score" in prompt
    assert "retrieval_score" in prompt
    assert "RAG Review Summary:" in prompt
    assert rag_answer in prompt


def test_generate_final_answer_normal_path(monkeypatch):
    captured = {}

    class DummyModel:
        def generate_content(self, prompt: str):
            captured["prompt"] = prompt

            class Resp:
                text = "FINAL-ANSWER"

            return Resp()

    # Force merge_ml_and_rag to return a non-empty list
    def fake_merge_ml_and_rag(ml_candidates, rag_products):
        return [
            {
                "product_id": "p1",
                "product_name": "Prod 1",
                "ml_score": 1.0,
                "price": 123,
                "rating": 4.5,
                "retrieval_score": 0.9,
                "snippet": "snippet",
            }
        ]

    monkeypatch.setattr(advisor_module, "ADVISOR_MODEL", DummyModel())
    monkeypatch.setattr(advisor_module, "merge_ml_and_rag", fake_merge_ml_and_rag)

    out = advisor_module.generate_final_answer(
        user_query="Which phone?",
        ml_candidates=[{"product_id": "p1"}],
        rag_result={"products": [], "rag_answer": "RAG SUMMARY"},
    )

    assert out == "FINAL-ANSWER"
    # Should be using the advisor-style prompt
    assert "Candidate Products (ML + RAG):" in captured["prompt"]
    assert "Which phone?" in captured["prompt"]


def test_generate_final_answer_fallback_when_no_overlap(monkeypatch):
    captured = {}

    class DummyModel:
        def generate_content(self, prompt: str):
            captured["prompt"] = prompt

            class Resp:
                text = "FALLBACK-ANSWER"

            return Resp()

    # Force empty merged_products
    def fake_merge_ml_and_rag(ml_candidates, rag_products):
        return []

    monkeypatch.setattr(advisor_module, "ADVISOR_MODEL", DummyModel())
    monkeypatch.setattr(advisor_module, "merge_ml_and_rag", fake_merge_ml_and_rag)

    out = advisor_module.generate_final_answer(
        user_query="Rewrite this for me",
        ml_candidates=[],
        rag_result={"rag_answer": "Original RAG Answer"},
    )

    assert out == "FALLBACK-ANSWER"
    # Should be using the "rewrite RAG answer" style prompt
    assert "Rewrite the following RAG answer" in captured["prompt"]
    assert "Original RAG Answer" in captured["prompt"]
