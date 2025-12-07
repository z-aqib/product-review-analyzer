# src/tests/test_pipeline_orchestration.py

import src.pipeline as pipeline_module


def test_run_pipeline_wires_components(monkeypatch):
    calls = {}

    def fake_get_ml_candidates_for_user(user_id: str, k: int):
        calls["ml_user_id"] = user_id
        calls["ml_k"] = k
        return [{"product_id": "p1", "score": 0.9}]

    def fake_ask(question: str, k: int):
        calls["rag_question"] = question
        calls["rag_k"] = k
        return {
            "question": question,
            "products": [],
            "rag_answer": "RAG-ANSWER",
        }

    def fake_generate_final_answer(user_query: str, ml_candidates, rag_result):
        calls["advisor_query"] = user_query
        calls["advisor_candidates"] = list(ml_candidates)
        calls["advisor_rag"] = dict(rag_result)
        return "FINAL-ANSWER"

    monkeypatch.setattr(
        pipeline_module,
        "get_ml_candidates_for_user",
        fake_get_ml_candidates_for_user,
    )
    monkeypatch.setattr(pipeline_module, "ask", fake_ask)
    monkeypatch.setattr(
        pipeline_module,
        "generate_final_answer",
        fake_generate_final_answer,
    )

    result = pipeline_module.run_pipeline("user-123", "Which phone?")

    # Result dict structure
    assert result["user_query"] == "Which phone?"
    assert result["ml_candidates"] == [{"product_id": "p1", "score": 0.9}]
    assert result["rag_result"] == {
        "question": "Which phone?",
        "products": [],
        "rag_answer": "RAG-ANSWER",
    }
    assert result["final_answer"] == "FINAL-ANSWER"

    # Wiring correctness
    assert calls["ml_user_id"] == "user-123"
    assert calls["ml_k"] == 5
    assert calls["rag_question"] == "Which phone?"
    assert calls["rag_k"] == 5
    assert calls["advisor_query"] == "Which phone?"
    assert calls["advisor_candidates"] == [{"product_id": "p1", "score": 0.9}]
    assert calls["advisor_rag"] == {
        "question": "Which phone?",
        "products": [],
        "rag_answer": "RAG-ANSWER",
    }
