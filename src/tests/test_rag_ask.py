# src/tests/test_rag_ask.py

import math

import pytest
import src.rag.rag as rag_module


def test_ask_builds_products_and_rag_answer(monkeypatch):
    class DummyEmbedder:
        def encode(self, texts, normalize_embeddings=None):
            # shape doesn't matter, we just pass it to DummyIndex
            return ["dummy-embedding"]

    class DummyIndex:
        def search(self, q_emb, k):
            # Return two results (distances and indices)
            return [[0.9, 0.8]], [[0, 1]]

    documents = ["doc for p1", "doc for p2"]
    metadatas = [
        {
            "product_id": "p1",
            "name": "Prod1",
            "price": "1000",
            "rating": "4.5",
        },
        {
            "product_id": "p2",
            "name": "Prod2",
            "price": "2000",
            # Use NaN to test rating == rating check
            "rating": math.nan,
        },
    ]

    def fake_parse_price(value):
        if value is None:
            return None
        return float(value)

    captured = {}

    class DummyModel:
        def generate_content(self, prompt: str):
            captured["prompt"] = prompt

            class Resp:
                text = "RAG-ANSWER"

            return Resp()

    monkeypatch.setattr(rag_module, "embedder", DummyEmbedder())
    monkeypatch.setattr(rag_module, "index", DummyIndex())
    monkeypatch.setattr(rag_module, "documents", documents)
    monkeypatch.setattr(rag_module, "metadatas", metadatas)
    monkeypatch.setattr(rag_module, "parse_price", fake_parse_price)
    monkeypatch.setattr(rag_module, "RAG_MODEL", DummyModel())

    result = rag_module.ask("some question", k=2)

    assert result["question"] == "some question"
    assert result["rag_answer"] == "RAG-ANSWER"
    assert len(result["products"]) == 2

    p1, p2 = result["products"]

    assert p1["product_id"] == "p1"
    assert p1["name"] == "Prod1"
    assert p1["price"] == 1000.0
    assert p1["rating"] == 4.5
    assert p1["retrieval_score"] == 0.9
    assert p1["document"] == "doc for p1"

    assert p2["product_id"] == "p2"
    # second rating is NaN in metadata, so code should set it to None
    assert p2["rating"] is None
    assert p2["price"] == 2000.0
    assert p2["retrieval_score"] == 0.8
    assert p2["document"] == "doc for p2"

    # Prompt sent to Gemini must contain context and question
    assert "doc for p1" in captured["prompt"]
    assert "doc for p2" in captured["prompt"]
    assert "some question" in captured["prompt"]


def test_ask_raises_when_not_initialized(monkeypatch):
    monkeypatch.setattr(rag_module, "embedder", None)
    monkeypatch.setattr(rag_module, "index", None)

    with pytest.raises(RuntimeError):
        rag_module.ask("q")
