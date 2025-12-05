import pytest
from fastapi.testclient import TestClient

# Import your FastAPI app
# Adjust the path ONLY IF your app file is different.
from app import app

client = TestClient(app)


# ----------------------------
# Basic health endpoint test
# ----------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ----------------------------
# RAG search endpoint test
# ----------------------------
def test_rag_search():
    payload = {"query": "best earphones under 2000 rupees"}
    response = client.post("/rag", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Expected RAG response shape
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0


# ----------------------------
# RAG + LLM pipeline test (optional)
# ----------------------------
@pytest.mark.integration
def test_full_pipeline():
    payload = {"user_id": "test-user", "query": "I want budget earphones for gaming"}
    response = client.post("/pipeline", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Basic shape of your pipeline output
    assert "final_answer" in data
    assert isinstance(data["final_answer"], str)
    assert len(data["final_answer"]) > 10
