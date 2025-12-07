# src/tests/test_app_api.py

from fastapi.testclient import TestClient
import src.app as app_module


def test_recommend_happy_path(monkeypatch):
    """
    - input validation passes, no flags
    - pipeline runs and returns final_answer
    - output moderation returns same text, no flags
    - response is 200 with empty guardrail_events
    """
    client = TestClient(app_module.app)

    def fake_validate_input_query(query: str):
        return {"flags": []}

    def fake_run_pipeline(user_id: str, user_query: str):
        return {
            "user_query": user_query,
            "ml_candidates": [{"product_id": "p1", "score": 1.0}],
            "rag_result": {"chunks": [{"id": 1, "text": "some context"}]},
            "final_answer": "raw answer from pipeline",
        }

    def fake_moderate_output_text(text: str):
        # No flags, but slightly modify text to show it went through moderation
        return {"text": text + " [SAFE]", "flags": []}

    monkeypatch.setattr(app_module, "validate_input_query", fake_validate_input_query)
    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(app_module, "moderate_output_text", fake_moderate_output_text)

    payload = {
        "user_id": "user-123",
        "user_query": "Recommend a laptop under 150k",
    }

    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["user_query"] == payload["user_query"]
    assert body["ml_candidates"] == [{"product_id": "p1", "score": 1.0}]
    assert body["rag_result"] == {"chunks": [{"id": 1, "text": "some context"}]}
    assert body["final_answer"] == "raw answer from pipeline [SAFE]"
    assert body["guardrail_events"] == []


def test_recommend_input_guardrail_violation_returns_400(monkeypatch):
    """
    - validate_input_query raises GuardrailViolation
    - endpoint should return 400 with detail including kind/message/details
    """
    client = TestClient(app_module.app)

    class FakeGuardrailViolation(Exception):
        def __init__(self, kind: str, message: str, details=None):
            super().__init__(message)
            self.kind = kind
            self.details = details or {}

    def fake_validate_input_query(query: str):
        raise FakeGuardrailViolation(
            kind="unsafe_text",
            message="Input not allowed",
            details={"reason": "blocked test"},
        )

    # Replace the GuardrailViolation type in the app module so the except block catches it
    monkeypatch.setattr(app_module, "GuardrailViolation", FakeGuardrailViolation)
    monkeypatch.setattr(app_module, "validate_input_query", fake_validate_input_query)

    payload = {
        "user_id": "user-123",
        "user_query": "bad input",
    }

    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 400

    body = resp.json()
    # FastAPI wraps HTTPException.detail into {"detail": {...}}
    assert body["detail"]["error"] == "unsafe_input"
    assert body["detail"]["kind"] == "unsafe_text"
    assert body["detail"]["message"] == "Input not allowed"
    assert body["detail"]["details"] == {"reason": "blocked test"}


def test_recommend_output_guardrail_violation_replaces_answer(monkeypatch):
    """
    - input validation passes
    - pipeline returns some final_answer
    - moderate_output_text raises GuardrailViolation
    - endpoint should still be 200 but with generic safe message and a guardrail_event
    """
    client = TestClient(app_module.app)

    def fake_validate_input_query(query: str):
        return {"flags": []}

    def fake_run_pipeline(user_id: str, user_query: str):
        return {
            "user_query": user_query,
            "ml_candidates": [],
            "rag_result": {},
            "final_answer": "potentially unsafe answer",
        }

    class FakeGuardrailViolation(Exception):
        def __init__(self, kind: str, message: str, details=None):
            super().__init__(message)
            self.kind = kind
            self.details = details or {}

    def fake_moderate_output_text(text: str):
        raise FakeGuardrailViolation(
            kind="unsafe_output",
            message="Output not allowed",
            details={"reason": "blocked output"},
        )

    monkeypatch.setattr(app_module, "validate_input_query", fake_validate_input_query)
    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(app_module, "GuardrailViolation", FakeGuardrailViolation)
    monkeypatch.setattr(app_module, "moderate_output_text", fake_moderate_output_text)

    payload = {
        "user_id": "user-123",
        "user_query": "normal query",
    }

    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    # Generic safe message from app.py
    assert "couldn't generate a safe response" in body["final_answer"]

    # We should have one guardrail event of type output_moderation
    assert len(body["guardrail_events"]) == 1
    event = body["guardrail_events"][0]
    assert event["type"] == "output_moderation"
    assert event["kind"] == "unsafe_output"
    assert event["message"] == "Output not allowed"
    assert event["details"] == {"reason": "blocked output"}
