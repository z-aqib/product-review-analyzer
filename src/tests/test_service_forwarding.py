import src.ml.service as service


def test_get_ml_candidates_for_user_forwards_to_model(monkeypatch):
    calls = {}

    class DummyModel:
        def recommend_for_user(self, user_id, k, exclude_seen):
            calls["user_id"] = user_id
            calls["k"] = k
            calls["exclude_seen"] = exclude_seen
            return [{"product_id": "p1", "score": 1.0, "product_name": "Dummy Product"}]

    # Replace the real global model with our dummy one
    monkeypatch.setattr(service, "model", DummyModel())

    result = service.get_ml_candidates_for_user("u123", k=7)

    # returned value is whatever DummyModel produced
    assert result == [{"product_id": "p1", "score": 1.0, "product_name": "Dummy Product"}]

    # and the call was made with correct arguments
    assert calls == {"user_id": "u123", "k": 7, "exclude_seen": True}
