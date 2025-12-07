# src/tests/test_train_mlflow.py

from contextlib import contextmanager

import src.train as train_module


def test_train_and_register_uses_mlflow_and_recommender(monkeypatch):
    calls = {
        "params": [],
        "metrics": [],
        "logged_model": False,
        "run_name": None,
    }

    # Fake MLflow functions
    def fake_log_param(key, value):
        calls["params"].append((key, value))

    def fake_log_metric(key, value):
        calls["metrics"].append((key, value))

    def fake_log_model(sk_model, artifact_path, registered_model_name):
        calls["logged_model"] = True
        # basic sanity: the model is the dummy recommender and paths are strings
        assert hasattr(sk_model, "fit")
        assert isinstance(artifact_path, str)
        assert isinstance(registered_model_name, str)

    @contextmanager
    def fake_start_run(run_name=None):
        calls["run_name"] = run_name
        yield

    # Patch mlflow inside train.py
    monkeypatch.setattr(train_module.mlflow, "log_param", fake_log_param)
    monkeypatch.setattr(train_module.mlflow, "log_metric", fake_log_metric)
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", fake_log_model)
    monkeypatch.setattr(train_module.mlflow, "start_run", fake_start_run)

    # Fake recommender so we don't touch the real data directory
    class DummyRecommender:
        def __init__(self, data_dir):
            self.data_dir = data_dir

        def fit(self):
            # mimic the real interface: returns self
            return self

    monkeypatch.setattr(train_module, "ItemItemRecommender", DummyRecommender)

    # Run the training function
    train_module.train_and_register()

    # Assertions on MLflow interaction
    assert calls["run_name"] == "item-item-cf-baseline"
    assert calls["logged_model"] is True

    # We should have at least the "algo" param
    assert ("algo", "item_item_cf") in calls["params"]

    # Metrics keys should include the ones defined in train.py
    metric_keys = {k for k, _ in calls["metrics"]}
    for expected_key in ["rmse", "mae", "auc", "accuracy"]:
        assert expected_key in metric_keys
