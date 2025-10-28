import mlflow
import mlflow.sklearn  # or mlflow.pyfunc if you wrap your CF model


def train_and_register():
    # ... your training code here ...
    params = {"algo": "user_knn", "k": 50}
    metrics = {"rmse": 0.892, "mae": 0.71}

    with mlflow.start_run(run_name="cf-baseline"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log your fitted model (replace with your actual model object)
        mlflow.sklearn.log_model(
            sk_model=fitted_model,
            artifact_path="model",
            registered_model_name="sentiment-recommender",
        )


if __name__ == "__main__":
    train_and_register()
