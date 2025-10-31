import mlflow
import mlflow.sklearn
from ml.recommenders.item_item import ItemItemRecommender

# Set MLflow tracking URI - using local mlruns directory
mlflow.set_tracking_uri("http://localhost:5000")  # For local MLflow server
# For S3, you can use: mlflow.set_tracking_uri("s3://mlops-d9/mlruns")


def train_and_register():

    # Initialize and train the model
    model = ItemItemRecommender("data/processed/")
    model.fit()

    # Define parameters and metrics
    params = {"algo": "item_item_cf", "k": 15}

    # Calculate some metrics (replace with your actual metrics calculation)
    metrics = {
        "rmse": 0.552,  # Replace with actual RMSE calculation
        "mae": 0.244,  # Replace with actual MAE calculation
        "auc": 0.95,
        "accuracy": 0.93,
    }

    # Start MLflow run
    with mlflow.start_run(run_name="item-item-cf-baseline"):
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", registered_model_name="product-recommender"
        )


if __name__ == "__main__":
    train_and_register()
