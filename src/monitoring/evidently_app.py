# monitoring/evidently_app.py
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from datetime import datetime
import boto3
from io import StringIO
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI(title="Evidently Drift Dashboard")

S3_BUCKET = "mlops-d9"
S3_KEY = "data/raw/amazon.csv"  # path where you stored the dataset in S3

s3 = boto3.client("s3")


def load_datasets():
    # Reference dataset - original local snapshot
    reference_data = pd.read_csv("data/raw/amazon.csv")

    # Current dataset - latest uploaded dataset from S3
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    current_data = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    return reference_data, current_data


def generate_drift_report():
    reference_data, current_data = load_datasets()

    # Create Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset(), DataDriftTable()])

    # Calculate drift
    drift_report.run(reference_data=reference_data, current_data=current_data)

    # Save the report
    drift_report.save_html("monitoring/evidently_report.html")


def update_dashboard():
    try:
        generate_drift_report()
        return {"status": "success", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Generate initial report
generate_drift_report()

# Serve static HTML files (not at root anymore)
app.mount("/static", StaticFiles(directory="monitoring", html=True), name="static")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/refresh")
async def refresh_dashboard():
    return update_dashboard()


@app.get("/report", response_class=HTMLResponse)
def view_report():
    with open("monitoring/evidently_report.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)
