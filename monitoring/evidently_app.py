import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Evidently Drift Dashboard")


# Load reference and current datasets
def load_datasets():
    # Load your training data as reference
    reference_data = pd.read_csv("data/splits/train/train_set.csv")

    # Load your test data as current
    current_data = pd.read_csv("data/splits/test/test_set.csv")

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

# Mount the static files directory
app.mount("/", StaticFiles(directory="monitoring", html=True), name="static")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/refresh")
async def refresh_dashboard():
    return update_dashboard()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)
