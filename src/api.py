from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time
from fastapi.responses import RedirectResponse

app = FastAPI(title="Sentiment API")

# Auto-instrument HTTP metrics at /metrics
Instrumentator().instrument(app).expose(app)

# Custom metrics
PREDICTION_COUNT = Counter("predictions_total", "Number of predictions served")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of predictions")


@app.get("/")
def home():
    return {"ok": True, "routes": ["/docs", "/metrics", "/predict"]}


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    start = time.time()
    # ... your model inference here ...
    result = {"label": "positive", "score": 0.93}
    duration = time.time() - start

    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(duration)
    return result
