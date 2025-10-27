from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Evidently Drift Dashboard")
app.mount("/", StaticFiles(directory="monitoring", html=True), name="static")
