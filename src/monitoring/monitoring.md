# Monitoring & Drift Detection

This project includes a full observability stack to monitor LLM performance, system health, and data drift using Prometheus, Grafana, and Evidently. All components are containerized via Docker Compose.

## Architecture Overview

```
           +--------------------+
           |  LLM FastAPI App   |
           |  (Exposes metrics) |
           +---------+----------+
                     |
                     | /metrics (Prometheus format)
                     v
           +--------------------+
           |    Prometheus      |
           | (Scrapes metrics)  |
           +---------+----------+
                     |
                     | Query metrics
                     v
           +--------------------+
           |      Grafana       |
           | (Visualize metrics)|
           +--------------------+
```

### Data Drift Monitoring:

```
           +--------------------+
           | Reference Dataset  |
           +--------------------+
                     |
                     v
           +--------------------+
           |  Evidently Service |
           | (Drift Detection)  |
           +---------+----------+
                     |
                     v
           +--------------------+
           |   Drift Report     |
           |  (/report endpoint)|
           +--------------------+
```

**FastAPI App** exposes metrics and serves LLM endpoints.

**Prometheus** scrapes `/metrics` to collect runtime and LLM metrics.

**Grafana** visualizes metrics from Prometheus with dashboards for request volume, latency, and guardrail violations.

**Evidently** compares production datasets against reference datasets and serves drift reports.

---

## 1. Prometheus — Metrics Collection

Prometheus collects metrics from the LLM backend (FastAPI app) and the Python runtime.

### Docker Service

```yaml
prometheus:
  image: prom/prometheus:latest
  container_name: llmops_prometheus
  volumes:
    - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
  command:
    - "--config.file=/etc/prometheus/prometheus.yml"
  ports:
    - "9091:9090"
  restart: unless-stopped
```

### Endpoints

- **Prometheus UI:** http://localhost:9091
- **Target Status:** http://localhost:9091/targets
- **Metrics scraped from LLM app:** `/metrics`

### Metrics Collected

| Metric | Type | Description |
|--------|------|-------------|
| `python_gc_objects_collected_total` | Counter | GC objects collected per generation |
| `python_gc_objects_uncollectable_total` | Counter | GC uncollectable objects |
| `python_gc_collections_total` | Counter | Number of GC cycles per generation |
| `python_info` | Gauge | Python runtime version info |
| `process_virtual_memory_bytes` | Gauge | Virtual memory usage |
| `process_resident_memory_bytes` | Gauge | RAM usage |
| `process_cpu_seconds_total` | Counter | CPU seconds consumed |
| `process_open_fds`, `process_max_fds` | Gauge | Open file descriptors |
| `llm_requests_total` | Counter | Number of LLM requests processed |
| `llm_request_latency_seconds` | Histogram | Distribution of LLM request latency |
| `llm_guardrail_violations_total` | Counter | Safety/guardrail violations |
| `http_requests_total` | Counter | HTTP requests count (method, status, handler) |
| `http_request_duration_seconds` | Histogram | Latency of HTTP requests |
| `http_response_size_bytes` | Summary | Response sizes |
| `http_request_size_bytes` | Summary | Request sizes |

✅ These metrics provide visibility into backend performance, resource usage, and model safety.

---

## 2. Grafana — Dashboard Visualization

Grafana visualizes metrics collected by Prometheus and displays real-time LLM operational metrics.

### Docker Service

```yaml
grafana:
  image: grafana/grafana:latest
  container_name: llmops_grafana
  ports:
    - "3001:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
    - GF_USERS_ALLOW_SIGN_UP=false
  volumes:
    - ./grafana/dashboards:/var/lib/grafana/dashboards
    - ./grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards
    - ./grafana/provisioning/datasources:/etc/grafana/provisioning/datasources
  depends_on:
    - prometheus
  restart: unless-stopped
```

### Dashboard URL

http://localhost:3001/

### Key Panels

| Panel | Prometheus Query | Description |
|-------|------------------|-------------|
| Total LLM Requests | `llm_requests_total` | Total number of LLM requests served |
| LLM Request Latency (95th percentile) | `histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[5m])) by (le))` | Measures LLM latency for requests |
| Guardrail Violations (Input Validation) | `llm_guardrail_violations_total{type="input_validation"}` | Tracks input validation safety violations |
| Guardrail Violations (Output Moderation) | `llm_guardrail_violations_total{type="output_moderation"}` | Tracks output moderation violations |

---

## 3. Evidently — Data Drift Monitoring

Evidently monitors data drift in production datasets, comparing live data against reference (training) datasets.

### Docker Service

```yaml
evidently_app:
  build:
    context: ../../
    dockerfile: src/llm/Dockerfile.evidently
  container_name: llmops_evidently
  ports:
    - "7000:7000"
  environment:
    - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
    - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    - AWS_DEFAULT_REGION=us-east-1
  depends_on:
    - llmops_app
  restart: unless-stopped
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/refresh` | POST | Regenerate data drift report |
| `/report` | GET | View HTML data drift report |

**Dashboard URL:** http://localhost:7000/report

### Example Drift Report Generation

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/raw/amazon.csv")
cur = pd.read_csv("data/raw/amazon.csv")  # can be current production batch

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("monitoring/evidently_report.html")
```

---

## 4. Workflow Overview

### Metrics Collection

- FastAPI + Prometheus client exposes metrics at `/metrics`.
- Prometheus scrapes metrics every interval (default 15s).

### Visualization

- Grafana reads metrics from Prometheus and visualizes:
  - LLM requests
  - Latency
  - Guardrail violations
  - System health

### Data Drift Monitoring

- Evidently compares live dataset with reference dataset.
- Reports are updated via FastAPI `/refresh` endpoint.
- Drift trends can be reviewed in the browser at `/report`.

---

## 5. Access URLs (Local)

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9091 |
| Grafana | http://localhost:3001 |
| Evidently Drift Report | http://localhost:7000/report |
