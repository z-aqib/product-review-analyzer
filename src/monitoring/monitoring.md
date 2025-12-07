Monitoring &amp; Drift Detection
This project includes a full observability stack to monitor LLM performance, system health, and data drift using Prometheus, Grafana, and Evidently. All components are containerized via Docker Compose.

Architecture Overview
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

Data Drift Monitoring:

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



FastAPI App exposes metrics and serves LLM endpoints.


Prometheus scrapes /metrics to collect runtime and LLM metrics.


Grafana visualizes metrics from Prometheus with dashboards for request volume, latency, and guardrail violations.


Evidently compares production datasets against reference datasets and serves drift reports.



1. Prometheus — Metrics Collection
Prometheus collects metrics from the LLM backend (FastAPI app) and the Python runtime.
Docker Service
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

Endpoints


Prometheus UI: http://localhost:9091


Target Status: http://localhost:9091/targets


Metrics scraped from LLM app: /metrics


Metrics Collected
MetricTypeDescriptionpython_gc_objects_collected_totalCounterGC objects collected per generationpython_gc_objects_uncollectable_totalCounterGC uncollectable objectspython_gc_collections_totalCounterNumber of GC cycles per generationpython_infoGaugePython runtime version infoprocess_virtual_memory_bytesGaugeVirtual memory usageprocess_resident_memory_bytesGaugeRAM usageprocess_cpu_seconds_totalCounterCPU seconds consumedprocess_open_fds, process_max_fdsGaugeOpen file descriptorsllm_requests_totalCounterNumber of LLM requests processedllm_request_latency_secondsHistogramDistribution of LLM request latencyllm_guardrail_violations_totalCounterSafety/guardrail violationshttp_requests_totalCounterHTTP requests count (method, status, handler)http_request_duration_secondsHistogramLatency of HTTP requestshttp_response_size_bytesSummaryResponse sizeshttp_request_size_bytesSummaryRequest sizes
✅ These metrics provide visibility into backend performance, resource usage, and model safety.

2. Grafana — Dashboard Visualization
Grafana visualizes metrics collected by Prometheus and displays real-time LLM operational metrics.
Docker Service
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

Dashboard URL


http://localhost:3001/


Key Panels
PanelPrometheus QueryDescriptionTotal LLM Requestsllm_requests_totalTotal number of LLM requests servedLLM Request Latency (95th percentile)histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[5m])) by (le))Measures LLM latency for requestsGuardrail Violations (Input Validation)llm_guardrail_violations_total{type="input_validation"}Tracks input validation safety violationsGuardrail Violations (Output Moderation)llm_guardrail_violations_total{type="output_moderation"}Tracks output moderation violations

3. Evidently — Data Drift Monitoring
Evidently monitors data drift in production datasets, comparing live data against reference (training) datasets.
Docker Service
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

Endpoints
EndpointMethodDescription/healthGETHealth check/refreshPOSTRegenerate data drift report/reportGETView HTML data drift report
Dashboard URL: http://localhost:7000/report
Example Drift Report Generation
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/raw/amazon.csv")
cur = pd.read_csv("data/raw/amazon.csv")  # can be current production batch

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("monitoring/evidently_report.html")


4. Workflow Overview


Metrics Collection


FastAPI + Prometheus client exposes metrics at /metrics.


Prometheus scrapes metrics every interval (default 15s).




Visualization


Grafana reads metrics from Prometheus and visualizes:


LLM requests


Latency


Guardrail violations


System health






Data Drift Monitoring


Evidently compares live dataset with reference dataset.


Reports are updated via FastAPI /refresh endpoint.


Drift trends can be reviewed in the browser at /report.





5. Access URLs (Local)
ServiceURLPrometheushttp://localhost:9091Grafanahttp://localhost:3001Evidently Drift Reporthttp://localhost:7000/report
