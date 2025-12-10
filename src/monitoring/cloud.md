# D7. Cloud Integration

Our project uses AWS Cloud extensively to host the full MLOps pipeline, integrating multiple cloud services to meet the D7 requirement. We use:

- **AWS EC2** — primary compute for model serving, RAG pipeline, Streamlit UI, monitoring stack
- **AWS S3** — cloud storage for embeddings, datasets, and ingestion artifacts

This section documents the setup, architecture, and deployment process.

---

## 1. AWS EC2 — Primary Cloud Compute

We deploy our entire pipeline on a single Ubuntu-based EC2 instance using Docker and Docker Compose.

### 1.1 Components Hosted on EC2

The EC2 instance runs the following:

| Component | Description | Port |
|-----------|-------------|------|
| Streamlit App | User-facing UI | `8501` |
| FastAPI Backend | ML + RAG + LLM API | `8001` |
| Evidently Dashboard | Data drift reports | `7000/report` |
| Grafana | Observability dashboards | `3001` |
| Prometheus | Metrics scraping | `9091` |
| Target Metrics Endpoint | Prometheus target | `/targets` |

### 1.2 Public Endpoints (NOTE: IP will change after restart)

These URLs were active during testing. The EC2 public IP rotates because the instance is stopped between runs.

- **Streamlit:** http://16.171.4.249:8501/
- **FastAPI docs:** http://16.171.4.249:8001/docs
- **Evidently:** http://16.171.4.249:7000/report
- **Grafana:** http://16.171.4.249:3001/
- **Prometheus:** http://16.171.4.249:9091/targets

### 1.3 EC2 Instance Configuration

- **Instance Type:** `t3.medium` (2 vCPU, 4 GB RAM)
- **OS:** Ubuntu 22.04 LTS
- **Storage:** 20 GB gp3 EBS
- **Security Group Rules:**
  - `22` — SSH
  - `8501` — Streamlit
  - `8001` — FastAPI
  - `7000` — Evidently
  - `3001` — Grafana
  - `9091` — Prometheus

### 1.4 Deployment Workflow

After SSHing into EC2:

```bash
sudo apt update
sudo apt install docker.io docker-compose -y
```

Clone the repository:

```bash
git clone <repo-url>
cd project/src/llm
```

Build & run the entire system using Docker Compose:

```bash
docker-compose up --build -d
```

This boots all services:

- Streamlit UI
- FastAPI backend
- RAG + embeddings logic
- ML recommendation service
- Prometheus & Grafana
- Evidently

A screenshot folder in `/cloud/` contains EC2 setup, security group, and running containers.

---

## 2. AWS S3 — Storage Layer

We use AWS S3 as our single source of truth for all data required by the backend.

### 2.1 What We Store in S3

Our S3 bucket holds:

- Embeddings (FAISS-based index)
- Primary dataset used for RAG ingestion
- Daily updated dataset used by Evidently for drift reports
- Any backend data our FastAPI service needs at runtime

### 2.2 Dataset Logic (Important)

- The original baseline dataset for Evidently is stored in the GitHub repo.
- The updated version (incoming data) is stored in S3 — this allows Evidently to detect drift.
- The main backend (RAG + ML) pulls all data from S3, so the EC2 instance remains stateless.

### 2.3 S3 Access Pattern in Code

During startup, our backend downloads required artifacts:

```python
import boto3

s3 = boto3.client("s3")
s3.download_file("our-bucket", "embeddings/index.faiss", "/app/data/index.faiss")
s3.download_file("our-bucket", "datasets/products.csv", "/app/data/products.csv")
```

This ensures all components run reliably directly from cloud storage.

### 2.4 Documentation Included

Our README includes:

- S3 bucket creation screenshot
- IAM role/policy setup
- Bucket folder structure
- Upload/Download examples

---

## 3. Cloud Architecture Summary

Below is a textual cloud architecture explanation (diagram can be added later):

```
User → Streamlit (EC2) → FastAPI Backend (EC2)
                       ↳ ML Recommender
                       ↳ RAG (Gemini + S3 dataset + embeddings)
                       ↳ LLM Advisor

Monitoring:
  - Prometheus ← FastAPI Metrics
  - Grafana ← Prometheus Data
  - Evidently ← Baseline (GitHub Repo) + Live Data (S3)
```

EC2 hosts everything in Docker. S3 stores all data needed for ingestion, RAG, and monitoring.

---

## 4. Compliance With D7 Requirements

✔ **Two full cloud services used:**
- AWS EC2
- AWS S3

✔ Model hosting on cloud via EC2  
✔ Data stored in cloud via S3  
✔ Monitoring tools deployed on cloud  
✔ Screenshots + setup steps included  
✔ Cloud-native deployment using Docker Compose