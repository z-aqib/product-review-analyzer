# Team Contributions

This document outlines the individual contributions of each team member to the Amazon Product Review RAG Pipeline with LLMOps project.

---

## 👥 Team Members

### Maham Junaid

#### Core Contributions
- **FastAPI Backend Development**
  - Developed `app.py` with complete FastAPI endpoints for RAG queries
  - Implemented `/query`, `/health`, `/refresh`, and `/report` endpoints
  - Designed API request/response schemas

- **Monitoring Stack**
  - Created Prometheus metrics collection for LLM latency, token usage & response rate
  - Built Grafana dashboards for real-time observability
  - Configured monitoring stack integration with FastAPI backend
  - Created `monitoring.md` documentation for metrics & drift detection (D4)

- **Cloud Infrastructure**
  - Led D7 Cloud Integration (AWS EC2 deployment, Docker Compose setup, S3 integration)
  - Configured EC2 security groups & deployment workflows
  - Contributed to infra documentation

- **Frontend Development**
  - Developed Streamlit UI with Maryam
  - Implemented query interface and visualization of retrieved results

- **Containerization & Deployment**
  - Contributed to local Dockerization setup (Dockerfiles & Docker Compose)
  - Assisted in multi-container orchestration

- **A/B Testing Feature (Bonus +5 pts)**
  - Implemented A/B testing dashboard to compare prompt variants and model responses
  - Integrated evaluation metrics for comparison

- *Safety & *Documentation**
  - Assisted in guardrails implementation
  - Contributed to Monitoring documentation
  - Contributed to README & cloud integration docs

---

### Maryam Ihsan

#### Core Contributions
- **FastAPI Backend Development**
  - Collaborated on `app.py` implementation for RAG query endpoints
  - Contributed to pipeline integration and API route design

- **Monitoring & Observability**
  - Developed Evidently monitoring dashboard for data drift detection
  - Collaborated on Prometheus/Grafana monitoring setup and configuration
  - Created monitoring documentation (D4)

- **Cloud Infrastructure**
  - Worked on D7 Cloud Integration (AWS EC2 deployment, S3 integration)
  - Contributed to cloud infrastructure documentation

- **Frontend Development**
  - Built Streamlit frontend application for RAG querying with Maham
  - Designed user-friendly UI for product recommendations

- **Containerization & Deployment**
  - Contributed to local Dockerization setup (Dockerfiles & Docker Compose)
  - Assisted in multi-container orchestration

- **Safety & Documentation**
  - Assisted in guardrails implementation
  - Contributed to guardrails documentation (D3)
  - Contributed to README & cloud integration docs

- **CI/CD Pipeline**
  - Initiated CI/CD workflows with GitHub Actions (D5)

---

### Zuha Aqib

#### Core Contributions
- **RAG Pipeline Development**
  - Developed `pipeline.py` for end-to-end RAG retrieval & LLM orchestration

- **Document Ingestion & Embeddings**
  - Implemented `ingest.py` for document ingestion & embedding generation
  - Built FAISS vector index handling

- **Prompt Engineering**
  - Completed D1 Prompt Engineering experimentation
  - Created multiple prompt strategies and evaluation metrics (F1, latency)

- **CI/CD Pipeline**
  - Configured automated testing & deployment workflows

- **Coverage**
  - Created tests for all running files and ran pytest --cov=src to get 88% coverage over all files in src.

- **Safety**
  - Implemented guardrails with Maryam & Maham

---

### Mohammed Haris

#### Core Contributions
- **RAG Implementation**
  - Implemented `rag.py` for semantic search & document retrieval
  - Performed retrieval experimentation, optimization & evaluation

- **LangChain Integration (Bonus +5 pts)**
  - Implemented LangChain workflow for modular pipeline routing
  - Built custom retrieval chain with assistance from Zuha

- **Security & Compliance**
  - Authored `SECURITY.md` under D8 Compliance
  - Worked on injection prevention & validation measures

- **Performance Testing**
  - Evaluated retrieval performance and optimized parameters

---

## 📝 Summary of Deliverables

| Deliverable | Owner(s) |
|------------|----------|
| D1 Prompt Engineering | Zuha |
| D2 RAG Pipeline | All (component-wise) |
| D3 Guardrails | Zuha, Maham, Maryam |
| D4 Monitoring | Maham (Prometheus/Grafana), Maryam (Evidently) |
| D5 CI/CD | Maryam (init), Zuha (expansion) |
| D6 Docs | Maham & Maryam |
| D7 Cloud | Maham & Maryam |
| D8 Security | Haris |

### Bonus Features (+5 pts)
- **LangChain Integration** — Haris
- **A/B Testing Dashboard** — Maham

---
