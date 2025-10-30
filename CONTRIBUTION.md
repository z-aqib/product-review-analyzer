# CONTRIBUTION.md

**Project:** *Product Review Analyzer — Collaborative Filtering*
**Course:** MLOps & LLMOps (Fall 2025, IBA Karachi)

---

## 👥 Team Members

| Name             | ERP ID | Role                             |
| ---------------- | ------ | -------------------------------- |
| **Zuha Aqib**    | 26106  | Team Lead — Backend + Monitoring |
| **Maham Junaid** | 26832  | Data Pipeline & Model Training   |
| **Maryam Ihsan** | 26948  | Evaluation & API Integration     |
| **Haris**        | 27110  | Infrastructure & CI/CD           |

---

## 🧩 Task Breakdown and Contributions

| Member           | Primary Responsibilities                       | Details of Work Done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Zuha Aqib**    | Backend Development, Monitoring, Documentation | <ul><li>Designed overall architecture and repo structure.</li><li>Implemented FastAPI service (`src/api.py`) with `/predict`, `/health`, `/metrics` endpoints.</li><li>Integrated **Prometheus FastAPI Instrumentator** for metrics.</li><li>Set up **Grafana dashboards** for latency, request counts, and uptime.</li><li>Wrote `Makefile` automation commands (`make dev`, `make stack-up`, etc.).</li><li>Prepared all documentation including **README.md** and **CONTRIBUTION.md**.</li></ul> |
| **Maham Junaid** | Data Preparation & Model Training              | <ul><li>Cleaned and preprocessed Amazon reviews dataset.</li><li>Structured processed data under `data/processed/` and ensured format consistency.</li><li>Implemented **Item–Item Collaborative Filtering** in `src/ml/recommenders/item_item.py` (fit, similarity matrix, recommend).</li><li>Generated pickled model artifacts for inference.</li></ul>                                                                                                                                          |
| **Maryam Ihsan** | Evaluation, Testing, and API Integration       | <ul><li>Built **Leave-One-Out evaluation** pipeline in `src/evaluate.py` and linked it to recommender outputs.</li><li>Created metrics summary CSVs (recall@k, NDCG, coverage).</li><li>Connected trained artifacts to FastAPI inference layer (`src/app/main.py`).</li><li>Implemented `/predict` logic and JSON response formatting.</li><li>Added **unit tests** for health endpoint (`src/tests/test_health.py`).</li></ul>                                                                     |
| **Haris**        | Infrastructure, CI/CD, and Automation          | <ul><li>Configured **Dockerfile** and `docker-compose.yml` for reproducible environments.</li><li>Integrated **Prometheus** + **Grafana** containers under `infra/` with correct service discovery.</li><li>Implemented GitHub Actions workflow for CI linting and pytest checks (to be finalized).</li><li>Handled **Evidently Drift Report** automation in `monitoring/generate_drift.py`.</li><li>Maintained overall repo hygiene and ensured all dependencies are pinned.</li></ul>             |

---

## 🌿 Branch-Naming Convention

| Branch Name                        | Prefix Category           | Purpose / Description                                                                                              |
| ---------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **`fix/structure`**                | `fix/`                    | Minor structural fixes and directory cleanup after initial setup (refined imports, paths, and relative structure). |
| **`infra/app-setup`**              | `infra/`                  | Configured application infrastructure — FastAPI service wiring, environment variables, and app-level organization. |
| **`infra/bootstrap-setup`**        | `infra/`                  | Initial repository bootstrap: virtual environment, Makefile, requirements, and local project scaffolding.          |
| **`infra/cloud-integration`**      | `infra/`                  | Cloud integration setup — connecting Dockerized services with cloud endpoints (planned deployment stage).          |
| **`ml-workflow-monitoring-setup`** | `ml-workflow/` *(custom)* | Integrated ML workflow monitoring — Prometheus, Grafana dashboards, and MLflow logging integration.                |
| **`main`**                         | —                         | Stable release branch for milestone submissions and final presentation.                                            |

**Workflow followed:**

1. Each member created a feature branch → committed changes → pushed → opened PR into `develop`.
2. After review and testing, `develop` was merged into `main` for stable releases.

---

## 🧠 Summary

This project was built collaboratively using **GitHub**, **Docker**, and **FastAPI**, following true MLOps principles — modular code, containerized infrastructure, and automated monitoring.
Each member contributed in a complementary domain (data, model, API, infra), ensuring smooth integration from dataset to deployment.
