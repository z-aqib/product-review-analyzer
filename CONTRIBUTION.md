# CONTRIBUTION.md

**Project:** *Product Review Analyzer —ML & LLMops*
**Course:** MLOps & LLMOps (Fall 2025, IBA Karachi)

---

## 👥 Team Members

| Name             | ERP ID | Role                                    |
| ---------------- | ------ | --------------------------------------- |
| **Zuha Aqib**    | 26106  | Team Lead — Data Pipeline & Model Training + CI/CD |
| **Maham Junaid** | 26909  | Model Integration, Cloud Deployment & Monitoring Setup    |
| **Maryam Ihsan** | 27152  | Model Integration, Cloud Deployment & Monitoring Setup    |
| **Muhammad Haaris** | 27083  | Data Pipeline & Model Training + CI/CD  |

---

## 🧩 Task Breakdown and Contributions

| Member           | Primary Responsibilities                       | Details of Work Done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Zuha Aqib**    | Data Pipeline, Model Training, and CI/CD | <ul><li>Led data cleaning and preprocessing of Amazon reviews dataset</li><li>Implemented core data pipeline architecture</li><li>Co-developed **Item–Item Collaborative Filtering** algorithm</li><li>Implemented GitHub Actions workflow for CI/CD pipeline</li><li>Set up automated testing and linting checks</li><li>Created data validation and model testing workflows</li><li>Managed model versioning and artifact tracking</li><li>Implemented automated deployment pipelines</li></ul> |
| **Muhammad Haaris** | Data Pipeline, Model Training, and CI/CD | <ul><li>Co-developed data preprocessing and cleaning workflows</li><li>Implemented train-test split methodology</li><li>Enhanced **Item–Item Collaborative Filtering** implementation</li><li>Set up Docker containerization for model training</li><li>Configured CI/CD pipelines for model deployment</li><li>Implemented automated model retraining workflows</li><li>Created data validation checks</li><li>Set up monitoring for model training pipelines</li></ul> |
| **Maham Junaid and Maryam Ihsan** | <ul><li>Integrated the trained recommendation model notebook into the project and generated all necessary serialized artifacts (.pkl files) for production use.</li><li>Configured AWS S3 for artifact and data storage, uploading processed model files for remote accessibility.</li><li>Developed the FastAPI application to serve model predictions through /health and /recommend endpoints.</li><li>Tested and verified API responses locally using cURL and FastAPI’s interactive /docs UI.</li><li>Dockerized the entire stack locally using Docker Compose, including services for the recommender API, Prometheus, and Grafana for observability.</li><li>Set up Prometheus for metrics collection and Grafana dashboards for system and model monitoring.</li><li>Created and configured AWS EC2 instance for deployment of all three containers (FastAPI, Prometheus, Grafana).</li><li>Pushed Docker images to Docker Hub and pulled them on EC2 for smooth deployment.</li><li>Resolved multiple EC2 free-tier storage issues by optimizing Docker image builds and using BuildKit for lightweight images.</li><li>Troubleshot container and network issues across the stack, ensuring all services ran correctly and communicated seamlessly.</li><li>Successfully deployed the full pipeline (API + Monitoring) on EC2 with working endpoints and dashboards.</li><li>Collaboratively handled iterative testing, error resolution, and documentation of the deployment workflow.</li></ul> |
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
