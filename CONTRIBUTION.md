# CONTRIBUTION.md

**Project:** *Product Review Analyzer —ML & LLMops*
**Course:** MLOps & LLMOps (Fall 2025, IBA Karachi)

---

## 👥 Team Members

| Name             | ERP ID | Role                                    |
| ---------------- | ------ | --------------------------------------- |
| **Zuha Aqib**    | 26106  | Team Lead — Data Pipeline & Model Training + CI/CD |
| **Maham Junaid** | 26909  | Cloud Integration & Monitoring setup    |
| **Maryam Ihsan** | 27152  | Evaluation & API Documentation    |
| **Muhammad Haaris** | 27083  | Data Pipeline & Model Training + CI/CD  |

---

## 🧩 Task Breakdown and Contributions

| Member           | Primary Responsibilities                       | Details of Work Done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Zuha Aqib**    | Data Pipeline, Model Training, and CI/CD | <ul><li>Led data cleaning and preprocessing of Amazon reviews dataset</li><li>Implemented core data pipeline architecture</li><li>Co-developed **Item–Item Collaborative Filtering** algorithm</li><li>Implemented GitHub Actions workflow for CI/CD pipeline</li><li>Set up automated testing and linting checks</li><li>Created data validation and model testing workflows</li><li>Managed model versioning and artifact tracking</li><li>Implemented automated deployment pipelines</li></ul> |
| **Muhammad Haaris** | Data Pipeline, Model Training, and CI/CD | <ul><li>Co-developed data preprocessing and cleaning workflows</li><li>Implemented train-test split methodology</li><li>Enhanced **Item–Item Collaborative Filtering** implementation</li><li>Set up Docker containerization for model training</li><li>Configured CI/CD pipelines for model deployment</li><li>Implemented automated model retraining workflows</li><li>Created data validation checks</li><li>Set up monitoring for model training pipelines</li></ul> |
| **Maham Junaid** | Cloud Integration & API Documentation | <ul><li>Implemented AWS EC2 instance setup for model deployment</li><li>Configured S3 buckets for data and model storage</li><li>Set up CloudWatch monitoring for model performance</li><li>Created comprehensive FastAPI documentation</li><li>Developed API schema and example cURL commands</li><li>Implemented automated API testing</li><li>Created cloud infrastructure documentation</li><li>Set up cloud-based monitoring dashboards</li></ul> |
| **Maryam Ihsan** | Cloud Integration & API Documentation | <ul><li>Configured AWS Lambda functions for serverless operations</li><li>Implemented automated cloud deployment scripts</li><li>Created cloud service integration documentation</li><li>Enhanced FastAPI documentation with detailed examples</li><li>Developed comprehensive API testing suite</li><li>Created cloud deployment guides in README.md</li><li>Documented cloud service interactions</li><li>Implemented cloud resource monitoring</li></ul> |

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
