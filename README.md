# Amazon Product Review RAG Pipeline with LLMOps

A complete Retrieval-Augmented Generation (RAG) system for Amazon product recommendations, combining machine learning, embeddings, and LLMs with comprehensive monitoring and safety guardrails.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Deployment Guide](#deployment-guide)
- [API Documentation](#api-documentation)
- [CI/CD Pipeline](#cicd-pipeline)
- [Prompt Engineering & Evaluation](#prompt-engineering--evaluation)
- [Monitoring & Observability](#monitoring--observability)
- [Data Drift Detection](#data-drift-detection)
- [Guardrails & Safety](#guardrails--safety)
- [Cloud Infrastructure](#cloud-infrastructure)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Performance Considerations](#performance-considerations)
- [Future Enhancements](#future-enhancements)
- [Contact & Support](#contact--support)

---

## Project Overview

This project implements an end-to-end LLMOps pipeline that helps users discover Amazon products through natural language queries. The system combines multiple stages:

### What Problem Does This Solve?

Traditional product search relies on keyword matching. Our system understands user intent and provides personalized, context-aware recommendations by:

- Using ML models to identify relevant products based on user history
- Leveraging RAG to retrieve and summarize actual product reviews
- Employing LLMs (Gemini) to generate natural, helpful recommendations
- Ensuring safety through input/output guardrails
- Monitoring system health and data drift in production

### Example Query

**User:** "I want a Dell laptop for programming under $1500 with good battery life"

**System Response:** A natural language recommendation combining ML scores, review summaries, and price/rating data from actual Amazon products.

---

## Architecture

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Input Validation   │
                  │    (Guardrails)     │
                  │ - Prompt Injection  │
                  │ - PII Detection     │
                  └──────────┬──────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         STAGE 1: ML RECOMMENDER       │
        │  - User-based collaborative filtering │
        │  - Returns top-K product candidates   │
        └────────────────┬──────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────────┐
        │      STAGE 2: RAG PIPELINE            │
        │  ┌─────────────────────────────────┐  │
        │  │  1. Embed query (Sentence BERT) │  │
        │  │  2. Search FAISS index          │  │
        │  │  3. Retrieve product reviews    │  │
        │  └─────────────────────────────────┘  │
        └────────────────┬──────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────────┐
        │      STAGE 3: LLM ADVISOR             │
        │  - Merge ML + RAG results             │
        │  - Build prompt for Gemini API        │
        │  - Generate final recommendation      │
        └────────────────┬──────────────────────┘
                         │
                         ▼
                  ┌─────────────────────┐
                  │  Output Moderation  │
                  │    (Guardrails)     │
                  │ - Toxicity filter   │
                  │ - Safety check      │
                  └──────────┬──────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         FINAL RESPONSE                │
        │  - Natural language recommendation    │
        │  - Product details + reasoning        │
        └───────────────────────────────────────┘
```

### Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWS EC2 INSTANCE                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Docker Compose Stack                    │  │
│  │                                                            │  │
│  │  ┌─────────────────┐      ┌──────────────────┐          │  │
│  │  │  Streamlit UI   │◄────►│  FastAPI Backend │          │  │
│  │  │   Port: 8501    │      │    Port: 8001    │          │  │
│  │  └─────────────────┘      └──────────┬───────┘          │  │
│  │                                       │                   │  │
│  │                                       │ /metrics          │  │
│  │                                       ▼                   │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │            MONITORING STACK                      │    │  │
│  │  │  ┌──────────────┐  ┌──────────────┐            │    │  │
│  │  │  │  Prometheus  │  │   Grafana    │            │    │  │
│  │  │  │  Port: 9091  │─►│  Port: 3001  │            │    │  │
│  │  │  └──────────────┘  └──────────────┘            │    │  │
│  │  │                                                  │    │  │
│  │  │  ┌──────────────────────────────────────────┐  │    │  │
│  │  │  │  Evidently (Data Drift Detection)       │  │    │  │
│  │  │  │  Port: 7000                              │  │    │  │
│  │  │  └──────────────────────────────────────────┘  │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │       AWS S3           │
              │  - Embeddings (FAISS)  │
              │  - Product datasets    │
              │  - Drift data          │
              └────────────────────────┘
```

---

## Features

### ✨ Core Capabilities

- **Natural Language Querying**: Ask for products in plain English
- **Hybrid Recommendations**: Combines ML collaborative filtering + semantic search
- **Review-Based Insights**: Uses actual Amazon reviews via RAG
- **LLM-Powered Responses**: Gemini generates natural, helpful answers
- **Safety First**: Input validation and output moderation guardrails
- **Production-Ready Monitoring**: Prometheus, Grafana, and Evidently integration
- **Cloud-Native Deployment**: Fully containerized with Docker Compose on AWS EC2

### 🛡️ Guardrails

- **Input Protection**: Prompt injection detection, PII filtering
- **Output Safety**: Toxicity detection, content moderation
- **Audit Logging**: All guardrail events tracked in monitoring

### 📊 Observability

- Real-time metrics (request volume, latency, errors)
- LLM-specific metrics (guardrail violations)
- Data drift detection between training and production data
- Grafana dashboards with 95th percentile latency tracking

---

## Prerequisites

### Required Accounts & Keys

1. **AWS Account** with:
   - EC2 access
   - S3 bucket access
   - IAM credentials

2. **Gemini API Key**:
   - Sign up at [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate an API key

### Local Development

- Docker & Docker Compose installed
- Python 3.9+
- Git

---

## Deployment Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/z-aqib/product-review-analyzer
cd product-review-analyzer/src/llm
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
```

### Step 3: AWS S3 Setup

Upload required files to S3:

```bash
# Example S3 structure
s3://your-bucket/
├── embeddings/
│   └── index.faiss
├── datasets/
│   ├── products.csv
│   └── current_data.csv  # For Evidently drift detection
```

Update your code to reference the correct S3 bucket name.

### Step 4: Local Deployment

```bash
# Build and start all services
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Verify services are running
docker-compose ps
```

### Step 5: AWS EC2 Deployment

#### Launch EC2 Instance

- **Instance Type**: `t3.medium` (2 vCPU, 4 GB RAM)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 20 GB gp3 EBS

#### Configure Security Group

Open the following ports:

| Port | Service | Protocol |
|------|---------|----------|
| 22 | SSH | TCP |
| 8501 | Streamlit | TCP |
| 8001 | FastAPI | TCP |
| 7000 | Evidently | TCP |
| 3001 | Grafana | TCP |
| 9091 | Prometheus | TCP |

#### SSH into EC2 and Deploy

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@13.60.49.144

# Install Docker
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu

# Clone and deploy
git clone https://github.com/z-aqib/product-review-analyzer
cd product-review-analyzer/src/llm

# Create .env file with your credentials
nano .env

# Deploy
docker-compose up --build -d
```

### Step 6: Access Services

| Service | URL |
|---------|-------------------------------------|
| Streamlit UI | `http://13.60.49.144:8501` |
| FastAPI Docs | `http://13.60.49.144:8001/docs` |
| Grafana | `http://13.60.49.144:3001` |
| Prometheus | `http://13.60.49.144:9091` |
| Evidently Report | `http://13.60.49.144:7000/report` |

**Default Grafana Credentials**: `admin` / `admin`

---

## API Documentation

### Base URL

```
http://13.60.49.144:8001
```

### Endpoints

#### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-07T10:30:00Z"
}
```

#### 2. RAG Query (Main Pipeline)

```bash
POST /query
Content-Type: application/json

{
  "user_id": "AG3D6O4STAQKAY2UVGEUV46KN35Q",
  "query": "I want a Dell laptop for programming under $1500"
}
```

**Response:**
```json
{
  "user_query": "I want a Dell laptop for programming under $1500",
  "ml_candidates": [
    {
      "product_id": "B08XYZ123",
      "product_name": "Dell XPS 15",
      "score": 0.87
    }
  ],
  "rag_result": {
    "products": [
      {
        "product_id": "B08XYZ123",
        "name": "Dell XPS 15",
        "price": 1399.99,
        "rating": 4.5,
        "retrieval_score": 0.92,
        "document": "Great laptop for programming..."
      }
    ],
    "rag_answer": "Based on reviews, the Dell XPS 15..."
  },
  "final_answer": "I recommend the Dell XPS 15 for your programming needs..."
}
```

#### 3. Refresh Evidently Report

```bash
POST /refresh
```

Regenerates the data drift report by comparing reference and current datasets.

**Response:**
```json
{
  "status": "success",
  "message": "Drift report regenerated"
}
```

#### 4. View Drift Report

```bash
GET /report
```

Returns HTML page with Evidently data drift analysis.

---

## CI/CD Pipeline

This project implements a comprehensive **GitHub Actions CI/CD pipeline** that automates testing, building, and deployment.

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                     GITHUB ACTIONS WORKFLOW                     │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  1. LINT         │
    │  - Ruff          │
    │  - Black         │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  2. TEST         │
    │  - pytest        │
    │  - Coverage 25%+ │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  3. PROMPT EVAL  │
    │  - LLM tests     │
    │  - Gemini API    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  4. BUILD        │
    │  - Docker image  │
    │  - Push to GHCR  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  5. CANARY       │
    │  - Deploy test   │
    │  - Health check  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  6. ACCEPTANCE   │
    │  - Golden tests  │
    │  - Query checks  │
    └──────────────────┘
```

### Job Details

#### 1. **Lint** 🧹
- **Tools**: Ruff (linting) + Black (formatting)
- **Purpose**: Enforce code quality standards
- **Command**:
  ```bash
  ruff check src --extend-exclude "rag/experiments,sft"
  black --line-length=100 --check src
  ```

#### 2. **Test** 🧪
- **Tools**: pytest + coverage
- **Requirement**: Minimum 25% code coverage
- **Command**:
  ```bash
  pytest --cov=src --cov-report=xml --cov-fail-under=25
  ```

#### 3. **Prompt Evaluation** 🧪
- **Purpose**: Test LLM experiments and prompt engineering strategies
- **Uses**: Gemini API + fine-tuned Qwen model for validation
- **Environment**: CI mode with limited test cases (full suite runs locally)
- **Metrics Tracked**: Token overlap F1, latency, response quality
- **Command**:
  ```bash
  python -m src.llm.experiments.run_experiments
  ```
- See [Prompt Engineering & Evaluation](#prompt-engineering--evaluation) section for details

#### 4. **Build & Push** 🐳
- **Registry**: GitHub Container Registry (GHCR)
- **Tags**:
  - `latest` (most recent)
  - `<commit-sha>` (version-specific)
- **Command**:
  ```bash
  docker build -t ghcr.io/<repo>:latest .
  docker push ghcr.io/<repo>:latest
  ```

#### 5. **Canary Deployment** 🚀
- **Purpose**: Deploy to test environment
- **Validation**: Health endpoint check (`/health`)
- **Timeout**: 50 seconds with retries
- **Command**:
  ```bash
  docker run -d --name rag-api-canary -p 8000:8000 <image>
  curl http://localhost:8000/health
  ```

#### 6. **Acceptance Tests** ✅
- **Purpose**: Validate golden test queries
- **Test Cases**:
  - Laptop query: "Dell laptop for programming under 150k"
  - Phone query: "Budget Android phone with good battery"
- **Endpoint**: `/recommend`
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/recommend" \
    -H "Content-Type: application/json" \
    -d '{
      "user_id": "AG3D6O4STAQKAY2UVGEUV46KN35Q",
      "user_query": "I want a Dell laptop for programming"
    }'
  ```

### Pipeline Triggers

- **Push to main**: Full pipeline runs
- **Pull Requests**: Full validation before merge

### Required Secrets

Configure in GitHub repository settings:

```
GEMINI_API_KEY       # For LLM testing and experiments
GITHUB_TOKEN         # Auto-provided for GHCR access
```

### Local CI Testing

Run CI checks locally before pushing:

```bash
# Lint
ruff check src
black --line-length=100 --check src

# Test
pytest --cov=src --cov-fail-under=25

# Build Docker
docker build -t rag-api-local .
docker run -d -p 8000:8000 rag-api-local

# Health check
curl http://localhost:8000/health
```

---

## Prompt Engineering & Evaluation

This project implements a **comprehensive prompt engineering framework** with automated evaluation and experimentation.

### Overview

The prompt evaluation system tests multiple prompting strategies, tracks quantitative metrics, and integrates with MLflow for experiment tracking.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROMPT EVALUATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  Experiment Config   │
    │  (CSV with variants) │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Load Eval Dataset  │
    │   (eval.jsonl)       │
    │   - scenario_id      │
    │   - ideal_answer     │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  For Each Experiment:                │
    │  1. Build prompt with strategy       │
    │  2. Run ML + RAG pipeline            │
    │  3. Generate response (Qwen/Gemini)  │
    │  4. Compute metrics                  │
    │  5. Log to MLflow                    │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │         Results & Metrics            │
    │  - experiment_results.csv            │
    │  - MLflow tracking UI                │
    │  - Token overlap F1 scores           │
    │  - Latency measurements              │
    └──────────────────────────────────────┘
```

### Prompting Strategies

The system supports multiple prompting strategies for comparison:

#### 1. **Zero-Shot**
- **Description**: Direct instruction without examples
- **Use Case**: Baseline performance
- **Example**:
  ```
  You are an expert product advisor. Use the ML ranking and RAG
  snippets to recommend the best product(s) for the user.
  ```

#### 2. **Few-Shot (3-shot / 5-shot)**
- **Description**: Include example conversations before the query
- **Use Case**: Improve response quality and consistency
- **Configuration**: `few_shot_k=3` or `few_shot_k=5`
- **Example**:
  ```
  Here are example conversations:

  Example 1:
  User: I need a laptop for gaming
  Assistant: Based on the reviews, I recommend...

  [2-4 more examples]

  Now answer the next user query...
  ```

#### 3. **Chain-of-Thought (CoT)**
- **Description**: Encourages step-by-step reasoning
- **Use Case**: Complex comparisons requiring explicit trade-off analysis
- **Example**:
  ```
  When answering, first think step-by-step about the options using
  the ML and RAG information. Explicitly compare the top candidates.
  Then end with a short section titled 'Final Recommendation'.
  ```

#### 4. **Meta-Prompting**
- **Description**: Structured output format with pros/cons
- **Use Case**: Clear, scannable recommendations
- **Example**:
  ```
  You are a brutally honest product advisor. You must:
  - Pick at most 1-2 main options
  - Clearly list pros and cons based on reviews
  - Call out if information is missing

  Output format:
  1. Short answer (1-2 sentences)
  2. Bullet list of pros and cons
  3. Final recommendation
  ```

### Experiment Configuration

Experiments are defined in CSV format with the following structure:

```csv
experiment_id,strategy,few_shot_k,sample_type,scenario_id,user_query_variant,user_id
exp_001,zero_shot,0,,laptop_01,I need a Dell laptop for programming,AG3D6O4...
exp_002,few_shot,3,laptop,laptop_01,I need a Dell laptop for programming,AG3D6O4...
exp_003,cot,0,,laptop_01,I need a Dell laptop for programming,AG3D6O4...
exp_004,meta,0,,phone_01,Budget Android phone with good camera,AG3D6O4...
```

**Key Fields:**
- `experiment_id`: Unique identifier for the experiment
- `strategy`: Prompting strategy (zero_shot, few_shot, cot, meta)
- `few_shot_k`: Number of examples (0, 3, or 5)
- `sample_type`: Filter examples by category (laptop, phone, etc.)
- `scenario_id`: Links to eval.jsonl for quantitative metrics
- `user_query_variant`: The query to test
- `user_id`: User ID for ML recommender

### Evaluation Metrics

#### Automated Metrics

1. **Token Overlap F1**
   - Compares generated response with ideal answer from eval.jsonl
   - Formula: `F1 = 2 * (precision * recall) / (precision + recall)`
   - Normalized token-level comparison
   - Range: 0.0 - 1.0 (higher is better)

2. **Latency (seconds)**
   - End-to-end response time
   - Includes ML + RAG + LLM generation
   - Tracked per experiment

3. **Response Length**
   - Character count of generated response
   - Helps identify overly verbose or too-short answers

#### Manual Metrics (Human Evaluation)

After automated runs, team members manually score:

1. **Helpfulness Score (1-5)**
   - 1: Not helpful, missing key info
   - 3: Adequate, covers basics
   - 5: Extremely helpful, actionable insights

2. **Factuality Score (1-5)**
   - 1: Contains incorrect information
   - 3: Mostly accurate, minor issues
   - 5: Fully accurate, well-grounded in data

### Evaluation Dataset (eval.jsonl)

Located at `data/eval.jsonl`, each line contains:

```json
{
  "scenario_id": "laptop_01",
  "user_query": "I need a Dell laptop for programming under $1500",
  "ideal_answer": "Based on your requirements, I recommend the Dell XPS 15...",
  "context": "Programming laptop, budget $1500, Dell preferred"
}
```

### MLflow Integration

All experiments are logged to MLflow for tracking and comparison:

**Logged Parameters:**
- `strategy`: Prompting strategy used
- `few_shot_k`: Number of examples
- `sample_type`: Example category filter
- `scenario_id`: Evaluation scenario

**Logged Metrics:**
- `latency_seconds`: Response time
- `metric_token_overlap_f1`: Automated quality score

**Accessing MLflow UI:**
```bash
# Start MLflow server
mlflow ui --port 5000

# Open in browser
# http://localhost:5000
```

### Running Experiments

#### Local (Full Suite)

```bash
# Run all experiments from experiments_config.csv
python -m src.llm.experiments.run_experiments

# View results
cat src/llm/experiments/experiment_results.csv

# Start MLflow UI to compare runs
mlflow ui
```

#### CI Mode (Subset)

In CI/CD, a smaller config (`experiments_config_ci.csv`) runs automatically:

```bash
# Triggered by GitHub Actions
# Uses experiments_config_ci.csv for speed
CI=true python -m src.llm.experiments.run_experiments
```

### Results Analysis

Results are saved to `src/llm/experiments/experiment_results.csv` with columns:

| Column | Description |
|--------|-------------|
| experiment_id | Unique experiment identifier |
| strategy | Prompting strategy used |
| few_shot_k | Number of few-shot examples |
| user_query_variant | The query tested |
| response_text | Full LLM response |
| latency_seconds | Response time |
| metric_token_overlap_f1 | Automated quality score |
| helpfulness_score | Manual rating (1-5) |
| factuality_score | Manual rating (1-5) |
| error | Error message if failed |

#### Sample Analysis Workflow

```python
import pandas as pd

# Load results
df = pd.read_csv("src/llm/experiments/experiment_results.csv")

# Compare strategies by F1 score
strategy_performance = df.groupby("strategy")["metric_token_overlap_f1"].mean()
print(strategy_performance)

# Find best performing experiments
top_experiments = df.nlargest(5, "metric_token_overlap_f1")
print(top_experiments[["experiment_id", "strategy", "metric_token_overlap_f1"]])

# Analyze latency by strategy
latency_by_strategy = df.groupby("strategy")["latency_seconds"].describe()
print(latency_by_strategy)
```

### Few-Shot Example Management

Sample responses are stored in `sample_responses.json`:

```json
{
  "items": [
    {
      "type": "laptop",
      "query": "I need a laptop for programming",
      "response": "Based on your needs, I recommend...",
      "review_of_response": "This is a great response because..."
    },
    {
      "type": "phone",
      "query": "Budget phone with good battery",
      "response": "For battery life on a budget...",
      "review_of_response": "Helpful and specific"
    }
  ]
}
```

**Selection Logic:**
- Filter by `sample_type` (laptop, phone, etc.)
- Exclude negative examples (reviews containing "not very good")
- Select top-k most relevant examples

### Fine-Tuned Model Integration

The system uses a **fine-tuned Qwen model** hosted on Hugging Face Spaces:

**Configuration:**
```python
SPACE_ID = "MuhammadHaaris/mlops"
SPACE_API_NAME = "/predict"
```

**Calling the Model:**
```python
from gradio_client import Client

client = Client(SPACE_ID)
result = client.predict(
    user_input=final_prompt,
    api_name=SPACE_API_NAME
)
```

**Retry Logic:**
- 3 attempts with exponential backoff
- 5-second delay between retries
- Graceful error handling and logging

### Best Practices

1. **Start with Zero-Shot**: Establish baseline performance
2. **Add Few-Shot Gradually**: Test 3-shot before 5-shot
3. **Use CoT for Complex Queries**: Multi-product comparisons benefit from step-by-step reasoning
4. **Validate with eval.jsonl**: Always include scenario_id for quantitative metrics
5. **Manual Review**: Automated metrics don't capture all quality aspects
6. **Track in MLflow**: Compare experiments systematically
7. **CI Integration**: Keep CI config small (5-10 experiments) for fast feedback

### Extending the Framework

**Adding New Strategies:**

Edit `build_llm_prompt_for_strategy()` in `run_experiments.py`:

```python
if strategy == "your_new_strategy":
    prompt += "\n\nYour custom instruction here..."
```

**Adding New Metrics:**

Extend the results schema and computation logic:

```python
# In run_experiments.py
def compute_your_metric(response: str, reference: str) -> float:
    # Your metric logic
    return score

# Add to result_row
result_row["your_metric_name"] = compute_your_metric(response_text, ideal_answer)
```

---

## Monitoring & Observability

### Prometheus Metrics

Access at: `http://13.60.49.144:9091`

#### Available Metrics

**LLM-Specific Metrics:**

```promql
# Total LLM requests
llm_requests_total

# LLM latency histogram
llm_request_latency_seconds_bucket

# Guardrail violations
llm_guardrail_violations_total{type="input_validation"}
llm_guardrail_violations_total{type="output_moderation"}
```

**System Metrics:**

```promql
# Python runtime
python_gc_objects_collected_total
python_gc_collections_total
process_resident_memory_bytes
process_cpu_seconds_total

# HTTP metrics
http_requests_total
http_request_duration_seconds
```

### Grafana Dashboards

Access at: `http://13.60.49.144:3001`

#### Key Panels

1. **Total LLM Requests**
   - Query: `llm_requests_total`

2. **LLM Latency (95th Percentile)**
   - Query: `histogram_quantile(0.95, sum(rate(llm_request_latency_seconds_bucket[5m])) by (le))`

3. **Guardrail Violations**
   - Input: `llm_guardrail_violations_total{type="input_validation"}`
   - Output: `llm_guardrail_violations_total{type="output_moderation"}`

4. **System Health**
   - Memory: `process_resident_memory_bytes`
   - CPU: `rate(process_cpu_seconds_total[5m])`

---

## Data Drift Detection

### Evidently Configuration

Evidently monitors **data drift** by comparing reference (training) and current (production) datasets.

#### What is Monitored

1. **Data Drift**: Statistical changes in feature distributions
2. **Dataset Statistics**: Missing values, outliers, data quality
3. **Feature-Level Drift**: Individual feature distribution shifts

**Note: Current implementation focuses on data drift. Concept drift** (changes in the relationship between features and target) is not yet implemented but can be added as a future enhancement.

#### Drift Detection Workflow

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load datasets
ref = pd.read_csv("data/raw/amazon.csv")           # Reference (training)
cur = pd.read_csv("data/raw/amazon_current.csv")   # Current (production)

# Generate report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

# Save HTML report
report.save_html("monitoring/evidently_report.html")
```

#### Accessing Drift Reports

1. **Static Report** (generated locally):
   ```bash
   python generate_drift.py
   open monitoring/evidently_report.html
   ```

2. **Dynamic Report** (via API):
   - Regenerate: `POST http://13.60.49.144:7000/refresh`
   - View: `GET http://13.60.49.144:7000/report`

#### Drift Metrics Tracked

| Metric | Description |
|--------|-------------|
| Dataset Drift | Overall drift detection across all features |
| Feature Drift | Per-feature statistical drift (Kolmogorov-Smirnov test) |
| Missing Values | Changes in data completeness |
| Correlations | Changes in feature relationships |

#### S3 Integration

- **Reference Dataset**: Stored in GitHub repo (baseline)
- **Current Dataset**: Stored in S3 (updated daily/weekly)
- **Benefit**: Enables continuous drift monitoring without manual data uploads

### How to Use Evidently

Access at: `http://13.60.49.144:7000/report`

**What It Monitors:**

- **Dataset Drift**: Compares production data against reference (training) data
- **Feature Distribution Changes**: Detects shifts in product attributes
- **Data Quality**: Identifies missing values, outliers

**Workflow:**

1. **Initial Setup**: Reference dataset stored in GitHub repo
2. **Production Data**: Current dataset stored in S3
3. **Refresh Report**: POST to `/refresh` to regenerate
4. **View Report**: Navigate to `/report` in browser

**Example:**

```python
# Trigger report regeneration
import requests

response = requests.post("http://13.60.49.144:7000/refresh")
print(response.json())

# View in browser
# Open: http://13.60.49.144:7000/report
```

---

## Guardrails & Safety

### Input Validation

Protects against malicious or unsafe user queries.

#### Checks Performed

1. **Prompt Injection Detection**
   - Blocks: "ignore previous instructions", "act as an unfiltered model"
   - Action: Hard fail (query rejected)

2. **PII Detection**
   - Detects: Emails, phone numbers
   - Action: Soft fail (flagged but may proceed)

#### Code Example

```python
from guards.policy import validate_input_query, GuardrailViolation

try:
    report = validate_input_query(user_query)
    print("Input safe:", report)
except GuardrailViolation as e:
    print(f"Blocked: {e.kind} - {e.details}")
```

### Output Moderation

Ensures generated responses are safe and appropriate.

#### Checks Performed

1. **Toxicity Filter**
   - Detects: Harmful language, profanity
   - Keywords: "kill", "hate you", "stupid"
   - Action: Sanitize or block

#### Code Example

```python
from guards.policy import moderate_output_text, GuardrailViolation

try:
    safe_output = moderate_output_text(llm_response)
    print("Safe output:", safe_output["text"])
except GuardrailViolation as e:
    print(f"Moderated: {e.kind}")
```

### Pipeline Integration

```
USER QUERY
    │
    ▼
Input Validation (Guardrails) ──► Reject / Flag PII
    │
    ▼
RAG Retrieval & LLM Advisor
    │
    ▼
Output Moderation (Guardrails) ──► Sanitize / Block toxic content
    │
    ▼
FINAL RESPONSE
```

### Monitoring Guardrail Events

All violations are logged with:

```json
{
  "kind": "input_prompt_injection",
  "message": "Query contains prompt-injection style instructions",
  "details": {"pattern": "(?i)ignore previous instructions"}
}
```

These events are tracked in Prometheus:

```promql
llm_guardrail_violations_total{type="input_validation"}
llm_guardrail_violations_total{type="output_moderation"}
```

---

## Cloud Infrastructure

### AWS Services Used

#### 1. EC2 (Compute)

- **Purpose**: Host entire MLOps pipeline
- **Instance Type**: `t3.medium` (2 vCPU, 4 GB RAM)
- **Components Running**:
  - Streamlit UI
  - FastAPI backend
  - RAG pipeline
  - Prometheus, Grafana, Evidently

#### 2. S3 (Storage)

- **Purpose**: Cloud storage for datasets and artifacts
- **Contents**:
  - FAISS embeddings index
  - Product datasets
  - Current data for drift detection

#### 3. Security Groups

Configured ports for external access (see deployment section).

### Docker Compose Services

```yaml
services:
  llmops_app:
    # FastAPI backend + RAG + ML
    ports: ["8001:8001"]

  streamlit_app:
    # User interface
    ports: ["8501:8501"]

  prometheus:
    # Metrics collection
    ports: ["9091:9090"]

  grafana:
    # Visualization
    ports: ["3001:3000"]

  evidently_app:
    # Data drift monitoring
    ports: ["7000:7000"]
```

### Stateless Design

- EC2 instance pulls all data from S3 at startup
- No persistent local storage required
- Easy to redeploy or scale

---

Fine-Tuned Model & Hugging Face Integration

We didn't just rely on generic models — we trained our own! 🚀

We performed **Supervised Fine-Tuning (SFT)** on the **Qwen2.5-7B** architecture using a custom dataset of query-response pairs derived from our Amazon reviews data. This ensures the model speaks the language of e-commerce natively!

#### 🤖 The Model (Qwen2.5-7B-ProductReviewAnalyzer-SFT)

You can download, use, or evaluate our fine-tuned weights directly from Hugging Face.

> 👉 **[Access the SFT Model on Hugging Face](https://huggingface.co/MuhammadHaaris/Qwen2.5-7B-ProductReviewAnalyzer-SFT-FP16)**

**Model Details**:
- **Base Model**: Qwen2.5-7B
- **Training Data**: Custom query-response pairs from Amazon product reviews
- **Precision**: FP16 for efficient inference
- **Use Case**: Product recommendation and review analysis

**Download & Use**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "MuhammadHaaris/Qwen2.5-7B-ProductReviewAnalyzer-SFT-FP16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate response
inputs = tokenizer("What's a good laptop for programming?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### 🌌 Live Demo (Hugging Face Spaces)

Want to try it without coding? We hosted the fine-tuned model on a **Gradio** interface for interactive testing.

> 👉 **[Try the Live Demo Here](https://huggingface.co/spaces/MuhammadHaaris/mlops)**

**Features**:
- Interactive chat interface
- Real-time product recommendations
- No setup required — just start asking questions!

**Example Queries**:
- "Recommend a budget gaming laptop"
- "Best phone with long battery life under $500"
- "Compare Dell XPS vs MacBook for programming"

#### Integration in Project

The fine-tuned model is integrated in our prompt evaluation framework:

```python
from gradio_client import Client

SPACE_ID = "MuhammadHaaris/mlops"
SPACE_API_NAME = "/predict"

client = Client(SPACE_ID)
result = client.predict(
    user_input=prompt,
    api_name=SPACE_API_NAME
)
```

**Location**: `src/llm/experiments/run_experiments.py`

**Benefits**:
- Domain-specific knowledge for product recommendations
- Better understanding of e-commerce terminology
- Improved response quality compared to generic models
- Cost-effective inference with FP16 precision

---

### Code Standards

- **Linting**: Use Ruff for Python linting
- **Formatting**: Use Black (line length: 100)
- **Testing**: Maintain minimum 25% code coverage
- **Documentation**: Update README for new features
- **Commits**: Use conventional commit messages

---

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check logs
docker-compose logs -f

# Restart specific service
docker-compose restart llmops_app

# Rebuild from scratch
docker-compose down
docker-compose up --build -d
```

#### 2. Gemini API Errors

**Error**: `GEMINI_API_KEY is not set`

**Solution**: Check `.env` file exists and contains valid key

```bash
# Verify .env
cat .env | grep GEMINI_API_KEY

# Restart services after updating .env
docker-compose down
docker-compose up -d
```

#### 3. S3 Access Issues

**Error**: `Unable to download from S3`

**Solution**: Verify AWS credentials and bucket permissions

```bash
# Test AWS credentials
aws s3 ls s3://your-bucket/

# Update .env with correct credentials
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

#### 4. Out of Memory on EC2

**Symptom**: Services crashing, slow response

**Solution**: Monitor memory usage

```bash
# Check memory
free -h

# Check Docker stats
docker stats

# Consider upgrading to t3.large (8 GB RAM)
```

#### 5. Port Already in Use

**Error**: `Port 8501 is already allocated`

**Solution**: Kill existing processes or change ports

```bash
# Find process using port
sudo lsof -i :8501

# Kill process
sudo kill -9 <PID>

# Or change port in docker-compose.yml
```

### Health Check Commands

```bash
# Check all services status
docker-compose ps

# Test FastAPI
curl http://localhost:8001/health

# Test Prometheus targets
curl http://localhost:9091/api/v1/targets

# View Grafana datasources
curl http://admin:admin@localhost:3001/api/datasources
```

### Logs Location

```bash
# Application logs
docker-compose logs llmops_app

# Prometheus logs
docker-compose logs prometheus

# Grafana logs
docker-compose logs grafana
```

---

## Project Structure

```
product-review-analyzer/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│
├── M1images/
│
├── M2images/
│
├── notebooks/
│   ├── data-clean-product-names.ipynb
│   ├── data-cleaning.ipynb
│   ├── data-extract-product-names.ipynb
│   ├── data-remove-non-pakistani.ipynb
│   ├── data-rename-brands.ipynb
│   ├── item-item-collabfiltering.ipynb
│   ├── item-item-collabfiltering.py
│   └── rag_with_gemini_api.ipynb
│
├── src/
│   ├── app/
│   │   ├── item-item-collabfiltering.py
│   │   └── main.py
│   │
│   ├── guards/
│   │   ├── _pycache_/
│   │   ├── guardrails.md
│   │   ├── policy.py
│   │   └── _init_.py
│   │
│   ├── llm/
│   │   ├── experiments/
│   │   │   ├── experiments_config.csv
│   │   │   ├── experiments_config_ci.csv
│   │   │   ├── experiments_results.txt
│   │   │   ├── experiment_results.csv
│   │   │   ├── run_experiments.py
│   │   │   ├── sample_responses.json
│   │   │   └── test_run_hf.py
│   │   │
│   │   ├── grafana/
│   │   │   ├── dashboards/
│   │   │   │   └── dashboard-llm.json
│   │   │   └── provisioning/
│   │   │       ├── dashboards/
│   │   │       │   └── dashboards.yml
│   │   │       └── datasources/
│   │   │           └── prometheus.yml
│   │   │
│   │   ├── prometheus/
│   │   │   └── prometheus.yml
│   │   │
│   │   ├── src/
│   │   │   └── llm/
│   │   │       ├── grafana/
│   │   │       │   ├── dashboards/
│   │   │       │   └── provisioning/
│   │   │       ├── grafana-dashboards/
│   │   │       └── prometheus/
│   │   │           └── prometheus.yml
│   │   │
│   │   ├── _pycache_/
│   │   ├── .env
│   │   ├── advisor.py
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.evidently
│   │   └── Dockerfile.frontend
│   │
│   ├── ml/
│   │   ├── eval/
│   │   │   ├── eval_dataset.py
│   │   │   └── metrics.py
│   │   │
│   │   ├── grafana-dashboards/
│   │   │   ├── dashboard-1.json
│   │   │   └── dashboard-1.png
│   │   │
│   │   ├── prometheus/
│   │   │   └── prometheus.yml
│   │   │
│   │   ├── recommenders/
│   │   │   ├── _pycache_/
│   │   │   └── item_item.py
│   │   │
│   │   ├── _pycache_/
│   │   ├── docker-compose.yml
│   │   └── service.py
│   │
│   ├── monitoring/
│   │   ├── _pycache_/
│   │   ├── cloud.md
│   │   ├── evidently_app.py
│   │   ├── evidently_report.html
│   │   ├── generate_drift.py
│   │   └── monitoring.md
│   │
│   ├── rag/
│   │   ├── experiments/
│   │   │   ├── 1-embedding-model-comparison-py.ipynb
│   │   │   ├── 2-generation-model-comparison-mistral-py.ipynb
│   │   │   ├── 2-generation-model-comparison-Qwen-py.ipynb
│   │   │   ├── 3-top-k-retrieval-ablation-py.ipynb
│   │   │   ├── 4-temperature-ablation-py.ipynb
│   │   │   ├── 5-max-new-tokens-test-py.ipynb
│   │   │   ├── 6-top-p-nucleus-sampling-test-py.ipynb
│   │   │   ├── 7-prompt-engineering-comparison-py.ipynb
│   │   │   ├── 8-context-chunking-strategy-py.ipynb
│   │   │   ├── 9-reranking-with-cross-encoder-py.ipynb
│   │   │   ├── 10-similarity-metric-comparison-py.ipynb
│   │   │   └── grid-search-hyperparameter-selection.ipynb
│   │   │
│   │   ├── results/
│   │   │   ├── grid search results/
│   │   │   │   ├── rag_grid_search_results_advanced.csv
│   │   │   │   ├── rag_model_summary_advanced.csv
│   │   │   │   └── rag_statistics_summary.csv
│   │   │   ├── 1_embedding_comparison.csv
│   │   │   ├── 2_generation_model_comparison_mistral.csv
│   │   │   ├── 2_generation_model_comparison_Qwen.csv
│   │   │   ├── 3_top_k_retrieval_ablation.csv
│   │   │   ├── 4_temperature_ablation.csv
│   │   │   ├── 5_max_new_tokens_test.csv
│   │   │   ├── 6_top_p_nucleus_sampling_test.csv
│   │   │   ├── 7_prompt_engineering_comparison.csv
│   │   │   ├── 8_context_chunking_strategy.csv
│   │   │   ├── 9_reranking_with_cross_encoder.csv
│   │   │   └── 10_similarity_metric_comparison.csv
│   │   │
│   │   ├── _pycache_/
│   │   ├── ingest.py
│   │   ├── rag.py
│   │   └── rag_service.py
│   │
│   ├── sft/
│   │   ├── complete_sft.ipynb
│   │   ├── query-response-pairs-SFT-training-data.csv
│   │   └── running-hugging-face-model.ipynb
│   │
│   ├── tests/
│   │   ├── test_metrics_and_policy.py
│   │   └── _init_.py
│   │
│   ├── _pycache_/
│   ├── app.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── requirements-evidently.txt
│   ├── requirements.txt
│   ├── requirements_frontend.txt
│   ├── streamlit_app.py
│   ├── test.py
│   ├── train.py
│   └── _init_.py
│
├── tests/
│   ├── small_eval_test.py
│   ├── test_prompts.py
│   └── test_rag_api.py
│
├── .gitignore
├── contribution.md
├── Makefile
├── README.md
├── requirements-all.txt
├── SECURITY.md
└── setup.md
```

---

## Performance Considerations

### Resource Usage

- **Memory**: ~3.5 GB under normal load
- **CPU**: ~40% utilization on t3.medium
- **Storage**: Embeddings (~500 MB), datasets (~100 MB)

### Optimization Tips

1. **FAISS Index**: Use IVF (Inverted File) indices for larger datasets
2. **Batch Processing**: Process multiple queries in parallel
3. **Caching**: Cache frequently requested products
4. **Connection Pooling**: Reuse database connections
5. **Async I/O**: Use async/await for I/O-bound operations

### Scaling Considerations

- **Horizontal Scaling**: Deploy multiple EC2 instances behind load balancer
- **Database**: Consider PostgreSQL with pgvector for production
- **Embeddings**: Pre-compute and cache product embeddings
- **Rate Limiting**: Implement API rate limits to prevent abuse

---

### Acknowledgments

This project leverages several excellent open-source technologies:

- **Gemini API** by Google for LLM capabilities
- **Sentence Transformers** for embedding generation
- **FAISS** by Meta for efficient similarity search
- **Evidently AI** for drift detection
- **Prometheus & Grafana** for observability
- **LangChain** & **LlamaIndex**: RAG frameworks
- **FastAPI**: Modern Python web framework
- **Streamlit**: Rapid UI development
- **MLflow**: Experiment tracking

## 🎁 Bonus Features

### 1. LangChain + Custom Retriever Implementation

We implemented a complete **LangChain-based RAG pipeline** as an alternative to our main RAG system, showcasing advanced retrieval capabilities.

**Location**: `src/rag/langchain_rag.py`

**Features**:
- Uses the **exact same Amazon product dataset** and BGE embeddings as the main RAG
- Implements a **custom retriever** `RatingAwareRetriever` that:
  - Performs similarity search
  - Automatically prioritizes products with **rating ≥ 3.8**
  - Falls back gracefully if not enough high-rated items are found
- Provides `ask_langchain(question: str, k: int = 5)` — same output format as original RAG
- Adds `"engine": "langchain_rating_aware"` field to identify the version

**How to Run**:
```bash
# Run the LangChain RAG demo
python -m src.rag.langchain_rag
```

**Example Usage**:
```python
from src.rag.langchain_rag import ask_langchain

response = ask_langchain("I need a laptop for programming", k=5)
print(response)
# Output includes products with ratings ≥ 3.8 prioritized
```

### 2. A/B Testing Dashboard for Prompt Variants

We added **A/B testing capabilities** to compare different prompt strategies in production, complete with Grafana visualizations.

**Grafana Dashboard Panels**:

1. **Prompt Variant Usage (Bar Chart)**
   - Tracks usage count for each prompt variant
   - Metric: `prompt_variant_usage_total`
   - Horizontal bar chart showing distribution

2. **Prompt Variant Latency (Time Series)**
   - Monitors average response time per variant
   - Metric: `rate(prompt_response_seconds_sum[5m]) / rate(prompt_response_seconds_count[5m])`
   - Helps identify performance differences between strategies

**Dashboard Configuration** (`src/llm/grafana/dashboards/dashboard-llm.json`):
```json
{
  "id": 5,
  "type": "barchart",
  "title": "Prompt Variant Usage (A/B Testing)",
  "datasource": {"type": "prometheus", "uid": "prometheus"},
  "targets": [
    {
      "expr": "prompt_variant_usage_total",
      "legendFormat": "{{variant}}",
      "refId": "A"
    }
  ]
}
```

**How to Use**:
1. Access Grafana at `http://13.60.49.144:3001`
2. Navigate to "LLM Dashboard"
3. View "Prompt Variant Usage" and "Prompt Variant Latency" panels
4. Compare performance across zero-shot, few-shot, CoT, and meta-prompting strategies

---

## Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/z-aqib/product-review-analyzer
cd product-review-analyzer/src/llm
cp .env.example .env  # Add your API keys

# 2. Deploy locally
docker-compose up --build -d

# 3. Access services
# Streamlit: http://localhost:8501
# API: http://localhost:8001/docs
# Grafana: http://localhost:3001
# MLflow: http://localhost:5000

# 4. Run tests
pytest --cov=src --cov-fail-under=25

# 5. Run prompt experiments
python -m src.llm.experiments.run_experiments

# 6. Deploy to AWS EC2
# Follow the AWS deployment guide above
```

---

**Built with ❤️ by the LLMOps Team**
