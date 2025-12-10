# **Milestone 2 — LLMOps + RAG System**

### **D1 — Prompt Engineering Workflow**

* Folder: `experiments/prompts/`.
* Minimum **three prompt strategies**:

  * Zero-shot baseline.
  * Few-shot (k comparison, e.g., 3 vs. 5).
  * One advanced strategy (Chain-of-Thought or Meta-prompting).
* Evaluation dataset included (`data/eval.jsonl`).
* Two evaluation types:

  * Quantitative metric (BLEU / ROUGE / cosine similarity depending on task).
  * Human-in-the-loop scoring rubric (scale 1-5).
* All performance metrics logged (MLflow or Weights & Biases).
* A generated `prompt_report.md` summarizing:

  * Prompt structure and examples.
  * Quantitative and qualitative results.
  * Observed issues and failure behaviors.

### **D2 — RAG Pipeline**

* ingestion script (`src/ingest.py`) indexing documents using Chroma/FAISS/LlamaIndex.
* Inference API (`src/app.py`) implementing RAG (retrieval + LLM response).
* Two required diagrams:

  * System architecture diagram.
  * Data flow diagram showing document storage indexing and inference path.
* Makefile support: `make rag` executes ingestion + inference pipeline.

### **D3 — Guardrails & Safety**

* Content moderation implemented using:

  * Guardrails AI / NeMo / custom engine.
* At least two rule types enforced:

  * Input validation (PII detection, injection filters).
  * Output filtering (toxicity/hallucination threshold).
* Guardrail triggered events logged for monitoring.
* Document guardrail integration in architecture.

### **D4 — LLM Monitoring and Evaluation**

* Prometheus tracking model metrics including:

  * latency
  * token usage
  * cost estimate
  * guardrail events
* Grafana visualization/dashboard.
* Evidently monitoring for retriever corpus drift.
* Dashboard screenshots included in README.

### **D5 — CI/CD for LLMOps**

* Extend CI with:

  * Linting for prompt scripts.
  * Automated prompt evaluation on a small dataset.
  * Docker build & push for RAG service.
  * Canary deployment step.
* ≥80% total test coverage (unit + integration).

### **D6 — Documentation**

* Update README with:

  * LLMOps goals and overview.
  * Updated architecture diagrams.
  * Deployment instructions for RAG.
  * Example API usage.
* Add `EVALUATION.md` covering:

  * Methodology
  * Prompt comparison results
  * Insights + decisions based on findings

### **D7 — Cloud Integration (Required)**

* Use at least **two** cloud services (AWS/GCP/Azure)
* Include:

  * Setup steps
  * Screenshots of deployed services
  * Explanation of how they support the workflow

### **D8 — Security & Compliance**

* SECURITY.md including:

  * Prompt injection defense description.
  * Data privacy considerations.
* Pip-audit integrated into CI.
* Documentation of responsible AI guardrail enforcement.

### **Milestone 2 Submission**

* GitHub tag: **`v2.0-milestone2`**
* All workflows must pass on tagged commit.
