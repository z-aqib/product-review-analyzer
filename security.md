# Security Policy

**Version:** 1.0
**Last Updated:** 2025
**Scope:** Product Review Analyzer - MLOps & LLMOps Milestone 2

## Table of Contents

1. [Purpose & Scope](#purpose--scope)
2. [Reporting Vulnerabilities](#reporting-vulnerabilities)
3. [Dependency & Supply-Chain Security](#dependency--supply-chain-security)
4. [Secrets Management](#secrets-management)
5. [Access Control & Infrastructure](#access-control--infrastructure)
6. [Data Protection & Privacy](#data-protection--privacy)
7. [Prompt Injection & Model Safety](#prompt-injection--model-safety)
8. [RAG-Specific Security](#rag-specific-security)
9. [Monitoring & Observability](#monitoring--observability)
10. [CI/CD Hardening](#cicd-hardening)
11. [Testing & Evaluation](#testing--evaluation)
12. [Incident Response](#incident-response)
13. [Developer Checklist](#developer-checklist)
14. [References](#references)

---

## Purpose & Scope

This document outlines security, privacy, and responsible-AI practices for the **product-review-analyzer** repository. It addresses:

- Prompt injection defenses and LLM safety guardrails
- Data privacy and PII handling
- Dependency scanning with pip-audit (failing CI on critical CVEs)
- Supply-chain security and container scanning
- Incident response and responsible disclosure

### Coverage

| Area | Components |
|------|-----------|
| **Code & Infrastructure** | `src/app.py`, `src/ingest.py`, `.github/workflows/`, `Dockerfile`, `docker-compose.yml` |
| **Data & Models** | Raw reviews, embeddings, FAISS/Chroma indices, model outputs, logs |
| **Third-Party** | AWS S3/GCP GCS, Prometheus/Grafana, MLflow/W&B, LLM providers (LangChain/LlamaIndex) |

---

## Reporting Vulnerabilities

### Security Contact

- **Primary:** Repository maintainer via [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories) (preferred) or email
- **Label:** Use `security` label on private issues
- **Email:** `security@<org>` (update before publishing)

### Timeline

- **Acknowledge:** 3 business days
- **Triage:** 7 days
- **Remediate:** 30 days (critical issues)

### How to Report

1. **Do not** open public issues for security vulnerabilities
2. Use [GitHub Security Advisories](https://github.com/YOUR-ORG/product-review-analyzer/security/advisories) or email with:
   - Summary & reproduction steps
   - Affected files/components
   - Environment details
   - Proof-of-concept (if safe)
3. **Exclude** sensitive data (API keys, PII, secrets)
4. We support responsible disclosure and provide **safe harbor** for good-faith researchers

---

## Dependency & Supply-Chain Security

### pip-audit (Python Vulnerability Scanning)

Scan for vulnerable dependencies daily and in CI/CD:

```bash
pip install pip-audit
pip-audit -r requirements.txt --format=json -o pip_audit_report.json
```

**CI Integration:** `.github/workflows/ci.yml` must fail on `CRITICAL` or `HIGH` CVEs.

### Container Image Scanning

Scan Docker images with [Trivy](https://github.com/aquasecurity/trivy):

```bash
trivy image --severity CRITICAL,HIGH <image:tag>
```

### Dependency Pinning

- Pin versions in `requirements.txt` and `pyproject.toml`
- Use `pip-audit` to detect outdated/vulnerable packages
- Update PRs reviewed before merge

### CI Policy

- ❌ Fail on `CRITICAL` CVE or `HIGH` CVE (unless explicitly waived with justification)
- ✅ Pass on `LOW` or `MEDIUM` (with tracking)

---

## Secrets Management

### No Secrets in Repository

- **Never commit:** API keys, tokens, credentials, database passwords
- **Use instead:**
  - Environment variables (`.env.local`, not in git)
  - GitHub Secrets or AWS Secrets Manager
  - Azure KeyVault or HashiCorp Vault

### Pre-Commit Hooks

Integrate `detect-secrets` or `git-secrets`:

```bash
# .pre-commit-config.yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
    - id: detect-secrets
      args: ['--baseline', '.secrets.baseline']
```

### Credential Rotation

- Rotate credentials every 90 days (or on suspected compromise)
- Use short-lived tokens where possible (e.g., AWS SigV4)
- Apply **least-privilege principle** to all credentials

---

## Access Control & Infrastructure

### RBAC & Least Privilege

- Restrict S3/GCS bucket access; no public-read on indexed documents
- Enable bucket policies, versioning, and SSE-KMS encryption
- Limit IAM roles to required permissions only

### Network Security

- Run services behind reverse proxy or API Gateway
- **Enforce TLS/HTTPS** on all API endpoints
- For cloud: restrict ingress to required IP ranges, use VPCs/subnets
- Disable public endpoints for internal services

### Container Security

- **Run as non-root user** in Dockerfile:
  ```dockerfile
  RUN useradd -m appuser
  USER appuser
  ```
- Use minimal base images (`python:3.11-slim`)
- Multi-stage builds to reduce attack surface
- Scan final image with Trivy

---

## Data Protection & Privacy

### PII Detection & Redaction

**Before indexing, classify and redact PII:**

- Names, emails, phone numbers, SSN, national IDs, credit cards
- **Tools:** Presidio, spaCy NER, regex patterns
- **Integration:** `src/ingest.py` preprocessing step
- **Logging:** Document redaction decisions for transparency

### Encryption

| Layer | Method |
|-------|--------|
| **At-Rest** | S3 SSE-KMS, database encryption, encrypted backups |
| **In-Transit** | TLS 1.3 for all network communication |
| **Key Management** | AWS KMS, Azure KeyVault; rotate keys quarterly |

### Data Retention & Deletion

- Define retention periods (e.g., 6 months for processed artifacts)
- Implement user data deletion on request
- Document in `README.md` under "Data Retention & Privacy"
- Maintain audit logs of deletion events

---

## Prompt Injection & Model Safety

### Threat Model

Adversaries may craft queries to:
- Override system instructions
- Exfiltrate secrets or sensitive data
- Generate unsafe, biased, or hallucinated outputs
- Poison the retrieval corpus

**Mitigation:** Treat both user inputs and ingested documents as untrusted sources.

### Input Validation & Sanitation

**Rules:**

1. **Length Limits:** Enforce max token/character limits per request
2. **Pattern Detection:** Reject suspicious payloads (HTML, JS, encoded attacks)
3. **Allowlist Fields:** Accept only expected request fields in API
4. **Encoding:** Normalize UTF-8; reject unusual encodings

**Implementation in `src/app.py`:**

```python
# Pseudo-code
def validate_input(prompt: str) -> bool:
    if len(prompt) > MAX_TOKENS:
        raise HTTPException(400, "Input exceeds token limit")
    if detect_html_payload(prompt) or detect_script_injection(prompt):
        raise HTTPException(400, "Suspicious input pattern")
    return True
```

### PII & Sensitive Content Detection

- Detect PII at query-time and ingestion-time
- Block or flag queries containing PII
- Return sanitized error message
- Log for compliance audit

### Prompt Injection Detector

**Heuristic Rules:**
- Detect phrases: "ignore previous instructions", "system prompt", "forget all rules"
- Detect embedded role-play attempts
- Escalate to human review or refuse request

**Optional ML-Based Detection:**
- Use a lightweight classifier to evaluate injection likelihood
- Example: sentence-transformers + cosine similarity to known injection patterns

### Output Moderation

**Checks:**

1. **Toxicity Filtering:** Use Detoxify, Perspective API (threshold: 0.7)
2. **Profanity Filters:** Block or mask offensive language
3. **Hallucination Detection:** For RAG, require citations to source documents
4. **Citation Enforcement:** Include top-K retrieved doc IDs/URLs in response

**Fallback for Low Confidence:**
```
"I'm not sure based on the available reviews. Here are the closest matches: [doc_ids]"
```

### Guardrail Enforcement Points

| Point | Action |
|-------|--------|
| **API Ingress** | Validate & normalize incoming prompts |
| **Retrieval** | Scan retrieved chunks for malicious content |
| **Generation** | Apply output filters & structured templates |
| **Logging** | Log all guardrail events to Prometheus + MLflow |

**Logging Template:**
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "user_id": "anon_hash",
  "rule_triggered": "prompt_injection",
  "prompt_length": 150,
  "action": "rejected",
  "reason": "contains 'ignore previous'"
}
```

### Implementation: Guardrails AI / NeMo Guardrails

Integrate guardrail middleware in FastAPI:

```python
from guardrails import Guard
# ... define rules ...
@app.post("/query")
async def query(prompt: str):
    guard = Guard()
    validated = guard.validate(prompt)
    if not validated.valid:
        return JSONResponse({"error": "Input validation failed"}, 400)
    # ... proceed to LLM ...
```
