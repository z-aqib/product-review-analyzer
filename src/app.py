# src/app.py

from typing import Any, Dict, List

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import run_pipeline
from .guards.policy import (
    validate_input_query,
    moderate_output_text,
    GuardrailViolation,
)

# ---------- Logging setup ----------

logger = logging.getLogger("rag_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------- FastAPI app ----------

app = FastAPI(
    title="MLOps Milestone 2 API",
    description=(
        "Unified ML + RAG + LLM advisor pipeline with basic guardrails "
        "for input validation and output moderation."
    ),
    version="2.0.0",
)


# ---------- Request / Response models ----------


class QueryRequest(BaseModel):
    user_id: str = Field(..., example="AG3D6O4STAQKAY2UVGEUV46KN35Q")
    user_query: str = Field(
        ...,
        example="I want a Dell laptop for programming under 150k with good battery.",
    )


class GuardrailEvent(BaseModel):
    type: str
    kind: str
    message: str
    details: Dict[str, Any] | None = None


class QueryResponse(BaseModel):
    user_query: str
    ml_candidates: Any
    rag_result: Any
    final_answer: str
    guardrail_events: List[GuardrailEvent]


# ---------- Health check ----------


@app.get("/health")
def health() -> dict[str, str]:
    """
    Simple health check for Docker/CI.
    """
    return {"status": "ok"}


# ---------- Endpoint ----------


@app.post("/recommend", response_model=QueryResponse)
async def recommend(request: QueryRequest) -> QueryResponse:
    guardrail_events: List[GuardrailEvent] = []

    # 1) Input validation guardrails
    try:
        input_report = validate_input_query(request.user_query)
        if input_report["flags"]:
            logger.info("Input guardrail flags: %s", input_report["flags"])
            guardrail_events.append(
                GuardrailEvent(
                    type="input_validation",
                    kind="input_flags",
                    message="Input query triggered guardrail flags.",
                    details=input_report,
                )
            )
    except GuardrailViolation as e:
        logger.warning("Input guardrail violation: %s", e, exc_info=True)
        # You can log this into Prometheus / Grafana or MLflow as a guardrail metric
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsafe_input",
                "kind": e.kind,
                "message": str(e),
                "details": e.details,
            },
        )

    # 2) Main business logic pipeline
    try:
        pipeline_result = run_pipeline(
            user_id=request.user_id,
            user_query=request.user_query,
        )
    except Exception as e:
        logger.exception("Error while running pipeline: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "Pipeline failed."},
        )

    # 3) Output moderation guardrails
    final_answer = pipeline_result.get("final_answer", "")
    try:
        output_report = moderate_output_text(final_answer)
        if output_report["flags"]:
            logger.info("Output guardrail flags: %s", output_report["flags"])
            guardrail_events.append(
                GuardrailEvent(
                    type="output_moderation",
                    kind="output_flags",
                    message="Final answer triggered guardrail flags.",
                    details=output_report,
                )
            )
        safe_answer = output_report["text"]
    except GuardrailViolation as e:
        logger.warning("Output guardrail violation: %s", e, exc_info=True)
        guardrail_events.append(
            GuardrailEvent(
                type="output_moderation",
                kind=e.kind,
                message=str(e),
                details=e.details,
            )
        )
        # Replace model answer with a generic safe message
        safe_answer = (
            "Sorry, I couldn't generate a safe response for this query. "
            "Please try rephrasing your question."
        )

    return QueryResponse(
        user_query=pipeline_result.get("user_query", request.user_query),
        ml_candidates=pipeline_result.get("ml_candidates"),
        rag_result=pipeline_result.get("rag_result"),
        final_answer=safe_answer,
        guardrail_events=guardrail_events,
    )
