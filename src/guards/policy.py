# src/guards/policy.py

import re
from typing import Any, Dict, List


class GuardrailViolation(Exception):
    """Raised when an input or output violates a safety policy."""

    def __init__(self, kind: str, message: str, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or {}


# ---------- Input validation ----------

_INJECTION_PATTERNS = [
    r"(?i)ignore previous instructions",
    r"(?i)disregard previous instructions",
    r"(?i)you are now a different system",
    r"(?i)pretend you are not bound by",
    r"(?i)act as an unfiltered model",
]

# Simple PII patterns: emails + phone numbers
_EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_REGEX = re.compile(r"\b(\+?\d[\d\-\s]{7,})\b")


def validate_input_query(user_query: str) -> Dict[str, Any]:
    """
    Perform input validation:
      - Check for obvious prompt injection phrases
      - Check for PII (emails, phone-like patterns)

    Returns a small report dict. Raises GuardrailViolation on hard failures.
    """
    flags: List[str] = []

    # Prompt injection patterns
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, user_query):
            flags.append("prompt_injection")
            # Hard fail: we don't want to process this query
            raise GuardrailViolation(
                kind="input_prompt_injection",
                message="Query contains prompt-injection style instructions.",
                details={"pattern": pattern},
            )

    # PII detection
    has_email = bool(_EMAIL_REGEX.search(user_query))
    has_phone = bool(_PHONE_REGEX.search(user_query))

    if has_email or has_phone:
        flags.append("pii_detected")

    return {
        "flags": flags,
        "has_email": has_email,
        "has_phone": has_phone,
    }


# ---------- Output moderation ----------

# Extremely simple “toxicity list” for demo.
# In your real project, you can expand this or call an external moderation API.
_TOXIC_KEYWORDS = [
    "kill",
    "hate you",
    "stupid",
    "idiot",
]


def moderate_output_text(text: str) -> Dict[str, Any]:
    """
    Check the LLM output for basic toxicity.

    Returns a dict with:
        {
            "flags": [...],
            "text": possibly_sanitized_text
        }

    Raises GuardrailViolation for severe cases.
    """
    lowered = text.lower()
    toxic_hits = [kw for kw in _TOXIC_KEYWORDS if kw in lowered]
    flags: List[str] = []

    if toxic_hits:
        flags.append("toxic_language")
        # Example policy: we *soft*-fail and replace the answer
        sanitized = (
            "Sorry, I couldn't generate a safe response to that query. "
            "Please try rephrasing your question."
        )
        raise GuardrailViolation(
            kind="output_toxicity",
            message="Generated answer contained toxic language.",
            details={"keywords": toxic_hits},
        )
    else:
        sanitized = text

    return {
        "flags": flags,
        "text": sanitized,
    }
