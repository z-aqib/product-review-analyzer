"""
Thin wrapper around the RAG notebook code so that the rest of the
project can simply do:

    from rag.rag_service import ask

and get a nice Python function.
"""

from .rag_with_gemini_api import ask  # noqa: F401 # re-export the ask() function
