"""Optional Langfuse tracing — enabled only when LANGFUSE_* keys are set."""

import os


def langfuse_callbacks() -> list:
    """[CallbackHandler] when Langfuse keys are configured, else [] (no-op)."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
