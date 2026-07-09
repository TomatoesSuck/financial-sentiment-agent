"""
MCP server exposing the fine-tuned sentiment model to any MCP client
(Claude Code, Claude Desktop, ...) over stdio.

Thin wrapper over the FastAPI service — the /predict contract stays the
single source of sentiment logic. Requires the service to be running
(default http://localhost:8000, override with SENTIMENT_SERVICE_URL).

Register in Claude Code:
    claude mcp add financial-sentiment -- <repo>/.venv/bin/python <repo>/mcp_server/server.py
"""

import os
from typing import Any, Dict

import requests
from mcp.server.fastmcp import FastMCP

SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL", "http://localhost:8000").rstrip("/")

mcp = FastMCP("financial-sentiment")


@mcp.tool()
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Classify financial text as positive / negative / neutral.

    Returns label, confidence and server-side latency_ms from a
    DistilBERT+LoRA model fine-tuned on FinancialPhraseBank.
    """
    r = requests.post(f"{SENTIMENT_SERVICE_URL}/predict", json={"text": text}, timeout=10)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    mcp.run()
