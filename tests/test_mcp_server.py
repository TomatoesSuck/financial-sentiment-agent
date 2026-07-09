"""
MCP server tests.

Tool listing runs the real server over stdio — no sentiment service needed.
The end-to-end call test only runs when the FastAPI service is reachable.
"""

import asyncio
import sys

import pytest
import requests

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=["mcp_server/server.py"])


def _service_up() -> bool:
    try:
        requests.get("http://localhost:8000/docs", timeout=1)
        return True
    except requests.RequestException:
        return False


async def _with_session(fn):
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def test_lists_analyze_sentiment_tool():
    async def check(session):
        tools = await session.list_tools()
        return [t.name for t in tools.tools]

    names = asyncio.run(_with_session(check))
    assert names == ["analyze_sentiment"]


@pytest.mark.skipif(not _service_up(), reason="sentiment service not running on :8000")
def test_scores_text_through_mcp():
    async def check(session):
        return await session.call_tool(
            "analyze_sentiment",
            {"text": "Revenue declined sharply amid rising costs."},
        )

    result = asyncio.run(_with_session(check))
    assert not result.isError
    text = "".join(c.text for c in result.content if hasattr(c, "text"))
    assert "negative" in text
