"""
Financial sentiment agent.

Pipeline: user question -> search_news -> analyze_sentiment per article ->
synthesized answer with sentiment distribution.

Public entry point: `run(question: str) -> str`
Bottom of file: REPL for interactive use.
"""

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent

from src.prompts import SYSTEM_PROMPT
from src.tools import analyze_sentiment, search_news

load_dotenv()

OPENAI_MODEL = "gpt-5.4-mini"

# Bounds the LangGraph agent loop (~2 steps per tool round-trip); replaces
# AgentExecutor's max_iterations=15. 30 covers 10 articles + synthesis.
RECURSION_LIMIT = 30


def _build_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set - fill .env first.")
    return create_agent(
        model=f"openai:{OPENAI_MODEL}",
        tools=[search_news, analyze_sentiment],
        system_prompt=SYSTEM_PROMPT,
    )


_agent = None


def _final_text(messages) -> str:
    """Final answer text; handles both plain-string and content-block messages."""
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def run(question: str) -> str:
    """Answer one financial question. Builds the agent on first call (lazy)."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return _final_text(result["messages"])


if __name__ == "__main__":
    print("Financial Sentiment Agent. Type 'quit' or 'exit' to leave.")
    while True:
        try:
            q = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if q.lower() in {"quit", "exit"}:
            sys.exit(0)
        if not q:
            continue
        try:
            print("\n" + run(q))
        except Exception as e:
            print(f"[error] {e}")
