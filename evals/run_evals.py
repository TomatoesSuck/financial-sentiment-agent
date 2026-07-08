"""
Trajectory evals for the agent's tool-orchestration policy.

Division of labor: tests/ is the regression gate (does the agent still run);
evals/ measures per-scenario behaviour (does it follow the policy) and records
the actual tool-call trajectory as evidence.

HTTP boundaries (NewsAPI, sentiment service) are mocked; the LLM is real.
Skips cleanly (exit 0) when OPENAI_API_KEY is absent, so CI stays green
without secrets.

Run:    .venv/bin/python -m evals.run_evals
Writes: evals/results.json
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import requests
from dotenv import load_dotenv

load_dotenv()

RESULTS_PATH = Path(__file__).parent / "results.json"


# ── mocked HTTP boundary ─────────────────────────────────────────────────────
def _news_response(articles):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"status": "ok", "articles": articles}
    r.raise_for_status.return_value = None
    return r


def _sentiment_response(label):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"label": label, "confidence": 0.9, "latency_ms": 40.0}
    r.raise_for_status.return_value = None
    return r


def _articles(n, topic="Apple"):
    return [
        {
            "title":       f"{topic} market development {i}",
            "description": f"Report {i}: analysts comment on {topic}'s recent results.",
            "url":         f"https://example.com/{i}",
            "publishedAt": "2026-06-15T08:00:00Z",
        }
        for i in range(1, n + 1)
    ]


# ── checks (each takes the run record, returns bool) ─────────────────────────
def news_called_first(rec):
    return bool(rec["timeline"]) and rec["timeline"][0] == "search_news"


def sentiment_once_per_article(rec):
    n = rec["n_articles"]
    return n <= rec["sentiment_calls"] <= n + 2


def no_sentiment_calls(rec):
    return rec["sentiment_calls"] == 0


def answer_nonempty(rec):
    return rec["error"] is None and len(rec["answer"].strip()) > 0


def answer_mentions_distribution(rec):
    return re.search(r"\d+\s+(positive|negative|neutral)", rec["answer"].lower()) is not None


def answer_says_no_news(rec):
    text = rec["answer"].lower()
    return any(p in text for p in ("could not find", "no relevant", "no recent", "no news"))


# ── scenarios ────────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    name: str
    question: str
    articles: List[Dict[str, Any]]
    labels: List[str] = field(default_factory=lambda: ["positive"])
    news_error: Optional[Exception] = None
    sentiment_error: Optional[Exception] = None
    checks: List[Callable] = field(default_factory=list)


HAPPY_CHECKS = [news_called_first, sentiment_once_per_article,
                answer_nonempty, answer_mentions_distribution]

SCENARIOS = [
    Scenario("all_positive", "What is the sentiment around Apple stock?",
             _articles(3), ["positive"], checks=HAPPY_CHECKS),
    Scenario("all_negative", "How bad is the news for Boeing right now?",
             _articles(3, "Boeing"), ["negative"], checks=HAPPY_CHECKS),
    Scenario("mixed_sentiment", "What's the current view on Tesla?",
             _articles(4, "Tesla"), ["positive", "negative", "neutral", "positive"],
             checks=HAPPY_CHECKS),
    Scenario("single_article", "Any news about Shopify?",
             _articles(1, "Shopify"), ["neutral"],
             checks=[news_called_first, sentiment_once_per_article, answer_nonempty]),
    Scenario("ten_articles", "What is the sentiment around NVIDIA?",
             _articles(10, "NVIDIA"), ["positive", "neutral"], checks=HAPPY_CHECKS),
    Scenario("empty_news_short_circuit", "Sentiment for XYZNonexistentTicker?",
             [], checks=[news_called_first, no_sentiment_calls, answer_says_no_news]),
    Scenario("sentiment_service_timeout", "How is Apple doing?",
             _articles(2), sentiment_error=requests.exceptions.Timeout("timed out"),
             checks=[news_called_first, answer_nonempty]),
    Scenario("newsapi_down", "What's happening with Amazon stock?",
             _articles(0), news_error=requests.exceptions.ConnectionError("connection refused"),
             checks=[no_sentiment_calls, answer_nonempty]),
]


# ── runner ───────────────────────────────────────────────────────────────────
def run_scenario(sc: Scenario) -> Dict[str, Any]:
    import src.agent as agent_module
    import src.tools as tools_module

    agent_module._agent = None          # fresh agent per scenario
    tools_module.NEWS_API_KEY = "eval-key"

    timeline: List[str] = []
    call_idx = {"i": 0}

    def fake_get(*args, **kwargs):
        timeline.append("search_news")
        if sc.news_error:
            raise sc.news_error
        return _news_response(sc.articles)

    def fake_post(*args, **kwargs):
        timeline.append("analyze_sentiment")
        if sc.sentiment_error:
            raise sc.sentiment_error
        label = sc.labels[call_idx["i"] % len(sc.labels)]
        call_idx["i"] += 1
        return _sentiment_response(label)

    answer, error = "", None
    with patch("src.tools.requests.get", side_effect=fake_get), \
         patch("src.tools.requests.post", side_effect=fake_post):
        try:
            answer = agent_module.run(sc.question)
        except Exception as e:  # a scenario failure, not a crash of the runner
            error = f"{type(e).__name__}: {e}"

    rec = {
        "answer":          answer,
        "error":           error,
        "timeline":        timeline,
        "news_calls":      timeline.count("search_news"),
        "sentiment_calls": timeline.count("analyze_sentiment"),
        "n_articles":      len(sc.articles),
    }
    checks = {fn.__name__: bool(fn(rec)) for fn in sc.checks}
    return {
        "scenario": sc.name,
        "passed":   all(checks.values()),
        "checks":   checks,
        "timeline": timeline,
        "error":    error,
    }


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping trajectory evals.")
        return 0

    import src.agent as agent_module

    results = []
    for sc in SCENARIOS:
        r = run_scenario(sc)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {sc.name}  checks={r['checks']}")

    n_pass = sum(r["passed"] for r in results)
    summary = {
        "model":     agent_module.OPENAI_MODEL,
        "passed":    n_pass,
        "total":     len(results),
        "scenarios": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{n_pass}/{len(results)} scenarios passed → {RESULTS_PATH}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
