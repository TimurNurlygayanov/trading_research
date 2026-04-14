"""
Idea Generator: fetches papers from free scientific sources, extracts tradeable
strategy ideas using LLM, and saves them to the generated_ideas table for review.

Sources:
  - arXiv q-fin.TR (Trading and Market Microstructure) — completely free, no key needed
  - Semantic Scholar API (free tier, rate-limited to ~1 req/sec)

Flow:
  run_idea_generator()
    → fetch arXiv + Semantic Scholar papers
    → deduplicate against already-seen URLs in DB
    → LLM (Haiku) extracts concrete strategy ideas from abstracts
    → save to generated_ideas table (status='pending')

The user reviews ideas at /research and clicks "Use this idea" to queue them.
"""
from __future__ import annotations

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Any

import anthropic
import requests
from dotenv import load_dotenv

from db import supabase_client as db

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_PAPERS_PER_SOURCE = 12

_SYSTEM_PROMPT = """You are a quantitative trading researcher. You read academic papers and extract concrete, testable trading strategy ideas.

For each paper that contains a tradeable strategy, extract ONE idea with:
- A clear actionable description: entry/exit signals, which indicators, timeframe, asset class
- The core insight that makes the strategy work

Return ONLY a JSON array. Each element:
{
  "paper_index": 1,
  "title": "RSI Mean Reversion on 4H EURUSD",
  "summary": "Buy when RSI(14) crosses above 30 from oversold on 4H bars, sell when RSI crosses below 70 from overbought. Use 1.5×ATR stop loss and 3×ATR take profit. Works best during London/NY overlap.",
  "asset_class": "forex",
  "confidence": "high"
}

RULES:
- asset_class must be one of: forex, equity, crypto, multi
- confidence: high = paper shows concrete backtest results, medium = idea is clear but untested, low = speculative
- Only include papers with CONCRETE trading signals (entry/exit rules). Skip pure theory, market microstructure, or regulation papers.
- summary must be specific enough that a programmer can implement it directly
- Return [] if no paper has a concrete tradeable strategy."""


def run_idea_generator() -> list[dict[str, Any]]:
    """
    Main entry point called by the scheduler every 4 hours.
    Fetches papers, extracts ideas, saves to DB.
    Returns list of inserted idea records.
    """
    all_papers: list[dict] = []

    try:
        arxiv_papers = fetch_arxiv_papers(max_results=MAX_PAPERS_PER_SOURCE)
        all_papers += arxiv_papers
        log.info(f"idea_generator: fetched {len(arxiv_papers)} arXiv papers")
    except Exception as e:
        log.warning(f"idea_generator: arXiv fetch failed: {e}")

    try:
        ss_papers = fetch_semantic_scholar_papers(max_results=MAX_PAPERS_PER_SOURCE)
        all_papers += ss_papers
        log.info(f"idea_generator: fetched {len(ss_papers)} Semantic Scholar papers")
    except Exception as e:
        log.warning(f"idea_generator: Semantic Scholar fetch failed: {e}")

    if not all_papers:
        log.warning("idea_generator: no papers fetched from any source")
        return []

    # Deduplicate against already-processed URLs
    seen_urls = _get_seen_urls()
    new_papers = [p for p in all_papers if p.get("url") not in seen_urls]
    log.info(
        f"idea_generator: {len(new_papers)} new papers "
        f"(skipping {len(all_papers) - len(new_papers)} already seen)"
    )

    if not new_papers:
        return []

    # Extract strategy ideas with LLM
    ideas = extract_strategy_ideas(new_papers)
    log.info(f"idea_generator: extracted {len(ideas)} strategy ideas")

    inserted: list[dict] = []
    for idea in ideas:
        try:
            row = db.insert_generated_idea(idea)
            inserted.append(row)
        except Exception as e:
            log.error(f"idea_generator: failed to insert idea: {e}")

    log.info(f"idea_generator: inserted {len(inserted)} new ideas")
    return inserted


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_arxiv_papers(max_results: int = 12) -> list[dict]:
    """
    Fetch recent quantitative finance papers from arXiv.
    q-fin.TR = Trading and Market Microstructure
    q-fin.PM = Portfolio Management
    """
    url = (
        "http://export.arxiv.org/api/query"
        "?search_query=cat:q-fin.TR+OR+cat:q-fin.PM"
        f"&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ns):
        title_el    = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        link_el     = entry.find("atom:id", ns)

        title    = (title_el.text or "").strip().replace("\n", " ")
        abstract = (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""
        url_str  = (link_el.text or "").strip() if link_el is not None else ""

        if not title or len(abstract) < 50 or not url_str:
            continue

        papers.append({
            "source_type":  "arxiv",
            "source_title": title,
            "abstract":     abstract[:1500],
            "url":          url_str,
            "year":         None,
        })

    return papers


def fetch_semantic_scholar_papers(max_results: int = 12) -> list[dict]:
    """
    Fetch strategy papers from Semantic Scholar (free API, no key required).
    Tries two different queries to diversify coverage.
    """
    queries = [
        "quantitative trading strategy momentum mean reversion backtest",
        "algorithmic trading machine learning alpha signal generation",
    ]
    papers: list[dict] = []

    for query in queries:
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query":  query,
                    "fields": "title,abstract,year,externalIds",
                    "limit":  max_results // 2,
                },
                timeout=20,
                headers={"User-Agent": "trading-research-idea-generator/1.0"},
            )
            if resp.status_code == 429:
                log.warning("idea_generator: Semantic Scholar rate limited, skipping")
                break
            resp.raise_for_status()

            for paper in resp.json().get("data", []):
                abstract = (paper.get("abstract") or "").strip()
                title    = (paper.get("title") or "").strip()
                if not abstract or len(abstract) < 50 or not title:
                    continue

                ext = paper.get("externalIds") or {}
                if ext.get("ArXiv"):
                    paper_url = f"https://arxiv.org/abs/{ext['ArXiv']}"
                elif ext.get("DOI"):
                    paper_url = f"https://doi.org/{ext['DOI']}"
                else:
                    pid = paper.get("paperId", "")
                    paper_url = f"https://www.semanticscholar.org/paper/{pid}"

                papers.append({
                    "source_type":  "semantic_scholar",
                    "source_title": title,
                    "abstract":     abstract[:1500],
                    "url":          paper_url,
                    "year":         paper.get("year"),
                })

            time.sleep(1.5)  # respect rate limit
        except Exception as e:
            log.warning(f"idea_generator: Semantic Scholar query failed: {e}")

    return papers[:max_results]


# ── LLM extraction ───────────────────────────────────────────────────────────

def extract_strategy_ideas(papers: list[dict]) -> list[dict]:
    """
    Pass paper titles + abstracts to LLM and extract concrete strategy ideas.
    Returns dicts ready for DB insertion.
    """
    if not papers:
        return []

    papers_text = "\n\n".join(
        f"[{i + 1}] {p['source_type'].upper()} {p.get('year') or ''}\n"
        f"TITLE: {p['source_title']}\n"
        f"ABSTRACT: {p['abstract'][:900]}\n"
        f"URL: {p['url']}"
        for i, p in enumerate(papers)
    )

    from agents.utils import call_claude
    response = call_claude(
        model=MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content":
            f"Extract tradeable strategy ideas from these {len(papers)} papers:\n\n{papers_text}"}],
    )

    usage = response.usage
    cost = (usage.input_tokens * 0.00025 + usage.output_tokens * 0.00125) / 1000
    try:
        db.log_spend("idea_generator", MODEL, usage.input_tokens, usage.output_tokens, cost, None)
    except Exception:
        pass

    raw = response.content[0].text.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        for part in raw.split("```")[1::2]:
            if part.strip().startswith("["):
                raw = part.strip()
                break

    try:
        extracted = json.loads(raw)
        if not isinstance(extracted, list):
            extracted = []
    except json.JSONDecodeError:
        log.error(f"idea_generator: JSON parse failed. Raw: {raw[:300]}")
        extracted = []

    ideas: list[dict] = []
    for item in extracted:
        idx = item.get("paper_index", 0) - 1
        if not (0 <= idx < len(papers)):
            continue
        paper  = papers[idx]
        title   = (item.get("title") or paper["source_title"])[:200]
        summary = (item.get("summary") or "").strip()
        if not summary:
            continue
        ideas.append({
            "title":        title,
            "summary":      summary[:2000],
            "source_type":  paper["source_type"],
            "source_title": paper["source_title"][:300],
            "source_url":   paper["url"],
            "asset_class":  item.get("asset_class", "multi"),
            "confidence":   item.get("confidence", "medium"),
            "status":       "pending",
        })

    return ideas


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_seen_urls() -> set[str]:
    try:
        return db.get_generated_idea_urls()
    except Exception:
        return set()
