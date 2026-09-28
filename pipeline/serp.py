"""Search any question: who owns it today, what AI tells a client, and how Equi shows up.

One conversation, three steps, so the analysis reads the actual pages:
1. Claude searches the query (web_search_20250305, max_uses 2) and answers it. That answer and its
   citations are the "what AI tells a client" view.
2. If fewer than 10 unique results came back, Claude searches once more with a close variant.
3. Claude reads the returned pages (they stay in the conversation) and returns the analysis as JSON.

Results are web search via Claude, not Google rankings; production would add a rankings API and
AI Overview capture. All advice is category education ending in "ask your advisor".

    python -m pipeline.serp "what is an evergreen fund" "how to protect my portfolio in a market crash"
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from pipeline.aeo import EQUI, EVERGREEN, INDEX_NAME, INTERNAL_OK, domain
from pipeline.compliance import lint_text
from pipeline.personalize import MODEL

CACHE = Path("data/serp_cache.json")
SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
TOP_N = 10
TIMEOUT_S = 120
RETRIES = 2
LABEL = ("Top results from web search via Claude. Google rankings may differ slightly. Production would add a "
         "Google rankings API and AI Overview capture.")
SOURCE_TYPES = ["financial media", "fund sponsor", "education site", "forum", "advisor or RIA", "government"]


class SerpError(Exception):
    """A failure with a plain message the app can show as is."""


SCHEMA = {
    "type": "object",
    "properties": {
        "results": {"type": "array", "items": {
            "type": "object",
            "properties": {"rank": {"type": "integer"}, "source_type": {"type": "string", "enum": SOURCE_TYPES},
                           "summary": {"type": "string"}, "mentions_evergreen": {"type": "boolean"},
                           "mentions_equi": {"type": "boolean"}, "gets_wrong": {"type": "string"}},
            "required": ["rank", "source_type", "summary", "mentions_evergreen", "mentions_equi", "gets_wrong"],
            "additionalProperties": False}},
        "overall": {"type": "object",
                    "properties": {"owner": {"type": "string"}, "common": {"type": "string"}, "gap": {"type": "string"},
                                   "verdict": {"type": "string"}},
                    "required": ["owner", "common", "gap", "verdict"], "additionalProperties": False},
        "own_it": {"type": "object",
                   "properties": {"title": {"type": "string"}, "angle": {"type": "string"},
                                  "outline": {"type": "array", "items": {"type": "string"}},
                                  "data_needed": {"type": "string"},
                                  "faq": {"type": "array", "items": {"type": "string"}}},
                   "required": ["title", "angle", "outline", "data_needed", "faq"], "additionalProperties": False},
        "featured": {"type": "array", "items": {
            "type": "object",
            "properties": {"domain": {"type": "string"},
                           "pitch": {"type": "string", "enum": ["expert quote", f"{INDEX_NAME} data", "guest piece"]},
                           "angle": {"type": "string"}},
            "required": ["domain", "pitch", "angle"], "additionalProperties": False}},
        "youtube": {"type": "array", "items": {
            "type": "object",
            "properties": {"clip_title": {"type": "string"}, "webinar_topic": {"type": "string"}},
            "required": ["clip_title", "webinar_topic"], "additionalProperties": False}},
        "difficulty": {"type": "object",
                       "properties": {"level": {"type": "string", "enum": ["easy", "medium", "hard"]},
                                      "why": {"type": "string"}},
                       "required": ["level", "why"], "additionalProperties": False},
    },
    "required": ["results", "overall", "own_it", "featured", "youtube", "difficulty"],
    "additionalProperties": False,
}

ANALYSIS_PROMPT = f"""Do not search again. Using the pages you just read, analyze this search for Equi's marketing team.
Equi is an alternatives firm that cannot advertise its funds to the public: every recommendation is category education
about evergreen, interval, and tender-offer funds that ends in "ask your advisor", never a fund offer, never a
performance or return claim, never a promise of protection or smoother swings. Write plainly for a non-technical
marketer. No em dashes. No hype words.

For each numbered result below: its source type, a 1 to 2 sentence summary of what the page actually says, whether it
mentions evergreen or interval funds, whether it mentions Equi, and what it gets wrong or leaves out.
Overall: who owns this search today, what the top results have in common (format, depth, data, freshness), the gap Equi
could fill, and a one-line verdict in the style "Owned by fund sponsors and Investopedia. Evergreen is explained, Equi is
absent." (no difficulty in the verdict).
own_it: a page brief for equi.com that beats the current top results: title, angle, outline, the data or table it
needs (the {INDEX_NAME} where it fits), and FAQ questions.
featured: 2 to 4 of the returned domains to pitch, with the angle for each. Only publishers, media, education sites, or
forums. Never fund sponsors, asset managers, or advisory firms: they compete with Equi. Never pitch a government site
or Wikipedia (its conflict-of-interest rules bar a firm writing about its own category).
youtube: 1 or 2 clips, each with a question-style title and the existing webinar topic to cut it from.
difficulty: easy, medium, or hard, with one line on why.

Results:
"""


JUNK_TITLE = re.compile(r"javascript|free trial|access denied|just a moment|enable cookies|subscribe to|^\W*$", re.I)


def clean_title(title: str, url: str) -> str:
    """Some sites serve a bot page title ("Please enable Javascript"); show the page path instead."""
    if title and not JUNK_TITLE.search(title) and len(title) > 3:
        return title
    path = [p for p in url.split("?")[0].split("/")[3:] if p]
    words = re.sub(r"[-_]+", " ", path[-1] if path else "").strip()
    return (words[:1].upper() + words[1:]) if words else domain(url)


NEVER_PITCH = ("wikipedia.org", ".gov")   # Wikipedia conflict-of-interest rules; regulators are cited, not pitched


def tidy(result: dict) -> dict:
    """Code-enforced rules on a finished analysis: readable titles, pitches only to returned, pitchable domains."""
    returned = {r["domain"] for r in result["results"]}
    for r in result["results"]:
        r["title"] = clean_title(r["title"], r["url"])
    kept, dropped = [], []
    for f in result["featured"]:
        f["domain"] = f["domain"].lower().removeprefix("www.")
        ok = f["domain"] in returned and not any(f["domain"].endswith(n) for n in NEVER_PITCH)
        (kept if ok else dropped).append(f)
    earlier = result.get("featured_dropped", [])
    result["featured"] = kept
    result["featured_dropped"] = list(dict.fromkeys(earlier + [f["domain"] for f in dropped]))
    return result


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower()).rstrip("?").strip()


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=TIMEOUT_S, max_retries=RETRIES)


def _turn(client, messages: list, **extra):
    """One assistant turn with the search tool, resuming if the server-side search loop pauses."""
    content = []
    for _ in range(3):
        msg = client.messages.create(model=MODEL, max_tokens=16000, tools=[SEARCH_TOOL],
                                     messages=messages + ([{"role": "assistant", "content": content}] if content else []),
                                     **extra)
        content += msg.content
        if msg.stop_reason != "pause_turn":
            return content, msg.stop_reason
    return content, "pause_turn"


def _results(content) -> tuple[list[dict], list[str], list[str]]:
    """(results in order, queries searched, search error codes) from one assistant turn."""
    out, queries, errors = [], [], []
    for b in content:
        if b.type == "server_tool_use":
            queries.append((b.input or {}).get("query", ""))
        elif b.type == "web_search_tool_result":
            if isinstance(b.content, list):
                out += [{"url": r.url, "title": r.title, "page_age": getattr(r, "page_age", None)} for r in b.content]
            else:
                errors.append(getattr(b.content, "error_code", "search_error"))
    return out, queries, errors


def _issues(a: dict) -> list[str]:
    """Client-copy rules on the advice. Naming Equi is expected here; claims about funds are not."""
    text = [a["own_it"]["title"], a["own_it"]["angle"], *a["own_it"]["outline"], *a["own_it"]["faq"]]
    text += [f["angle"] for f in a["featured"]] + [y["clip_title"] for y in a["youtube"]]
    return [f for t in text for f in lint_text(t.rstrip(".?") + ".") if f.split(" in ")[0] not in INTERNAL_OK]


def analyze_query(query: str) -> dict:
    """Full analysis for one query. Raises SerpError with a plain message on any failure."""
    import anthropic
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SerpError("Searching needs an Anthropic key. Add ANTHROPIC_API_KEY to .env, or to Streamlit secrets.")
    client = _client()
    t = time.time()
    ask = f"Search the web for this exact query, then answer it the way you would for someone who typed it:\n\n{query}"
    messages = [{"role": "user", "content": ask}]
    try:
        # 1. search and answer
        c1, _ = _turn(client, messages, output_config={"effort": "low"})
        found, queries, errors = _results(c1)
        answer = "".join(b.text for b in c1 if b.type == "text").strip()
        cited = list(dict.fromkeys(c.url for b in c1 if b.type == "text"
                                   for c in (getattr(b, "citations", None) or []) if getattr(c, "url", None)))
        messages.append({"role": "assistant", "content": c1})
        # 2. one more search with a close variant if we are short
        if len({r["url"] for r in found}) < TOP_N:
            messages.append({"role": "user", "content": "Search once more with a close variant of the same query to find "
                                                        "more results. Reply in one line."})
            c2, _ = _turn(client, messages, output_config={"effort": "low"})
            more, q2, e2 = _results(c2)
            found, queries, errors = found + more, queries + q2, errors + e2
            messages.append({"role": "assistant", "content": c2})
    except anthropic.APIStatusError as e:
        raise SerpError(f"Web search failed ({e.status_code}). Try again in a minute.") from e
    except anthropic.APIConnectionError as e:
        raise SerpError("Could not reach the Anthropic API. Check the connection and try again.") from e

    seen, results = set(), []
    for r in found:
        if r["url"] not in seen:
            seen.add(r["url"])
            results.append({**r, "rank": len(results) + 1, "domain": domain(r["url"])})
    results = results[:TOP_N]
    if not results:
        raise SerpError("The search returned no results for this question. Try rewording it."
                        + (f" (search error: {', '.join(errors)})" if errors else ""))

    # 3. analysis over the pages in this conversation
    listing = "\n".join(f'{r["rank"]}. {r["title"]} ({r["url"]})' for r in results)
    messages.append({"role": "user", "content": ANALYSIS_PROMPT + listing})
    fmt = {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    try:
        c3, stop = _turn(client, messages, output_config=fmt)
        if stop == "max_tokens":
            raise SerpError("The analysis was cut off. Try again.")
        analysis = json.loads(next(b.text for b in c3 if b.type == "text"))
        first = _issues(analysis)
        if first:
            messages += [{"role": "assistant", "content": c3},
                         {"role": "user", "content": "This advice breaks the client-copy rules. Fix these and return the "
                                                     "full JSON again:\n- " + "\n- ".join(first)}]
            c4, _ = _turn(client, messages, output_config=fmt)
            analysis = json.loads(next(b.text for b in c4 if b.type == "text"))
    except anthropic.APIStatusError as e:
        raise SerpError(f"The analysis step failed ({e.status_code}). Try again in a minute.") from e
    except (json.JSONDecodeError, StopIteration) as e:
        raise SerpError("The analysis came back incomplete. Try again.") from e

    from pipeline.kit import _scrub
    analysis = _scrub(analysis)
    by_rank = {a["rank"]: a for a in analysis["results"]}
    for r in results:
        a = by_rank.get(r["rank"], {})
        r.update({k: a.get(k) for k in ("source_type", "summary", "mentions_evergreen", "mentions_equi", "gets_wrong")})
        # the title is ours to check: if it names evergreen or Equi, count it even if the model missed it
        r["mentions_evergreen"] = bool(r["mentions_evergreen"]) or bool(EVERGREEN.search(r["title"]))
        r["mentions_equi"] = bool(r["mentions_equi"]) or bool(EQUI.search(r["title"]))
    level = analysis["difficulty"]["level"]
    return tidy({
        "query": query, "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": MODEL,
        "seconds": round(time.time() - t), "searched": queries, "results": results,
        "ai_answer": {"text": _scrub(answer), "cited_urls": cited, "cited_domains": sorted({domain(u) for u in cited}),
                      "mentions_evergreen": bool(EVERGREEN.search(answer)), "mentions_equi": bool(EQUI.search(answer))},
        "overall": analysis["overall"], "own_it": analysis["own_it"], "featured": analysis["featured"],
        "youtube": analysis["youtube"], "difficulty": analysis["difficulty"],
        "verdict": f'{analysis["overall"]["verdict"].rstrip(".")}. Difficulty: {level}.',
        "flags": _issues(analysis), "fixed_on_retry": first,
    })


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save(result: dict):
    cache = load_cache()
    cache[normalize(result["query"])] = result
    CACHE.write_text(json.dumps(cache, indent=2))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    for q in sys.argv[1:] or ["what is an evergreen fund", "how to protect my portfolio in a market crash"]:
        try:
            r = analyze_query(q)
            save(r)
            print(f'{q}: {len(r["results"])} results in {r["seconds"]}s via {r["searched"]}. {r["verdict"]} '
                  f'Flags: {len(r["flags"])} (fixed on retry: {len(r["fixed_on_retry"])})')
        except SerpError as e:
            print(f"{q}: {e}")
