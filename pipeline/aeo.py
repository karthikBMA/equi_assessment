"""Answer-engine tracker: what do AI assistants tell wealthy clients about alternatives?

Clients increasingly ask an assistant before they ask their advisor. If the answer
never mentions evergreen or interval funds, the advisor conversation starts from
zero. This asks Claude (with live web search) the questions clients actually type,
records what comes back and who gets cited, and turns the gaps into pages worth
creating. Equi cannot advertise its funds, so the pages are category education.

    python -m pipeline.aeo --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from pipeline.compliance import DISCLOSURE, lint_text
from pipeline.personalize import MODEL

OUT = Path("data/aeo.json")
SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
TIMEOUT_S = 120      # a web-search answer takes ~25s; searches run server-side inside one call
RETRIES = 2          # SDK retries 429 / 5xx / timeouts with backoff
MAX_CONTINUATIONS = 3

QUESTIONS = {
    "Downturn protection": [
        "how do i protect my portfolio from a market crash",
        "should i move to cash if i think a recession is coming",
        "what do rich people invest in during a downturn",
        "is there a way to invest that doesnt drop as much when stocks fall",
        "how do family offices hedge against a stock market drop",
        "best way to protect $10 million from a bear market",
    ],
    "Alternatives to 60/40": [
        "is the 60/40 portfolio dead",
        "what should i invest in besides stocks and bonds",
        "what are alternative investments and should i own them",
        "how much of my portfolio should be in alternatives",
        "what do endowments invest in that regular investors dont",
        "private equity vs public stocks for high net worth investors",
    ],
    "Evergreen and interval funds": [
        "what is an evergreen fund",
        "what is an interval fund and how does it work",
        "are evergreen private equity funds a good idea",
        "interval fund vs mutual fund difference",
        "what is a tender offer fund",
        "are semi liquid private funds safe",
    ],
    "Liquidity and lock-ups": [
        "can i get my money out of a private equity fund early",
        "how long is money locked up in private equity",
        "what happens if a private fund gates redemptions",
        "is there a private credit fund i can withdraw from quarterly",
        "how liquid are private real estate funds",
        "what does lock up period mean for hedge funds",
    ],
    "Preserving family wealth": [
        "how do wealthy families preserve wealth across generations",
        "what is a multi family office and do i need one",
        "how should i invest an inheritance of 5 million",
        "how do family offices invest their money",
        "what is a qualified purchaser and why does it matter",
        "how to diversify a concentrated stock position after selling a business",
    ],
}

EVERGREEN = re.compile(r"evergreen|interval fund|tender[- ]offer fund|semi[- ]liquid|perpetual[- ]life", re.I)
EQUI = re.compile(r"\bEqui\b")          # case-sensitive: "equity" and "Equitable" do not match


def ordered(n: int) -> list[tuple[str, str]]:
    """First n questions, round-robin across groups so a small run still covers every group."""
    groups = list(QUESTIONS.items())
    out, i = [], 0
    while len(out) < min(n, sum(len(q) for _, q in groups)):
        for g, qs in groups:
            if i < len(qs) and len(out) < n:
                out.append((g, qs[i]))
        i += 1
    return out


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=TIMEOUT_S, max_retries=RETRIES)


def ask(group: str, question: str) -> dict:
    """One question, answered with live web search. Never raises; failures come back as an error row."""
    t = time.time()
    row = {"group": group, "question": question}
    try:
        client = _client()
        messages = [{"role": "user", "content": question}]
        content = []
        for _ in range(MAX_CONTINUATIONS + 1):
            msg = client.messages.create(model=MODEL, max_tokens=16000, output_config={"effort": "medium"},
                                         tools=[SEARCH_TOOL], messages=messages)
            content += msg.content
            if msg.stop_reason != "pause_turn":
                break
            # server-side search loop paused: resend with the partial turn and it resumes
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": msg.content}]
        answer, cited, searched, errors = [], [], [], []
        for b in content:
            if b.type == "text":
                answer.append(b.text)
                cited += [c.url for c in (getattr(b, "citations", None) or []) if getattr(c, "url", None)]
            elif b.type == "web_search_tool_result":
                if isinstance(b.content, list):
                    searched += [r.url for r in b.content if getattr(r, "url", None)]
                else:
                    errors.append(getattr(b.content, "error_code", "search_error"))
        text = "".join(answer).strip()
        cited = list(dict.fromkeys(cited))
        row.update({
            "answer": text, "cited_urls": cited, "cited_domains": sorted({domain(u) for u in cited}),
            "searched_urls": list(dict.fromkeys(searched)),
            "mentions_evergreen": bool(EVERGREEN.search(text)),
            "evergreen_terms": sorted({m.lower() for m in EVERGREEN.findall(text)}),
            "mentions_equi": bool(EQUI.search(text)) or any(domain(u).split(".")[0] == "equi" for u in cited),
            "search_errors": errors, "seconds": round(time.time() - t, 1), "error": None,
        })
    except Exception as e:
        row.update({"error": f"{type(e).__name__}: {str(e)[:200]}", "seconds": round(time.time() - t, 1)})
    return row


def metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    share = lambda rs, k: round(sum(r[k] for r in rs) / len(rs), 3) if rs else None
    by_group = {g: share([r for r in ok if r["group"] == g], "mentions_evergreen") for g in QUESTIONS}
    domains = Counter(d for r in ok for d in r["cited_domains"])
    return {
        "asked": len(rows), "answered": len(ok), "failed": len(rows) - len(ok),
        "evergreen_share": share(ok, "mentions_evergreen"),
        "evergreen_by_group": by_group,
        "equi_share": share(ok, "mentions_equi"),
        "top_domains": domains.most_common(15),
        "gaps": [{"group": r["group"], "question": r["question"]} for r in ok if not r["mentions_evergreen"]],
    }


PAGES_SCHEMA = {
    "type": "object",
    "properties": {"pages": {"type": "array", "items": {
        "type": "object",
        "properties": {"target_question": {"type": "string"}, "page_title": {"type": "string"},
                       "why_cited": {"type": "string"},
                       "pitch_domains": {"type": "array", "items": {"type": "string"}}},
        "required": ["target_question", "page_title", "why_cited", "pitch_domains"],
        "additionalProperties": False}}},
    "required": ["pages"], "additionalProperties": False,
}

PAGES_SYSTEM = """You plan educational web content for an alternatives firm that is not allowed to advertise its funds
to the public. Every page is category education about evergreen and interval-style alternatives: how they work, the
trade-offs, and the risks, ending with "ask your advisor". No fund names, no performance or return claims, no
recommendations. Write plainly. No em dashes. No hype words."""


def pages_to_create(m: dict) -> list[dict]:
    gaps = "\n".join(f'- [{g["group"]}] {g["question"]}' for g in m["gaps"]) or "- none"
    doms = ", ".join(f"{d} ({n})" for d, n in m["top_domains"])
    prompt = (f"Questions wealthy clients ask where an AI answer never mentioned evergreen, interval, or tender-offer "
              f"funds:\n{gaps}\n\nMost-cited domains across all answers (citation count): {doms}\n\n"
              "Propose 5 to 8 pages. For each: the target question (from the list, or a close variant), a plain page "
              "title, one or two sentences on why an answer engine would cite it (what gap it fills, what format "
              "they cite), and 2 or 3 of the cited domains above to pitch for a feature or a quote. Pitch only "
              "publishers, media, and education or regulator sites. Never pitch fund managers, asset managers, or "
              "advisory firms, since they compete with Equi; if no cited domain fits, return an empty list.\n"
              "Page titles are client-facing: never promise protection, smoother swings, or lower volatility.")

    def ask(p):
        msg = _client().messages.create(model=MODEL, max_tokens=16000, system=PAGES_SYSTEM,
                                        output_config={"effort": "medium",
                                                       "format": {"type": "json_schema", "schema": PAGES_SCHEMA}},
                                        messages=[{"role": "user", "content": p}])
        return json.loads(next(b.text for b in msg.content if b.type == "text"))["pages"]

    def issues(pages):
        return [f'"{pg["page_title"]}": {f}' for pg in pages for f in lint_text(pg["page_title"] + ".")]

    pages = ask(prompt)
    first = issues(pages)
    if first:
        pages = ask(prompt + "\n\nA first draft broke the client-copy rules. Fix these and keep the rest:\n- "
                    + "\n- ".join(first) + "\n\nFirst draft:\n" + json.dumps(pages))
    for pg in pages:
        pg["flags"] = [f for f in issues([pg])]
        pg["disclosure"] = DISCLOSURE
    return pages


def run(n: int = 30, workers: int = 4, progress=print) -> dict:
    todo = ordered(n)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(lambda gq: ask(*gq), todo):
            rows.append(row)
            status = "error" if row["error"] else ("evergreen" if row["mentions_evergreen"] else "no evergreen")
            progress(f"  [{len(rows)}/{len(todo)}] {row['seconds']:>5}s  {status:12}  {row['question']}")
    m = metrics(rows)
    try:
        pages, pages_error = pages_to_create(m), None
    except Exception as e:
        pages, pages_error = [], f"{type(e).__name__}: {str(e)[:200]}"
    out = {"run_date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": MODEL, "tool": SEARCH_TOOL,
           "metrics": m, "pages": pages, "pages_error": pages_error, "results": rows}
    OUT.write_text(json.dumps(out, indent=2))
    return out


def load() -> dict | None:
    return json.loads(OUT.read_text()) if OUT.exists() else None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pages-only", action="store_true", help="rebuild Pages to create from the cached answers")
    args = ap.parse_args()
    t = time.time()
    if args.pages_only:
        out = load()
        out["pages"], out["pages_error"] = pages_to_create(out["metrics"]), None
        OUT.write_text(json.dumps(out, indent=2))
        print(f"Rebuilt {len(out['pages'])} pages in {time.time()-t:.0f}s")
        raise SystemExit
    out = run(args.n, args.workers)
    m = out["metrics"]
    print(f"\nDone in {time.time()-t:.0f}s -> {OUT}. Answered {m['answered']}/{m['asked']}.")
    print(f"Evergreen mentioned in {m['evergreen_share']:.0%} of answers; Equi in {m['equi_share']:.0%}.")
