"""Answer-engine tracker: what do AI assistants tell wealthy clients about alternatives?

Clients increasingly ask an assistant before they ask their advisor. If the answer
never mentions evergreen or interval funds, the advisor conversation starts from
zero. This asks Claude (with live web search) the questions clients actually type,
records what comes back and who gets cited, and turns the gaps into pages worth
creating. Equi cannot advertise its funds, so the pages are category education.
A sixth group asks what advisors type when they look for a provider, reported
separately, since those answers are where Equi itself should appear.

    python -m pipeline.aeo            # all 40 questions
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
    # advisors searching for a provider: this is where Equi can be named, as long as nothing offers a fund
    "Advisors researching": [
        "best evergreen alternatives platforms for RIAs",
        "white label fund of funds for wealth managers",
        "how RIAs add liquid alternatives for clients",
        "alternative investment platforms for independent advisors",
        "how to explain evergreen funds to clients as an advisor",
        "client materials for advisors on private markets",
        "interval fund vs evergreen fund for RIA clients",
        "how multi family offices source alternative managers",
        "due diligence checklist for evergreen funds for advisors",
        "turnkey alternatives solution for RIAs",
    ],
}
ADVISOR_GROUPS = {"Advisors researching"}
AUDIENCES = {"client": [g for g in QUESTIONS if g not in ADVISOR_GROUPS], "advisor": sorted(ADVISOR_GROUPS)}

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
    by_audience = {}
    for aud, groups in AUDIENCES.items():
        rs = [r for r in ok if r["group"] in groups]
        by_audience[aud] = {"answered": len(rs), "evergreen": sum(r["mentions_evergreen"] for r in rs),
                            "equi": sum(r["mentions_equi"] for r in rs),
                            "evergreen_share": share(rs, "mentions_evergreen"), "equi_share": share(rs, "mentions_equi")}
    return {
        "by_audience": by_audience,
        "asked": len(rows), "answered": len(ok), "failed": len(rows) - len(ok),
        "evergreen_share": share(ok, "mentions_evergreen"),
        "evergreen_by_group": by_group,
        "equi_share": share(ok, "mentions_equi"),
        "top_domains": domains.most_common(15),
        "gaps": [{"group": r["group"], "question": r["question"]} for r in ok if not r["mentions_evergreen"]],
    }


ROW = {
    "type": "object",
    "properties": {"target_question": {"type": "string"}, "action": {"type": "string"}, "where": {"type": "string"},
                   "why_cited": {"type": "string"}, "metric": {"type": "string"}},
    "required": ["target_question", "action", "where", "why_cited", "metric"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"publish": {"type": "array", "items": ROW},
                   "featured": {"type": "array", "items": ROW},
                   "youtube": {"type": "array", "items": ROW}},
    "required": ["publish", "featured", "youtube"],
    "additionalProperties": False,
}
INDEX_NAME = "Evergreen Fund Index"

PLAN_SYSTEM = f"""You write an answer-engine (AEO) action plan for the marketing team at Equi, an alternatives firm that
is not allowed to advertise its funds to the public. Everything client-facing is category education about evergreen,
interval, and tender-offer funds: how they work, trade-offs, risks, ending with "ask your advisor". No fund names, no
performance or return claims, no recommendations, never promise protection or smoother swings.

Write for a non-technical marketer: every action is a concrete thing a person can do this month. Plain words, short.
No em dashes. No hype words.

Three lanes:
1. publish: pages on equi.com. The first row is always the {INDEX_NAME}: a quarterly comparison of evergreen fund fees,
   liquidity terms, and minimums compiled from public SEC filings. It is the anchor asset the other rows link to.
   Then 5 to 6 question-led pages, each answering one client question directly. "where" is the equi.com URL path.
2. featured: get Equi quoted or published where AI already looks. One row per cited domain you choose, using only
   publishers, media, and education or regulator sites from the list given. Never fund managers, asset managers, or
   advisory firms: they compete with Equi. "action" is the pitch angle: an expert quote, {INDEX_NAME} data for a
   reporter, or a guest piece. For a regulator site, the action is to cite it and match its definitions, not pitch it.
3. youtube: 4 to 5 short clips cut from existing Equi webinars and long-form videos, each titled with a client question
   and published with a full transcript. "where" is the playlist or channel section.

Some gaps are advisor questions (an RIA or family office looking for a provider). Pages and pitches for those can
explain how Equi works with advisors (client materials built for them, Equi-branded or white-label funds), but still
never offer a fund to the public or cite performance.

"metric" is how to tell it worked, in plain words (for example, "cited in the weekly AI answer check within 6 weeks")."""


# Plan actions are instructions to Equi's marketer, not client copy: naming Equi or its research team is expected.
# The claim rules (protection, returns, predictions) still apply, because the actions become client-facing pages.
INTERNAL_OK = {"a mention of Equi", "internal process (clients have no committee)"}


def is_index_row(row: dict) -> bool:
    """The Evergreen Fund Index row, named in the text or only in its page path."""
    text = " ".join(row.get(k, "") for k in ("target_question", "action", "where")).lower().replace("-", " ")
    return INDEX_NAME.lower() in text


def lint_plan(plan: dict) -> list[str]:
    found = [f'{lane} "{r["target_question"]}": {f}' for lane in ("publish", "featured", "youtube")
             for r in plan[lane] for f in lint_text(r["action"] + ".") if f.split(" in ")[0] not in INTERNAL_OK]
    if not plan["publish"] or not is_index_row(plan["publish"][0]):
        found.append(f"The first publish row must be the {INDEX_NAME}.")
    return found


def action_plan(m: dict) -> dict:
    """Three-lane AEO plan from the tracker results, in one call."""
    gaps = "\n".join(f'- [{"advisor" if g["group"] in ADVISOR_GROUPS else "client"}: {g["group"]}] {g["question"]}'
                     for g in m["gaps"]) or "- none"
    groups = ", ".join(f"{g} {v:.0%}" for g, v in m["evergreen_by_group"].items())
    doms = ", ".join(f"{d} ({n})" for d, n in m["top_domains"])
    prompt = (f"AI answers mentioned evergreen, interval, or tender-offer funds in {m['evergreen_share']:.0%} of "
              f"{m['answered']} client questions (by group: {groups}), and Equi in {m['equi_share']:.0%}.\n\n"
              f"Questions where evergreen never came up:\n{gaps}\n\n"
              f"Most-cited domains (citation count): {doms}\n\nWrite the three-lane plan.")

    def ask(p):
        msg = _client().messages.create(model=MODEL, max_tokens=16000, system=PLAN_SYSTEM,
                                        output_config={"effort": "medium",
                                                       "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
                                        messages=[{"role": "user", "content": p}])
        return json.loads(next(b.text for b in msg.content if b.type == "text"))

    issues = lint_plan

    plan = ask(prompt)
    first = issues(plan)
    if first:
        plan = ask(prompt + "\n\nA first draft broke these rules. Fix them and keep the rest:\n- "
                   + "\n- ".join(first) + "\n\nFirst draft:\n" + json.dumps(plan))
    plan["flags"] = issues(plan)
    plan["fixed_on_retry"] = first
    plan["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    plan["disclosure"] = DISCLOSURE
    return plan


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
        plan, plan_error = action_plan(m), None
    except Exception as e:
        plan, plan_error = None, f"{type(e).__name__}: {str(e)[:200]}"
    out = {"run_date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": MODEL, "tool": SEARCH_TOOL,
           "metrics": m, "action_plan": plan, "plan_error": plan_error, "results": rows}
    OUT.write_text(json.dumps(out, indent=2))
    return out


def load() -> dict | None:
    return json.loads(OUT.read_text()) if OUT.exists() else None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=sum(len(q) for q in QUESTIONS.values()))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plan-only", action="store_true", help="rebuild the AEO action plan from the cached answers")
    args = ap.parse_args()
    t = time.time()
    if args.plan_only:
        out = load()
        out["action_plan"], out["plan_error"] = action_plan(out["metrics"]), None
        out.pop("pages", None)
        OUT.write_text(json.dumps(out, indent=2))
        p = out["action_plan"]
        print(f"Plan in {time.time()-t:.0f}s: {len(p['publish'])} publish, {len(p['featured'])} featured, "
              f"{len(p['youtube'])} clips. Fixed on retry: {len(p['fixed_on_retry'])}. Flags left: {len(p['flags'])}")
        raise SystemExit
    out = run(args.n, args.workers)
    m = out["metrics"]
    print(f"\nDone in {time.time()-t:.0f}s -> {OUT}. Answered {m['answered']}/{m['asked']}.")
    for aud, a in m["by_audience"].items():
        print(f"  {aud}: evergreen in {a['evergreen']} of {a['answered']}, Equi in {a['equi']} of {a['answered']}")
