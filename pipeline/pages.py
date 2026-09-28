"""Publish-ready education pages for equi.com, one per client question.

Built the way answer engines quote pages: the direct answer in the first two
sentences, a comparison table, and an FAQ marked up as FAQPage JSON-LD. The model
writes the words; this file builds the HTML, the schema, the byline placeholder,
the date, and the disclosure, so none of those can drift.

    python -m pipeline.pages "What is an evergreen fund?"
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from pipeline.compliance import DISCLOSURE, lint_text
from pipeline.kit import RETRIES, TIMEOUT_S, _scrub
from pipeline.personalize import MODEL, STYLE_GUIDE

OUT = Path("data/pages.json")
BYLINE = "[Equi PM name], Portfolio Manager, Equi"
CLOSING = "ask your advisor"

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "meta_description": {"type": "string"},
        "direct_answer": {"type": "string"},
        "sections": {"type": "array", "items": {
            "type": "object",
            "properties": {"heading": {"type": "string"}, "paragraphs": {"type": "array", "items": {"type": "string"}}},
            "required": ["heading", "paragraphs"], "additionalProperties": False}},
        "comparison": {"type": "object",
                       "properties": {"caption": {"type": "string"},
                                      "columns": {"type": "array", "items": {"type": "string"}},
                                      "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}},
                       "required": ["caption", "columns", "rows"], "additionalProperties": False},
        "faq": {"type": "array", "items": {
            "type": "object",
            "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
            "required": ["question", "answer"], "additionalProperties": False}},
        "closing": {"type": "string"},
    },
    "required": ["title", "meta_description", "direct_answer", "sections", "comparison", "faq", "closing"],
    "additionalProperties": False,
}

SYSTEM = f"""You write education pages for equi.com, the site of an alternatives firm that is not allowed to advertise
its funds to the public. Each page answers one question wealthy investors ask about evergreen, interval, and
tender-offer funds and how they compare with closed-end private funds and public funds.

{STYLE_GUIDE}

Write the way answer engines quote pages:
- direct_answer: the complete answer in exactly two sentences, standalone, no preamble.
- sections: 3 or 4 short sections that go one level deeper (how it works, liquidity, costs and minimums, risks).
- comparison: a table comparing evergreen, interval, tender-offer, and traditional closed-end funds on liquidity,
  how often you can redeem, typical minimums in words, fee structure in words, and who can invest.
  Every row has exactly one cell per column.
- faq: 5 questions a client would ask next, each with a two to three sentence answer.
- closing: one or two sentences ending with "{CLOSING} whether this fits your situation."

Hard rules: category education only. Never name Equi's funds or any fund or manager. No performance, returns, yields,
or targets. Never promise protection, smoother swings, or lower volatility, and never say private holdings are
shielded from market moves. Never recommend buying or selling."""


def slug(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")


def check(page: dict) -> list[str]:
    issues = []
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", page["direct_answer"].strip()) if x]
    if len(sentences) > 2:
        issues.append(f"The direct answer is {len(sentences)} sentences; it must be two.")
    cols = page["comparison"]["columns"]
    for row in page["comparison"]["rows"]:
        if len(row) != len(cols):
            issues.append(f"Comparison row {row[:1]} has {len(row)} cells for {len(cols)} columns.")
    if len(page["faq"]) < 4:
        issues.append("The FAQ needs at least 4 questions.")
    if CLOSING not in page["closing"].lower():
        issues.append(f'The closing must end with "{CLOSING}".')
    text = [page["title"], page["direct_answer"], page["closing"]]
    text += [p for s in page["sections"] for p in s["paragraphs"]]
    text += [c for row in page["comparison"]["rows"] for c in row]
    text += [f["question"] + " " + f["answer"] for f in page["faq"]]
    for t in text:
        issues += lint_text(t)
    return issues


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=TIMEOUT_S, max_retries=RETRIES)


def _ask(prompt: str) -> dict:
    msg = _client().messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                                    output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                                    messages=[{"role": "user", "content": prompt}])
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Page was cut off at the token limit.")
    return _scrub(json.loads(next(b.text for b in msg.content if b.type == "text")))


def draft(question: str, context: str = "") -> dict:
    prompt = f"Question: {question}\n{context}\nWrite the page."
    page = _ask(prompt)
    first = check(page)
    if first:
        page = _ask(prompt + "\n\nA first draft broke these rules. Fix every one and keep the rest:\n- "
                    + "\n- ".join(first) + "\n\nFirst draft:\n" + json.dumps(page))
    page.update({"question": question, "flags": check(page), "fixed_on_retry": first,
                 "updated": date.today().isoformat(), "model": MODEL})
    return page


def faq_schema(page: dict) -> dict:
    return {"@context": "https://schema.org", "@type": "FAQPage",
            "mainEntity": [{"@type": "Question", "name": f["question"],
                            "acceptedAnswer": {"@type": "Answer", "text": f["answer"]}} for f in page["faq"]]}


def render_html(page: dict) -> str:
    e = html.escape
    # "</" inside JSON-LD could close the script tag early, so escape it
    ld = json.dumps(faq_schema(page), indent=2).replace("</", "<\\/")
    updated = date.fromisoformat(page["updated"]).strftime("%B %-d, %Y")
    cols = page["comparison"]["columns"]
    table = ("<table><caption>" + e(page["comparison"]["caption"]) + "</caption><thead><tr>"
             + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr></thead><tbody>"
             + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>" for row in page["comparison"]["rows"])
             + "</tbody></table>")
    sections = "".join(f"<h2>{e(s['heading'])}</h2>" + "".join(f"<p>{e(p)}</p>" for p in s["paragraphs"])
                       for s in page["sections"])
    faq = "".join(f"<h3>{e(f['question'])}</h3><p>{e(f['answer'])}</p>" for f in page["faq"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(page['title'])}</title>
<meta name="description" content="{e(page['meta_description'])}">
<script type="application/ld+json">
{ld}
</script>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; background:#FAFAF7; color:#2E3440; font-family:'Source Serif 4',serif; }}
  main {{ max-width:760px; margin:0 auto; padding:48px 24px 64px; }}
  h1 {{ font-size:32px; line-height:1.2; color:#1F2A44; margin:0 0 12px; }}
  h2 {{ font-size:22px; color:#1F2A44; margin:32px 0 8px; }}
  h3 {{ font-size:17px; color:#1F2A44; margin:20px 0 4px; font-family:Inter,sans-serif; font-weight:600; }}
  p {{ font-size:17px; line-height:1.65; margin:0 0 12px; }}
  .meta {{ font-family:Inter,sans-serif; font-size:13px; color:#6B7280; margin-bottom:24px; }}
  .answer {{ font-size:19px; line-height:1.6; border-left:3px solid #A8843A; padding-left:16px; margin:0 0 24px; }}
  table {{ width:100%; border-collapse:collapse; font-family:Inter,sans-serif; font-size:14px; margin:24px 0; }}
  caption {{ text-align:left; font-weight:600; color:#1F2A44; padding-bottom:8px; }}
  th, td {{ text-align:left; vertical-align:top; padding:8px 10px; border-bottom:1px solid #DEDCD3; }}
  th {{ background:#F1F0EA; color:#1F2A44; }}
  .disclosure {{ font-family:Inter,sans-serif; font-size:12px; color:#6B7280; border-top:1px solid #DEDCD3;
                margin-top:40px; padding-top:12px; }}
</style></head><body><main>
<h1>{e(page['title'])}</h1>
<div class="meta">By {e(BYLINE)} &nbsp;·&nbsp; Updated {updated}</div>
<p class="answer">{e(page['direct_answer'])}</p>
{sections}
{table}
<h2>Frequently asked questions</h2>
{faq}
<p>{e(page['closing'])}</p>
<p class="disclosure">{e(DISCLOSURE)}</p>
</main></body></html>"""


def load() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(page: dict):
    pages = load()
    pages[slug(page["question"])] = page
    OUT.write_text(json.dumps(pages, indent=2))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    q = sys.argv[1] if len(sys.argv) > 1 else "What is an evergreen fund?"
    page = draft(q)
    save(page)
    print(f"{q}: {len(page['faq'])} FAQs, fixed on retry {len(page['fixed_on_retry'])}, flags {len(page['flags'])}")
