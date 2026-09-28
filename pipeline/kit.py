"""Firm-branded client kit, written by Claude and rendered by us.

The kit is the Part 1 hook ("we already did the work"): a one-page letter the
advisor can send their own clients, talking points for the advisor, and for
committee-led firms an IC memo outline the research lead can take to committee.

Claude returns structured JSON; layout, branding, and the compliance footer are
ours, so the model cannot drop or rewrite the disclosures. Branding here is a
palette hashed from the firm name. In production we would pull the logo and
colors from the firm's website.
"""
from __future__ import annotations

import colorsys
import hashlib
import html
import json
import os
import re
from datetime import date

from pipeline.personalize import MODEL, STYLE_GUIDE, firm_facts

TIMEOUT_S = 90   # per attempt; a kit is ~1,500 words and normally takes well under this
RETRIES = 1      # the SDK retries timeouts, 429s and 5xx once, so worst case is about 3 minutes
# Sonnet 5 thinks by default and thinking counts toward max_tokens. A letter needs little
# reasoning, so effort is medium; max_tokens leaves room for thinking plus the kit.
EFFORT = "medium"
MAX_TOKENS = 16000

COMPLIANCE_FOOTER = (
    "This letter is for educational purposes only. It is not an offer to sell or a solicitation of an offer "
    "to buy any security, and it is not investment advice for any individual. Alternative investments involve "
    "risk, including the loss of principal, and are not suitable for every investor. Evergreen funds offer "
    "periodic liquidity that may be limited or suspended, so investors may not be able to redeem when they wish. "
    "Any investment would be made only through the fund's offering documents, which describe its risks, fees, "
    "and eligibility requirements."
)

SYSTEM = f"""You write educational materials that an independent wealth advisory firm sends to its own clients,
under the firm's name. Equi prepares these for the firm as a draft. The firm's compliance team reviews before use.

{STYLE_GUIDE}

Hard rules:
- Educational only. Explain what evergreen alternative funds are, how their periodic liquidity works
  (subscriptions and redemptions at set intervals, with limits the fund can apply), how they differ from
  traditional closed-end private funds, what they can and cannot do, and the main risks.
- Never mention Equi, any fund name, any manager, or any performance, return, yield, or target.
  No numbers about returns or volatility. No guarantees, no "protect", no "safe".
- Never tell clients what they should own. The letter explains and invites a conversation with their advisor.
- Use only the firm facts given. Never invent the firm's history, people, holdings, or client counts, and never
  claim what clients have asked, said, or think.
- The letter goes to the firm's individual and family clients. Never mention the firm's investment
  committee, research team, or internal process in the letter or the talking points; clients do not have one.
- Avoid specific figures in the letter (fund lives in years, percentages, dates). Describe ranges in words.
- Write as the firm ("we", "our clients"). The closing is only a sign-off phrase such as "With best regards,";
  the advisor's name is added separately.
- Return only JSON, no prose, no markdown fences."""

USER_TMPL = """Firm facts:
{facts}

Write the kit. The letter must fit on one printed page: 300 to 380 words across the paragraphs.
{memo_ask}
Return JSON:
{{"letter": {{"title": "short plain title for the letter", "greeting": "Dear clients,",
             "paragraphs": ["...", "..."], "closing": "sign-off phrase only, no name"}},
  "talking_points": ["6 to 8 short points, each something the advisor can say to a client, including 2 or 3
                      likely client questions with a plain answer. No notes about the kit itself."],
  {memo_schema}
  "tailoring": "one sentence on how this kit was fitted to this firm's clients"}}"""

MEMO_ASK = """This firm decides by investment committee. Also write an IC memo outline the research lead can take to
committee: 7 to 9 sections, each with 2 to 4 bullets on what the section should cover. Where a bullet needs data
only the fund manager can supply (terms, fees, track record, service providers), end it with "[Equi to provide]"."""
MEMO_SCHEMA = '"ic_memo": [{"section": "...", "points": ["...", "..."]}],'


# What compliance would flag in client-facing text (the letter and talking points).
# The IC memo is exempt from the performance rule: it is meant to ask the manager for track record.
LINT_RULES = [
    (r"\breturns? (potential|profile|stream|target|expectation)|\boutperform"
     r"|\bperformance\b(?![- ](based[- ])?(fee|component|allocation))|\byields?\b|\bprotect",
     "return or performance language"),
    (r"\d", "a specific figure"),
    (r"committee|research team", "internal process (clients have no committee)"),
    (r"\bequi\b", "a mention of Equi"),
    (r"many of you (have )?asked|you(\'ve| have) asked|clients (have )?asked|you may be wondering",
     "a claim about what clients asked or think"),
]


def lint(kit: dict) -> list[str]:
    """Plain-English problems in the client-facing text. Empty list means clean."""
    L = kit["letter"]
    parts = {"letter": " ".join([L["title"], *L["paragraphs"]]), "talking points": " ".join(kit["talking_points"])}
    found = []
    for where, text in parts.items():
        for pattern, label in LINT_RULES:
            m = re.search(r"[^.]*(" + pattern + r")[^.]*\.?", text, re.I)
            if m:
                found.append(f'{where.capitalize()}: {label} in "{m.group(0).strip()}"')
    return found


def is_committee(record: dict) -> bool:
    return record.get("decision_structure") == "Committee-led"


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=TIMEOUT_S, max_retries=RETRIES)


def _ask(prompt: str) -> dict:
    msg = _client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Kit was cut off at the token limit. Try again.")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return _scrub(json.loads(text))


def generate(record: dict) -> dict:
    """Write the kit, lint it, and fix once if the client-facing text breaks a rule.
    Anything still flagged is stored in kit["flags"] for the reviewer."""
    committee = is_committee(record)
    prompt = USER_TMPL.format(facts=firm_facts(record), memo_ask=MEMO_ASK if committee else "",
                              memo_schema=MEMO_SCHEMA if committee else "")
    kit = _ask(prompt)
    problems = lint(kit)
    if problems:
        fix = ("\n\nA first draft broke these rules. Rewrite the kit so none of them apply, keeping everything "
               "else:\n- " + "\n- ".join(problems) + "\n\nFirst draft:\n" + json.dumps(kit))
        kit = _ask(prompt + fix)
        kit["fixed_on_retry"] = problems
    kit["flags"] = lint(kit)
    kit["firm_name"] = record["firm_name"]
    kit["committee"] = committee
    kit["generated"] = date.today().isoformat()
    kit["model"] = MODEL
    return kit


def _scrub(x):
    """No em or en dashes anywhere in what we render."""
    if isinstance(x, str):
        return x.replace(" \u2014 ", ", ").replace("\u2014", ", ").replace("\u2013", "-")
    if isinstance(x, list):
        return [_scrub(i) for i in x]
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items()}
    return x


# ---------- branding ----------

def palette(firm_name: str) -> dict:
    """Stable, muted colors from the firm name. Dark primary for type, soft accent, pale tint."""
    h = int(hashlib.md5(firm_name.encode()).hexdigest()[:6], 16)
    hue = (h % 360) / 360

    def hls(hh, l, s):
        r, g, b = colorsys.hls_to_rgb(hh % 1, l, s)
        return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
    # low saturation keeps every hue muted; bright greens and yellows read as consumer, not private bank
    return {"primary": hls(hue, 0.24, 0.34), "accent": hls(hue + 0.08, 0.42, 0.30),
            "tint": hls(hue, 0.96, 0.30), "rule": hls(hue, 0.85, 0.20)}


def monogram(firm_name: str) -> str:
    words = [w for w in firm_name.replace("&", " ").split() if w[0].isupper()]
    return "".join(w[0] for w in words[:2])


# ---------- rendering ----------

def possessive(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def render_html(kit: dict, record: dict) -> str:
    p = palette(kit["firm_name"])
    e = html.escape
    firm = e(kit["firm_name"])
    loc = e(", ".join(x for x in (record.get("city"), record.get("state")) if x))
    L = kit["letter"]
    letter = "".join(f"<p>{e(par)}</p>" for par in L["paragraphs"])
    points = "".join(f"<li>{e(t)}</li>" for t in kit["talking_points"])
    memo = ""
    if kit.get("ic_memo"):
        secs = "".join(
            f'<div class="sec"><h3>{i}. {e(s["section"])}</h3><ul>'
            + "".join(f"<li>{e(pt).replace('[Equi to provide]', '<span class=tbd>[Equi to provide]</span>')}</li>"
                      for pt in s["points"]) + "</ul></div>"
            for i, s in enumerate(kit["ic_memo"], 1))
        memo = f"""<section class="page"><div class="kicker">For the investment committee. Draft.</div>
<h2>IC memo outline: evergreen alternatives</h2><p class="sub">Prepared for {e(possessive(kit['firm_name']))} research team.
Items marked <span class="tbd">[Equi to provide]</span> come from the manager's DDQ.</p>{secs}</section>"""
    today = date.fromisoformat(kit.get("generated", date.today().isoformat())).strftime("%B %-d, %Y")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{firm}: client kit</title>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; background:#ECEBE6; font-family:Inter,system-ui,sans-serif; color:#2E3440; }}
  .page {{ background:#fff; max-width:760px; margin:24px auto; padding:56px 64px; box-shadow:0 1px 3px rgba(0,0,0,.08);
          page-break-after:always; }}
  .head {{ display:flex; align-items:center; gap:14px; border-bottom:2px solid {p['primary']}; padding-bottom:14px; }}
  .mono {{ width:44px; height:44px; background:{p['primary']}; color:#fff; display:flex; align-items:center;
          justify-content:center; font-family:'Source Serif 4',serif; font-weight:600; font-size:18px; letter-spacing:1px; }}
  .firm {{ font-family:'Source Serif 4',serif; font-size:22px; color:{p['primary']}; font-weight:600; }}
  .loc {{ font-size:12px; color:#6B7280; letter-spacing:.04em; text-transform:uppercase; }}
  .date {{ margin:28px 0 6px; font-size:13px; color:#6B7280; }}
  h1 {{ font-family:'Source Serif 4',serif; font-weight:600; font-size:24px; color:{p['primary']}; margin:4px 0 18px; }}
  h2 {{ font-family:'Source Serif 4',serif; font-weight:600; font-size:22px; color:{p['primary']}; margin:4px 0 8px; }}
  h3 {{ font-size:14px; color:{p['primary']}; margin:18px 0 6px; }}
  p, li {{ font-family:'Source Serif 4',serif; font-size:15.5px; line-height:1.6; }}
  .sign {{ margin-top:22px; }}
  .foot {{ margin-top:34px; padding-top:12px; border-top:1px solid {p['rule']}; font-family:Inter,sans-serif;
          font-size:10.5px; line-height:1.5; color:#6B7280; }}
  .kicker {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:{p['accent']}; font-weight:600; }}
  .sub {{ font-family:Inter,sans-serif; font-size:13px; color:#6B7280; }}
  .page.alt {{ background:{p['tint']}; }}
  .tbd {{ color:{p['accent']}; font-family:Inter,sans-serif; font-size:12px; font-weight:600; }}
  ul {{ padding-left:20px; }}
  @media print {{ body {{ background:#fff; }} .page {{ box-shadow:none; margin:0; }} }}
</style></head><body>
<section class="page">
  <div class="head"><div class="mono">{e(monogram(kit['firm_name']))}</div>
    <div><div class="firm">{firm}</div><div class="loc">{loc}</div></div></div>
  <div class="date">{today}</div>
  <h1>{e(L['title'])}</h1>
  <p>{e(L.get('greeting') or 'Dear clients,')}</p>
  {letter}
  <p class="sign">{e(closing(kit))}<br><br>[Advisor name]<br>{firm}</p>
  <div class="foot">{e(COMPLIANCE_FOOTER)}</div>
</section>
<section class="page alt"><div class="kicker">For advisors only. Not for client distribution.</div>
  <h2>Talking points</h2><p class="sub">For conversations with {firm} clients about evergreen alternatives.</p>
  <ul>{points}</ul></section>
{memo}
</body></html>"""


def closing(kit: dict) -> str:
    """Sign-off phrase without the name; the model sometimes adds the placeholder itself."""
    c = (kit["letter"].get("closing") or "").replace("[Advisor name]", "").strip()
    return c or "With best regards,"


def letter_text(kit: dict) -> str:
    L = kit["letter"]
    return "\n\n".join([L["title"], L.get("greeting") or "Dear clients,", *L["paragraphs"],
                        closing(kit), "[Advisor name]"])


if __name__ == "__main__":
    # python -m pipeline.kit   writes the two demo kits (principal-led and committee-led) to data/kits.json
    import argparse
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", nargs="+", default=["Marchetti Ruiz Family Advisors", "Thornbury Wealth Advisors"])
    args = ap.parse_args()
    recs = {r["firm_name"]: r for r in json.loads(Path("data/scored.json").read_text())}
    out = {}
    for name in args.firms:
        k = generate(recs[name])
        out[name] = k
        print(f"{name}: {'retried, ' if k.get('fixed_on_retry') else ''}{len(k['flags'])} flags left")
    Path("data/kits.json").write_text(json.dumps(out, indent=2))
