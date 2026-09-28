"""Market-signal triggers and the notes they fire (Part 1 experiment 2: market-drop rapid response).

Rules (no daily noise):
- S&P 500 (SPY) down 2% or more on the day
- VIX up 25% or more day over day, or VIX closing above 30
Otherwise nothing fires and the firm gets the weekly digest.

Firing a trigger drafts, for each Tier A/B firm, a short client-ready note the
advisor can forward under their own name, plus a two-line ping from Equi to the
advisor. Educational only: market facts and how structures behave, never Equi
performance, predictions, or advice. Uses closing prices, so an intraday spike
(VIX touched 65 on Aug 5 2024) shows as the close.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from pipeline.compliance import DISCLOSURE, lint_text
from pipeline.kit import LINT_RULES, TIMEOUT_S, RETRIES
from pipeline.personalize import MODEL, STYLE_GUIDE

SPY_DROP = -2.0
VIX_JUMP = 25.0
VIX_LEVEL = 30.0

EVENTS = {
    "2024-08-05": {"label": "Aug 5 2024: VIX spike", "days": ["2024-08-05"],
                   "context": "Global selloff after the Bank of Japan rate hike unwound the yen carry trade; "
                              "Japan's Nikkei fell about 12% that day."},
    "2025-04-03": {"label": "Apr 3-4 2025: tariff selloff", "days": ["2025-04-03", "2025-04-04"],
                   "context": "Two-day selloff after the US announced broad new tariffs on Apr 2 2025."},
}


# ---------- market data ----------

def fetch(start: str, end: str) -> list[dict]:
    """Daily closes for SPY and ^VIX with day-over-day change. Raises on no data."""
    import yfinance as yf
    # a blocked or slow Yahoo connection must not stall the page; the app falls back to cached closes
    df = yf.download(["SPY", "^VIX"], start=start, end=end, progress=False, auto_adjust=False,
                     timeout=10)["Close"].dropna()
    if df.empty:
        raise RuntimeError("No market data returned.")
    rows = []
    for i in range(1, len(df)):
        d, p = df.iloc[i], df.iloc[i - 1]
        rows.append({"date": df.index[i].date().isoformat(),
                     "spy": round(float(d["SPY"]), 2), "spy_pct": round((d["SPY"] / p["SPY"] - 1) * 100, 2),
                     "vix": round(float(d["^VIX"]), 2), "vix_pct": round((d["^VIX"] / p["^VIX"] - 1) * 100, 2)})
    return rows


def latest() -> dict:
    """Most recent trading day versus the one before."""
    today = date.today()
    return fetch((today - timedelta(days=14)).isoformat(), (today + timedelta(days=1)).isoformat())[-1]


def event_days(key: str) -> list[dict]:
    days = EVENTS[key]["days"]
    start = (date.fromisoformat(days[0]) - timedelta(days=7)).isoformat()
    end = (date.fromisoformat(days[-1]) + timedelta(days=1)).isoformat()
    return [r for r in fetch(start, end) if r["date"] in days]


def triggers(day: dict) -> list[str]:
    fired = []
    if day["spy_pct"] <= SPY_DROP:
        fired.append(f"S&P 500 down {abs(day['spy_pct']):.1f}% (rule: 2% or more)")
    if day["vix_pct"] >= VIX_JUMP:
        fired.append(f"VIX up {day['vix_pct']:.0f}% day over day (rule: 25% or more)")
    if day["vix"] > VIX_LEVEL:
        fired.append(f"VIX closed at {day['vix']:.1f} (rule: above 30)")
    return fired


# ---------- notes ----------

SYSTEM = f"""You write two things for an alternatives firm's sales team on a volatile market day.

1. A short client-ready note an independent wealth advisory firm can forward to its own clients under its own
   name. 90 to 140 words. Calm, factual: what happened in markets today (use the numbers given), why days like
   this happen, and how the structure of what clients hold (for example private or evergreen funds with
   periodic liquidity) behaves differently from daily-priced public markets. Educational only.
2. A two-line ping to the advisor from Equi's team: line one names the move plainly, line two offers the
   note (already written under their firm's name) and, if useful, a short call. No pitch.

{STYLE_GUIDE}

Hard rules:
- Never mention Equi, any fund, or any performance, return, yield, or target in the client note. Do not use the
  words "performance" or "returns" at all, even to disclaim them.
- Periodic valuation means reported values update less often and can lag public markets. It does not mean the
  holdings are unaffected. Never say or imply that private or evergreen holdings are insulated, shielded,
  smoother, steadier, or less risky on days like this.
- Never predict markets ("will recover", "buying opportunity") and never suggest a trade or allocation.
- Use only the market numbers and firm facts given. Do not claim what clients asked or think.
- Do not sign the client note or add a greeting; the firm's signature is added separately.
- advisor_ping is exactly two lines."""

USER_TMPL = """Market day: {date}
S&P 500 (SPY) close {spy} ({spy_pct:+.2f}% on the day). VIX close {vix} ({vix_pct:+.1f}% day over day).
Context: {context}

Write one entry per firm below, fitted to each firm's clients:
{firms}

Return JSON: {{"notes": [{{"firm": "exact firm name", "subject": "...", "client_note": "...",
  "advisor_ping": ["line one", "line two"]}}]}}"""

PREDICTION = (r"will (recover|rebound|bounce)|buying opportunity|buy the dip|\bbottom(ed)? out", "a market prediction")
SHIELDED = (r"insulat|\bsmooth|shield|cushion|immune|unaffected|less risky|\bsteadier|\bbuffer"
            r"|less (price )?sensitiv|less volatil|lower volatil|dampen",
            "a claim that private holdings are shielded from volatility")
NOTE_RULES = [r for r in LINT_RULES if r[1] != "a specific figure"] + [PREDICTION, SHIELDED]


def signed(n: dict) -> str:
    """The note as the client sees it: body, the firm's signature, and the disclosure, all added by us."""
    return f'{n["client_note"]}\n\n[Advisor name]\n{n["firm"]}\n\n{DISCLOSURE}'


# "It does not mean these holdings are unaffected" is the disclaimer we want, not a claim.
NEGATABLE = {"a claim that private holdings are shielded from volatility", "return or performance language"}
NEGATION = re.compile(r"\b(not|never|no|without)\b|n't", re.I)


def lint_note(n: dict) -> list[str]:
    return lint_text(n["client_note"])


def _firm_line(r: dict) -> str:
    def m(x):
        return "unknown" if not x else f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M"
    return (f"- {r['firm_name']} ({r['firm_type']}, {r.get('city')}). Contact: {r.get('contact_name')}, "
            f"{r.get('contact_title')}. Average client {m(r.get('avg_client_usd'))}. "
            f"Alternatives held: {r.get('alts_exposure') or 'unknown'}.")


NOTES_SCHEMA = {
    "type": "object",
    "properties": {"notes": {"type": "array", "items": {
        "type": "object",
        "properties": {"firm": {"type": "string"}, "subject": {"type": "string"},
                       "client_note": {"type": "string"},
                       "advisor_ping": {"type": "array", "items": {"type": "string"}}},
        "required": ["firm", "subject", "client_note", "advisor_ping"],
        "additionalProperties": False}}},
    "required": ["notes"],
    "additionalProperties": False,
}


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=TIMEOUT_S, max_retries=RETRIES)


def _batch(day: dict, context: str, firms: list[dict]) -> list[dict]:
    msg = _client().messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        # structured output guarantees valid JSON; a batch of free-form JSON once came back with trailing text
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": NOTES_SCHEMA}},
        messages=[{"role": "user", "content": USER_TMPL.format(
            **day, context=context, firms="\n".join(_firm_line(r) for r in firms))}])
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Notes were cut off at the token limit.")
    text = next(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return json.loads(text)["notes"]


def draft_notes(day: dict, context: str, firms: list[dict], batch_size: int = 7) -> dict[str, dict]:
    """Notes for every firm, keyed by firm name, each with its lint flags."""
    from pipeline.kit import _scrub
    groups = [firms[i:i + batch_size] for i in range(0, len(firms), batch_size)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda g: _batch(day, context, g), groups))
    names = {r["firm_name"] for r in firms}
    out = {}
    for n in (n for batch in results for n in batch):
        n = _scrub(n)
        # drop a trailing "[Name], Firm" signature if the model added one; signed() adds ours
        n["client_note"] = re.sub(r"\s*\[[^\]]*\][^\n]*$", "", n["client_note"].strip())
        if n.get("firm") in names:
            n["flags"] = lint_note(n)
            out[n["firm"]] = n
    return out
