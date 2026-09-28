"""Market-moment ad sets, one per fired Market signals trigger.

On a panic day, affluent investors search "should I sell my stocks". Equi cannot
advertise its funds, so these ads are category education: what evergreen
alternatives are and why periodic liquidity behaves differently, ending in "ask
your advisor". Google responsive search ad limits are enforced here in code,
not trusted to the model.

    python -m pipeline.ads --event 2024-08-05     # writes the ad set into data/signals.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline.compliance import DISCLOSURE, lint_text
from pipeline.kit import RETRIES, TIMEOUT_S, _scrub
from pipeline.personalize import MODEL, STYLE_GUIDE

HEADLINE_MAX = 30       # Google responsive search ads
DESCRIPTION_MAX = 90
CLOSING = "Ask your advisor about evergreen alternatives."

SCHEMA = {
    "type": "object",
    "properties": {
        "search_ads": {"type": "array", "items": {
            "type": "object",
            "properties": {"angle": {"type": "string"},
                           "headlines": {"type": "array", "items": {"type": "string"}},
                           "descriptions": {"type": "array", "items": {"type": "string"}}},
            "required": ["angle", "headlines", "descriptions"], "additionalProperties": False}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "social_ads": {"type": "array", "items": {
            "type": "object",
            "properties": {"platform": {"type": "string"}, "audience": {"type": "string"},
                           "headline": {"type": "string"}, "primary_text": {"type": "string"}},
            "required": ["platform", "audience", "headline", "primary_text"], "additionalProperties": False}},
        "landing_page": {"type": "object",
                         "properties": {"title": {"type": "string"},
                                        "sections": {"type": "array", "items": {
                                            "type": "object",
                                            "properties": {"heading": {"type": "string"},
                                                           "points": {"type": "array", "items": {"type": "string"}}},
                                            "required": ["heading", "points"], "additionalProperties": False}}},
                         "required": ["title", "sections"], "additionalProperties": False},
    },
    "required": ["search_ads", "keywords", "social_ads", "landing_page"],
    "additionalProperties": False,
}

SYSTEM = f"""You write paid ads for an alternatives firm on a volatile market day. The firm is not allowed to advertise
its funds to the public, so every ad is category education about evergreen alternatives (open-ended private funds
with periodic liquidity, interval and tender-offer funds): what they are, how their liquidity works, and the
trade-offs. Every ad leads the reader to talk to their own advisor.

{STYLE_GUIDE}

Hard rules:
- Never name the firm, any fund, or any manager. Never mention performance, returns, yield, or targets.
- Never promise safety or protection, never say private holdings are shielded from volatility, never predict markets,
  never tell anyone to buy or sell. Answer the panic with education, not reassurance.
- Search ads: exactly 3 variants, each with 5 headlines of at most {HEADLINE_MAX} characters and 2 descriptions of at
  most {DESCRIPTION_MAX} characters. Count characters, including spaces.
- Keywords: 15 to 20 queries a worried investor types on a day like this, written the way people type them.
- Social ads: exactly 2 variants for affluent investors (for example LinkedIn and Meta), with the audience in plain terms.
- Landing page: a title and 4 to 5 sections. The last section's last point is exactly "{CLOSING}"."""


def check(ad_set: dict) -> list[str]:
    """Character limits and structure, enforced in code, plus the client-copy rules on every line."""
    issues = []
    ads = ad_set.get("search_ads", [])
    if len(ads) != 3:
        issues.append(f"Expected 3 search ad variants, got {len(ads)}.")
    for i, ad in enumerate(ads, 1):
        for h in ad["headlines"]:
            if len(h) > HEADLINE_MAX:
                issues.append(f'Search ad {i}: headline is {len(h)} characters (max {HEADLINE_MAX}): "{h}"')
        for d in ad["descriptions"]:
            if len(d) > DESCRIPTION_MAX:
                issues.append(f'Search ad {i}: description is {len(d)} characters (max {DESCRIPTION_MAX}): "{d}"')
    if len(ad_set.get("social_ads", [])) != 2:
        issues.append(f'Expected 2 social ads, got {len(ad_set.get("social_ads", []))}.')
    page = ad_set.get("landing_page", {})
    last = (page.get("sections") or [{}])[-1].get("points", [""])[-1:] or [""]
    if last[0].strip() != CLOSING:
        issues.append(f'Landing page does not end with "{CLOSING}"')
    copy = [t for ad in ads for t in ad["headlines"] + ad["descriptions"]]
    copy += [s["headline"] + ". " + s["primary_text"] for s in ad_set.get("social_ads", [])]
    copy += [page.get("title", "")] + [p for s in page.get("sections", []) for p in s["points"]]
    for text in copy:
        issues += [f"Copy: {f}" for f in lint_text(text)]
    return issues


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=TIMEOUT_S, max_retries=RETRIES)


def _ask(prompt: str) -> dict:
    msg = _client().messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                                    output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                                    messages=[{"role": "user", "content": prompt}])
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Ad set was cut off at the token limit.")
    return _scrub(json.loads(next(b.text for b in msg.content if b.type == "text")))


def generate(day: dict, context: str) -> dict:
    """Ad set for one market day. One corrective pass if limits or rules are broken; the rest is flagged."""
    prompt = (f"Market day {day['date']}: S&P 500 (SPY) {day['spy_pct']:+.2f}%, VIX {day['vix']:.2f} "
              f"({day['vix_pct']:+.1f}% day over day). {context}\nWrite the ad set.")
    ad_set = _ask(prompt)
    issues = check(ad_set)
    if issues:
        ad_set = _ask(prompt + "\n\nA first draft broke these rules. Fix every one and keep everything else:\n- "
                      + "\n- ".join(issues) + "\n\nFirst draft:\n" + json.dumps(ad_set))
        ad_set["fixed_on_retry"] = issues
    ad_set["flags"] = check(ad_set)
    ad_set["disclosure"] = DISCLOSURE
    ad_set["model"] = MODEL
    return ad_set


if __name__ == "__main__":
    from datetime import date
    from dotenv import load_dotenv
    from pipeline import signals
    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="2024-08-05", choices=list(signals.EVENTS))
    args = ap.parse_args()
    path = Path("data/signals.json")
    data = json.loads(path.read_text())
    day = data["days"][signals.EVENTS[args.event]["days"][-1]]
    ad_set = generate(day, signals.EVENTS[args.event]["context"])
    ad_set["generated"] = str(date.today())
    data["fired"].setdefault(args.event, {})["ads"] = ad_set
    path.write_text(json.dumps(data, indent=2))
    print(f"Ad set for {args.event}: {len(ad_set['search_ads'])} search ads, {len(ad_set['keywords'])} keywords, "
          f"{len(ad_set['social_ads'])} social ads. Fixed on retry: {len(ad_set.get('fixed_on_retry', []))}. "
          f"Flags left: {len(ad_set['flags'])}")
    for f in ad_set["flags"]:
        print("  -", f)
