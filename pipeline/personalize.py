"""Write persona-specific outreach drafts with Claude.

Every firm worth pursuing gets two variants for the A/B test:
  A (kit-led):     "we already built your client materials"
  B (insight-led): "here's a gap we noticed in your lineup"

Guardrails:
- The model only gets facts from the scored record, and must return the facts
  it used so a rep can check them at a glance.
- No performance numbers, return claims, or guarantees (SEC marketing rule).
- Tone is taken from how Equi's own team writes.
"""
from __future__ import annotations

import json
import os
import re

MODEL = os.getenv("EQUI_MODEL", "claude-sonnet-5")

STYLE_GUIDE = """How Equi writes (taken from the team's own emails):
- Plain, direct, short sentences. Says what it means in the first line.
- Useful before it asks for anything. Their words: "sent as something genuinely useful rather than a pitch."
- Positions Equi as "a partner, not a vendor."
- Confident without hype. No "revolutionary", "cutting-edge", "unlock", "leverage", "synergy", "game-changer".
- No em dashes. No exclamation points. No "I hope this finds you well".
- Sounds like one busy professional writing to another."""

WHAT_EQUI_SELLS = """Equi: SEC-registered adviser offering institutional-grade liquid alternatives with managed risk and
downside protection, built as evergreen funds (no multi-year lock-up). Two shapes: Equi-branded funds, or a
white-label fund-of-funds built under the advisor's own brand. The key differentiator: Equi builds the
advisor's client-facing materials so the advisor can explain it to their own clients."""

PERSONA_PLAYBOOK = {
    "Founder-CIO": "Founder who is also the CIO. Can say yes alone. Respect their time: one clear idea, one easy next step. "
                   "Frame around what it saves them: a finished client story, not another manager to research.",
    "Principal": "Founder or principal. Can say yes without a committee. Lead with the business outcome for their practice "
                 "and the fact that the client materials are already done.",
    "CIO": "CIO. Sophisticated and already fluent in alternatives. Do not explain what alternatives are. "
           "Talk structure: evergreen liquidity, risk management, how it sits next to what they already own. Peer tone.",
    "Research lead": "Director of research or due-diligence lead at a committee-run firm. They are a champion, not the buyer, "
                     "and they have to defend the pick internally. Offer to equip them: DDQ answers, a draft IC memo, "
                     "risk and liquidity terms. Make their job easier, don't try to close them.",
}

VARIANTS = {
    "A": "Kit-led. The hook: Equi already drafted a client-facing piece under their firm's name, explaining evergreen "
         "alternatives to their clients. Offer to send it over. The email should make it feel like the work is done.",
    "B": "Insight-led. The hook: one specific observation about their firm from the facts provided (for example, a gap "
         "between their illiquid alts and a liquid sleeve, their client profile, or a transition toward a family-office "
         "model). State it plainly, say why it matters to their clients, and ask one question.",
}

SYSTEM = f"""You write first-touch outreach emails for Equi's sales team.

{WHAT_EQUI_SELLS}

{STYLE_GUIDE}

Hard rules:
- Use only the facts given about the firm. Never invent names, events, numbers, holdings, or relationships.
- Never state or imply fund performance, returns, yields, or guarantees.
- Under 110 words in the body. Subject under 8 words, lowercase is fine, no clickbait.
- One call to action, low effort (reply, or a 20-minute call).
- Two or three short paragraphs separated by blank lines. Greeting on its own line.
- Never tell the advisor what their clients should own or what their "next allocation" is. Observe, then ask.
- Refer to past meetings or intros only as stated in the notes, without embellishing.
- Sign off as {{sender_name}}.
- Return only JSON, no prose, no markdown fences."""

USER_TMPL = """Firm facts:
{facts}

Recipient persona guidance:
{persona}

Variant to write:
{variant}

{extra}
Return JSON:
{{"subject": "...", "body": "...", "angle": "one sentence on why this angle fits this firm",
  "facts_used": ["only facts about this firm that you used", "..."]}}"""


def firm_facts(r: dict) -> str:
    def m(x):
        if x is None or x != x:
            return "unknown"
        return f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M"
    lines = [
        f"Firm: {r['firm_name']} ({r['firm_type']}), {r['city']}, {r['state']}",
        f"Recipient: {r.get('contact_name') or 'unknown'}, {r.get('contact_title') or 'unknown'}",
        f"AUM: {m(r.get('aum_usd'))}",
        f"Average client size: {m(r.get('avg_client_usd'))}",
        f"Current alternatives: {r.get('alts_exposure') or 'unknown'}",
        f"Decision structure: {r.get('decision_structure')}",
    ]
    if r.get("mfo_transition"):
        lines.append("Signal: serves $10M+ clients as an RIA, a profile that is often moving toward a family-office model")
    if r.get("raw_notes"):
        lines.append(f"CRM notes: {r['raw_notes']}")
    return "\n".join(lines)


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _parse(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text)
    data["body"] = data["body"].replace("\u2014", ",").replace("\u2013", "-")
    return data


def draft(record: dict, variant: str = "A", sender_name: str = "the Equi team",
          instruction: str | None = None, previous: str | None = None) -> dict:
    """Write one draft. `instruction` + `previous` let a rep ask for a rewrite."""
    persona = PERSONA_PLAYBOOK.get(record.get("persona") or "", PERSONA_PLAYBOOK["CIO"])
    extra = ""
    if previous and instruction:
        extra = f"Rewrite this previous draft following the rep's note.\nRep's note: {instruction}\nPrevious draft:\n{previous}\n"
    msg = _client().messages.create(
        model=MODEL,
        max_tokens=3000,
        system=SYSTEM.replace("{sender_name}", sender_name),
        messages=[{"role": "user", "content": USER_TMPL.format(
            facts=firm_facts(record), persona=persona, variant=VARIANTS[variant], extra=extra)}],
    )
    out = _parse("".join(b.text for b in msg.content if getattr(b, "type", "") == "text"))
    out["variant"] = variant
    return out


def worth_drafting(record: dict) -> bool:
    return record.get("tier") in {"A", "B"} and record.get("contact_fit") != "Find contact" \
        and bool(record.get("contact_email"))


def assign_arms(records: list[dict], seed: int = 11) -> dict[str, str]:
    """Randomize each firm to variant A or B, stratified by tier and persona.

    Within each (tier, persona) group, firms are shuffled with a fixed seed and
    alternated A, B, A, ... so both arms get a similar mix. Same input, same arms.
    """
    import random
    rng = random.Random(seed)
    groups: dict[tuple, list[str]] = {}
    for r in records:
        groups.setdefault((r.get("tier"), r.get("persona")), []).append(r["firm_name"])
    arms = {}
    for key in sorted(groups, key=str):
        names = sorted(groups[key])
        rng.shuffle(names)
        start = rng.choice("AB")
        for i, n in enumerate(names):
            arms[n] = start if i % 2 == 0 else ("B" if start == "A" else "A")
    return arms
