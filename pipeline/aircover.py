"""Air cover planner and holdout test. SIMULATION: no ads are running.

Idea: before a Tier A firm's outbound sequence starts, run 2 to 3 weeks of
category-education ads to affluent households around that firm's city, so the
firm's own clients have seen "evergreen alternatives" before the founder or CIO
hears from Equi. Ads never name the firm or any fund (Equi cannot advertise its
funds), and they end in "ask your advisor".

Whether that lifts meetings is an open question, so it ships as a holdout test:
firms are paired and one of each pair gets no air cover.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from pipeline.ads import DESCRIPTION_MAX, HEADLINE_MAX
from pipeline.compliance import DISCLOSURE
from pipeline.score import TODAY

SIMULATION = "Simulation: no ads are running."
PLATFORM_NOTE = ("Ad platforms can restrict targeting for financial services ads (for example age, gender, and "
                 "minimum location radius). Check each platform's current rules before any flight goes live.")
RADIUS_MILES = 25
SEED = 2024

# Copy by firm type. Checked against the search ad limits below.
COPY = {
    "MFO": ("Evergreen funds for families",
            "How evergreen alternatives and periodic liquidity work. Ask your advisor if they fit."),
    "RIA": ("Evergreen alternatives 101",
            "Private markets with periodic liquidity: how it works, the trade-offs. Ask your advisor."),
}


def launch_date(today: date = TODAY) -> date:
    """First Monday at least 3 weeks out, so the first flight can run in full."""
    d = today + timedelta(days=21)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def plan(records: list[dict], arms: dict[str, str], today: date = TODAY) -> list[dict]:
    """One row per Tier A firm, in rank order. Two sequences start per week."""
    rows = []
    tier_a = [r for r in records if r["tier"] == "A"]
    for i, r in enumerate(tier_a):
        seq_start = launch_date(today) + timedelta(weeks=i // 2)
        weeks = 3 if r.get("decision_structure") == "Committee-led" else 2
        flight_start = seq_start - timedelta(weeks=weeks)
        headline, desc = COPY.get(r["firm_type"], COPY["RIA"])
        arm = arms.get(r["firm_name"], "unpaired")
        if arm == "control":
            status = "Holdout: control, no air cover"
        elif today < flight_start:
            status = "Scheduled (simulated)"
        elif today < seq_start:
            status = "Live (simulated)"
        else:
            status = "Flight ended (simulated)"
        rows.append({
            "firm": r["firm_name"], "metro": f'{r.get("city")}, {r.get("state")}',
            "audience": (f'Within {RADIUS_MILES} miles of {r.get("city")}, {r.get("state")}. Households in the top 5% '
                         "by income or estimated net worth, interested in wealth, estate, or business-exit planning. "
                         "No age or gender targeting. The firm's own client list is not used."),
            "flight_start": flight_start.isoformat(), "flight_end": (seq_start - timedelta(days=1)).isoformat(),
            "sequence_start": seq_start.isoformat(), "weeks": weeks,
            "headline": headline, "description": desc, "disclosure": DISCLOSURE,
            "arm": arm, "status": status,
        })
    return rows


def copy_issues() -> list[str]:
    return [f"{t}: {kind} is {len(s)} characters" for t, (h, d) in COPY.items()
            for kind, s, mx in (("headline", h, HEADLINE_MAX), ("description", d, DESCRIPTION_MAX)) if len(s) > mx]


def holdout(records: list[dict], seed: int = SEED) -> tuple[list[dict], list[str], dict[str, str]]:
    """Pair Tier A and B firms by tier and firm type (closest scores together), then flip a seeded coin per pair.
    Returns (pairs, unpaired firm names, arm by firm)."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["tier"] in {"A", "B"}:
            groups.setdefault((r["tier"], r["firm_type"]), []).append(r)
    pairs, unpaired, arms = [], [], {}
    for (tier, ftype) in sorted(groups):
        firms = sorted(groups[(tier, ftype)], key=lambda r: -(r["score"] or 0))
        for i in range(0, len(firms) - 1, 2):
            a, b = firms[i], firms[i + 1]
            cover, control = (a, b) if rng.random() < 0.5 else (b, a)
            arms[cover["firm_name"]], arms[control["firm_name"]] = "air cover", "control"
            pairs.append({"pair": len(pairs) + 1, "tier": tier, "type": ftype,
                          "air_cover": cover["firm_name"], "control": control["firm_name"],
                          "score_gap": round(abs((a["score"] or 0) - (b["score"] or 0)), 1)})
        if len(firms) % 2:
            unpaired.append(firms[-1]["firm_name"])
    return pairs, unpaired, arms


PRIMARY_METRIC = "Meeting rate from outbound: meetings booked within 30 days of the sequence start, per firm."
GUARDRAIL = "Unsubscribe and negative reply rate on the outbound sequence, compared across arms."


def power_note(n_pairs: int) -> str:
    return (f"This list is too small for significance: {n_pairs} pairs means {n_pairs} firms per arm, so only a very "
            "large lift would show. The real test needs roughly 40+ firms per arm. Until then, treat any read as "
            "directional.")
