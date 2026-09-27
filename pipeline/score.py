"""Score cleaned firms against Equi's ICP.

Design:
- Hard gates run first. A gated firm never gets a fit score, only a reason.
- Six weighted criteria. Each returns a 0-1 value, a plain-English reason,
  and (when data is missing) the lowest and highest value it could be.
  That gives every firm a score *range*, which drives the confidence label
  and the "what would move this firm" line.
- Caps and modifiers run last (SFO, sub-band AUM, platform-gated, cross-border).
- Weights live in DEFAULT_WEIGHTS so the app can let a user re-weight live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

TODAY = date(2026, 9, 27)

DEFAULT_WEIGHTS = {
    "firm_type": 25,
    "client_wealth": 20,
    "decision_speed": 15,
    "alts_gap": 20,
    "aum": 10,
    "warmth": 10,
}

CRITERIA_LABELS = {
    "firm_type": "Firm type",
    "client_wealth": "Client wealth (QP fit)",
    "decision_speed": "Decision speed",
    "alts_gap": "Alts experience",
    "aum": "AUM band",
    "warmth": "Warmth and reach",
}

WHY_WEIGHT = {
    "firm_type": "Equi said firm type matters more than size. MFOs and RIAs becoming MFOs are the core buyer.",
    "client_wealth": "Fund eligibility is gated on qualified purchasers ($5M+). No QP clients, no sale.",
    "decision_speed": "Principal-led vs committee-led is Equi's best predictor of deal speed.",
    "alts_gap": "The win is in the middle: fluent enough to move, not so deep they built it themselves. Liquid-alts gap is the sweet spot.",
    "aum": "Band is $1B to $30B with a $5B+ bullseye. Equi said type beats size, and a $3B MFO in the alts sweet spot beats a $12B RIA that built its own alts desk, so size is a tiebreaker, not a driver.",
    "warmth": "A warm path shortens the cycle but never makes a bad-fit firm good, so it gets the least weight.",
}

TIERS = [(80, "A"), (60, "B"), (40, "C"), (0, "Park")]

ILLIQUID = {"private equity", "pe", "real estate", "re", "venture", "real assets",
            "infra", "art", "private credit"}
LIQUID = {"hedge funds"}


@dataclass
class Criterion:
    value: float | None          # 0-1, None if unknown
    reason: str
    lo: float = 0.0
    hi: float = 1.0
    known: bool = True

    def bounds(self):
        return (self.value, self.value) if self.known else (self.lo, self.hi)


@dataclass
class Result:
    gate: str | None = None
    criteria: dict[str, Criterion] = field(default_factory=dict)
    modifiers: list[tuple[int, str]] = field(default_factory=list)
    cap: tuple[str, str] | None = None       # (max tier, reason)
    decision_structure: str = "Unknown"
    mfo_transition: bool = False
    alts_list: list[str] = field(default_factory=list)


# ---------- helpers ----------

def classify_title(title: str | None) -> str:
    if not title:
        return "Unknown"
    t = title.lower()
    if t.startswith("ea ") or "assistant" in t:
        return "Gatekeeper"
    if any(k in t for k in ["research", "due diligence", "manager selection", "external managers"]):
        return "Research lead"
    founderish = any(k in t for k in ["founder", "principal", "owner", "managing partner", "president"])
    cio = "chief investment" in t or t.strip() in {"cio"} or "cio" in t.split() or "/cio" in t or "head of investments" in t
    if founderish and cio:
        return "Founder-CIO"
    if founderish:
        return "Principal"
    if cio:
        return "CIO"
    if "head of alternative" in t or "director of investment" in t or "product" in t or "portfolio manager" in t:
        return "Non-buyer seat"
    if "managing director" in t:
        return "Principal"
    return "Unknown"


PERSONA_RANK = {"Founder-CIO": 0, "Principal": 1, "CIO": 1, "Research lead": 2,
                "Unknown": 4, "Gatekeeper": 5, "Non-buyer seat": 6}


def best_contact(contacts: list[dict]) -> dict | None:
    if not contacts:
        return None
    ranked = sorted(contacts, key=lambda c: PERSONA_RANK[classify_title(c.get("title"))])
    return ranked[0]


def parse_alts(raw) -> list[str]:
    if pd.isna(raw):
        return []
    s = str(raw).lower().replace("/", ",")
    if s.strip() in {"none"}:
        return ["none"]
    if s.strip() in {"everything"}:
        return ["everything"]
    return [x.strip() for x in s.split(",") if x.strip()]


def fmt_b(x):
    return f"${x/1e9:.1f}B" if x else "unknown"


def fmt_m(x):
    if x is None or pd.isna(x):
        return "unknown"
    return f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"


def ramp(x, x0, x1, y0, y1):
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# ---------- the model ----------

def evaluate(f: pd.Series) -> Result:
    r = Result()
    ftype = f.firm_type
    aum = f.aum_usd if pd.notna(f.aum_usd) else None
    client = f.avg_client_usd if pd.notna(f.avg_client_usd) else None
    notes = (f.raw_notes or "").lower()
    custodian = str(f.custodian or "")

    # --- hard gates ---
    gates = {
        "Wirehouse": "Wirehouse. Has in-house alts teams and platform rules; Equi's partner model does not fit.",
        "TAMP": "TAMP. Sells model portfolios to advisors, which makes it a competitor channel, not a buyer.",
        "Broker-Dealer": "Broker-dealer. Product-shelf driven with in-house structuring; out of scope.",
        "Hedge Fund Manager": "Investment-side firm. A manager, not a buyer.",
        "Private Credit Manager": "Investment-side firm. A manager, not a buyer.",
    }
    if ftype in gates:
        r.gate = gates[ftype]
        return r
    if aum and aum > 30e9:
        r.gate = (f"{fmt_b(aum)} AUM, above the $30B ceiling. Firms this size staff their own "
                  "manager research and source the same managers Equi does.")
        return r

    contact = best_contact(f.contacts)
    persona = classify_title(contact.get("title")) if contact else "Unknown"

    # --- decision structure ---
    if "committee" in notes or "ic-driven" in notes:
        ds = "Committee-led"
    elif "principal-led" in notes or persona in {"Founder-CIO", "Principal"}:
        ds = "Principal-led"
    elif persona == "Research lead":
        ds = "Committee-led"
    elif persona == "CIO":
        ds = "CIO-led"
    else:
        ds = "Unknown"
    r.decision_structure = ds

    # --- MFO transition signal (inferred) ---
    r.mfo_transition = ftype == "RIA" and client is not None and client >= 10e6

    # 1. firm type
    if ftype == "MFO":
        c = Criterion(1.0, "Multi-family office, Equi's core buyer.")
    elif ftype == "RIA" and r.mfo_transition:
        c = Criterion(0.88, f"Independent RIA serving {fmt_m(client)} average clients. That profile is "
                            "rebranding toward multi-family office; treated as a transition signal (inferred, verify on ADV and site).")
    elif ftype == "RIA":
        c = Criterion(0.72, "Independent RIA. Right model, but not yet showing family-office traits.")
    elif ftype == "SFO":
        c = Criterion(0.32, "Single-family office. One family's money, fewer mandates to build for; weaker fit than an MFO.")
    else:
        c = Criterion(0.2, f"Unrecognized firm type '{ftype}'.")
    r.criteria["firm_type"] = c

    # 2. client wealth
    if client is None:
        c = Criterion(None, "Average client size unknown. This is the biggest open question for eligibility.",
                      lo=0.1, hi=1.0, known=False)
    elif 5e6 <= client <= 30e6:
        c = Criterion(1.0, f"Average client {fmt_m(client)}, inside the $5M to $30M sweet spot. Most clients are likely qualified purchasers.")
    elif 3e6 <= client < 5e6:
        v = ramp(client, 3e6, 5e6, 0.4, 0.8)
        c = Criterion(v, f"Average client {fmt_m(client)}, just under the $5M QP line. A meaningful slice of clients qualify, not most.")
    elif client > 30e6:
        c = Criterion(0.6, f"Average client {fmt_m(client)}, above the sweet spot. These clients usually get direct institutional access already.")
    else:
        c = Criterion(0.1, f"Average client {fmt_m(client)}. Few, if any, qualified purchasers.")
    r.criteria["client_wealth"] = c

    # 3. decision speed
    ds_map = {
        "Principal-led": (1.0, "Principal-led. The founder can say yes without a committee."),
        "CIO-led": (0.8 if ftype == "MFO" else 0.72,
                    "CIO-led. One sophisticated decision-maker, likely with a light approval step."),
        "Committee-led": (0.45, "Committee-led. Slower; win by equipping the research lead as a champion, not by closing them."),
    }
    if ds in ds_map:
        v, why = ds_map[ds]
        r.criteria["decision_speed"] = Criterion(v, why)
    else:
        r.criteria["decision_speed"] = Criterion(None, "Decision structure unknown.", lo=0.45, hi=1.0, known=False)

    # 4. alts gap
    alts = parse_alts(f.alts_exposure)
    r.alts_list = alts
    if not alts:
        c = Criterion(None, "Alts exposure unknown.", lo=0.2, hi=1.0, known=False)
    elif alts == ["none"]:
        c = Criterion(0.45, "No alternatives yet. Needs the most help, but it's a long education curve: pipeline, not a quick win.")
    elif alts == ["everything"] or "alts-native" in notes or len(alts) >= 3 and "hedge funds" in alts:
        c = Criterion(0.2, "Deep in alternatives, including hedge funds. Likely has its own manager-research infrastructure.")
    elif "model portfolios" in alts or "structured notes" in alts:
        c = Criterion(0.3, "Alts via packaged products only.")
    elif any(a in LIQUID for a in alts):
        c = Criterion(0.65, f"Already uses hedge funds ({', '.join(alts)}). Fluent, but Equi would be replacing or adding to a current manager.")
    else:
        c = Criterion(1.0, f"Has illiquid alts ({', '.join(alts)}) but no liquid or hedge-fund sleeve. Fluent enough to move fast, with a clear gap an evergreen absolute-return fund fills.")
    r.criteria["alts_gap"] = c

    # 5. AUM
    if aum is None:
        c = Criterion(None, "AUM not disclosed. Pull from Form ADV before outreach.", lo=0.2, hi=1.0, known=False)
    elif aum >= 5e9:
        c = Criterion(1.0, f"{fmt_b(aum)} AUM, in the $5B+ bullseye.")
    elif aum >= 1e9:
        v = ramp(aum, 1e9, 5e9, 0.55, 0.9)
        c = Criterion(v, f"{fmt_b(aum)} AUM, inside the band but under the $5B bullseye.")
    else:
        c = Criterion(0.15, f"{fmt_b(aum)} AUM, below the $1B floor.")
        r.cap = ("C", "Below the $1B floor; nurture until they grow into the band.")
    r.criteria["aum"] = c

    # 6. warmth and reach
    pts, bits = 0.0, []
    if "warm intro" in notes:
        pts += 0.5; bits.append("warm intro available")
    if "met at" in notes:
        pts += 0.4; bits.append("met in person")
    lt = f.last_touch_date if pd.notna(f.last_touch_date) else None
    if lt:
        days = (TODAY - lt).days
        if days <= 90:
            pts += 0.3; bits.append(f"touched {days} days ago")
        elif days <= 240:
            pts += 0.15; bits.append(f"last touch {days} days ago")
        else:
            bits.append(f"cold, last touch {days} days ago")
    else:
        bits.append("never contacted")
    has_named_email = contact and contact.get("email") and not f.generic_inbox
    if has_named_email:
        pts += 0.2; bits.append("direct email on file")
    else:
        bits.append("no direct email")
    r.criteria["warmth"] = Criterion(min(pts, 1.0), "; ".join(bits).capitalize() + ".")

    # --- modifiers and caps ---
    if "competitor platform" in notes:
        r.modifiers.append((-10, "Recently adopted a competitor platform. Displacement takes quarters, not weeks."))
    if f.country != "USA":
        r.modifiers.append((-15, "Non-US firm. Cross-border eligibility for US fund structures needs compliance review first."))
        r.cap = ("B", "Cross-border; not a Tier A until compliance clears it.")
    if "lpl" in custodian.lower():
        r.cap = ("C", "LPL-affiliated. Products generally need LPL platform approval before advisors can use them.")
    if ftype == "SFO":
        r.cap = ("C", "Single-family office; capped per Equi's guidance.")
    return r


def tier_for(score: float) -> str:
    for cut, t in TIERS:
        if score >= cut:
            return t
    return "Park"


TIER_ORDER = {"A": 0, "B": 1, "C": 2, "Park": 3}


def apply_cap(tier: str, cap):
    if cap and TIER_ORDER[tier] < TIER_ORDER[cap[0]]:
        return cap[0]
    return tier


def contact_fit(f, res: Result):
    contact = best_contact(f.contacts)
    if not contact or not contact.get("name"):
        return "Find contact", "No named decision-maker on file. Research the founder or CIO before any outreach."
    persona = classify_title(contact.get("title"))
    ideal = {"MFO": {"Founder-CIO", "CIO", "Principal"},
             "RIA": {"Founder-CIO", "Principal", "CIO"},
             "SFO": {"CIO", "Founder-CIO", "Principal"}}.get(f.firm_type, set())
    if res.decision_structure == "Committee-led":
        ideal = ideal | {"Research lead"}
    if persona in ideal:
        return "Right person", f"{contact['name']} ({contact['title']}) is the buyer Equi wants at a firm like this."
    if persona == "Research lead":
        return "Champion, not buyer", f"{contact['name']} is a research lead. Equip them to defend the pick; also map the CIO."
    return "Wrong seat", f"{contact.get('title')} is not the decision-maker. Route to the founder or CIO."


def route(tier, res: Result, cfit):
    if tier == "DQ":
        return "Disqualified"
    if cfit[0] == "Find contact":
        return "Research: find decision-maker"
    if tier == "A":
        if res.decision_structure == "Committee-led":
            return "Committee Kit to champion"
        return "Pre-built Kit + founder direct" if res.decision_structure == "Principal-led" else "Pre-built Kit + CIO peer sequence"
    if tier == "B":
        if res.decision_structure == "Committee-led":
            return "Committee Kit to champion, slower cadence"
        return "Market-signal alerts + Kit on engagement"
    if tier == "C":
        return "Education nurture"
    return "Park"


def score_firms(firms: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.values()) or 1
    rows = []
    for _, f in firms.iterrows():
        res = evaluate(f)
        contact = best_contact(f.contacts)
        base = {
            "firm_name": f.firm_name, "firm_type": f.firm_type,
            "aum_usd": f.aum_usd, "avg_client_usd": f.avg_client_usd,
            "city": f.city, "state": f.state, "country": f.country,
            "contact_name": contact.get("name") if contact else None,
            "contact_title": contact.get("title") if contact else None,
            "contact_email": contact.get("email") if contact else None,
            "contacts": f.contacts, "cleaning_notes": f.cleaning_notes,
            "raw_notes": f.raw_notes, "source_rows": f.source_rows,
            "alts_exposure": f.alts_exposure, "custodian": f.custodian,
        }
        if res.gate:
            rows.append({**base, "score": None, "score_lo": None, "score_hi": None,
                         "tier": "DQ", "gate": res.gate, "confidence": None,
                         "breakdown": {}, "modifiers": [], "cap": None,
                         "contact_fit": None, "contact_fit_why": None,
                         "route": "Disqualified", "decision_structure": None,
                         "mfo_transition": False, "mover": None, "persona": None})
            continue
        pts, lo, hi, breakdown, unknown = 0.0, 0.0, 0.0, {}, []
        for k, crit in res.criteria.items():
            weight = w[k] * 100 / total_w
            a, b = crit.bounds()
            if crit.known:
                pts += crit.value * weight
            else:
                pts += (a + b) / 2 * weight
                unknown.append(k)
            lo += a * weight
            hi += b * weight
            breakdown[k] = {"points": round((crit.value if crit.known else (a + b) / 2) * weight, 1),
                            "max": round(weight, 1), "reason": crit.reason, "known": crit.known,
                            "lo": round(a * weight, 1), "hi": round(b * weight, 1)}
        mod = sum(m for m, _ in res.modifiers)
        score, lo, hi = [max(0, min(100, x + mod)) for x in (pts, lo, hi)]
        tier = apply_cap(tier_for(score), res.cap)
        tier_hi = apply_cap(tier_for(hi), res.cap)
        known_share = 1 - len(unknown) / len(res.criteria)
        confidence = "High" if known_share == 1 else "Medium" if known_share >= 0.66 else "Low"
        mover = None
        if unknown and tier_hi != tier:
            names = [CRITERIA_LABELS[u].split(" (")[0].lower() for u in unknown]
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            verb = "checks" if len(names) == 1 else "check"
            mover = f"Could reach Tier {tier_hi} if {joined} {verb} out. Confirm before outreach."
        elif res.cap and tier_for(score) != tier:
            mover = f"Scores as Tier {tier_for(score)} on fit, held at {tier}: {res.cap[1]}"
        cfit = contact_fit(f, res)
        rows.append({**base, "score": round(score, 1), "score_lo": round(lo, 1), "score_hi": round(hi, 1),
                     "tier": tier, "gate": None, "confidence": confidence, "breakdown": breakdown,
                     "modifiers": res.modifiers, "cap": res.cap,
                     "contact_fit": cfit[0], "contact_fit_why": cfit[1],
                     "route": route(tier, res, cfit), "decision_structure": res.decision_structure,
                     "mfo_transition": res.mfo_transition, "mover": mover,
                     "persona": classify_title(contact.get("title")) if contact else None,
                     "alts_list": res.alts_list})
    out = pd.DataFrame(rows)
    out["_t"] = out.tier.map({**TIER_ORDER, "DQ": 9})
    out = out.sort_values(["_t", "score"], ascending=[True, False]).drop(columns="_t")
    return out.reset_index(drop=True)


if __name__ == "__main__":
    from pipeline.clean import clean_leads
    s = score_firms(clean_leads("data/sample-leads.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(s[["firm_name", "firm_type", "score", "score_lo", "score_hi", "tier", "confidence",
             "decision_structure", "contact_fit", "route"]].to_string())
    print()
    for _, r in s[s.mover.notna()].iterrows():
        print(r.firm_name, "->", r.mover)


# ---------- weight stress test ----------

def stability(firms: pd.DataFrame, weights: dict | None = None, n: int = 500,
              concentration: float = 20, seed: int = 7) -> pd.Series:
    """Share of randomly perturbed weightings in which each firm lands in Tier A.

    Weights are drawn from a Dirichlet centered on the chosen weights, which moves
    each weight roughly 5 to 8 points either way. High share = the ranking is
    driven by the firm, not by our weight choices.
    """
    import numpy as np
    w = weights or DEFAULT_WEIGHTS
    keys = list(w)
    base = np.array([w[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    hits: dict[str, int] = {}
    for _ in range(n):
        draw = rng.dirichlet(base / base.sum() * concentration) * 100
        s = score_firms(firms, dict(zip(keys, draw)))
        for name in s.loc[s.tier == "A", "firm_name"]:
            hits[name] = hits.get(name, 0) + 1
    return pd.Series({k: v / n for k, v in hits.items()}, name="tier_a_share")


# ---------- red flags and the bubble ----------

def red_flags(rec: dict) -> list[str]:
    """Plain-English reasons a firm is not a clean Tier A."""
    if rec.get("tier") == "DQ":
        return [rec["gate"]]
    flags = []
    for k, b in (rec.get("breakdown") or {}).items():
        if not b["known"]:
            flags.append(f"{CRITERIA_LABELS[k]}: unknown. {b['reason']}")
        elif b["max"] and b["points"] / b["max"] < (0.3 if k == "warmth" else 0.7):
            flags.append(f"{CRITERIA_LABELS[k]}: {b['reason']}")
    for pts, why in rec.get("modifiers") or []:
        flags.append(f"{pts} pts. {why}")
    if rec.get("cap"):
        flags.append(f"Capped at Tier {rec['cap'][0]}. {rec['cap'][1]}")
    if rec.get("contact_fit") in {"Find contact", "Wrong seat"}:
        flags.append(rec["contact_fit_why"])
    return flags


def bubble_reason(rec: dict) -> str | None:
    """Why a firm needs a human decision, or None if the call is clear."""
    share = rec.get("tier_a_share") or 0
    if rec.get("tier") == "DQ":
        return None
    if rec.get("tier") == "A" and share < 0.9:
        return f"Tier A under our weights, but only in {share:.0%} of alternative weightings."
    if rec.get("tier") != "A" and share >= 0.2:
        return f"Lands in Tier A in {share:.0%} of alternative weightings. Your priorities decide this one."
    if rec.get("cap") and (rec.get("score") or 0) >= 60:
        return "Scores well on fit but held back by a rule. Worth a human look."
    if rec.get("mover"):
        return rec["mover"]
    if any(p <= -10 for p, _ in rec.get("modifiers") or []):
        return "Good fit with a specific blocker."
    if rec.get("contact_fit") == "Find contact" and rec.get("tier") in {"B", "C"}:
        return "No named decision-maker yet. Find the right person before judging this one."
    return None


def bubble_type(rec: dict) -> str | None:
    """Which human a bubble firm needs.

    research: the tier hinges on missing data or a missing contact. Fill it first.
    decision: the tier hinges on Equi's priorities (Tier A in 20% to 90% of
              weightings) or on a rule holding back a firm that scores 60+.
    Research wins a tie, since a decision on missing data gets remade later.
    """
    if not rec.get("bubble"):
        return None
    if (rec.get("mover") or "").startswith("Could reach") or rec.get("contact_fit") == "Find contact":
        return "research"
    return "decision"
