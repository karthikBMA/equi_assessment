"""Equi demand engine. Run with: streamlit run app.py

Reads data/scored.json (from build.py) for enrichment, and re-scores live from
the CSV when the sidebar weights change.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline import kit as kits_mod
from pipeline import personalize
from pipeline import signals as sig
from pipeline import ads as ads_mod
from pipeline import aeo, aircover
from pipeline.compliance import DISCLOSURE
from pipeline.clean import clean_leads
from pipeline.score import (CRITERIA_LABELS, DEFAULT_WEIGHTS, WHY_WEIGHT, bubble_reason,
                            bubble_type,
                            red_flags, route, score_firms, stability, tier_for)
from pipeline.sequences import as_rows, sequence_for, short_route

CSV = "data/sample-leads.csv"
SCORED = Path("data/scored.json")
DRAFTS = Path("data/drafts.json")
KITS = Path("data/kits.json")
SIGNALS = Path("data/signals.json")

# Key comes from .env locally or st.secrets on Streamlit Cloud. Never shown.
load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
NO_KEY_MSG = "Writing needs an Anthropic key: add ANTHROPIC_API_KEY to .env, or to Streamlit secrets when deployed."

INK, PAPER, BRASS, SLATE, RED = "#1F2A44", "#FAFAF7", "#A8843A", "#2E3440", "#B4553F"
TIER_COLOR = {"A": BRASS, "B": INK, "C": "#5E6675", "Park": "#8A8F99", "DQ": RED}
SOURCE_COLOR = {"csv": "#5E6675", "derived": INK, "sec_adv": "#3E6B4F", "needs_lookup": RED}
FIELD_LABEL = {"aum_usd": "AUM", "avg_client_usd": "Avg client", "mfo_transition": "MFO transition",
               "sec_registration": "SEC registration"}
SOURCE_LABEL = {"csv": "CSV", "derived": "Derived", "sec_adv": "SEC ADV", "needs_lookup": "Needs lookup"}

st.set_page_config(page_title="Equi demand engine", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1900px; }}
  .pill {{ display:inline-block; padding:1px 8px; border-radius:3px; font-size:0.78rem;
          font-weight:600; letter-spacing:0.02em; border:1px solid currentColor; }}
  .muted {{ color:#6B7280; font-size:0.9rem; }}
  .summary {{ font-size:1.05rem; color:{SLATE}; margin:0.2rem 0 1rem 0; }}
  .crit {{ margin:0 0 0.9rem 0; }}
  .crit-head {{ display:flex; justify-content:space-between; font-size:0.9rem; font-weight:600; }}
  .bar {{ height:6px; background:#E6E4DC; border-radius:3px; margin:4px 0 4px 0; position:relative; }}
  .bar > .fill {{ height:6px; background:{INK}; border-radius:3px; }}
  .bar > .range {{ position:absolute; top:0; height:6px; background:{INK}; opacity:0.18; border-radius:3px; }}
  .crit-why {{ font-size:0.86rem; color:#4B5563; }}
  .flag {{ color:{RED}; font-size:0.9rem; margin-bottom:0.35rem; }}
  .src-row {{ display:grid; grid-template-columns: 150px 1fr 110px; gap:8px; font-size:0.88rem;
             padding:5px 0; border-bottom:1px solid #ECEAE3; }}
  .seq-row {{ display:grid; grid-template-columns: 80px 80px 1fr; gap:8px; font-size:0.88rem;
             padding:6px 0; border-bottom:1px solid #ECEAE3; }}
  .lede {{ font-family:'Source Serif 4',serif; font-size:1.18rem; line-height:1.6; color:{SLATE};
           max-width:900px; margin:0 0 0.7rem 0; }}
  .jump-line {{ font-size:0.98rem; color:{SLATE}; padding-top:0.5rem; line-height:1.45; }}
  .finding {{ font-size:1rem; line-height:1.55; margin:0 0 0.55rem 0; max-width:1000px; }}
  .src-note {{ grid-column: 2 / 4; color:#6B7280; font-size:0.8rem; margin-top:-3px; }}
</style>
""", unsafe_allow_html=True)


def pill(text, color):
    return f'<span class="pill" style="color:{color}">{html.escape(str(text))}</span>'


def money(x):
    if x is None or pd.isna(x):
        return "unknown"
    return f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"


ALTS_SHORT = {"private equity": "PE", "pe": "PE", "real estate": "RE", "re": "RE", "venture": "VC",
              "private credit": "private credit", "real assets": "real assets", "infra": "infra", "art": "art"}


def first_sentence(text: str) -> str:
    """First sentence, or the first two when the first is too short to stand alone."""
    parts = text.split(". ")
    head = parts[0] if len(parts[0]) >= 25 or len(parts) == 1 else ". ".join(parts[:2])
    return head if head.endswith(".") else head + "."


def strength_line(r) -> str:
    """Compact case for a Tier A firm, e.g. 'MFO, $25M clients, PE/RE only, principal-led'."""
    ftype = "RIA moving to MFO" if r.get("mfo_transition") else r["firm_type"]
    bits = [ftype]
    if r.get("avg_client_usd"):
        m = r["avg_client_usd"] / 1e6
        bits.append(f"${m:.0f}M clients" if m >= 10 or m == int(m) else f"${m:.1f}M clients")
    alts = r.get("alts_list") or []
    gap = (r.get("breakdown") or {}).get("alts_gap", {})
    if not alts:
        bits.append("alts unknown")
    elif alts == ["none"]:
        bits.append("no alts yet")
    elif "hedge funds" in alts:
        bits.append("already in hedge funds")
    elif gap.get("known") and gap.get("points") == gap.get("max"):
        bits.append("/".join(dict.fromkeys(ALTS_SHORT.get(a, a) for a in alts)) + " only")
    if r.get("decision_structure") and r["decision_structure"] != "Unknown":
        ds = r["decision_structure"]
        bits.append(ds if ds.startswith("CIO") else ds.lower())
    notes = (r.get("raw_notes") or "").lower()
    if "warm intro" in notes:
        bits.append("warm intro")
    elif "met at" in notes:
        bits.append("met in person")
    return ", ".join(bits)


def biggest_flag(r) -> str:
    """The one thing holding a non-A firm back: a binding cap, then the biggest penalty,
    then the criterion that loses the most points."""
    if r.get("cap") and tier_for(r["score"]) != r["tier"]:
        return f'Held at Tier {r["cap"][0]}: {first_sentence(r["cap"][1])}'
    if r.get("modifiers"):
        pts, why = min(r["modifiers"])
        return f"{pts} pts: {first_sentence(why)}"
    bd = r.get("breakdown") or {}
    if bd:
        k, b = max(bd.items(), key=lambda kv: kv[1]["max"] - kv[1]["points"])
        if k == "decision_speed" and r.get("decision_structure") == "Committee-led":
            return "Committee-led, so slower to decide. Equip the research lead."
        if k == "warmth":
            return "Cold relationship: " + b["reason"].split(";")[0].lower().rstrip(".") + "."
        return first_sentence(b["reason"])
    return r["red_flags"][0] if r.get("red_flags") else ""


def why_line(r) -> str:
    if r["tier"] == "DQ":
        return r["gate"]
    b = r.get("bubble")
    if b:
        # specific bubble reasons stand as is; generic ones get the actual issue
        if b.startswith(("Could reach", "No named")):
            return b
        if "alternative weightings" in b:
            return f'Tier A in {r["tier_a_share"]:.0%} of weightings. {biggest_flag(r)}'
        return biggest_flag(r)
    if r["tier"] == "A":
        return strength_line(r)
    return biggest_flag(r)


# ---------- data ----------

@st.cache_data
def load_firms():
    return clean_leads(CSV)


@st.cache_data
def load_enrichment():
    if not SCORED.exists():
        return {}
    return {r["firm_name"]: r for r in json.loads(SCORED.read_text())}


@st.cache_data(show_spinner="Running the weight stress test...")
def tier_a_share(weights: tuple) -> dict:
    w = dict(weights)
    n = 500 if w == DEFAULT_WEIGHTS else 200
    return stability(load_firms(), w, n=n).to_dict()


def rescore(weights: dict) -> list[dict]:
    df = score_firms(load_firms(), weights)
    records = json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))
    share = tier_a_share(tuple(weights.items()))
    enriched = load_enrichment()
    for r in records:
        r["tier_a_share"] = share.get(r["firm_name"], 0.0)
        r["red_flags"] = red_flags(r)
        r["bubble"] = bubble_reason(r)
        r["bubble_type"] = bubble_type(r)
        e = enriched.get(r["firm_name"], {})
        r["sources"] = e.get("sources", {})
        r["sec_adv"] = e.get("sec_adv")
        if e.get("contacts"):
            r["contacts"] = e["contacts"]
        r["sequence"] = sequence_for(r["route"])
        r["why"] = why_line(r)
    apply_overrides(records)
    return records


# ---------- overrides from Your call ----------

DEMOTE = {"A": "B", "B": "C", "C": "Park", "Park": "Park"}


def overrides() -> dict:
    return st.session_state.setdefault("overrides", {})


def apply_overrides(records):
    """Apply rep decisions on top of the model. The model tier is kept for export."""
    for r in records:
        o = overrides().get(r["firm_name"])
        r["model_tier"] = r["tier"]
        r["decision"] = o
        if not o or o["tier"] == r["tier"] or r["tier"] == "DQ":
            continue
        r["tier"] = o["tier"]
        r["overridden"] = True
        r["route"] = route(o["tier"], SimpleNamespace(decision_structure=r["decision_structure"]),
                           (r["contact_fit"], ""))
        r["sequence"] = sequence_for(r["route"])
        r["why"] = f'Overridden from Tier {r["model_tier"]}. {o["note"] or r["why"]}'



# ---------- sidebar ----------

def reset_weights():
    for k, v in DEFAULT_WEIGHTS.items():
        st.session_state[f"w_{k}"] = v


with st.sidebar:
    st.header("Weights")
    st.caption("Move a weight and every firm re-scores. Weights are rescaled to total 100.")
    weights = {}
    for k, v in DEFAULT_WEIGHTS.items():
        st.session_state.setdefault(f"w_{k}", v)
        weights[k] = st.slider(CRITERIA_LABELS[k], 0, 50, key=f"w_{k}", help=WHY_WEIGHT[k])
    total = sum(weights.values())
    custom = weights != DEFAULT_WEIGHTS
    st.button("Reset to Equi defaults", on_click=reset_weights, disabled=not custom,
              width="stretch")
    if total == 0:
        st.warning("All weights are zero. Using defaults.")
        weights = dict(DEFAULT_WEIGHTS)

    records = rescore(weights)
    st.header("Filters")
    all_tiers = ["A", "B", "C", "Park", "DQ"]
    all_routes = sorted({r["route"] for r in records})
    tiers = st.multiselect("Tier", all_tiers, placeholder="All tiers") or all_tiers
    routes = st.multiselect("Route", all_routes, format_func=short_route,
                            placeholder="All routes") or all_routes

if not load_enrichment():
    st.info("No enrichment found. Run `python build.py --offline` to add source tags and sequences.")

st.title("Equi demand engine")
st.markdown('<div class="muted">Demand plan and lead pipeline for independent RIAs and multi-family offices.'
            + (" Custom weights in use." if custom else "") + "</div>", unsafe_allow_html=True)

TABS = ["Start here", "Kit Studio", "Market signals", "Air cover", "Shortlist", "Your call", "Drafts", "Review queue"]
tab_start, tab_kit, tab_signals, tab_air, tab_shortlist, tab_call, tab_drafts, tab_queue = st.tabs(TABS, key="tab",
                                              on_change="rerun")


# ---------- firm detail ----------

def render_breakdown(r):
    parts = []
    for k, b in r["breakdown"].items():
        mx = b["max"] or 1
        fill = b["points"] / mx * 100
        rng = ""
        extra = ""
        if not b["known"]:
            rng = (f'<div class="range" style="left:{b["lo"]/mx*100:.0f}%;'
                   f'width:{(b["hi"]-b["lo"])/mx*100:.0f}%"></div>')
            extra = f' <span class="muted">(unknown, could be {b["lo"]:.0f} to {b["hi"]:.0f})</span>'
        parts.append(
            f'<div class="crit"><div class="crit-head"><span>{CRITERIA_LABELS[k]}</span>'
            f'<span>{b["points"]:.1f} / {b["max"]:.0f}</span></div>'
            f'<div class="bar">{rng}<div class="fill" style="width:{fill:.0f}%"></div></div>'
            f'<div class="crit-why">{html.escape(b["reason"])}{extra}</div></div>')
    for pts, why in r.get("modifiers") or []:
        parts.append(f'<div class="flag">{pts} pts. {html.escape(why)}</div>')
    if r.get("cap"):
        parts.append(f'<div class="flag">Capped at Tier {r["cap"][0]}. {html.escape(r["cap"][1])}</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_sources(r):
    src = r.get("sources") or {}
    if not src:
        st.caption("No source tags. Run build.py to enrich.")
        return
    rows = []
    for field, t in src.items():
        v = t["value"]
        if field in {"aum_usd", "avg_client_usd"}:
            v = money(v)
        elif isinstance(v, bool):
            v = "Yes" if v else "No"
        elif isinstance(v, list):
            v = ", ".join(map(str, v))
        v = "not found" if v is None else v
        label = FIELD_LABEL.get(field, field.replace("_", " ").capitalize())
        note = f'<div class="src-note">{html.escape(t["note"])}</div>' if t.get("note") else ""
        rows.append(f'<div class="src-row"><span class="muted">{label}</span><span>{html.escape(str(v))}</span>'
                    f'<span>{pill(SOURCE_LABEL[t["source"]], SOURCE_COLOR[t["source"]])}</span>{note}</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_detail(r):
    st.divider()
    tier = r["tier"]
    st.subheader(r["firm_name"])
    loc = ", ".join(x for x in (r.get("city"), r.get("state")) if x)
    head = [pill("Disqualified" if tier == "DQ" else f"Tier {tier}", TIER_COLOR[tier])
            + (f' <span class="muted">overridden from Tier {r["model_tier"]}</span>' if r.get("overridden") else ""),
            html.escape(f'{r["firm_type"]} in {loc}'), f'AUM {money(r["aum_usd"])}',
            f'avg client {money(r["avg_client_usd"])}']
    if tier != "DQ":
        head += [f'score {r["score"]:.0f} (range {r["score_lo"]:.0f} to {r["score_hi"]:.0f})',
                 f'{r["confidence"]} confidence', f'Tier A in {r["tier_a_share"]:.0%} of weightings']
    st.markdown(f'<div>{" &nbsp;·&nbsp; ".join(head)}</div>', unsafe_allow_html=True)

    if tier == "DQ":
        st.markdown(f'<div class="flag">{html.escape(r["gate"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div><b>Why:</b> {html.escape(r["why"])}</div>', unsafe_allow_html=True)
        st.markdown(f'**Route:** {short_route(r["route"])}. {r["route"]}.')
        if r.get("mover"):
            st.markdown(f'<div class="muted">{html.escape(r["mover"])}</div>', unsafe_allow_html=True)
        if r.get("bubble"):
            label = "Needs research" if r["bubble_type"] == "research" else "Needs a call from your team"
            st.markdown(f'<div class="muted">{label}: {html.escape(r["bubble"])}</div>',
                        unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        if tier != "DQ":
            st.markdown("#### Score breakdown")
            render_breakdown(r)
            st.markdown("#### Contact fit")
            st.markdown(f'**{r["contact_fit"]}.** {r["contact_fit_why"]}')
            for c in r.get("contacts") or []:
                src = c.get("email_source")
                tagged = f' {pill(SOURCE_LABEL[src], SOURCE_COLOR[src])}' if src else ""
                st.markdown(f'<div class="muted">{html.escape(str(c.get("name") or "No name"))}, '
                            f'{html.escape(str(c.get("title") or "no title"))}, '
                            f'{html.escape(str(c.get("email") or "no email"))}{tagged}</div>',
                            unsafe_allow_html=True)
            st.markdown("#### Sequence")
            seq = r["sequence"]
            st.markdown(f'<div class="muted">{html.escape(seq["goal"])}</div>', unsafe_allow_html=True)
            rows = "".join(f'<div class="seq-row"><span>{t["Day"]}</span><span class="muted">{t["Channel"]}</span>'
                           f'<span>{html.escape(t["What"])}</span></div>' for t in as_rows(r["route"]))
            st.markdown(rows, unsafe_allow_html=True)
    with right:
        st.markdown("#### Red flags")
        if r["red_flags"]:
            st.markdown("".join(f'<div class="flag">{html.escape(f)}</div>' for f in r["red_flags"]),
                        unsafe_allow_html=True)
        else:
            st.caption("None. Clean fit.")
        st.markdown("#### Where each field came from")
        render_sources(r)
        st.markdown("#### Cleaning notes")
        if r.get("cleaning_notes"):
            for n in r["cleaning_notes"]:
                st.markdown(f'<div class="muted">{html.escape(n)}</div>', unsafe_allow_html=True)
        else:
            st.caption("Nothing changed from the CSV.")
        st.caption(f'Source rows: {", ".join(map(str, r["source_rows"]))}')


# ---------- shortlist ----------

with tab_shortlist:
    n_dq = sum(r["tier"] == "DQ" for r in records)
    n_a = sum(r["tier"] == "A" for r in records)
    n_stable = sum((r["tier_a_share"] or 0) >= 0.9 for r in records if r["tier"] != "DQ")
    n_decide = sum(r.get("bubble_type") == "decision" and not r["decision"] for r in records)
    n_research = sum(r.get("bubble_type") == "research" and not r["decision"] for r in records)
    st.markdown(
        f'<div class="summary">{len(records)} firms scored, {n_dq} disqualified. '
        f'<span style="color:{BRASS};font-weight:600">Call {n_a} now</span>: the Tier A firms.<br>'
        f'{n_stable} hold Tier A in 90%+ of alternative weightings, so they do not depend on our weight choices.<br>'
        f'{n_decide} need a call from your team. {n_research} need research before anyone decides.</div>',
        unsafe_allow_html=True)
    st.button("Open Your call", type="tertiary", on_click=lambda: st.session_state.update(tab="Your call"))

    shown = [r for r in records if r["tier"] in tiers and r["route"] in routes]
    table = pd.DataFrame([{
        "Firm": r["firm_name"], "Type": r["firm_type"],
        "AUM": r["aum_usd"] / 1e9 if r["aum_usd"] else None,
        "Avg client": r["avg_client_usd"] / 1e6 if r["avg_client_usd"] else None,
        "Score": r["score"],
        "Range": "" if r["score"] is None else f'{r["score_lo"]:.0f} to {r["score_hi"]:.0f}',
        "Tier": r["tier"] + (" (overridden)" if r.get("overridden") else ""),
        "A stability": (r["tier_a_share"] or 0) * 100 if r["tier"] != "DQ" else None,
        "Route": short_route(r["route"]), "Confidence": r["confidence"], "Why": r["why"],
    } for r in shown])

    if table.empty:
        st.caption("No firms match these filters.")
    else:
        styled = table.style.map(lambda t: f"color:{TIER_COLOR.get(t.split()[0], SLATE)};font-weight:600",
                                 subset=["Tier"])
        event = st.dataframe(
            styled, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 640),
            on_select="rerun", selection_mode="single-row", key=f"shortlist-{hash((tuple(tiers), tuple(routes), tuple(weights.items())))}",
            column_config={
                "Type": st.column_config.TextColumn(width=48),
                "AUM": st.column_config.NumberColumn(format="$%.1fB", width=62),
                "Avg client": st.column_config.NumberColumn(format="$%.1fM", width=76),
                "Score": st.column_config.NumberColumn(format="%.0f", width=52),
                "Range": st.column_config.TextColumn(width=74),
                "Tier": st.column_config.TextColumn(width=40 if not any(r.get("overridden") for r in shown) else 120),
                "Confidence": st.column_config.TextColumn(width=84),
                "Route": st.column_config.TextColumn(width=150),
                "Firm": st.column_config.TextColumn(width=210),
                "Why": st.column_config.TextColumn(width="large", help="Why this tier, in one line."),
                "A stability": st.column_config.NumberColumn(
                    format="%.0f%%", width=84, help="Share of perturbed weightings (500 at defaults, 200 at custom weights) where the firm lands in Tier A."),
            })
        picked = event.selection.rows
        if picked:
            st.session_state["firm"] = shown[picked[0]]["firm_name"]
        current = next((r for r in shown if r["firm_name"] == st.session_state.get("firm")), None)
        if current:
            render_detail(current)
        else:
            st.caption("Select a row to open the firm.")


# ---------- your call ----------

def lookup_fields(r) -> list[str]:
    return [FIELD_LABEL.get(k, k.replace("_", " ")) for k, t in (r.get("sources") or {}).items()
            if t["source"] == "needs_lookup" and k != "sec_registration"]


def the_question(r) -> str:
    share = r["tier_a_share"] or 0
    flag = biggest_flag(r)
    if r["bubble_type"] == "research":
        if r.get("contact_fit") == "Find contact":
            return "Who is the decision-maker? There is no named founder or CIO on file."
        return f'{r["mover"].split(". ")[0]}. Missing: {", ".join(lookup_fields(r)) or "see red flags"}.'
    if r["model_tier"] == "A":
        return f"Should this stay Tier A? It drops out in {1 - share:.0%} of weightings. The weak spot: {flag}"
    if share >= 0.2:
        return f"Should this be Tier A? It makes A in {share:.0%} of weightings. What holds it back: {flag}"
    return f"Should the rule hold? {flag}"


def recommend(r) -> tuple[str, str]:
    """(action, reason). Action is one of Keep, Promote to A, Demote."""
    share = r["tier_a_share"] or 0
    t = r["model_tier"]
    if r["bubble_type"] == "research":
        if r.get("contact_fit") == "Find contact":
            return "Keep", "Keep. Find the founder or CIO first; the route updates once a contact is added."
        return "Keep", (f"Keep at Tier {t} until someone confirms {', '.join(lookup_fields(r))}. "
                        "Form ADV or a first call answers most of it.")
    if t == "A":
        return "Keep", f"Keep at A. It holds A in {share:.0%} of weightings, and the weak spot is worth probing on a call, not a reason to wait."
    if r.get("cap") and tier_for(r["score"]) != t:
        return "Keep", f"Keep at Tier {t}. The rule is there for a reason: {first_sentence(r['cap'][1])} Revisit when it no longer applies."
    if r.get("modifiers") and share < 0.2:
        pts, why = min(r["modifiers"])
        return "Keep", f"Keep at Tier {t} until this is resolved: {first_sentence(why)}"
    if share >= 0.4:
        return "Promote to A", (f"Promote. It makes A in {share:.0%} of weightings, so the call is close, and a first "
                                "conversation is cheap next to missing a good fit.")
    return "Keep", f"Keep at Tier {t}. It makes A in only {share:.0%} of weightings."


def decide(firm, action, tier, model_tier):
    note = st.session_state.get(f"note|{firm}", "").strip()
    overrides()[firm] = {"action": action, "tier": tier, "model_tier": model_tier, "note": note,
                         "at": datetime.now().strftime("%Y-%m-%d %H:%M")}


def undo(firm):
    overrides().pop(firm, None)


def render_call_row(r, rec_action, rec_why):
    firm, t = r["firm_name"], r["model_tier"]
    o = r["decision"]
    status = f'{o["action"]}{", now Tier " + o["tier"] if o["tier"] != t else ""}' if o else "Open"
    with st.expander(f'{firm}  ·  Tier {t}  ·  {status}', expanded=False):
        st.markdown(f'<div><b>The question.</b> {html.escape(the_question(r))}</div>', unsafe_allow_html=True)
        if r["red_flags"]:
            st.markdown("".join(f'<div class="flag">{html.escape(f)}</div>' for f in r["red_flags"]),
                        unsafe_allow_html=True)
        st.markdown(f'<div><b>Our recommendation.</b> {html.escape(rec_why)}</div>',
                    unsafe_allow_html=True)
        if o:
            st.caption(f'Decided {o["at"]}: {o["action"]}' + (f'. Note: {o["note"]}' if o["note"] else ""))
            st.button("Undo", key=f"undo|{firm}", on_click=undo, args=(firm,))
            return
        st.text_input("Note (optional, saved with the decision)", key=f"note|{firm}",
                      placeholder="e.g. Spoke to their COO, committee meets monthly")
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        c1.button("Keep", key=f"keep|{firm}", on_click=decide, args=(firm, "Keep", t, t),
                  type="primary" if rec_action == "Keep" else "secondary", width="stretch")
        c2.button("Promote to A", key=f"promote|{firm}", on_click=decide, args=(firm, "Promote to A", "A", t),
                  disabled=t == "A", type="primary" if rec_action == "Promote to A" else "secondary", width="stretch")
        c3.button(f"Demote to {DEMOTE[t]}", key=f"demote|{firm}", on_click=decide,
                  args=(firm, f"Demote to {DEMOTE[t]}", DEMOTE[t], t), width="stretch")


with tab_call:
    in_play = [r for r in records if r.get("bubble_type") or r["decision"]]
    decision = [r for r in in_play if r.get("bubble_type") != "research"]
    research = [r for r in in_play if r.get("bubble_type") == "research"]
    n_open = sum(not r["decision"] for r in in_play)
    st.markdown(f'<div class="summary">{n_open} open, {len(in_play) - n_open} decided. '
                "Decisions apply across the app and reset when the session ends; export them to keep a record.</div>",
                unsafe_allow_html=True)

    st.markdown("### Your call")
    st.caption("The tier depends on Equi's priorities or on a rule. The data is complete; someone has to decide.")
    for r in decision:
        render_call_row(r, *recommend(r))

    st.markdown("### Needs research")
    st.caption("The tier depends on data we do not have yet. Find it, then decide.")
    for r in research:
        render_call_row(r, *recommend(r))

    if overrides():
        by_firm = {r["firm_name"]: r for r in records}
        export = pd.DataFrame([{
            "firm": f, "model_tier": o["model_tier"], "your_tier": o["tier"], "action": o["action"],
            "our_recommendation": recommend(by_firm[f])[0] if f in by_firm and by_firm[f].get("bubble_type") else "",
            "note": o["note"], "decided_at": o["at"],
            "weights": "default" if not custom else json.dumps(weights),
        } for f, o in overrides().items()])
        agree = (export.action == export.our_recommendation).mean()
        st.caption(f"You agreed with our recommendation on {agree:.0%} of decisions. "
                   "A low number means the weights need retuning.")
        st.download_button("Export decisions as CSV", export.to_csv(index=False), "equi_decisions.csv", "text/csv")


# ---------- drafts ----------

@st.cache_data
def load_drafts() -> dict:
    return json.loads(DRAFTS.read_text()) if DRAFTS.exists() else {}


def draft_store() -> dict:
    """Working copy of every draft, keyed 'firm|variant'. Survives switching firms."""
    store = st.session_state.setdefault("drafts", {})
    if not store:
        for firm, vs in load_drafts().items():
            for v, d in vs.items():
                if "error" not in d:
                    store[f"{firm}|{v}"] = dict(d)
    return store


def approvals() -> dict:
    return st.session_state.setdefault("approved", {})


def sync_edit(firm, v):
    d = draft_store()[f"{firm}|{v}"]
    d["subject"] = st.session_state[f"subj|{firm}|{v}"]
    d["body"] = st.session_state[f"body|{firm}|{v}"]
    d["edited"] = True


def reseed(firm, v):
    """Drop widget state so the text boxes reload from the store on the next run."""
    for k in (f"subj|{firm}|{v}", f"body|{firm}|{v}"):
        st.session_state.pop(k, None)


def rewrite(r, v):
    firm = r["firm_name"]
    note = st.session_state.get(f"rw|{firm}|{v}", "").strip()
    if not note:
        st.session_state["draft_msg"] = ("warning", "Add a note first, e.g. 'shorter, mention the Future Proof meeting'.")
        return
    d = draft_store()[f"{firm}|{v}"]
    prev = f'Subject: {d["subject"]}\n\n{d["body"]}'
    try:
        new = personalize.draft(r, v, sender_name=st.session_state.get("sender", "Karthik"),
                                instruction=note, previous=prev)
    except Exception as e:
        st.session_state["draft_msg"] = ("error", f"Rewrite failed: {type(e).__name__}: {str(e)[:200]}")
        return
    draft_store()[f"{firm}|{v}"] = {**new, "rewritten_with": note}
    if approvals().get(firm) == v:
        approvals().pop(firm)  # the approved text changed, so it needs a fresh look
    reseed(firm, v)
    st.session_state[f"rw|{firm}|{v}"] = ""
    st.session_state["draft_msg"] = ("success", f"Variant {v} rewritten. Review it before approving.")


def write_both(r):
    try:
        for v in ("A", "B"):
            draft_store()[f'{r["firm_name"]}|{v}'] = personalize.draft(
                r, v, sender_name=st.session_state.get("sender", "Karthik"))
    except Exception as e:
        st.session_state["draft_msg"] = ("error", f"Drafting failed: {type(e).__name__}: {str(e)[:200]}")


def approve(firm, v):
    approvals()[firm] = v


def unapprove(firm):
    approvals().pop(firm, None)


def recipient(r) -> tuple[str | None, str | None]:
    src = (r.get("sources") or {}).get("contact_email") or {}
    email = src.get("value") or r.get("contact_email")
    return email, (src.get("note") if src.get("source") == "derived" else None)


def mailto(email, subject, body) -> str:
    return f"mailto:{quote(email or '', safe='@')}?subject={quote(subject)}&body={quote(body)}"


def render_variant(r, v, arm):
    firm = r["firm_name"]
    d = draft_store()[f"{firm}|{v}"]
    approved = approvals().get(firm) == v
    title = f'Variant {v}: {"kit-led" if v == "A" else "insight-led"}'
    tags = (" " + pill("test arm", BRASS) if v == arm else "") + (" " + pill("approved", "#3E6B4F") if approved else "")
    st.markdown(f"#### {title}{tags}", unsafe_allow_html=True)
    st.session_state.setdefault(f"subj|{firm}|{v}", d["subject"])
    st.session_state.setdefault(f"body|{firm}|{v}", d["body"])
    st.text_input("Subject", key=f"subj|{firm}|{v}", on_change=sync_edit, args=(firm, v))
    st.text_area("Body", key=f"body|{firm}|{v}", height=300, on_change=sync_edit, args=(firm, v))
    if d.get("angle"):
        st.caption(f'Angle: {d["angle"]}')
    st.markdown('<div class="muted"><b>Facts used</b> (check each against the firm record)</div>'
                + "".join(f'<div class="muted">· {html.escape(f)}</div>' for f in d.get("facts_used", [])),
                unsafe_allow_html=True)
    if d.get("rewritten_with"):
        st.caption(f'Rewritten with note: "{d["rewritten_with"]}"')

    st.text_input("Rewrite with note", key=f"rw|{firm}|{v}", placeholder="e.g. shorter, lead with the client letter",
                  disabled=not HAS_KEY)
    c1, c2 = st.columns(2)
    c1.button("Rewrite", key=f"rwb|{firm}|{v}", on_click=rewrite, args=(r, v), disabled=not HAS_KEY,
              width="stretch", help=None if HAS_KEY else NO_KEY_MSG)
    if approved:
        c2.button("Unapprove", key=f"un|{firm}|{v}", on_click=unapprove, args=(firm,), width="stretch")
        email, _ = recipient(r)
        st.link_button("Open in email", mailto(email, d["subject"], d["body"]), width="stretch")
        st.caption("Copy the text (icon at top right):")
        st.code(f'To: {email}\nSubject: {d["subject"]}\n\n{d["body"]}', language=None, wrap_lines=True)
        if v != arm:
            st.caption(f"This is off the test arm ({arm}). Fine to send, but it will not count toward the A/B read.")
    else:
        c2.button("Approve", key=f"ap|{firm}|{v}", on_click=approve, args=(firm, v), type="primary",
                  width="stretch", help="One variant per firm. Approving this replaces any other approval.")


with tab_drafts:
    store = draft_store()
    by_firm = {r["firm_name"]: r for r in records}
    drafted = [n for n in by_firm if f"{n}|A" in store or f"{n}|B" in store]
    missing = [r["firm_name"] for r in records if personalize.worth_drafting(r) and r["firm_name"] not in drafted]
    # arms come from the default-weight build so they do not move with the sliders
    arms = personalize.assign_arms([r for r in load_enrichment().values() if r["firm_name"] in drafted])
    arms.update(personalize.assign_arms([by_firm[n] for n in missing], seed=12))

    msg = st.session_state.pop("draft_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    n_ok = len(approvals())
    st.markdown(f'<div class="summary">{len(drafted)} firms with drafts, {n_ok} approved. '
                "Each firm is randomly assigned a test arm (A kit-led, B insight-led), balanced by tier and persona. "
                "Send the test arm unless there is a reason not to.</div>", unsafe_allow_html=True)
    if not HAS_KEY:
        st.caption(NO_KEY_MSG + " Pre-written drafts still load, and editing, approving and export all work.")

    top1, top2 = st.columns([3, 1])
    st.session_state.setdefault("sender", "Karthik")
    top2.text_input("Sign rewrites as", key="sender")

    # labels stay fixed: a label that changes on approve makes Streamlit lose the selection
    options = drafted + missing
    pick = top1.selectbox("Firm", options, key="draft_firm") if options else None
    if pick:
        r = by_firm[pick]
        state = (f"Approved variant {approvals()[pick]}" if pick in approvals()
                 else "No drafts yet" if pick in missing else "Not reviewed")
        st.markdown(f'<div class="muted">Tier {r["tier"]} &nbsp;·&nbsp; {state}</div>', unsafe_allow_html=True)
        email, inferred = recipient(r)
        st.markdown(f'<div>To <b>{html.escape(str(r.get("contact_name")))}</b>, {html.escape(str(r.get("contact_title")))}, '
                    f'{html.escape(str(email))} &nbsp;·&nbsp; Route: {short_route(r["route"])} &nbsp;·&nbsp; '
                    f'Test arm: <b>{arms.get(pick, "A")}</b></div>', unsafe_allow_html=True)
        if inferred:
            st.caption(f"Email is inferred. {inferred}")
        if r["tier"] not in {"A", "B"}:
            st.caption(f'Now Tier {r["tier"]}, so this firm is off the outreach list. Drafts kept for reference.')
        if pick in missing:
            st.button("Write both drafts", on_click=write_both, args=(r,), disabled=not HAS_KEY,
                      help=None if HAS_KEY else NO_KEY_MSG)
        else:
            left, right = st.columns(2, gap="large")
            for col, v in ((left, "A"), (right, "B")):
                with col:
                    if f"{pick}|{v}" in store:
                        render_variant(r, v, arms.get(pick, "A"))

    if approvals():
        rows = []
        for firm, v in approvals().items():
            r, d = by_firm.get(firm), store[f"{firm}|{v}"]
            if not r:
                continue
            email, _ = recipient(r)
            rows.append({"email": email, "first_name": (r.get("contact_name") or "").split()[0],
                         "firm": firm, "subject": d["subject"], "body": d["body"], "variant": v,
                         "route": short_route(r["route"])})
        st.divider()
        st.download_button(f"Export {len(rows)} approved as CSV", pd.DataFrame(rows).to_csv(index=False),
                           "equi_approved_drafts.csv", "text/csv",
                           help="Columns: email, first_name, firm, subject, body, variant, route. Ready for a sequencer.")


# ---------- review queue (shared by every tab that produces something to approve) ----------

def queue() -> list[dict]:
    return st.session_state.setdefault("queue", [])


def queue_add(kind: str, firm: str, title: str, payload: dict):
    """Add, or replace the pending item of this kind for this firm (and the same market event, for notes)."""
    q = queue()
    q[:] = [i for i in q if not (i["kind"] == kind and i["firm"] == firm and i["status"] == "pending"
                                 and i["payload"].get("event") == payload.get("event"))]
    q.append({"kind": kind, "firm": firm, "title": title, "payload": payload, "status": "pending",
              "added": datetime.now().strftime("%Y-%m-%d %H:%M")})


# ---------- kit studio ----------

@st.cache_data
def load_kits() -> dict:
    return json.loads(KITS.read_text()) if KITS.exists() else {}


def kit_store() -> dict:
    store = st.session_state.setdefault("kits", {})
    for firm, k in load_kits().items():
        store.setdefault(firm, k)
    return store


def make_kit(r):
    try:
        kit_store()[r["firm_name"]] = kits_mod.generate(r)
        st.session_state["kit_msg"] = ("success", "Kit written. Read it before sending it to review.")
    except Exception as e:
        st.session_state["kit_msg"] = ("error", f"Kit failed: {type(e).__name__}: {str(e)[:200]}")


with tab_kit:
    msg = st.session_state.pop("kit_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    st.markdown('<div class="summary">A client letter under the firm\'s own name, talking points for the advisor, '
                "and for committee-led firms an IC memo outline. Educational only, with a fixed compliance footer. "
                "Branding is a palette derived from the firm name; production would pull the logo and colors "
                "from the firm's website.</div>", unsafe_allow_html=True)
    eligible = [r for r in records if r["tier"] in {"A", "B"}]
    by_name = {r["firm_name"]: r for r in eligible}
    ks = kit_store()
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Firm (Tier A and B)", list(by_name), key="kit_firm")
    if pick:
        r = by_name[pick]
        committee = kits_mod.is_committee(r)
        pal = kits_mod.palette(pick)
        sw = "".join(f'<span style="display:inline-block;width:14px;height:14px;background:{v};'
                     f'border:1px solid #DDD;margin-right:4px;vertical-align:middle"></span>' for v in pal.values())
        st.markdown(f'<div class="muted">Tier {r["tier"]} &nbsp;·&nbsp; {r["decision_structure"]} &nbsp;·&nbsp; '
                    f'{"letter, talking points and IC memo outline" if committee else "letter and talking points"}'
                    f' &nbsp;·&nbsp; palette {sw}</div>', unsafe_allow_html=True)
        k = ks.get(pick)
        c2.button("Rewrite kit" if k else "Write kit", on_click=make_kit, args=(r,), disabled=not HAS_KEY,
                  width="stretch", help=None if HAS_KEY else NO_KEY_MSG)
        if not HAS_KEY and not k:
            st.caption(NO_KEY_MSG)
        if k:
            page = kits_mod.render_html(k, r)
            st.caption(f'Written {k.get("generated")} by {k.get("model")}. {k.get("tailoring", "")}')
            flags = k.get("flags", kits_mod.lint(k))
            if flags:
                st.warning("Check before sending. Our compliance check flagged:\n\n"
                           + "\n".join(f"- {f}" for f in flags))
            else:
                fixed = k.get("fixed_on_retry")
                st.caption("Compliance check: clean (no return language, figures, internal process, or claims "
                           "about clients)." + (f" Fixed on a second pass: {len(fixed)} issue(s)." if fixed else ""))
            b1, b2, _ = st.columns([1, 1, 3])
            b1.download_button("Download HTML", page, f'{pick.lower().replace(" ", "_").replace("&", "and")}_kit.html',
                               "text/html", width="stretch")
            # a rewritten kit can be re-sent; the pending older version is replaced
            queued = any(i["kind"] == "kit" and i["firm"] == pick and i["status"] == "pending"
                         and i["payload"]["kit"] == k for i in queue())
            b2.button("In review queue" if queued else "Send to review queue", disabled=queued, width="stretch",
                      on_click=queue_add, args=("kit", pick, f"Client kit for {pick}", {"kit": k}))
            # tall enough to show the letter through its compliance footer without scrolling the frame.
            # A data: URL gives the frame its own origin, so kit HTML (model text, escaped) cannot reach the app.
            st.iframe("data:text/html;base64," + base64.b64encode(page.encode()).decode(), height=1500)


# ---------- market signals ----------

@st.cache_data(ttl=900, show_spinner="Checking SPY and VIX...")
def live_market():
    """Latest close vs the prior close, cached 15 minutes. Returns (day, error)."""
    try:
        return sig.latest(), None
    except Exception as e:
        return None, f"{type(e).__name__}"


@st.cache_data
def load_signals() -> dict:
    return json.loads(SIGNALS.read_text()) if SIGNALS.exists() else {"days": {}, "fired": {}}


@st.cache_data(show_spinner="Pulling that day's real closes...")
def replay_days(key: str) -> list[dict]:
    """Real closes for a replay event: cached from build.py, else fetched live."""
    cached = [load_signals()["days"].get(d) for d in sig.EVENTS[key]["days"]]
    return cached if all(cached) else sig.event_days(key)


def alerts() -> list[dict]:
    return st.session_state.setdefault("alerts", [])


def day_line(d: dict) -> str:
    return (f'{date_label(d["date"])}: S&P 500 (SPY) {d["spy"]:,.2f}, {d["spy_pct"]:+.2f}% &nbsp;·&nbsp; '
            f'VIX {d["vix"]:.2f}, {d["vix_pct"]:+.1f}% day over day')


def date_label(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%a %b %-d, %Y")


def fire(key: str, label: str, day: dict, context: str, firms: list[dict]):
    """Queue a note per Tier A/B firm: pre-written where we have it, written now otherwise."""
    pre = load_signals()["fired"].get(key, {}).get("notes", {}) if key in sig.EVENTS else {}
    notes = {f["firm_name"]: pre[f["firm_name"]] for f in firms if f["firm_name"] in pre}
    todo = [f for f in firms if f["firm_name"] not in notes]
    if todo and HAS_KEY:
        try:
            notes.update(sig.draft_notes(day, context, todo))
        except Exception as e:
            st.session_state["sig_msg"] = ("error", f"Writing notes failed: {type(e).__name__}: {str(e)[:200]}")
            return
    for firm, n in notes.items():
        queue_add("signal", firm, f"Market note: {label}", {"note": n, "event": key})
    ad_set = load_signals()["fired"].get(key, {}).get("ads")
    if not ad_set and HAS_KEY:
        try:
            ad_set = ads_mod.generate(day, context)
        except Exception as e:
            st.session_state["sig_msg"] = ("error", f"Ad set failed: {type(e).__name__}: {str(e)[:200]}")
    if ad_set:
        queue_add("ads", "Market-wide", f"Ad set: {label}", {"ads": ad_set, "event": key})
    alerts().insert(0, {"label": label, "date": day["date"], "triggers": sig.triggers(day), "notes": len(notes),
                        "missing": len(firms) - len(notes), "fired": datetime.now().strftime("%H:%M"),
                        "source": "pre-written" if key in load_signals()["fired"] and not todo else "written now"})
    missing = len(firms) - len(notes)
    st.session_state["sig_msg"] = ("success", f"{len(notes)} notes" + (" and the ad set" if ad_set else "")
                                   + " are in the review queue."
                                   + (f" {missing} firms have no note: {NO_KEY_MSG}" if missing else ""))
    st.session_state["sig_event"] = key


def render_ads(a: dict):
    """Ad set with character counts against the Google limits."""
    for i, ad in enumerate(a["search_ads"], 1):
        st.markdown(f'**Search ad {i}**: <span class="muted">{html.escape(ad["angle"])}</span>', unsafe_allow_html=True)
        lines = [f'{html.escape(h)} <span class="muted">({len(h)}/{ads_mod.HEADLINE_MAX})</span>' for h in ad["headlines"]]
        lines += [f'<i>{html.escape(d)}</i> <span class="muted">({len(d)}/{ads_mod.DESCRIPTION_MAX})</span>'
                  for d in ad["descriptions"]]
        st.markdown("<br>".join(lines), unsafe_allow_html=True)
    st.caption(aircover.PLATFORM_NOTE)
    st.markdown("**Keywords (panic-day queries)**")
    st.markdown(f'<div class="muted">{html.escape("; ".join(a["keywords"]))}</div>', unsafe_allow_html=True)
    for sa in a["social_ads"]:
        st.markdown(f'**{html.escape(sa["platform"])}** <span class="muted">for {html.escape(sa["audience"])}</span>',
                    unsafe_allow_html=True)
        st.markdown(f'{html.escape(sa["headline"])}<br>{html.escape(sa["primary_text"])}<br>'
                    f'<span class="muted">{html.escape(DISCLOSURE)}</span>', unsafe_allow_html=True)
    lp = a["landing_page"]
    st.markdown(f'**Landing page: {html.escape(lp["title"])}**')
    for sec in lp["sections"]:
        st.markdown(f'<div><b>{html.escape(sec["heading"])}</b><br>'
                    + "<br>".join(f"· {html.escape(p)}" for p in sec["points"]) + "</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="muted">{html.escape(DISCLOSURE)}</div>', unsafe_allow_html=True)
    if a.get("flags"):
        st.warning("Check flagged:\n\n" + "\n".join(f"- {f}" for f in a["flags"]))
    else:
        st.caption("Character limits and copy rules: clean." + (f' Fixed on a second pass: {len(a["fixed_on_retry"])}.'
                                                                if a.get("fixed_on_retry") else ""))


def render_status(day: dict, context: str | None = None):
    fired = sig.triggers(day)
    st.markdown(f"<div>{day_line(day)}</div>", unsafe_allow_html=True)
    if fired:
        st.markdown("".join(f'<div class="flag">Trigger: {html.escape(t)}</div>' for t in fired), unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No trigger. Firms get the weekly digest, not a daily note.</div>',
                    unsafe_allow_html=True)
    if context:
        st.caption(context)
    return fired


with tab_signals:
    msg = st.session_state.pop("sig_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    ab = [r for r in records if r["tier"] in {"A", "B"}]
    st.markdown('<div class="summary">Triggers: S&P 500 down 2% or more in a day, or VIX up 25% or more day over '
                "day, or VIX above 30. When one fires, every Tier A and B firm gets a short client-ready note under "
                "its own name plus a two-line ping to the advisor. Educational only. Otherwise, a weekly digest.</div>",
                unsafe_allow_html=True)

    st.markdown("### Today")
    live, err = live_market()
    if live:
        fired_today = render_status(live)
        st.caption("Latest close from Yahoo Finance, refreshed every 15 minutes.")
        if fired_today:
            st.button(f"Fire today's trigger for {len(ab)} firms", disabled=not HAS_KEY,
                      help=None if HAS_KEY else NO_KEY_MSG,
                      on_click=fire, args=(live["date"], f'{date_label(live["date"])} market move', live,
                                           "Live market day.", ab))
    else:
        st.markdown(f'<div class="muted">Live market data is unavailable right now ({err}). You may be offline '
                    "or Yahoo Finance is not responding. The replays below use real closes cached in the repo.</div>",
                    unsafe_allow_html=True)

    st.markdown("### Replay a real event")
    cols = st.columns(len(sig.EVENTS) + 2)
    for col, (key, ev) in zip(cols, sig.EVENTS.items()):
        col.button(ev["label"], key=f"replay|{key}", width="stretch",
                   on_click=lambda k=key: st.session_state.update(sig_event=k))
    key = st.session_state.get("sig_event")
    if key in sig.EVENTS:
        ev = sig.EVENTS[key]
        try:
            days = replay_days(key)
        except Exception as e:
            days = []
            st.warning(f"Could not load the closes for this day ({type(e).__name__}).")
        fire_day = None
        for d in days:
            if render_status(d) and fire_day is None:
                fire_day = d
        st.caption(ev["context"])
        if fire_day:
            # a two-day event fires once, on the last day, with the first day as context
            last = days[-1]
            context = ev["context"] + "".join(f' Earlier: {date_label(d["date"])}, SPY {d["spy_pct"]:+.2f}%, '
                                              f'VIX {d["vix"]:.2f}.' for d in days[:-1])
            pre = load_signals()["fired"].get(key)
            have = sum(r["firm_name"] in (pre or {}).get("notes", {}) for r in ab)
            note = (f"Pre-written notes for {have} of {len(ab)} firms (written {pre['generated']} by {pre['model']})."
                    if pre else ("Notes will be written now, about 30 seconds." if HAS_KEY else NO_KEY_MSG))
            st.caption(note)
            st.button(f"Fire trigger: queue notes for {len(ab)} Tier A and B firms", type="primary",
                      disabled=not (pre or HAS_KEY), key=f"fire|{key}",
                      on_click=fire, args=(key, ev["label"], last, context, ab))

    sent_notes = [i for i in queue() if i["kind"] == "signal" and i["payload"]["event"] == key]
    if sent_notes:
        st.markdown("### Notes from this event")
        pick = st.selectbox("Firm", [i["firm"] for i in sent_notes], key="sig_firm")
        n = next(i for i in sent_notes if i["firm"] == pick)["payload"]["note"]
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown(f'**{html.escape(n["subject"])}**')
            st.markdown(f'<div>{html.escape(sig.signed(n)).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**Ping to the advisor**")
            st.markdown("<br>".join(html.escape(l) for l in n["advisor_ping"]), unsafe_allow_html=True)
            if n.get("flags"):
                st.warning("Compliance check flagged:\n\n" + "\n".join(f"- {f}" for f in n["flags"]))
            else:
                st.caption("Compliance check: clean.")

    ad_item = next((i for i in queue() if i["kind"] == "ads" and i["payload"]["event"] == key), None)
    if ad_item:
        with st.expander("Ad set for this event (search, social, landing page)"):
            render_ads(ad_item["payload"]["ads"])

    st.markdown("### Alert feed")
    if alerts():
        for a in alerts():
            st.markdown(f'<div><b>{html.escape(a["label"])}</b> &nbsp;·&nbsp; fired {a["fired"]} &nbsp;·&nbsp; '
                        f'{a["notes"]} notes queued ({a["source"]})'
                        + (f', {a["missing"]} missing' if a["missing"] else "")
                        + f'<br><span class="muted">{html.escape("; ".join(a["triggers"]))}</span></div>',
                        unsafe_allow_html=True)
    else:
        st.caption("Nothing fired this session.")


# ---------- review queue ----------

def sent_drafts() -> set:
    return st.session_state.setdefault("sent_drafts", set())


def engagement(item_id: str, kind: str) -> tuple[bool, bool | None]:
    """Simulated opens and forwards, stable per item. Drafts cannot be forwarded to clients."""
    h = int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16) % 100
    opened = h < 62
    forwarded = (h < 28) if kind not in {"draft", "ads"} else None
    return opened, forwarded


def queue_rows(by_firm: dict) -> list[dict]:
    rows = []
    store, apps = draft_store(), approvals()
    for key in store:
        firm, v = key.split("|")
        if v != "A" or firm not in by_firm:
            continue
        chosen = apps.get(firm)
        status = "sent" if firm in sent_drafts() else "approved" if chosen else "pending"
        rows.append({"id": f"draft|{firm}", "kind": "draft", "firm": firm, "status": status,
                     "title": f"First-touch email (variant {chosen or 'A or B'})", "added": "", "flags": []})
    for i, item in enumerate(queue()):
        pl = item["payload"]
        flags = (pl.get("note") or pl.get("kit") or pl.get("ads") or {}).get("flags", [])
        rows.append({"id": f'{item["kind"]}|{item["firm"]}|{i}', "kind": item["kind"], "firm": item["firm"],
                     "status": item["status"], "title": item["title"], "added": item["added"], "flags": flags,
                     "ref": item})
    for r in rows:
        tier = by_firm.get(r["firm"], {}).get("tier", "-")
        r["tier"] = tier
        opened, fwd = engagement(r["id"], r["kind"]) if r["status"] == "sent" else (None, None)
        r["opened"], r["forwarded"] = opened, fwd
        r["priority"] = "Top: forwarded to clients" if fwd else "High" if tier == "A" else "Normal"
        r["_p"] = (0 if fwd else 1 if tier == "A" else 2, 0 if tier == "A" else 1,
                   {"pending": 0, "approved": 1, "sent": 2}[r["status"]])
    return sorted(rows, key=lambda r: r["_p"])


def set_status(row: dict, status: str):
    if row["kind"] == "draft":
        firm = row["firm"]
        if status == "pending":
            approvals().pop(firm, None); sent_drafts().discard(firm)
        elif status == "approved":
            approvals().setdefault(firm, "A"); sent_drafts().discard(firm)
        else:
            approvals().setdefault(firm, "A"); sent_drafts().add(firm)
    else:
        row["ref"]["status"] = status


def bulk(kind: str, frm: str, to: str, by_firm: dict):
    for r in queue_rows(by_firm):
        if r["kind"] == kind and r["status"] == frm:
            set_status(r, to)


KIND_LABEL = {"draft": "Email draft", "kit": "Client kit", "signal": "Market note", "ads": "Ad set"}

with tab_queue:
    by_firm = {r["firm_name"]: r for r in records}
    rows = queue_rows(by_firm)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("pending", "approved", "sent")}
    n_fwd = sum(bool(r["forwarded"]) for r in rows)
    st.markdown(f'<div class="summary">{counts["pending"]} pending, {counts["approved"]} approved, {counts["sent"]} sent.'
                + (f' <span style="color:{BRASS};font-weight:600">{n_fwd} forwarded to clients</span>: call those first.'
                   if n_fwd else "") + "</div>", unsafe_allow_html=True)
    st.caption("Opened and forwarded are simulated for this demo, so the priority logic can be shown. In production "
               "they come from tracked links in each sent email and note.")

    f1, f2, f3 = st.columns([2, 2, 3])
    kinds = f1.multiselect("Type", list(KIND_LABEL), format_func=KIND_LABEL.get, placeholder="All types") or list(KIND_LABEL)
    stats = f2.multiselect("Status", ["pending", "approved", "sent"], placeholder="All statuses") or ["pending", "approved", "sent"]
    with f3:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.button("Approve all market notes", on_click=bulk, args=("signal", "pending", "approved", by_firm),
                  disabled=not any(r["kind"] == "signal" and r["status"] == "pending" for r in rows), width="stretch")
        b2.button("Mark approved notes sent", on_click=bulk, args=("signal", "approved", "sent", by_firm),
                  disabled=not any(r["kind"] == "signal" and r["status"] == "approved" for r in rows), width="stretch")

    shown = [r for r in rows if r["kind"] in kinds and r["status"] in stats]
    yn = lambda v: "" if v is None else ("Yes" if v else "No")
    table = pd.DataFrame([{
        "Priority": r["priority"], "Firm": r["firm"], "Tier": r["tier"], "Type": KIND_LABEL[r["kind"]],
        "Item": r["title"], "Status": r["status"], "Check": "Flagged" if r["flags"] else "",
        "Opened (sim)": yn(r["opened"]), "Forwarded to clients (sim)": yn(r["forwarded"]),
    } for r in shown])
    if table.empty:
        st.caption("Nothing here yet. Approve drafts, send a kit to review, or fire a market signal.")
    else:
        styled = table.style.map(lambda v: f"color:{BRASS};font-weight:600" if str(v).startswith("Top") else "",
                                 subset=["Priority"])
        ev = st.dataframe(styled, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 560),
                          on_select="rerun", selection_mode="single-row",
                          key=f"queue-{hash((tuple(kinds), tuple(stats), len(rows), counts['sent']))}",
                          column_config={"Priority": st.column_config.TextColumn(width=190),
                                         "Firm": st.column_config.TextColumn(width=230),
                                         "Tier": st.column_config.TextColumn(width=40),
                                         "Item": st.column_config.TextColumn(width="large")})
        if ev.selection.rows:
            st.session_state["queue_pick"] = shown[ev.selection.rows[0]]["id"]
        row = next((r for r in shown if r["id"] == st.session_state.get("queue_pick")), None)
        if row:
            st.divider()
            st.markdown(f'#### {html.escape(row["firm"])}: {html.escape(row["title"])}')
            if row["kind"] == "draft":
                v = approvals().get(row["firm"], "A")
                d = draft_store()[f'{row["firm"]}|{v}']
                st.markdown(f'**{html.escape(d["subject"])}**')
                st.markdown(f'<div style="white-space:pre-wrap">{html.escape(d["body"])}</div>', unsafe_allow_html=True)
                st.caption("Edit or rewrite this in the Drafts tab.")
            elif row["kind"] == "kit":
                st.markdown(f'<div style="white-space:pre-wrap">{html.escape(kits_mod.letter_text(row["ref"]["payload"]["kit"]))}</div>',
                            unsafe_allow_html=True)
                st.caption("Full kit with talking points is in Kit Studio.")
            elif row["kind"] == "ads":
                render_ads(row["ref"]["payload"]["ads"])
            else:
                n = row["ref"]["payload"]["note"]
                st.markdown(f'**{html.escape(n["subject"])}**')
                st.markdown(f'<div>{html.escape(sig.signed(n)).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                st.markdown("**Ping to the advisor:** " + html.escape(" ".join(n["advisor_ping"])))
            for f in (row["flags"] if row["kind"] != "ads" else []):
                st.markdown(f'<div class="flag">Check: {html.escape(f)}</div>', unsafe_allow_html=True)
            if row["forwarded"]:
                st.markdown(f'<div style="color:{BRASS};font-weight:600">Forwarded to clients (simulated). '
                            "Their clients are reading it: this firm moves to the top of today's calls.</div>",
                            unsafe_allow_html=True)
            c1, c2, c3, _ = st.columns([1, 1, 1, 3])
            c1.button("Approve", key=f'qa|{row["id"]}', on_click=set_status, args=(row, "approved"),
                      disabled=row["status"] != "pending", type="primary", width="stretch")
            c2.button("Mark sent", key=f'qs|{row["id"]}', on_click=set_status, args=(row, "sent"),
                      disabled=row["status"] == "sent", width="stretch")
            c3.button("Back to pending", key=f'qp|{row["id"]}', on_click=set_status, args=(row, "pending"),
                      disabled=row["status"] == "pending", width="stretch")


# ---------- air cover ----------

@st.cache_data
def load_aeo():
    return aeo.load()


def run_aeo():
    with st.status("Asking 30 client questions with live web search. About 3 minutes.", expanded=True) as box:
        log = st.empty()
        lines = []

        def progress(line):
            lines.append(line)
            log.code("\n".join(lines[-8:]), language=None)
        try:
            aeo.run(30, progress=progress)
            load_aeo.clear()
            box.update(label="Done. Results saved to data/aeo.json.", state="complete")
        except Exception as e:
            box.update(label=f"Run failed: {type(e).__name__}", state="error")


with tab_air:
    st.markdown("### What clients hear from AI")
    st.caption("Clients ask an assistant before they ask their advisor. We ask the questions they actually type, "
               "with live web search, and track whether evergreen, interval, or tender-offer funds come up and who "
               "gets cited.")
    d = load_aeo()
    c1, c2 = st.columns([4, 1])
    if c2.button("Run now", disabled=not HAS_KEY, width="stretch", help=None if HAS_KEY else NO_KEY_MSG):
        run_aeo()
        d = load_aeo()
    if not d:
        st.caption("No results yet. Run `python -m pipeline.aeo --n 30` or press Run now.")
    else:
        m = d["metrics"]
        c1.markdown(
            f'<div class="summary">Asked {m["answered"]} client questions on {d["run_date"]} ({d["model"]}, '
            f'{d["tool"]["type"]}). Evergreen, interval, or tender-offer funds came up in '
            f'<b>{m["evergreen_share"]:.0%}</b> of answers. Equi came up in <b>{m["equi_share"]:.0%}</b>.'
            + (f' {m["failed"]} questions failed.' if m["failed"] else "") + "</div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown("**By question group**")
            st.dataframe(pd.DataFrame([{"Group": g, "Evergreen mentioned": v * 100}
                                       for g, v in m["evergreen_by_group"].items()]),
                         hide_index=True, width="stretch",
                         column_config={"Evergreen mentioned": st.column_config.ProgressColumn(
                             format="%.0f%%", min_value=0, max_value=100)})
            st.caption("Evergreen comes up when the client already knows the word, and almost never when they "
                       "describe the problem it solves.")
        with g2:
            st.markdown("**Most-cited domains**")
            st.dataframe(pd.DataFrame(m["top_domains"], columns=["Domain", "Citations"]), hide_index=True,
                         width="stretch", height=320)
            st.caption(f"From {sum(len(r.get('cited_urls') or []) for r in d['results'])} citations across "
                       f"{m['answered']} answers, so the ranking is thin. Rerun weekly and pool.")

        with st.expander(f'Gaps: {len(m["gaps"])} questions where evergreen never came up'):
            for g in m["gaps"]:
                st.markdown(f'<div class="muted">{html.escape(g["group"])}: {html.escape(g["question"])}</div>',
                            unsafe_allow_html=True)

        st.markdown("**Pages to create**")
        st.caption("Category education only, each ending in \"ask your advisor\". Pitch lists exclude fund managers "
                   "and advisory firms.")
        for i, pg in enumerate(d.get("pages") or [], 1):
            st.markdown(
                f'<div style="margin:0 0 0.8rem 0"><b>{i}. {html.escape(pg["page_title"])}</b><br>'
                f'<span class="muted">Answers: {html.escape(pg["target_question"])}</span><br>'
                f'{html.escape(pg["why_cited"])}<br>'
                f'<span class="muted">Pitch: {html.escape(", ".join(pg["pitch_domains"]) or "no cited domain fits")}</span>'
                + "".join(f'<div class="flag">Check: {html.escape(f)}</div>' for f in pg.get("flags", []))
                + "</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="muted">Every page carries: {html.escape(DISCLOSURE)}</div>', unsafe_allow_html=True)

        with st.expander("Read the answers"):
            ok = [r for r in d["results"] if not r.get("error")]
            q = st.selectbox("Question", [r["question"] for r in ok], key="aeo_q")
            r = next(x for x in ok if x["question"] == q)
            st.markdown(f'{"Mentions " + ", ".join(r["evergreen_terms"]) if r["mentions_evergreen"] else "No evergreen mention"}'
                        f' &nbsp;·&nbsp; {len(r["cited_urls"])} citations', unsafe_allow_html=True)
            st.markdown(f'<div style="max-height:360px;overflow:auto;border:1px solid #ECEAE3;padding:10px;font-size:0.9rem">'
                        f'{html.escape(r["answer"]).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            st.markdown("<br>".join(f'<span class="muted">{html.escape(u)}</span>' for u in r["cited_urls"]),
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("### Air cover plan")
    st.markdown(f'<div class="flag" style="font-weight:600">{aircover.SIMULATION}</div>', unsafe_allow_html=True)
    st.caption("Before a Tier A firm's sequence starts, 2 to 3 weeks of category-education ads run to affluent "
               "households around its city, so its clients have heard of evergreen alternatives before Equi calls. "
               "Ads never name the firm or any fund. Uses the default-weight tiers so the test arms stay fixed.")
    base = list(load_enrichment().values())
    pairs, unpaired, arms = aircover.holdout(base)
    plan_rows = aircover.plan(base, arms)
    st.dataframe(pd.DataFrame([{
        "Firm": p["firm"], "Metro": p["metro"], "Flight": f'{p["flight_start"]} to {p["flight_end"]}',
        "Sequence starts": p["sequence_start"], "Status": p["status"], "Audience": p["audience"],
        "Ad copy": f'{p["headline"]} | {p["description"]}',
    } for p in plan_rows]), hide_index=True, width="stretch",
        column_config={"Firm": st.column_config.TextColumn(width=220), "Metro": st.column_config.TextColumn(width=130),
                       "Flight": st.column_config.TextColumn(width=190), "Status": st.column_config.TextColumn(width=230),
                       "Audience": st.column_config.TextColumn(width="large"),
                       "Ad copy": st.column_config.TextColumn(width="large")})
    issues = aircover.copy_issues()
    st.caption("Ad copy is within Google's 30 and 90 character limits." if not issues else "Copy over limit: " + "; ".join(issues))
    st.caption(aircover.PLATFORM_NOTE)
    st.markdown(f'<div class="muted">Every ad links to a page carrying: {html.escape(DISCLOSURE)}</div>',
                unsafe_allow_html=True)

    st.markdown("**Holdout test**")
    st.markdown(f"Tier A and B firms are paired by tier and firm type, closest scores together. A seeded coin flip "
                f"(seed {aircover.SEED}) puts one of each pair in air cover and the other in control. "
                f"**Primary metric:** {aircover.PRIMARY_METRIC} **Guardrail:** {aircover.GUARDRAIL}")
    st.dataframe(pd.DataFrame([{"Pair": p["pair"], "Tier": p["tier"], "Type": p["type"], "Air cover": p["air_cover"],
                                "Control": p["control"], "Score gap": p["score_gap"]} for p in pairs]),
                 hide_index=True, width="stretch")
    if unpaired:
        st.caption(f'Unpaired (odd one out in its group, not in the test, gets air cover): {", ".join(unpaired)}.')
    st.markdown(f'<div class="flag">{html.escape(aircover.power_note(len(pairs)))}</div>', unsafe_allow_html=True)


# ---------- start here ----------

def jump(tab: str):
    st.session_state["tab"] = tab


def jump_row(tab: str, line: str):
    # top-aligned with a small offset: centering is thrown off by the markdown block's own margin
    b, t = st.columns([1.1, 5], gap="medium", vertical_alignment="top")
    b.button(tab, key=f"jump|{tab}", on_click=jump, args=(tab,), width="stretch")
    t.markdown(f'<div class="jump-line">{line}</div>', unsafe_allow_html=True)


def findings() -> list[str]:
    """Plain-sentence findings, computed from the current data and weights."""
    out = []
    stable = [r for r in records if r["tier"] != "DQ" and (r["tier_a_share"] or 0) >= 0.9]
    if stable:
        floor = int(min(r["tier_a_share"] for r in stable) * 100)
        draws = 500 if not custom else 200
        out.append(f"<b>{len(stable)} firms hold Tier A in {floor}%+ of {draws} alternative weightings</b>, so the top "
                   "of the list comes from the firms, not from our choice of weights.")
    d = load_aeo()
    if d:
        m = d["metrics"]
        low_g, low_v = min(m["evergreen_by_group"].items(), key=lambda kv: kv[1])
        out.append(f"<b>AI answers mention evergreen alternatives in {m['evergreen_share']:.0%} of answers to "
                   f"{m['answered']} client questions</b>, and {low_v:.0%} of the time when the question is about "
                   f"{low_g.lower()}. Equi came up in {m['equi_share']:.0%}.")
    fired = load_signals()["fired"].get("2024-08-05")
    if fired:
        day = fired["day"]
        n_trig = {1: "one", 2: "two", 3: "three"}.get(len(fired["triggers"]), len(fired["triggers"]))
        out.append(f"<b>On Aug 5 2024, all {n_trig} market triggers would have fired</b> (S&P 500 "
                   f"{day['spy_pct']:+.1f}%, VIX {day['vix']:.1f}, up {day['vix_pct']:.0f}%), with "
                   f"{len(fired['notes'])} firm-branded client notes ready for review that afternoon.")
    raw_rows = len(pd.read_csv(CSV))
    n_dq = sum(r["tier"] == "DQ" for r in records)
    n_a = sum(r["tier"] == "A" for r in records)
    n_decide = sum(r.get("bubble_type") == "decision" and not r["decision"] for r in records)
    out.append(f"<b>{raw_rows} CSV rows became {len(records)} firms</b>: {n_dq} disqualified by hard gates, "
               f"{n_a} Tier A to call now, and {n_decide} that need a call from your team before outreach.")
    return out


with tab_start:
    st.markdown("### The idea")
    for line in [
        "Equi's differentiator is that it builds the advisor's client materials. So the materials become the outreach: "
        "the first touch is a finished client letter under the prospect firm's own name.",
        "Outreach is timed to market drops, when downside protection matters most to clients and advisors field the "
        "hardest calls.",
        "Clients hear about evergreen alternatives first, through AI answers and market-day ads, so advisors walk "
        "into an easier conversation.",
        "The pipeline decides who gets all of it.",
    ]:
        st.markdown(f'<div class="lede">{line}</div>', unsafe_allow_html=True)

    st.markdown("### What the data says")
    for f in findings():
        st.markdown(f'<div class="finding">{f}</div>', unsafe_allow_html=True)
    st.caption("Computed from the current data and weights. Move a sidebar weight and these update.")

    st.markdown("### Part 1: demand engine")
    jump_row("Kit Studio", "A client letter, advisor talking points, and for committee firms an IC memo outline, "
                           "all under the prospect firm's own name.")
    jump_row("Market signals", "A big down day fires a client-ready note for every Tier A and B firm, a two-line "
                               "advisor ping, and a market-day ad set.")
    jump_row("Air cover", "What AI tells clients today, the pages that would change it, and a simulated ad plan "
                          "tested against a holdout.")

    st.markdown("### Part 2: lead pipeline")
    jump_row("Shortlist", f"{len(records)} firms scored against Equi's ICP, each with a score range, a Tier A "
                          "stability check, and a route.")
    jump_row("Your call", "Firms whose tier depends on your priorities, a rule, or data we do not have yet.")
    jump_row("Drafts", "Two first-touch emails per firm, kit-led and insight-led, randomized for the A/B test.")
    jump_row("Review queue", "Everything waiting for a human before it goes out: drafts, kits, notes, and ad sets.")
