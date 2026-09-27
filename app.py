"""Equi lead desk. Run with: streamlit run app.py

Reads data/scored.json (from build.py) for enrichment, and re-scores live from
the CSV when the sidebar weights change.
"""
from __future__ import annotations

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

from pipeline import personalize
from pipeline.clean import clean_leads
from pipeline.score import (CRITERIA_LABELS, DEFAULT_WEIGHTS, WHY_WEIGHT, bubble_reason,
                            bubble_type,
                            red_flags, route, score_firms, stability, tier_for)
from pipeline.sequences import as_rows, sequence_for, short_route

CSV = "data/sample-leads.csv"
SCORED = Path("data/scored.json")
DRAFTS = Path("data/drafts.json")

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

st.set_page_config(page_title="Equi lead desk", layout="wide")

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

st.title("Equi lead desk")
st.markdown('<div class="muted">Independent RIAs and multi-family offices, scored against Equi\'s ICP.'
            + (" Custom weights in use." if custom else "") + "</div>", unsafe_allow_html=True)

tab_shortlist, tab_call, tab_drafts = st.tabs(["Shortlist", "Your call", "Drafts"], key="tab",
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
