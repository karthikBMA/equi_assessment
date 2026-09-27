"""Equi lead desk. Run with: streamlit run app.py

Reads data/scored.json (from build.py) for enrichment, and re-scores live from
the CSV when the sidebar weights change.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.clean import clean_leads
from pipeline.score import (CRITERIA_LABELS, DEFAULT_WEIGHTS, WHY_WEIGHT, bubble_reason,
                            bubble_type,
                            red_flags, score_firms, stability, tier_for)
from pipeline.sequences import as_rows, sequence_for, short_route

CSV = "data/sample-leads.csv"
SCORED = Path("data/scored.json")

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
    return records


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

(tab_shortlist,) = st.tabs(["Shortlist"])


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
    head = [pill("Disqualified" if tier == "DQ" else f"Tier {tier}", TIER_COLOR[tier]),
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
    n_decide = sum(r.get("bubble_type") == "decision" for r in records)
    n_research = sum(r.get("bubble_type") == "research" for r in records)
    st.markdown(
        f'<div class="summary">{len(records)} firms scored, {n_dq} disqualified. '
        f'<span style="color:{BRASS};font-weight:600">Call {n_a} now</span>: the Tier A firms.<br>'
        f'{n_stable} hold Tier A in 90%+ of alternative weightings, so they do not depend on our weight choices.<br>'
        f'{n_decide} need a call from your team. {n_research} need research before anyone decides.</div>',
        unsafe_allow_html=True)

    shown = [r for r in records if r["tier"] in tiers and r["route"] in routes]
    table = pd.DataFrame([{
        "Firm": r["firm_name"], "Type": r["firm_type"],
        "AUM": r["aum_usd"] / 1e9 if r["aum_usd"] else None,
        "Avg client": r["avg_client_usd"] / 1e6 if r["avg_client_usd"] else None,
        "Score": r["score"],
        "Range": "" if r["score"] is None else f'{r["score_lo"]:.0f} to {r["score_hi"]:.0f}',
        "Tier": r["tier"],
        "A stability": (r["tier_a_share"] or 0) * 100 if r["tier"] != "DQ" else None,
        "Route": short_route(r["route"]), "Confidence": r["confidence"], "Why": r["why"],
    } for r in shown])

    if table.empty:
        st.caption("No firms match these filters.")
    else:
        styled = table.style.map(lambda t: f"color:{TIER_COLOR.get(t, SLATE)};font-weight:600",
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
                "Tier": st.column_config.TextColumn(width=40),
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
