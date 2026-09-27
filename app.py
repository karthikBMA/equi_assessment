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
                            red_flags, score_firms, stability)
from pipeline.sequences import as_rows, sequence_for

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
  .block-container {{ padding-top: 2rem; max-width: 1400px; }}
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
        e = enriched.get(r["firm_name"], {})
        r["sources"] = e.get("sources", {})
        r["sec_adv"] = e.get("sec_adv")
        if e.get("contacts"):
            r["contacts"] = e["contacts"]
        r["sequence"] = sequence_for(r["route"])
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
    routes = st.multiselect("Route", all_routes, placeholder="All routes") or all_routes

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
        st.markdown(f'**Route:** {r["route"]}')
        if r.get("mover"):
            st.markdown(f'<div class="muted">{html.escape(r["mover"])}</div>', unsafe_allow_html=True)
        if r.get("bubble"):
            st.markdown(f'<div class="muted">Needs a human call: {html.escape(r["bubble"])}</div>',
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
    n_b = sum(r["tier"] == "B" for r in records)
    st.markdown(f'<div class="summary">{len(records)} firms, {n_dq} disqualified, '
                f'<span style="color:{BRASS};font-weight:600">{n_a} Tier A</span>, {n_b} Tier B.</div>',
                unsafe_allow_html=True)

    shown = [r for r in records if r["tier"] in tiers and r["route"] in routes]
    table = pd.DataFrame([{
        "Firm": r["firm_name"], "Type": r["firm_type"],
        "AUM ($B)": r["aum_usd"] / 1e9 if r["aum_usd"] else None,
        "Avg client ($M)": r["avg_client_usd"] / 1e6 if r["avg_client_usd"] else None,
        "Score": r["score"],
        "Range": "" if r["score"] is None else f'{r["score_lo"]:.0f} to {r["score_hi"]:.0f}',
        "Tier": r["tier"], "Confidence": r["confidence"],
        "Tier A stability": (r["tier_a_share"] or 0) * 100 if r["tier"] != "DQ" else None,
        "Route": r["route"],
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
                "AUM ($B)": st.column_config.NumberColumn(format="$%.1fB"),
                "Avg client ($M)": st.column_config.NumberColumn(format="$%.1fM"),
                "Score": st.column_config.NumberColumn(format="%.0f"),
                "Tier A stability": st.column_config.NumberColumn(
                    format="%.0f%%", help="Share of perturbed weightings (500 at defaults, 200 at custom weights) where the firm lands in Tier A."),
            })
        picked = event.selection.rows
        if picked:
            st.session_state["firm"] = shown[picked[0]]["firm_name"]
        current = next((r for r in shown if r["firm_name"] == st.session_state.get("firm")), None)
        if current:
            render_detail(current)
        else:
            st.caption("Select a row to open the firm.")
