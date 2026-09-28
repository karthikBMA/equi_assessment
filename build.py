"""Run the full pipeline and cache results for the app.

    python build.py              # clean, score, stress test, enrich (queries SEC IAPD)
    python build.py --offline    # same, but skip the SEC lookup
    python build.py --signals    # cache real SPY/VIX for the replay events, pre-write Aug 5 2024 notes
    python build.py --drafts     # also pre-write A/B drafts with Claude (needs ANTHROPIC_API_KEY)
"""
import argparse
import json
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from pipeline.clean import clean_leads
from pipeline.score import score_firms, stability, red_flags, bubble_reason, bubble_type
from pipeline import ads, personalize, signals
from pipeline.enrich import enrich, source_summary
from pipeline.sequences import SEQUENCES, sequence_for

load_dotenv()
DATA = Path("data")


def to_records(df):
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample-leads.csv")
    ap.add_argument("--drafts", action="store_true")
    ap.add_argument("--sender", default="the Equi team")
    ap.add_argument("--offline", action="store_true", help="skip the SEC IAPD lookup")
    ap.add_argument("--signals", action="store_true", help="cache replay market data and pre-write Aug 5 2024 notes")
    args = ap.parse_args()

    firms = clean_leads(args.csv)
    scored = score_firms(firms)
    share = stability(firms, n=500)
    scored["tier_a_share"] = scored.firm_name.map(share).fillna(0.0)
    records = to_records(scored)
    for r in records:
        r["red_flags"] = red_flags(r)
        r["bubble"] = bubble_reason(r)
        r["bubble_type"] = bubble_type(r)
    enrich(records, offline=args.offline)
    missing = {r["route"] for r in records} - set(SEQUENCES)
    if missing:
        raise SystemExit(f"No sequence defined for routes: {missing}")
    for r in records:
        r["sequence"] = sequence_for(r["route"])
    (DATA / "scored.json").write_text(json.dumps(records, indent=2))
    if args.signals:
        build_signals(records)
    print(f"Scored {len(records)} firms -> data/scored.json")
    adv = [r["sec_adv"]["status"] for r in records]
    print(f"Sources tagged: {source_summary(records)}")
    print(f"SEC IAPD: {adv.count('match')} matched, {adv.count('no_match')} no match, "
          f"{adv.count('error')} errors, {adv.count('skipped')} skipped")

    if args.drafts:
        todo = [(r, v) for r in records if personalize.worth_drafting(r) for v in ("A", "B")]
        print(f"Writing {len(todo)} drafts...")

        def run(item):
            r, v = item
            try:
                return r["firm_name"], personalize.draft(r, v, sender_name=args.sender)
            except Exception as e:  # keep going; the app can regenerate
                return r["firm_name"], {"variant": v, "error": str(e)}

        drafts: dict = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for name, d in pool.map(run, todo):
                drafts.setdefault(name, {})[d["variant"]] = d
        (DATA / "drafts.json").write_text(json.dumps(drafts, indent=2))
        errors = sum("error" in d for v in drafts.values() for d in v.values())
        print(f"Wrote drafts for {len(drafts)} firms -> data/drafts.json ({errors} errors)")


def build_signals(records):
    """Real closes for each replay event, plus pre-written notes for Aug 5 2024 so the demo runs without a key."""
    out = {"days": {}, "fired": {}}
    for key in signals.EVENTS:
        for day in signals.event_days(key):
            out["days"][day["date"]] = day
    firms = [r for r in records if r["tier"] in {"A", "B"}]
    key = "2024-08-05"
    day = out["days"][key]
    notes = signals.draft_notes(day, signals.EVENTS[key]["context"], firms)
    out["fired"][key] = {"day": day, "triggers": signals.triggers(day), "notes": notes,
                         "model": signals.MODEL, "generated": str(date.today())}
    # the ad set lives in the same file; write it here too so --signals never drops it
    ad_set = ads.generate(day, signals.EVENTS[key]["context"])
    ad_set["generated"] = str(date.today())
    out["fired"][key]["ads"] = ad_set
    (DATA / "signals.json").write_text(json.dumps(out, indent=2))
    flagged = sum(bool(n["flags"]) for n in notes.values())
    print(f"Signals: {len(out['days'])} replay days cached; {len(notes)}/{len(firms)} Aug 5 notes "
          f"({flagged} flagged), ad set ({len(ad_set['flags'])} flagged) -> data/signals.json")


if __name__ == "__main__":
    main()
