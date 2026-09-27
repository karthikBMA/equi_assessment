"""Run the full pipeline and cache results for the app.

    python build.py              # clean, score, stress test
    python build.py --drafts     # also pre-write A/B drafts with Claude (needs ANTHROPIC_API_KEY)
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from pipeline.clean import clean_leads
from pipeline.score import score_firms, stability, red_flags, bubble_reason
from pipeline import personalize

load_dotenv()
DATA = Path("data")


def to_records(df):
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample-leads.csv")
    ap.add_argument("--drafts", action="store_true")
    ap.add_argument("--sender", default="the Equi team")
    args = ap.parse_args()

    firms = clean_leads(args.csv)
    scored = score_firms(firms)
    share = stability(firms, n=500)
    scored["tier_a_share"] = scored.firm_name.map(share).fillna(0.0)
    records = to_records(scored)
    for r in records:
        r["red_flags"] = red_flags(r)
        r["bubble"] = bubble_reason(r)
    (DATA / "scored.json").write_text(json.dumps(records, indent=2))
    print(f"Scored {len(records)} firms -> data/scored.json")

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


if __name__ == "__main__":
    main()
