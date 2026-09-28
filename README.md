# Equi demand engine

<!-- KARTHIK: your 3-line intro goes here. -->
_[Placeholder: 3-line intro]_

The demand plan (Part 1) is in [PLAN.md](PLAN.md). This README covers setup, how it works, and the calls I made.

## Setup

**Local**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python build.py --drafts      # clean, score, stress test, enrich, and write A/B drafts
streamlit run app.py
```

`python build.py --offline` skips the SEC lookup. `--signals` caches the replay market data and rewrites the Aug 5 2024 notes and ad set. `python -m pipeline.kit` rewrites the demo kits, `python -m pipeline.aeo --n 30` reruns the tracker and action plan, and `python -m pipeline.pages "question"` drafts a page. The app runs without a key from `data/`; buttons that write new text say so.

**Streamlit Cloud:** deploy `app.py` with Python 3.14 (Advanced settings; tested on 3.14.7, see `.python-version`), then add `ANTHROPIC_API_KEY = "sk-ant-..."` under Secrets. The app reads `.env` locally and `st.secrets` on Cloud.

## How it works

**Part 1, demand engine.** Equi's edge is that it builds the advisor's client materials, so the materials are the outreach. Kit Studio writes a client letter under the prospect firm's name. Market signals watches SPY and VIX and, on a real drop, queues a client-ready note for every Tier A and B firm plus an ad set. Client demand tracks what AI tells wealthy clients and plans the pages and ads to change it. Search lab does it per question.

**Part 2, lead pipeline.** `clean.py` normalizes the CSV and dedupes it (50 rows become 45 firms, every change logged). `score.py` applies hard gates, six weighted criteria, caps, and a confidence range, then stress-tests the weights. `enrich.py` tags every field by source. `sequences.py` defines the touches for each route. `personalize.py` writes two first-touch variants per firm.

**Tabs**
- **Start here:** the thesis and live findings.
- **Kit Studio:** firm-branded letter, talking points, IC memo outline for committee firms.
- **Market signals:** today's status, real-event replays, the alert feed.
- **Client demand:** what AI tells clients, a three-lane AEO action plan with page drafts, and the simulated air cover plan.
- **Search lab:** any client question: who owns it, what AI says, Equi's plan.
- **Shortlist:** ranked firms with score range, stability, route, and a full detail view.
- **Your call:** firms that need a human decision, with a recommendation and overrides.
- **Drafts:** variants A and B side by side, rewrite with a note, approve, sequencer CSV.
- **Review queue:** everything awaiting approval, with simulated engagement.

## Decisions and tradeoffs

**Weights.** Each weight traces to something Equi said. The wording below is my paraphrase of the brief, from `WHY_WEIGHT` in `score.py`.

| Criterion | Weight | What Equi said (paraphrased) |
|---|---|---|
| Firm type | 25 | Firm type matters more than size. MFOs and RIAs becoming MFOs are the core buyer. |
| Client wealth | 20 | Eligibility is gated on qualified purchasers ($5M+). No QP clients, no sale. |
| Alts experience | 20 | The win is in the middle: fluent enough to move, not so deep they built it themselves. |
| Decision speed | 15 | Principal-led vs committee-led is the best predictor of deal speed. |
| AUM | 10 | $1B to $30B, $5B+ bullseye, but type beats size, so size is a tiebreaker. |
| Warmth | 10 | A warm path shortens the cycle but never makes a bad fit good. |

**Stress test.** I re-scored the list under 500 weightings, each moving every weight roughly 5 to 8 points. Seven firms hold Tier A in 96%+ of them, and at 1,000 draws the same seven hold in 95%+. Equal weights give the same Tier A list as ours. So the top of the list comes from the firms, not my weights. Firms that flip go to **Your call** as decisions for Equi (11 today), apart from firms that need research first (6).

**The liquid-alts gap.** Equi sells liquid, evergreen alternatives. A firm with private equity and real estate but no liquid sleeve speaks the language and has an obvious gap, so it scores highest. A firm already in hedge funds is a replacement sale. A firm in everything has built its own research desk.

**Caps.** LPL-affiliated firms max out at Tier C, because products generally need LPL platform approval first. Non-US firms max out at B with a -15 modifier until compliance clears cross-border eligibility. Firms under $1B max out at C: nurture until they grow into the band. Single-family offices are capped at C per Equi's guidance.

**Strict SEC matching.** The public IAPD search is fuzzy and matches former names. "Harborstone" returned Viant Capital, an inactive broker-dealer once called Harborstone Capital. A wrong match feeds a wrong AUM into the score, so a match must be an active adviser with the same normalized name in the same state. The synthetic firms correctly return no match.

**Drafts and an approval queue, not live sending.** Every draft, kit, note, and ad passes a human. The model occasionally writes what compliance would stop ("structurally insulated from volatility"), so code checks catch known patterns, retry once, and flag the rest.

**No Clay on synthetic data.** Enriching made-up firms returns nothing, or someone else's data. In production: SEC Form ADV data files for AUM (Item 5.F) and high-net-worth client counts and assets (Item 5.D, which gives real average client size), and Clay for contacts, emails, and job changes.

**Session state and a cost guard.** Overrides, approvals, and the queue live in the browser session, with CSV exports; production would write them to BigQuery. Each session gets 25 Claude calls across all buttons, so the 32-call weekly AEO check runs from the command line.

**Client-side work is category education only.** Equi cannot advertise its funds to the public. Letters, notes, ads, and pages explain evergreen alternatives, never name a fund, never cite performance, end in "ask your advisor", and carry a disclosure line.

## A/B test design and metrics

- **Unit:** firm. Randomized to variant A (kit-led) or B (insight-led), stratified by tier and persona, fixed seed.
- **Primary metric:** positive reply rate. **Secondary:** meetings booked. **Guardrail:** unsubscribe and negative reply rate.
- **Size:** about 100 firms per arm for a real read on reply rate; with this list, treat early reads as directional.
- **Also test:** sender (founder vs SDR), subject style, and send timing around market triggers.
- **Air cover geo holdout:** Tier A and B firms paired by tier and type, one of each pair gets air cover. Primary metric is meeting rate from outbound. Thirteen pairs cannot show significance; the real test needs roughly 40+ firms per arm.

**Pipeline health**
- Tier A meeting rate vs B and C. If A does not beat B, fix the scoring.
- Time to meeting, principal-led vs committee-led. This validates the decision-speed weight.
- Override rate in Your call. If reps override often, retune the weights.

## Other ideas to land meetings

- **Co-host a prospect firm's client evening** on evergreen alternatives, then cut the recording into firm-branded clips.
- **Answer the DDQ first.** Send research leads a completed due-diligence questionnaire with the first email.
- **Conference follow-up within 48 hours,** with a kit built for every firm met.
- **Referral loop:** when a firm forwards a note to clients, ask which peers should see it.

## What I would do with more time

- Run the real enrichment path (Form ADV files, Clay) and re-score on real data.
- Persist state to BigQuery and wire real engagement from tracked links into the queue.
- Rerun the AEO tracker weekly and track Equi's pages over time.
- Move kit generation to structured output, like everything else.
- Calibrate weights against real meeting outcomes.
