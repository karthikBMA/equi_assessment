# Equi demand engine

Live app: https://equiassessment-bqwmtkyccgewnfq9fno3ia.streamlit.app/

Part 1 plan: see [PLAN.md](PLAN.md)

<!-- KARTHIK: your 3-line intro goes here. -->
_[Placeholder: 3-line intro]_

## Setup

**You need**
- Python 3.14 (the version this was tested on, 3.14.7). Download it from python.org.
- git, to copy the repo. Download it from git-scm.com.
- An Anthropic API key (optional: without one, the app runs from cached results in `data/`).

**Run it.** Open Terminal (Mac) or PowerShell (Windows) and paste each command.

1. `git clone https://github.com/kdev792/Equi_Assessment.git` then `cd Equi_Assessment`. This copies the project.
2. Create a private Python environment and turn it on.
   - Mac: `python3.14 -m venv .venv` then `source .venv/bin/activate`
   - Windows: `py -3.14 -m venv .venv` then `.venv\Scripts\activate`
3. `pip install -r requirements.txt`. This installs the tested package versions.
4. Add your key (skip this to run without one). Mac: `cp .env.example .env`. Windows: `copy .env.example .env`. Then open `.env` in a text editor and replace `sk-ant-...` with your key.
5. `streamlit run app.py`. This starts the app.
6. Open http://localhost:8501 if the browser does not open by itself.

**Optional**
- Rebuild the data: `python build.py` cleans, scores, and stress-tests the list (`--offline` skips the SEC lookup). `--drafts` and `--signals` rewrite the drafts and the Aug 5 2024 notes; both need a key.
- Rerun the AI answer check: `python -m pipeline.aeo` (about 4 minutes, needs a key).
- Deploy to Streamlit Community Cloud: connect the GitHub repo, pick `app.py`, choose Python 3.14 under Advanced settings, and add `ANTHROPIC_API_KEY = "sk-ant-..."` under Secrets.

**If something goes wrong**
- *Wrong Python version:* `python --version` should say 3.14. On Windows, use `py -3.14` as in step 2; on Mac, `python3.14`.
- *Windows says running scripts is disabled:* run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then repeat step 2.
- *Key not found:* the sidebar says "No Anthropic key set". `.env` must sit in the project folder with the line `ANTHROPIC_API_KEY=sk-ant-...` (name, equals sign, key; no spaces or quotes). On Windows, check it did not save as `.env.txt`.
- *Port already in use:* run `streamlit run app.py --server.port 8502` and open http://localhost:8502.
- *Market data blocked:* if Yahoo Finance is unreachable, Market signals says so and replays use cached closes.

## How the code is organized

```
app.py                  the Streamlit app
build.py                runs the pipeline, writes data/
requirements.txt        exact package versions
.python-version         the tested Python version
.env.example            template for .env (never commit the key)
.gitignore              keeps .env and caches out of git
.streamlit/config.toml  colors and fonts
pipeline/
  __init__.py           marks the folder as a package
  clean.py              normalizes the CSV, merges duplicates, logs every change
  score.py              gates, weights, caps, tiers, stress test
  enrich.py             source tags and the SEC adviser lookup
  sequences.py          the touches for each route
  personalize.py        first-touch email variants, written with the Claude API
  kit.py                client letter, talking points, and IC memo outline
  signals.py            market triggers and the notes they fire
  ads.py                market-day ad sets, character limits checked in code
  aeo.py                what AI tells clients and advisors, and the action plan
  pages.py              publish-ready equi.com pages with FAQ schema
  serp.py               Search lab analysis
  aircover.py           simulated air cover plan and holdout pairs
  compliance.py         disclosure line and client-copy checks
data/
  sample-leads.csv      the input list
  scored.json           scored, enriched firms
  drafts.json, kits.json, signals.json, aeo.json, pages.json, serp_cache.json
                        cached results, so the app works without a key
```

**Data flow**
1. `sample-leads.csv` goes through `clean.py`: 50 rows become 45 firms.
2. `score.py` scores every firm and stress-tests the weights.
3. `enrich.py`, `sequences.py`, and `personalize.py` add source tags, routes, and drafts; the other modules cache kits, notes, and AI answers.
4. `app.py` reads `data/` and re-scores live when the weights move.

## How it works

**Part 1, demand engine.** Clients hear about evergreen alternatives first, through AI answers and ads; advisors get a note on market-drop days; and the outreach is a kit under the prospect firm's name, since Equi builds client materials.

**Part 2, lead pipeline.** Clean and dedupe the list, score every firm with a confidence range, stress-test the weights, tag every field by source, route each firm to a sequence, and write two first-touch drafts.

**Tabs**
- **Start here:** the thesis and live findings.
- **Kit Studio:** firm-branded letter, talking points, IC memo outline for committee firms.
- **Market signals:** today's status, real-event replays, the alert feed.
- **Client demand:** what AI tells clients and advisors, a three-lane AEO action plan with page drafts, and the simulated air cover plan.
- **Search lab:** any client question: who owns it, what AI says, Equi's plan.
- **Shortlist:** ranked firms with score range, stability, route, and detail.
- **Your call:** firms that need a human decision, with a recommendation.
- **Drafts:** variants A and B, rewrite with a note, approve, sequencer CSV.
- **Review queue:** everything awaiting approval, with simulated engagement.

## Decisions and tradeoffs

**Weights.** Each traces to something Equi said, paraphrased from the brief (`WHY_WEIGHT` in `score.py`).

| Criterion | Weight | What Equi said (paraphrased) |
|---|---|---|
| Firm type | 25 | Firm type matters more than size. MFOs and RIAs becoming MFOs are the core buyer. |
| Client wealth | 20 | Eligibility is gated on qualified purchasers ($5M+). No QP clients, no sale. |
| Alts experience | 20 | The win is in the middle: fluent enough to move, not so deep they built it themselves. |
| Decision speed | 15 | Principal-led vs committee-led is the best predictor of deal speed. |
| AUM | 10 | $1B to $30B, $5B+ bullseye, but type beats size, so size is a tiebreaker. |
| Warmth | 10 | A warm path shortens the cycle but never makes a bad fit good. |

**Stress test.** I re-scored the list under 500 weightings, each moving every weight roughly 5 to 8 points. Seven firms hold Tier A in 96%+ of them, and at 1,000 draws the same seven hold in 95%+. Equal weights give the same Tier A list as ours. Firms that flip go to **Your call** as decisions for Equi (11 today), apart from firms that need research first (6).

**The liquid-alts gap.** A firm with private equity and real estate but no liquid sleeve speaks the language and has an obvious gap for Equi, so it scores highest. A firm in hedge funds is a replacement sale; a firm in everything has its own research desk.

**Caps.** LPL-affiliated firms max out at C (products need LPL platform approval). Non-US firms max out at B with -15 until compliance clears cross-border eligibility. Firms under $1B and single-family offices max out at C.

**Strict SEC matching.** The public IAPD search is fuzzy: "Harborstone" returned Viant Capital, an inactive broker-dealer once called Harborstone Capital. A match must be an active adviser with the same normalized name in the same state. The synthetic firms correctly return no match.

**Drafts and an approval queue, not live sending.** Every draft, kit, note, and ad passes a human. The model occasionally writes what compliance would stop ("structurally insulated from volatility"), so code checks catch known patterns, retry once, and flag the rest.

**No Clay on synthetic data.** Enriching made-up firms returns nothing, or someone else's data. In production: SEC Form ADV data for AUM (Item 5.F) and high-net-worth client counts and assets (Item 5.D, real average client size), and Clay for contacts and job changes.

**Session state and a cost guard.** Overrides, approvals, and the queue live in the browser session, with CSV exports; production would write them to BigQuery. Each session gets 25 Claude calls across all buttons, so the 42-call weekly AEO check runs from the command line.

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
- Persist state to BigQuery and wire tracked-link engagement into the queue.
- Rerun the AEO tracker weekly and track Equi's pages over time.
- Move kit generation to structured output, like everything else.
- Calibrate weights against real meeting outcomes.
