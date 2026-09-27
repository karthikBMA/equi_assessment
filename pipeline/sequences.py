"""What each route actually does, touch by touch.

Keys match the route strings from `score.route()`. `short` is the label the app
shows in tables. A touch is (day, channel, what).
`day` is None for touches that fire on an event instead of a schedule.
"""
from __future__ import annotations

SEQUENCES = {
    "Pre-built Kit + founder direct": {
        "short": "Kit + founder",
        "goal": "Meeting with the founder inside two weeks.",
        "touches": [
            (0, "Email", "Send the approved draft from the founder or senior Equi person."),
            (3, "Email", "Send the firm-branded kit: client letter, talking points."),
            (7, "LinkedIn", "Short note referencing the kit. No pitch."),
            (12, "Call", "Call the founder. Ask if the client letter is usable as is."),
        ],
    },
    "Pre-built Kit + CIO peer sequence": {
        "short": "Kit + CIO",
        "goal": "Peer conversation between the CIO and an Equi investment lead.",
        "touches": [
            (0, "Email", "Send the approved draft, framed as one investment person to another."),
            (4, "Email", "Send the kit plus a one-page note on how the fund fits a liquid-alts sleeve."),
            (9, "LinkedIn", "Connect from the Equi investment lead, not the SDR."),
            (14, "Call", "Offer 30 minutes with the Equi PM on portfolio construction."),
        ],
    },
    "Committee Kit to champion": {
        "short": "Committee kit",
        "goal": "Get the fund onto the next investment committee agenda.",
        "touches": [
            (0, "Email", "Email the research lead offering the DDQ and a draft IC memo."),
            (5, "Email", "Send the committee pack: DDQ, IC memo outline, client letter."),
            (14, "Email", "Check in: what does the committee still need?"),
            (14, "Research", "Map the CIO and other committee members; plan a second thread."),
        ],
    },
    "Committee Kit to champion, slower cadence": {
        "short": "Committee kit (slow)",
        "goal": "Equip the research lead so the fund reaches a committee agenda within two quarters.",
        "touches": [
            (0, "Email", "Email the research lead offering the DDQ and a draft IC memo."),
            (10, "Email", "Send the committee pack: DDQ, IC memo outline, client letter."),
            (28, "Email", "Check in: what does the committee still need?"),
            (28, "Research", "Map the CIO and other committee members."),
            (None, "Email", "Between touches, send the champion market-signal notes they can circulate."),
        ],
    },
    "Market-signal alerts + Kit on engagement": {
        "short": "Signal alerts",
        "goal": "Be useful on the days advisors field worried client calls.",
        "touches": [
            (None, "Email", "On a market trigger: short client-ready note plus a two-line advisor ping."),
            (None, "Email", "If they open or forward a note, offer the full kit."),
            (None, "Weekly", "Otherwise a weekly digest. No daily noise."),
        ],
    },
    "Education nurture": {
        "short": "Nurture",
        "goal": "Stay known until the firm grows into fit.",
        "touches": [
            (0, "Email", "Monthly explainer on evergreen alternatives. Educational only."),
            (90, "Review", "Re-score quarterly. Move to alerts if the tier improves."),
        ],
    },
    "Research: find decision-maker": {
        "short": "Find contact",
        "goal": "Find the founder or CIO before any outreach.",
        "touches": [
            (0, "Research", "Check the firm website team page and Form ADV Schedule A."),
            (2, "Review", "Re-score with the contact added; the route updates from there."),
        ],
    },
    "Park": {
        "short": "Park",
        "goal": "No outreach.",
        "touches": [(180, "Review", "Re-score in six months.")],
    },
    "Disqualified": {
        "short": "Disqualified",
        "goal": "No outreach.",
        "touches": [],
    },
}


def sequence_for(route: str) -> dict:
    return SEQUENCES.get(route, {"short": route, "goal": "No sequence defined for this route.", "touches": []})


def short_route(route: str) -> str:
    return sequence_for(route)["short"]


def as_rows(route: str) -> list[dict]:
    """Touches as table rows for the app."""
    return [{"Day": "On trigger" if d is None else f"D{d}", "Channel": ch, "What": what}
            for d, ch, what in sequence_for(route)["touches"]]
