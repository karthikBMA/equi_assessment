"""Shared rules for anything a client might read.

Equi cannot advertise its funds to the public, so client-facing copy is category
education that ends in "ask your advisor": no fund names, no Equi performance,
no return claims. Every client-facing piece carries DISCLOSURE (kits carry the
longer COMPLIANCE_FOOTER in pipeline/kit.py, which covers the same ground).
"""
from __future__ import annotations

import re

DISCLOSURE = ("Educational only, not an offer or a recommendation. Alternative investments involve risk, "
              "including loss of principal. Ask your advisor whether they fit your situation.")


NEGATION_WINDOW = 60   # characters before the claim word


def lint_text(text: str) -> list[str]:
    """Sentence-level check with the market-note rules. Negated disclaimers pass, and so do questions:
    an FAQ asking "Can any fund guarantee protection?" is not a claim that one can."""
    from pipeline.signals import NEGATABLE, NEGATION, NOTE_RULES
    found = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for pattern, label in NOTE_RULES:
        for sent in sentences:
            m = re.search(pattern, sent, re.I)
            if not m:
                continue
            # the negation has to sit in the same clause, just before the claim word: "no lock-up" later
            # in the sentence, or "no fixed end date," in an earlier clause, does not excuse "downside protection"
            before = re.split(r"[,;:()]", sent[max(0, m.start() - NEGATION_WINDOW):m.start()])[-1]
            excused = label in NEGATABLE and (NEGATION.search(before) or sent.rstrip().endswith("?"))
            if not excused:
                found.append(f'{label} in "{sent.strip()}"')
                break
    return found
