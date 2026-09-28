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


def lint_text(text: str) -> list[str]:
    """Sentence-level check with the market-note rules. Negated disclaimers pass, and so do questions:
    an FAQ asking "Can any fund guarantee protection?" is not a claim that one can."""
    from pipeline.signals import NEGATABLE, NEGATION, NOTE_RULES
    found = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for pattern, label in NOTE_RULES:
        for sent in sentences:
            excused = label in NEGATABLE and (NEGATION.search(sent) or sent.rstrip().endswith("?"))
            if re.search(pattern, sent, re.I) and not excused:
                found.append(f'{label} in "{sent.strip()}"')
                break
    return found
