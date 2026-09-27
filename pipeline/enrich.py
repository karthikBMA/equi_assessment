"""Enrich scored firms and tag where every field came from.

Source tags:
- csv:          taken from the lead list (after cleaning)
- derived:      inferred by our own rules from csv fields (persona, decision structure, ...)
- sec_adv:      confirmed against the SEC's public adviser search (IAPD)
- needs_lookup: we could not fill it; a rep or a paid source has to

The IAPD search only returns identity (CRD, SEC number, office address). AUM and
Item 5.D (number of high-net-worth clients and their assets) live in the Form ADV
filing itself; in production we would pull those from the SEC's monthly Form ADV
bulk data by CRD. The sample CSV is synthetic, so expect no matches.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from pipeline.clean import GENERIC_INBOXES, norm_name
from pipeline.score import classify_title

IAPD_URL = "https://api.adviserinfo.sec.gov/search/firm?query={q}&nrows=10&start=0&wt=json"
ADV_PDF = "https://reports.adviserinfo.sec.gov/reports/ADV/{crd}/PDF/{crd}.pdf"

EMAIL_PATTERNS = {
    "{first}.{last}": lambda f, l: f"{f}.{l}",
    "{first}{last}": lambda f, l: f"{f}{l}",
    "{first}_{last}": lambda f, l: f"{f}_{l}",
    "{f}.{last}": lambda f, l: f"{f[0]}.{l}",
    "{f}{last}": lambda f, l: f"{f[0]}{l}",
    "{first}": lambda f, l: f,
    "{last}": lambda f, l: l,
}


# which cleaning notes explain a csv field
CLEANING_KEYWORDS = {
    "aum_usd": ("aum",),
    "avg_client_usd": ("avg client",),
    "alts_exposure": ("alts exposure",),
    "custodian": ("custodian",),
}


def tag(value, source: str, note: str | None = None) -> dict:
    return {"value": value, "source": source, "note": note}


def _name_parts(name: str | None):
    if not name:
        return None
    parts = [re.sub(r"[^a-z]", "", p) for p in name.lower().replace("-", "").split()]
    parts = [p for p in parts if p]
    return (parts[0], parts[-1]) if len(parts) >= 2 else None


def email_pattern(name: str | None, email: str | None) -> str | None:
    """Which pattern (if any) turns this person's name into their email local part."""
    parts = _name_parts(name)
    if not parts or not email:
        return None
    local = email.split("@")[0]
    for pat, fn in EMAIL_PATTERNS.items():
        if fn(*parts) == local:
            return pat
    return None


def infer_emails(contacts: list[dict]) -> list[dict]:
    """Fill missing or generic-inbox emails from a named colleague's format at the same domain."""
    known = [(c, email_pattern(c.get("name"), c.get("email"))) for c in contacts]
    known = [(c["email"].split("@")[1], p) for c, p in known if p]
    out = []
    for c in contacts:
        email = c.get("email")
        local = email.split("@")[0] if email else None
        generic = local in GENERIC_INBOXES
        if email and not generic:
            out.append({**c, "email_source": "csv", "email_note": None})
            continue
        parts = _name_parts(c.get("name"))
        if parts and known:
            domain, pat = known[0]
            guess = f"{EMAIL_PATTERNS[pat](*parts)}@{domain}"
            out.append({**c, "email": guess, "email_source": "derived",
                        "email_note": f"Inferred from colleague format {pat}@{domain}. Verify before sending."})
            continue
        why = ("Generic inbox only" if generic else "No email") + (
            "; no named colleague to copy the format from." if c.get("name") else "; no named contact.")
        out.append({**c, "email": email if generic else None, "email_source": "needs_lookup",
                    "email_note": why})
    return out


# ---------- SEC IAPD lookup ----------

def iapd_lookup(firm_name: str, state: str | None, timeout: float = 8) -> dict:
    """Strict match on the SEC adviser search. Returns {status, ...}.

    The search is fuzzy and matches former names, so we only accept an active
    investment adviser whose current name normalizes to ours.
    """
    url = IAPD_URL.format(q=urllib.parse.quote(firm_name))
    req = urllib.request.Request(url, headers={"User-Agent": "equi-gtm-takehome"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            hits = json.load(resp).get("hits", {}).get("hits", [])
    except Exception as e:
        return {"status": "error", "detail": f"IAPD unreachable ({type(e).__name__})"}
    target = norm_name(firm_name)
    for h in hits:
        s = h.get("_source", {})
        if s.get("firm_ia_scope") != "ACTIVE" or norm_name(s.get("firm_name", "")) != target:
            continue
        try:
            addr = json.loads(s.get("firm_ia_address_details") or "{}").get("officeAddress", {})
        except ValueError:
            addr = {}
        if state and addr.get("state") and addr["state"] != state:
            continue
        crd = s.get("firm_source_id")
        return {"status": "match", "crd": crd, "sec_number": s.get("firm_ia_full_sec_number"),
                "name": s.get("firm_name"), "city": addr.get("city"), "state": addr.get("state"),
                "adv_pdf": ADV_PDF.format(crd=crd)}
    near = hits[0]["_source"].get("firm_name") if hits else None
    return {"status": "no_match",
            "detail": f"No active adviser named '{firm_name}'." + (f" Closest result was '{near}', rejected." if near else "")}


# ---------- per-firm enrichment ----------

def _decision_basis(rec: dict) -> str:
    notes = (rec.get("raw_notes") or "").lower()
    if "committee" in notes or "ic-driven" in notes:
        return "From the CSV notes (committee language)."
    if "principal-led" in notes:
        return "From the CSV notes ('principal-led')."
    if rec.get("persona"):
        return f"From the contact's title ({rec['persona']})."
    return "No signal in notes or title."


def enrich_record(rec: dict, offline: bool = False) -> dict:
    """Return {field: {value, source, note}} for one scored firm, plus the fixed contacts."""
    s: dict[str, dict] = {}
    cleaned = " ".join(rec.get("cleaning_notes") or [])

    def csv_or_lookup(key, value, missing_note):
        if value is None:
            s[key] = tag(None, "needs_lookup", missing_note)
        else:
            words = CLEANING_KEYWORDS.get(key, ())
            note = next((n for n in rec.get("cleaning_notes") or []
                         if any(w in n.lower() for w in words)), None)
            s[key] = tag(value, "csv", note)

    csv_or_lookup("firm_type", rec.get("firm_type"), "Firm type missing.")
    csv_or_lookup("aum_usd", rec.get("aum_usd"), "Not in the CSV. Form ADV Item 5.F has it.")
    csv_or_lookup("avg_client_usd", rec.get("avg_client_usd"),
                  "Not in the CSV. Form ADV Item 5.D (HNW client count and assets) gives it.")
    csv_or_lookup("alts_exposure", rec.get("alts_exposure"), "Not in the CSV. Ask on the first call.")
    csv_or_lookup("custodian", rec.get("custodian"), "Not in the CSV. Form ADV Item 9 lists custodians.")
    s["location"] = tag(", ".join(x for x in (rec.get("city"), rec.get("state"), rec.get("country")) if x),
                        "csv", None)

    # contact and email
    contacts = infer_emails(rec.get("contacts") or [])
    s["contact_name"] = tag(rec.get("contact_name"), "csv" if rec.get("contact_name") else "needs_lookup",
                            None if rec.get("contact_name") else "No named decision-maker. Check the firm's team page or ADV Schedule A.")
    s["contact_title"] = tag(rec.get("contact_title"), "csv" if rec.get("contact_title") else "needs_lookup", None)
    primary = next((c for c in contacts if c.get("name") == rec.get("contact_name")), contacts[0] if contacts else None)
    if primary:
        s["contact_email"] = tag(primary.get("email"), primary["email_source"], primary["email_note"])
    else:
        s["contact_email"] = tag(None, "needs_lookup", "No contact on file.")

    # derived fields
    if rec.get("tier") != "DQ":
        s["persona"] = (tag(rec["persona"], "derived", f"From title '{rec.get('contact_title')}'.")
                        if rec.get("persona") else tag(None, "needs_lookup", "No title on file."))
        s["decision_structure"] = tag(rec.get("decision_structure"), "derived", _decision_basis(rec))
        s["mfo_transition"] = tag(bool(rec.get("mfo_transition")), "derived",
                                  "RIA with $10M+ average client; verify on ADV and website."
                                  if rec.get("mfo_transition") else "Not an RIA with $10M+ average client.")
    elif rec.get("contact_title"):
        s["persona"] = tag(classify_title(rec["contact_title"]), "derived", f"From title '{rec['contact_title']}'.")

    # SEC ADV
    if offline:
        adv = {"status": "skipped", "detail": "Offline run; IAPD not queried."}
    elif rec.get("country") != "USA":
        adv = {"status": "skipped", "detail": "Non-US firm; not SEC-registered as an adviser here."}
    else:
        adv = iapd_lookup(rec["firm_name"], rec.get("state"))
    if adv["status"] == "match":
        s["sec_registration"] = tag(f"CRD {adv['crd']} / {adv['sec_number']}", "sec_adv",
                                    f"Matched '{adv['name']}' in {adv.get('city')}, {adv.get('state')}. "
                                    f"AUM and Item 5.D: {adv['adv_pdf']}")
    else:
        s["sec_registration"] = tag(None, "needs_lookup", adv.get("detail"))
    if "Merged" in cleaned:
        s["source_rows"] = tag(rec.get("source_rows"), "csv", next(n for n in rec["cleaning_notes"] if n.startswith("Merged")))
    return {"sources": s, "contacts": contacts, "sec_adv": adv}


def enrich(records: list[dict], offline: bool = False, workers: int = 6) -> list[dict]:
    """Enrich in place. Adds `sources`, `sec_adv`, and inferred emails on `contacts`."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1 if offline else workers) as pool:
        results = list(pool.map(lambda r: enrich_record(r, offline), records))
    for r, e in zip(records, results):
        r["sources"], r["contacts"], r["sec_adv"] = e["sources"], e["contacts"], e["sec_adv"]
    return records


def source_summary(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for t in r["sources"].values():
            counts[t["source"]] = counts.get(t["source"], 0) + 1
    return counts
