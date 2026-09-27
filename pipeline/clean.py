"""Normalize and dedupe the raw lead list.

Every judgment call made while cleaning is logged per firm in `cleaning_notes`
so a rep can see exactly what was changed and why.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
from dateutil import parser as dateparser

STATE_NAMES = {
    "north carolina": "NC", "georgia": "GA", "colorado": "CO",
    "massachusetts": "MA", "new york": "NY", "california": "CA",
}

FIRM_TYPE_MAP = {
    "ria": "RIA",
    "mfo": "MFO", "multi-family office": "MFO",
    "single family office": "SFO", "sfo": "SFO",
    "wirehouse": "Wirehouse",
    "tamp": "TAMP",
    "broker-dealer": "Broker-Dealer",
    "hedge fund manager": "Hedge Fund Manager",
    "private credit manager": "Private Credit Manager",
}

NAME_SUFFIXES = r"\b(llc|inc|lp|ltd|co)\b"
NAME_SYNONYMS = {"grp": "group", "advisers": "advisors", "offices": "office"}
GENERIC_INBOXES = {"info", "contact", "hello", "admin", "office"}


# ---------- field parsers (each returns value, note|None) ----------

def parse_money(raw, kind: str):
    """Parse '4.2B', '$4,200,000,000', '2.5 billion', '880M', '900K' into dollars."""
    if pd.isna(raw) or str(raw).strip() == "":
        return None, None
    s = str(raw).strip().lower().replace("$", "").replace(",", "")
    m = re.match(r"^([\d.]+)\s*(b|billion|m|million|k|thousand)?$", s)
    if not m:
        return None, f"Could not parse {kind} '{raw}'"
    num, unit = float(m.group(1)), m.group(2)
    mult = {"b": 1e9, "billion": 1e9, "m": 1e6, "million": 1e6,
            "k": 1e3, "thousand": 1e3}.get(unit)
    if mult:
        return num * mult, None
    # bare number: raw dollars if huge, otherwise ambiguous
    if num >= 1e7:
        return num, None
    if kind == "AUM":
        return num * 1e6, f"AUM '{raw}' had no unit; read as ${num/1000:.1f}B (millions convention)"
    return num * 1e6, f"{kind} '{raw}' had no unit; read as millions"


def parse_date(raw):
    if pd.isna(raw):
        return None
    try:
        return dateparser.parse(str(raw), default=pd.Timestamp("2026-01-01")).date()
    except (ValueError, OverflowError):
        return None


def clean_email(raw):
    if pd.isna(raw):
        return None, None
    e = str(raw).strip().lower()
    note = None
    if " " in e:
        e = e.replace(" ", "")
        note = f"Fixed stray space in email ({raw})"
    if not re.match(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$", e):
        return None, f"Invalid email dropped ({raw})"
    return e, note


def phone_digits(raw):
    if pd.isna(raw):
        return None
    d = re.sub(r"\D", "", str(raw))
    return d[-10:] if len(d) >= 10 else None


def norm_name(name: str) -> str:
    s = re.sub(r"[^\w\s&]", " ", str(name).lower())
    s = re.sub(NAME_SUFFIXES, " ", s)
    words = [NAME_SYNONYMS.get(w, w) for w in s.split()]
    return " ".join(words)


# ---------- dedupe ----------

def _cluster(df: pd.DataFrame) -> list[list[int]]:
    """Union-find on normalized name, email domain, or phone."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    keys: dict[str, int] = {}
    for i, row in df.iterrows():
        for k in (f"n:{row._norm}",
                  f"d:{row._domain}" if isinstance(row._domain, str) else None,
                  f"p:{row._phone}" if isinstance(row._phone, str) else None):
            if not k:
                continue
            if k in keys:
                union(i, keys[k])
            else:
                keys[k] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(df)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _completeness(row) -> int:
    return int(row.notna().sum())


def clean_leads(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["_row"] = raw.index + 2  # spreadsheet row number incl. header
    rows = []
    for _, r in raw.iterrows():
        notes = []
        aum, n1 = parse_money(r.aum, "AUM")
        client, n2 = parse_money(r.avg_client_size, "Avg client size")
        email, n3 = clean_email(r.email)
        notes += [n for n in (n1, n2, n3) if n]
        ftype = FIRM_TYPE_MAP.get(str(r.firm_type).strip().lower(), str(r.firm_type))
        state = str(r.state).strip()
        state = STATE_NAMES.get(state.lower(), state.upper())
        local = email.split("@")[0] if email else None
        rows.append({
            **r.to_dict(),
            "firm_type": ftype,
            "state": state,
            "city": str(r.city).strip().title(),
            "country": "Canada" if str(r.country).strip() == "Canada" else "USA",
            "aum_usd": aum,
            "avg_client_usd": client,
            "email": email,
            "generic_inbox": bool(local in GENERIC_INBOXES) if local else False,
            "phone": phone_digits(r.phone),
            "last_touch_date": parse_date(r.last_touch),
            "_norm": norm_name(r.firm_name),
            "_domain": email.split("@")[1] if email else None,
            "_phone": phone_digits(r.phone),
            "_notes": notes,
        })
    df = pd.DataFrame(rows)

    firms = []
    for idx in _cluster(df):
        grp = df.loc[idx].copy()
        grp["_score"] = grp.apply(_completeness, axis=1)
        grp = grp.sort_values("_score", ascending=False)
        base = grp.iloc[0].to_dict()
        notes = [n for ns in grp._notes for n in ns]

        # fill gaps from the other rows
        for col in ["aum_usd", "avg_client_usd", "alts_exposure", "email", "phone",
                    "custodian", "last_touch_date", "title", "primary_contact"]:
            if pd.isna(base.get(col)) or base.get(col) is None:
                for _, other in grp.iloc[1:].iterrows():
                    if pd.notna(other[col]) and other[col] is not None:
                        base[col] = other[col]
                        notes.append(f"Filled {col.replace('_', ' ')} from duplicate row {other._row}")
                        break

        # corroborate an ambiguous unit-less AUM against the duplicate rows
        for _, other in grp.iloc[1:].iterrows():
            if other._row != grp.iloc[0]._row and pd.notna(other.aum) and pd.notna(base.get("aum_usd")):
                if abs((other.aum_usd or 0) - base["aum_usd"]) < 1e6:
                    notes = [n + f"; confirmed by row {other._row} ('{other.aum}')"
                             if n.startswith("AUM '") else n for n in notes]

        # contacts: keep every distinct person at the firm
        contacts = []
        seen = set()
        for _, c in grp.iterrows():
            if pd.isna(c.primary_contact) and pd.isna(c.title):
                continue
            key = (str(c.email or c.primary_contact)).lower()
            last = str(c.primary_contact).split()[-1].lower() if pd.notna(c.primary_contact) else None
            if key in seen or (last and any(last in s for s in seen)):
                continue
            seen.add(key)
            contacts.append({"name": c.primary_contact if pd.notna(c.primary_contact) else None,
                             "title": c.title if pd.notna(c.title) else None,
                             "email": c.email, "phone": c.phone})
        if len(grp) > 1:
            same_person = len(contacts) == 1
            notes.insert(0, f"Merged {len(grp)} rows ({', '.join(str(x) for x in sorted(grp._row))})"
                         + ("" if same_person else f"; kept {len(contacts)} distinct contacts"))
        all_notes = [str(n) for n in grp.notes if pd.notna(n)]
        base["raw_notes"] = "; ".join(dict.fromkeys(all_notes))
        base["contacts"] = contacts
        base["source_rows"] = sorted(int(x) for x in grp._row)
        base["cleaning_notes"] = list(dict.fromkeys(notes))
        firms.append(base)

    out = pd.DataFrame(firms)
    out["firm_name"] = out.firm_name.map(_display_name)
    keep = ["firm_name", "firm_type", "aum_usd", "city", "state", "country", "contacts",
            "custodian", "alts_exposure", "avg_client_usd", "last_touch_date",
            "generic_inbox", "raw_notes", "cleaning_notes", "source_rows"]
    return out[keep].reset_index(drop=True)


def _display_name(n: str) -> str:
    n = re.sub(r"\s+(LLC|Inc)\.?$", "", str(n), flags=re.I).strip()
    return n.title() if n.isupper() else n


if __name__ == "__main__":
    d = clean_leads("data/sample-leads.csv")
    print(len(d), "firms")
    for _, r in d[d.cleaning_notes.map(len) > 0].iterrows():
        print(r.firm_name, "|", r.cleaning_notes)
