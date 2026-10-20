"""Self-learning people directory (Roadmap #3).

"Message and mail Umesh" should just work — resolve to *Umesh Patil /
umesh@example.com* automatically, and the directory should GROW and HEAL
itself as Paimon meets people. This is the structured-resolution companion to the
memory wiki: the wiki holds *narrative* facts about a person ("vendor PM, slow to
respond"); this holds the *structured* fields you resolve against (email,
channels, aliases) and the fuzzy lookup that turns a casual name into a contact.

Store: ``data/contacts.json``
    {"contacts": [
       {id, full_name, aliases[], email, channels{teams,slack,...},
        notes, source, confidence, last_seen, created_at}
    ]}

Self-learning (provenance + confidence so new evidence overrides old):
  * EXPLICIT  — the user says "Umesh's email is X" → memory_learn.observe.
  * IMPLICIT  — comms sends mail (mailer) → observe the recipient; a follow-up
                logs a person (sync_from_board) → observe the name.
Resolution is fuzzy (email-exact > alias/name-exact > token overlap > first name
> small edit distance), returning a ranked list so the caller can pick the top
match or ask when it's genuinely ambiguous.

Dependency-free; degrades safely (every public call is wrapped by callers).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone

import store  # reuse DATA_DIR + atomic JSON IO

CONTACTS_FILE = os.path.join(store.DATA_DIR, "contacts.json")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ===========================================================================
#  storage
# ===========================================================================
def _load() -> dict:
    return store._read_json(CONTACTS_FILE, {"contacts": []})


def _save(data: dict) -> None:
    data["updated_at"] = _now()
    store._write_json(CONTACTS_FILE, data)


def _new_id() -> str:
    return "ct_" + uuid.uuid4().hex[:10]


# ===========================================================================
#  fuzzy matching
# ===========================================================================
def _lev(a: str, b: str) -> int:
    """Small Levenshtein distance (tolerant of whisper/typo mishears)."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _names(c: dict) -> list[str]:
    out = [c.get("full_name", "")] + list(c.get("aliases", []))
    return [_norm(n) for n in out if n]


def _score(contact: dict, q: str) -> float:
    """How well a contact matches query ``q`` (0 = no match). Confidence nudges
    ties so a well-established contact wins over a low-evidence one."""
    q = _norm(q)
    if not q:
        return 0.0
    email = _norm(contact.get("email", ""))
    if email and q == email:
        return 100.0
    names = _names(contact)
    if q in names:
        base = 92.0
    else:
        base = 0.0
        qt = set(q.split())
        for n in names:
            nt = set(n.split())
            if not nt:
                continue
            if n.split()[0] == q:                      # first-name hit
                base = max(base, 78.0)
            inter = qt & nt
            if inter:                                  # shared tokens
                base = max(base, 60.0 + 8.0 * len(inter))
            if q in n or n in q:                       # substring
                base = max(base, 55.0)
            if _lev(q, n) <= 2:                        # small mishear
                base = max(base, 50.0)
    if base == 0.0:
        return 0.0
    return base + 5.0 * float(contact.get("confidence", 0.7))


def lookup(query: str, limit: int = 5) -> list[dict]:
    """Ranked contacts matching a name/alias/email (best first); [] if none."""
    scored = [(s, c) for c in _load()["contacts"]
              if (s := _score(c, query)) > 0]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]


def resolve(query: str) -> dict:
    """Resolve ``query`` to one contact when confident, else flag ambiguity.

    Returns {"status": "resolved"|"ambiguous"|"none", "contact": <best|None>,
    "candidates": [...]}. 'resolved' when there's a single strong match (or the
    top clearly beats the rest); 'ambiguous' when two+ are close (the caller
    should ask which one)."""
    hits = lookup(query, limit=5)
    if not hits:
        return {"status": "none", "contact": None, "candidates": []}
    if len(hits) == 1:
        return {"status": "resolved", "contact": hits[0], "candidates": hits}
    top, second = _score(hits[0], query), _score(hits[1], query)
    if top >= 90 or top - second >= 18:
        return {"status": "resolved", "contact": hits[0], "candidates": hits}
    return {"status": "ambiguous", "contact": None, "candidates": hits}


# ===========================================================================
#  write — observe (self-learning) + explicit CRUD
# ===========================================================================
def _find_for_observe(contacts: list, name: str, email: str):
    """Conservative match for an OBSERVED (auto-learned) signal: merge only on
    strong evidence so two different people don't get fused.

      1. exact email always merges (same address = same person);
      2. an exact alias/full-name match merges — UNLESS the observed email
         conflicts with that contact's existing email (then it's likely a
         different person / a new account → keep them separate);
      3. a first-name match (in either direction, e.g. 'Umesh' ↔ 'Umesh Patil')
         merges when there's exactly ONE non-conflicting candidate.
    The email-conflict guard is what keeps two same-first-name people apart."""
    ne, nn = _norm(email), _norm(name)
    if ne:
        for c in contacts:
            if _norm(c.get("email", "")) == ne:
                return c
    if not nn:
        return None

    def _email_ok(c):
        return not (ne and _norm(c.get("email", "")) and _norm(c["email"]) != ne)

    exact = [c for c in contacts if nn in _names(c)]
    exact_ok = [c for c in exact if _email_ok(c)]
    if exact_ok:
        return exact_ok[0]
    if exact:                      # name matches but email conflicts → separate
        return None

    obs_first = nn.split()[0]
    cands = [c for c in contacts
             if c.get("full_name")
             and _norm(c["full_name"]).split()[0] == obs_first and _email_ok(c)]
    return cands[0] if len(cands) == 1 else None


def observe(name: str = "", email: str = "", channels: dict | None = None,
            note: str = "", source: str = "observed",
            confidence: float = 0.7) -> dict:
    """Idempotently fold an observed signal about a person into the directory.
    Merges into an existing contact on strong evidence (email/exact name), else
    creates one. New evidence fills blanks, adds aliases, bumps last_seen, and
    raises confidence — never silently overwrites a higher-confidence email."""
    name = (name or "").strip()
    email = (email or "").strip()
    if not name and not email:
        return {}
    data = _load()
    contacts = data["contacts"]
    c = _find_for_observe(contacts, name, email)
    created = False
    if c is None:
        c = {
            "id": _new_id(),
            "full_name": name or email.split("@")[0],
            "aliases": [], "email": "", "channels": {}, "notes": "",
            "source": source, "confidence": 0.0,
            "created_at": _now(), "last_seen": _now(),
        }
        contacts.append(c)
        created = True

    # full_name: promote a longer/more-complete name; keep the old one as an
    # alias. A shorter new name that isn't already known becomes an alias.
    if name:
        if not c["full_name"] or len(name) > len(c["full_name"]):
            old = c["full_name"]
            c["full_name"] = name          # promote first…
            if old and _norm(old) != _norm(name):
                _add_alias(c, old)          # …then keep the old name as an alias
        elif _norm(name) not in _names(c):
            _add_alias(c, name)

    # email: fill if empty; if different AND new evidence is at least as
    # confident, replace it (self-heal — new account overrides old).
    if email:
        cur = c.get("email", "")
        if not cur:
            c["email"] = email
        elif _norm(cur) != _norm(email) and confidence >= float(c.get("confidence", 0)):
            c["email"] = email

    if channels:
        c.setdefault("channels", {}).update({k: v for k, v in channels.items() if v})
    if note:
        c["notes"] = (c.get("notes", "") + ("; " if c.get("notes") else "") + note)[:400]
    c["source"] = source or c.get("source", "observed")
    c["confidence"] = max(float(c.get("confidence", 0.0)), float(confidence))
    c["last_seen"] = _now()
    _save(data)
    return c


def _add_alias(c: dict, alias: str) -> None:
    alias = (alias or "").strip()
    if not alias:
        return
    aliases = c.setdefault("aliases", [])
    if _norm(alias) != _norm(c.get("full_name", "")) and \
            alias not in aliases and _norm(alias) not in [_norm(a) for a in aliases]:
        aliases.append(alias)


def add_contact(full_name: str, email: str = "", aliases: list | None = None,
                channels: dict | None = None, notes: str = "",
                source: str = "user", confidence: float = 0.95) -> dict:
    """Explicitly add/update a contact (high confidence — the user said so)."""
    c = observe(name=full_name, email=email, channels=channels, note=notes,
                source=source, confidence=confidence)
    if aliases:
        data = _load()
        target = next((x for x in data["contacts"] if x["id"] == c["id"]), None)
        if target:
            for a in aliases:
                _add_alias(target, a)
            _save(data)
            return target
    return c


def update_contact(contact_id: str, **fields) -> dict | None:
    allowed = {"full_name", "email", "notes", "confidence"}
    data = _load()
    c = next((x for x in data["contacts"] if x["id"] == contact_id), None)
    if c is None:
        return None
    for k, v in fields.items():
        if k in allowed and v is not None:
            c[k] = v
        elif k == "channels" and isinstance(v, dict):
            c.setdefault("channels", {}).update(v)
        elif k == "aliases" and isinstance(v, list):
            for a in v:
                _add_alias(c, a)
    c["last_seen"] = _now()
    _save(data)
    return c


def list_contacts() -> list[dict]:
    return sorted(_load()["contacts"],
                  key=lambda c: c.get("last_seen", ""), reverse=True)


def get_contact(contact_id: str) -> dict | None:
    return next((c for c in _load()["contacts"] if c["id"] == contact_id), None)


def delete_contact(contact_id: str) -> bool:
    data = _load()
    n = len(data["contacts"])
    data["contacts"] = [c for c in data["contacts"] if c["id"] != contact_id]
    if len(data["contacts"]) != n:
        _save(data)
        return True
    return False


# ===========================================================================
#  implicit learning from the task board
# ===========================================================================
def sync_from_board() -> int:
    """Pull people the BA already tracks — anyone named in a follow-up or owning
    a task — into the directory at low confidence, so they're resolvable even
    before we learn their email. Idempotent (observe dedups). Returns the number
    of distinct names observed."""
    seen = set()
    try:
        for f in store.pending_follow_ups():
            person = (f.get("person") or "").strip()
            if person and _norm(person) not in seen:
                seen.add(_norm(person))
                observe(name=person, source="follow-up", confidence=0.5)
        for t in store.all_tasks():
            person = (t.get("person") or "").strip()
            if person and _norm(person) not in seen:
                seen.add(_norm(person))
                observe(name=person, source="task", confidence=0.5)
    except Exception as exc:  # pragma: no cover
        print(f"[people] sync_from_board skipped: {exc}")
    return len(seen)


def observe_email_recipient(address: str, source: str = "email") -> dict:
    """Record that we sent/received mail to an address (called by mailer). Keeps
    a contact's email + last_seen current even when we only know the address."""
    addr = (address or "").strip()
    if not _EMAIL_RE.fullmatch(addr):
        # tolerate "Name <addr>" forms
        m = _EMAIL_RE.search(addr)
        if not m:
            return {}
        addr = m.group(0)
    return observe(email=addr, channels={"email": addr}, source=source,
                   confidence=0.7)
