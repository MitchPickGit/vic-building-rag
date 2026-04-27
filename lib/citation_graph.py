"""
Cross-reference graph for the legislation corpus.

Parses every cross-reference appearing in the chunks (in body text and
in the parsers' cross_references field), resolves internal references
to actual chunk citations where possible, and flags external references
to the BCA/NCC, Australian Standards, and other Acts.

Usage:
    # One-shot build:
    python -m lib.citation_graph build

    # Programmatic lookup:
    from lib.citation_graph import CitationGraph
    g = CitationGraph()
    refs = g.outgoing("Building Act 1993 (Vic) s 16(1)")
    # → [{"to_citation": ..., "relationship_type": ..., "external": ..., "to_text": ...}, ...]

Schema (SQLite):
    citations(
        from_citation       TEXT NOT NULL,    -- the chunk doing the citing
        to_text             TEXT NOT NULL,    -- raw reference text, e.g. "section 16(1)"
        to_citation         TEXT,             -- resolved citation if internal, else NULL
        relationship_type   TEXT NOT NULL,    -- internal_section | internal_regulation
                                              --   | internal_part | internal_division
                                              --   | internal_schedule | external_act
                                              --   | external_standard | external_code
        external            INTEGER NOT NULL  -- 0/1
    )
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DB_PATH = "citation_graph.db"

# --- Reference patterns ----------------------------------------------------
#
# Order matters — more specific patterns first. We use anchored regexes
# with named groups so we can build typed records cleanly.

# Australian Standards: "AS 3959", "AS/NZS 1170.4-2007", "AS 1428.1"
RE_AS = re.compile(
    r"\b(AS(?:/NZS)?)\s+(\d{3,4}(?:\.\d+)?(?:[-–]\d{4})?)\b",
    re.IGNORECASE,
)

# BCA / NCC references: "BCA Volume One", "NCC Volume Two Part 3.7", "BCA Volume One clause A6G2"
RE_BCA_NCC = re.compile(
    r"\b(BCA|NCC)\s+(Volume\s+(?:One|Two|Three|1|2|3))(?:\s+(Part\s+\d+(?:\.\d+)*|clause\s+[A-Z]?\d+(?:\.\d+)*))?",
    re.IGNORECASE,
)

# Other Acts (state legislation referenced by name): match "<Title> Act <year>"
# where Title is one or more capitalised words. Excludes the host Acts —
# see is_external_act_name() filter below.
RE_OTHER_ACT = re.compile(
    r"\b((?:[A-Z][A-Za-z'’\-]*\s+){1,8}Act\s+\d{4})\b"
)

# Internal Act sections: "section 16(1)", "section 16(1)(a)", "s. 16(1)", "ss. 16 and 17"
RE_INTERNAL_SECTION = re.compile(
    r"(?:section|sections|ss?\.)\s+"
    r"(\d+[A-Z]*(?:\(\d+[A-Z]*\))?(?:\([a-z]+\))?)",
    re.IGNORECASE,
)

# Internal Regulations: "regulation 16", "reg. 16(1)", "regulations 25 and 26"
RE_INTERNAL_REGULATION = re.compile(
    r"(?:regulation|regulations|reg\.|regs\.?)\s+"
    r"(\d+[A-Z]*(?:\(\d+[A-Z]*\))?(?:\([a-z]+\))?)",
    re.IGNORECASE,
)

# Structural references
RE_INTERNAL_PART = re.compile(r"\bPart\s+(\d+[A-Z]*)\b")
RE_INTERNAL_DIVISION = re.compile(r"\bDivision\s+(\d+[A-Z]*)\b")
RE_INTERNAL_SCHEDULE = re.compile(r"\bSchedule\s+(\d+)\b")

# Sentinels for the host Acts so we don't classify them as "external"
HOST_ACTS = {
    "Building Act 1993",
    "Building Regulations 2018",
}


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def extract_references(text: str) -> list[dict]:
    """Return a list of structured reference records found in `text`.
    Each record: {to_text, relationship_type, external, to_anchor}.

    `to_anchor` is the structural target (e.g. "16(1)", "Part 3",
    "Schedule 3") — used downstream to resolve to a chunk citation.
    """
    if not text:
        return []
    refs: list[dict] = []
    seen: set[tuple] = set()  # dedupe (relationship_type, to_anchor)

    def add(to_text: str, rel: str, anchor: str, external: bool):
        key = (rel, anchor.strip().lower())
        if key in seen:
            return
        seen.add(key)
        refs.append({
            "to_text": to_text.strip(),
            "relationship_type": rel,
            "external": external,
            "to_anchor": anchor.strip(),
        })

    # External first (so e.g. "BCA Volume Two clause A6G2" doesn't get
    # eaten by the Part regex).
    for m in RE_AS.finditer(text):
        add(m.group(0), "external_standard", m.group(0), True)
    for m in RE_BCA_NCC.finditer(text):
        add(m.group(0), "external_code", m.group(0), True)
    for m in RE_OTHER_ACT.finditer(text):
        title = m.group(1).strip()
        # Skip host Acts and obvious false positives ("the Act", "this Act")
        if title in HOST_ACTS:
            continue
        if any(title.startswith(prefix) for prefix in ("the ", "this ", "an ", "a ")):
            continue
        add(title, "external_act", title, True)

    # Internal — these are all anchored to the host corpus
    for m in RE_INTERNAL_SECTION.finditer(text):
        add(m.group(0), "internal_section", m.group(1), False)
    for m in RE_INTERNAL_REGULATION.finditer(text):
        add(m.group(0), "internal_regulation", m.group(1), False)
    for m in RE_INTERNAL_PART.finditer(text):
        # Skip part headings and "this Part" — already structural in nature
        add(m.group(0), "internal_part", m.group(1), False)
    for m in RE_INTERNAL_DIVISION.finditer(text):
        add(m.group(0), "internal_division", m.group(1), False)
    for m in RE_INTERNAL_SCHEDULE.finditer(text):
        add(m.group(0), "internal_schedule", m.group(1), False)
    return refs


# ---------------------------------------------------------------------------
# Resolution: turn an internal reference's anchor into a chunk citation
# ---------------------------------------------------------------------------

def build_chunk_index(chunks: list[dict]) -> dict[tuple, str]:
    """Return a lookup keyed by (doc_type, structural_key) → citation.

    structural_key forms:
      ("section",  "16")
      ("section",  "16(1)")
      ("regulation", "23")
      ("part_act", "Part 3")
      ("part_regs", "Part 3")
      ("schedule_act", "1")
      ("schedule_regs_item", "16")
    """
    index: dict[tuple, str] = {}
    for c in chunks:
        cit = c["citation"]
        sn = (c.get("section_number") or "").strip()
        sub = (c.get("subsection") or "").strip()
        part = (c.get("part") or "").strip()
        doc = c.get("doc_type", "")

        if not sn:
            continue

        if doc == "act":
            if sub:
                index.setdefault(("section", f"{sn}{sub}"), cit)
            # Always also map the bare section number to ANY one of its chunks
            # — useful when the reference text is just "section 16" with no
            # subsection. We pick the "(1)" subsection chunk if available, else
            # any chunk with that section number.
            existing = index.get(("section", sn))
            if existing is None or (sub == "(1)"):
                index[("section", sn)] = cit
        elif doc == "regulation":
            if sub:
                index.setdefault(("regulation", f"{sn}{sub}"), cit)
            existing = index.get(("regulation", sn))
            if existing is None or (sub == "(1)"):
                index[("regulation", sn)] = cit
        elif doc == "regulation_schedule":
            # Schedule 3 items
            item = c.get("item_number")
            if item:
                index[("schedule_regs_item", str(item))] = cit

        # Part-level lookup (any chunk in that part is a representative)
        if part:
            m = re.match(r"Part\s+(\d+[A-Z]*)", part)
            if m:
                key = ("part_act" if doc == "act" else "part_regs", m.group(1))
                index.setdefault(key, cit)

    return index


def resolve_internal_ref(rel: str, anchor: str, from_chunk: dict,
                         chunk_index: dict[tuple, str]) -> str | None:
    """Return the citation of the chunk this reference points to, or None."""
    from_doc = from_chunk.get("doc_type", "")
    if rel == "internal_section":
        # Sections are an Act concept. Map "16(1)" → chunk; if no subsection
        # match, fall back to "16" (any chunk with that section number).
        if ("section", anchor) in chunk_index:
            return chunk_index[("section", anchor)]
        bare = re.match(r"^(\d+[A-Z]*)", anchor)
        if bare and ("section", bare.group(1)) in chunk_index:
            return chunk_index[("section", bare.group(1))]
        return None
    if rel == "internal_regulation":
        if ("regulation", anchor) in chunk_index:
            return chunk_index[("regulation", anchor)]
        bare = re.match(r"^(\d+[A-Z]*)", anchor)
        if bare and ("regulation", bare.group(1)) in chunk_index:
            return chunk_index[("regulation", bare.group(1))]
        return None
    if rel == "internal_part":
        # Use the from_chunk's doc_type to pick which document's Part X
        # the reference targets — assume same document unless explicit.
        key = ("part_act" if from_doc == "act" else "part_regs", anchor)
        return chunk_index.get(key)
    if rel == "internal_schedule":
        # Disambiguate: in the Regs, "Schedule 3" is the exemptions table,
        # which is broken into items. We only resolve to a chunk if it's
        # a Schedule 3 of the Regs — otherwise return None (we don't index
        # other Schedules yet, but the reference is still valuable as a
        # pointer for the user).
        if from_doc == "regulation" and anchor == "3":
            # Just return the first Sch 3 item chunk as a representative
            for k, v in chunk_index.items():
                if k[0] == "schedule_regs_item":
                    return v
        return None
    return None


# ---------------------------------------------------------------------------
# Building / loading the SQLite database
# ---------------------------------------------------------------------------

def build_graph(chunk_paths: Iterable[str], db_path: str = DB_PATH) -> dict:
    """Walk the chunks, extract references, resolve internals, write SQLite.

    Returns a stats dict for the caller to print.
    """
    chunks: list[dict] = []
    for p in chunk_paths:
        if not Path(p).exists():
            print(f"  warn: {p} missing, skipping")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {len(list(chunk_paths))} files")

    chunk_index = build_chunk_index(chunks)
    print(f"Built chunk index: {len(chunk_index)} keys")

    # Open SQLite
    if Path(db_path).exists():
        Path(db_path).unlink()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE citations (
            from_citation     TEXT NOT NULL,
            to_text           TEXT NOT NULL,
            to_citation       TEXT,
            relationship_type TEXT NOT NULL,
            external          INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_from ON citations(from_citation)")
    conn.execute("CREATE INDEX idx_to ON citations(to_citation)")
    conn.execute("CREATE INDEX idx_external ON citations(external)")

    rows = []
    type_counts: dict[str, int] = defaultdict(int)
    resolved_count = 0
    external_count = 0

    for c in chunks:
        # Search both text and the structural metadata that the parser
        # captured (cross_references field, plus section title for cases
        # like "Review by VCAT").
        haystack_parts = [
            c.get("text") or "",
            c.get("note") or "",
            c.get("amendment_history") or "",
            " ".join(c.get("cross_references") or []),
        ]
        haystack = "\n".join(haystack_parts)

        refs = extract_references(haystack)
        for r in refs:
            to_citation: str | None = None
            if not r["external"]:
                to_citation = resolve_internal_ref(
                    r["relationship_type"], r["to_anchor"], c, chunk_index
                )
                if to_citation:
                    resolved_count += 1
            else:
                external_count += 1
            rows.append((
                c["citation"],
                r["to_text"],
                to_citation,
                r["relationship_type"],
                1 if r["external"] else 0,
            ))
            type_counts[r["relationship_type"]] += 1

    conn.executemany(
        "INSERT INTO citations VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()

    stats = {
        "total_rows": len(rows),
        "resolved_internal": resolved_count,
        "external": external_count,
        "by_type": dict(type_counts),
    }
    return stats


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

class CitationGraph:
    def __init__(self, db_path: str = DB_PATH):
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"{db_path} missing. Run: python -m lib.citation_graph build"
            )
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def outgoing(self, from_citation: str) -> list[dict]:
        """References emitted by `from_citation` (i.e. what does this chunk cite)."""
        rows = self.conn.execute(
            "SELECT * FROM citations WHERE from_citation = ?", (from_citation,)
        ).fetchall()
        return [dict(r) for r in rows]

    def incoming(self, to_citation: str) -> list[dict]:
        """References that point AT `to_citation` (i.e. who cites this chunk)."""
        rows = self.conn.execute(
            "SELECT * FROM citations WHERE to_citation = ?", (to_citation,)
        ).fetchall()
        return [dict(r) for r in rows]

    def external_for(self, from_citation: str) -> list[dict]:
        """External references (NCC, AS, other Acts) emitted by this chunk."""
        rows = self.conn.execute(
            "SELECT * FROM citations WHERE from_citation = ? AND external = 1",
            (from_citation,),
        ).fetchall()
        return [dict(r) for r in rows]

    def related_for_chunks(self, citations: Iterable[str]) -> dict[str, list[dict]]:
        """For a batch of chunk citations, return a mapping cit → outgoing refs.
        Used to build the 'Related provisions' block in answer prompts."""
        out: dict[str, list[dict]] = {}
        for cit in citations:
            out[cit] = self.outgoing(cit)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_build():
    sys.stdout.reconfigure(encoding="utf-8")
    chunk_paths = [
        "building_act_chunks.jsonl",
        "building_regs_chunks.jsonl",
        "ncc_chunks.jsonl",
        "housing_provisions_chunks.jsonl",
    ]
    stats = build_graph(chunk_paths, db_path=DB_PATH)
    print(f"\nWrote {stats['total_rows']:,} reference rows to {DB_PATH}")
    print(f"  resolved internal: {stats['resolved_internal']:,}")
    print(f"  external (BCA/NCC/AS/other-Acts): {stats['external']:,}")
    print(f"  by relationship_type:")
    for k, v in sorted(stats["by_type"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22}  {v:>5}")


def _cli_inspect(citation: str):
    sys.stdout.reconfigure(encoding="utf-8")
    g = CitationGraph()
    out = g.outgoing(citation)
    inc = g.incoming(citation)
    print(f"\nFROM `{citation}` ({len(out)} outgoing):")
    for r in out:
        target = r["to_citation"] or f"(unresolved: {r['to_anchor'] if 'to_anchor' in r else r['to_text']})"
        marker = " [EXTERNAL]" if r["external"] else ""
        print(f"  {r['relationship_type']:<22}  '{r['to_text']}' → {target}{marker}")
    print(f"\nTO `{citation}` ({len(inc)} incoming):")
    for r in inc:
        print(f"  cited by {r['from_citation']} ({r['relationship_type']})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        _cli_build()
    elif len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        _cli_inspect(" ".join(sys.argv[2:]))
    else:
        print("Usage:")
        print("  python -m lib.citation_graph build")
        print("  python -m lib.citation_graph inspect <citation>")
