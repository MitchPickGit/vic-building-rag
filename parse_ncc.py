"""
Parser for the National Construction Code 2022 Volume Two
(Class 1 and 10 buildings — the Australian residential code).

NCC text is licensed under Creative Commons Attribution 4.0
International. The ABCB explicitly permits redistribution, derivative
works, and AI training under that licence — see
https://ncc.abcb.gov.au/faq/copyright. Attribution and a no-endorsement
disclaimer are required wherever NCC content is served.

This parser SKIPS embedded references to Australian Standards (AS / AS-NZS)
and other third-party documents. The references themselves are captured
in the chunk's cross_references field, but no AS text is ingested — those
standards are separately copyrighted by Standards Australia.

Scope:
  - Section A (Governing requirements) — universal
  - Section H (Class 1 and 10 buildings) — residential code
  - Schedule 1 (Definitions) — universal
  - Schedule 10 (Victoria) — Vic-specific variations
  - Schedule 2 (Referenced documents) — captured as a single index chunk
    listing the references; AS bodies are NOT ingested.

Other state Schedules (3-9, 11-12) are skipped — outside Vic scope.

Output: ncc_chunks.jsonl — one chunk per Provision (A1G1, H6P2, S1C5),
plus chunks for Part / Specification headings.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

import pdfplumber

PDF_PATH = "docs/ncc2022-volume-two.pdf"
OUT_PATH = "ncc_chunks.jsonl"

VERSION = "NCC 2022 Volume Two (Amendment 1, 1 May 2023)"
VERSION_DATE = "2023-05-01"
EDITION = "NCC 2022 Vol 2"
LICENCE = "CC BY 4.0 (Australian Building Codes Board)"

# In-scope page ranges (PDF pages, inclusive). Each Section/Schedule has
# inner-TOC pages at its start (provision-id lists, no body prose) — we
# skip those by starting BODY ranges later. Detected empirically.
PAGE_RANGES = [
    ("Section A",   32, 75,   "Section A Governing requirements"),
    ("Section H",   81, 154,  "Section H Class 1 and 10 buildings"),
    ("Schedule 1",  155, 194, "Schedule 1 Definitions"),
    ("Schedule 2",  197, 212, "Schedule 2 Referenced documents"),
    ("Schedule 10", 284, 298, "Schedule 10 Victoria"),
]

# Font classification
BODY_MIN_SIZE = 9.5            # NCC body is 10pt
MARGIN_MAX_SIZE = 9.0          # 9pt is the running header / footer
BOLD_FONT_SUBSTR = "Bold"

PART_HEADING_SIZE = 13.5       # 14pt bold for Section + Part headings
SUBHEADING_SIZE = 11.5         # 12pt bold for Subpart and Provision headings

# Regex for the various heading types.
RE_SECTION   = re.compile(r"^\s*Section\s+([A-Z])\s+(.+?)\s*$")
RE_PART      = re.compile(r"^\s*Part\s+([A-Z]\d+)\s+(.+?)\s*$")
RE_SCHEDULE  = re.compile(r"^\s*Schedule\s+(\d+)\s+(.+?)\s*$")
RE_SPEC      = re.compile(r"^\s*Specification\s+(\d+)\s+(.+?)\s*$")

# Provision identifiers like "A1G1", "H6P2", "S1C5" — Section/Part letter+digit,
# subpart letter, then a digit. Followed by the provision title.
# Schedule 10 (Vic) uses "VIC " prefix on its overrides: "VIC H1D10 Flood hazard areas".
RE_PROVISION = re.compile(
    r"^\s*(?:(VIC)\s+)?([A-Z]\d+[A-Z]\d+(?:[a-z])?)\s+(.+?)\s*$"
)
# Subprovision: "(1)" "(a)" — same as Vic legislation
RE_SUBPROV   = re.compile(r"^\s*\((\d+[A-Z]*)\)\s+(.*)$")

# Cross-version annotation, e.g. "[2019: A2.0, A2.1]"
RE_VERSION_TAG = re.compile(r"\[(\d{4}):\s*[^\]]+\]")

# AS/NZS references — capture but don't ingest. Same patterns the
# citation-graph module uses; duplicated here so the parser is self-
# contained.
RE_AS_REF = re.compile(
    r"\b(AS(?:/NZS)?\s+\d{3,4}(?:\.\d+)?(?:[-–]\d{4})?)\b",
    re.IGNORECASE,
)

# Other Acts (Vic-specific overrides reference state Acts)
RE_OTHER_ACT = re.compile(
    r"\b((?:[A-Z][A-Za-z'’\-]*\s+){1,8}Act\s+\d{4})\b"
)

# Running headers/footers to drop. NCC pages have section banner at top
# (e.g. "Governing requirements", "Class 1 and 10 buildings") and a footer
# with version stamp.
RUNNING_HEADER_PATTERNS = [
    re.compile(r"^\s*Governing requirements\s*$"),
    re.compile(r"^\s*Class 1 and 10 buildings\s*$"),
    re.compile(r"^\s*Schedule 1\s*$"),
    re.compile(r"^\s*Schedule 2\s*$"),
    re.compile(r"^\s*Schedule 10\s*$"),
    re.compile(r"^\s*NCC 2022 Volume Two.*Page\s+\d+\s*$"),
    re.compile(r"^\s*\d+\s*$"),  # bare page numbers
    re.compile(r"^\s*[A-Z]\d+[A-Z]\d+\s*$"),  # provision-id reprint at top of continuation page
]


def chars_to_lines(chars, y_tolerance=2.5):
    lines = defaultdict(list)
    for ch in chars:
        key = round(ch["top"] / y_tolerance) * y_tolerance
        lines[key].append(ch)
    out = []
    for y in sorted(lines.keys()):
        line_chars = sorted(lines[y], key=lambda c: c["x0"])
        text = ""
        prev_x1 = None
        for c in line_chars:
            if prev_x1 is not None and c["x0"] - prev_x1 > 2:
                text += " "
            text += c["text"]
            prev_x1 = c["x1"]
        text = text.rstrip()
        if not text:
            continue
        sizes = [round(c["size"], 1) for c in line_chars]
        fonts = [c["fontname"] for c in line_chars]
        dominant_size = max(set(sizes), key=sizes.count)
        dominant_font = max(set(fonts), key=fonts.count)
        is_bold = BOLD_FONT_SUBSTR in dominant_font
        out.append({
            "y": y,
            "x0": line_chars[0]["x0"],
            "text": text,
            "size": dominant_size,
            "font": dominant_font,
            "is_bold": is_bold,
        })
    return out


def is_running_header(text: str) -> bool:
    if not text.strip():
        return True
    for pat in RUNNING_HEADER_PATTERNS:
        if pat.match(text):
            return True
    return False


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_xrefs(text: str) -> list[str]:
    """Capture refs to AS/NZS standards and other Acts, plus internal
    NCC provision citations like "A1G1" or "Part H6"."""
    out = set()
    for m in RE_AS_REF.finditer(text):
        out.add(m.group(0).strip())
    for m in RE_OTHER_ACT.finditer(text):
        title = m.group(1).strip()
        # Skip the host references and noise
        if title.startswith(("the ", "this ", "an ", "a ")):
            continue
        out.add(title)
    # Internal NCC refs — provision IDs and Parts/Specs
    for m in re.finditer(r"\b([A-Z]\d+[A-Z]\d+)\b", text):
        out.add(m.group(0))
    for m in re.finditer(r"\bPart\s+([A-Z]\d+)\b", text):
        out.add(m.group(0))
    for m in re.finditer(r"\bSpecification\s+(\d+)\b", text):
        out.add(m.group(0))
    return sorted(out)


def parse():
    chunks: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        # State carried across pages within a section
        current_section: Optional[str] = None    # e.g. "Section A Governing requirements"
        current_part: Optional[str] = None       # e.g. "Part A1 Interpreting the NCC"
        current_subpart: Optional[str] = None    # e.g. "Governing Requirements", "Performance Requirements"
        current_schedule: Optional[str] = None   # e.g. "Schedule 1 Definitions" or "Schedule 10 Victoria"
        current_spec: Optional[str] = None       # e.g. "Specification 1 Fire-resistance ..."

        # Provision being accumulated
        current_prov_id: Optional[str] = None
        current_prov_title: Optional[str] = None
        current_subprov: Optional[str] = None    # "(1)" if a subprovision is in progress

        buffer_text: list[str] = []
        buffer_pages: list[int] = []

        def flush():
            nonlocal buffer_text, buffer_pages
            if not buffer_text or current_prov_id is None:
                buffer_text, buffer_pages = [], []
                return
            text = normalize_text(" ".join(buffer_text))
            if not text:
                buffer_text, buffer_pages = [], []
                return

            # Strip any [YYYY: ref] cross-version tags from the body but
            # keep them in a separate field (useful for builders).
            version_tags = RE_VERSION_TAG.findall(text)
            text = RE_VERSION_TAG.sub("", text).strip()
            text = re.sub(r"\s+", " ", text)

            # Build citation
            doc_label = current_schedule or current_section or "NCC 2022 Volume Two"
            citation = f"NCC 2022 Vol 2 {current_prov_id}"
            if current_subprov:
                citation += current_subprov

            chunks.append({
                "doc_type": "ncc",
                "authority_level": 3,    # NCC adopted by Vic Building Regs (Act=1, Regs=2, NCC=3)
                "ncc_volume": "Volume Two",
                "version": VERSION,
                "version_date": VERSION_DATE,
                "edition": EDITION,
                "licence": LICENCE,
                "section": current_section,
                "part": current_part,
                "subpart": current_subpart,
                "schedule": current_schedule,
                "specification": current_spec,
                "section_number": current_prov_id,
                "section_title": current_prov_title,
                "subsection": current_subprov,
                "citation": citation,
                "text": text,
                "previous_edition_refs": version_tags or None,
                "penalty": None,            # NCC doesn't impose penalties; the Vic Building Act/Regs do
                "amendment_history": None,
                "cross_references": extract_xrefs(text),
                "pages": sorted(set(buffer_pages)),
            })
            buffer_text, buffer_pages = [], []

        for label, start, end, header_text in PAGE_RANGES:
            # Seed the Section/Schedule context for this range, since the
            # heading line lives on the inner-TOC page that we skip.
            flush()
            if label.startswith("Section"):
                current_section = header_text
                current_schedule = None
            else:
                current_schedule = header_text
                current_section = None
            current_part = None
            current_subpart = None
            current_spec = None
            current_prov_id = None
            current_prov_title = None
            current_subprov = None

            for pdf_p in range(start, end + 1):
                if pdf_p > len(pdf.pages):
                    continue
                page = pdf.pages[pdf_p - 1]

                body_chars = [c for c in page.chars if c["size"] >= BODY_MIN_SIZE]
                lines = chars_to_lines(body_chars)
                lines = [ln for ln in lines if not is_running_header(ln["text"])]

                for ln in lines:
                    text = ln["text"].strip()
                    if not text:
                        continue

                    # --- Heading detection (highest precedence first) ---
                    if ln["is_bold"]:
                        m_section = RE_SECTION.match(text)
                        m_part    = RE_PART.match(text)
                        m_sched   = RE_SCHEDULE.match(text)
                        m_spec    = RE_SPEC.match(text)
                        m_prov    = RE_PROVISION.match(text)

                        if m_section and ln["size"] >= PART_HEADING_SIZE:
                            flush()
                            current_section = text
                            current_part = None
                            current_subpart = None
                            current_spec = None
                            current_prov_id = None
                            current_prov_title = None
                            current_subprov = None
                            continue
                        if m_sched and ln["size"] >= PART_HEADING_SIZE:
                            flush()
                            current_schedule = text
                            current_section = None
                            current_part = None
                            current_subpart = None
                            current_spec = None
                            current_prov_id = None
                            current_prov_title = None
                            current_subprov = None
                            continue
                        if m_part and ln["size"] >= SUBHEADING_SIZE:
                            flush()
                            current_part = text
                            current_subpart = None
                            current_spec = None
                            current_prov_id = None
                            current_prov_title = None
                            current_subprov = None
                            continue
                        if m_spec and ln["size"] >= SUBHEADING_SIZE:
                            flush()
                            current_spec = text
                            current_part = None
                            current_subpart = None
                            current_prov_id = None
                            current_prov_title = None
                            current_subprov = None
                            continue
                        # Subpart heading — known names: "Governing Requirements",
                        # "Performance Requirements", "Deemed-to-Satisfy
                        # Provisions". Bold, size 10-12, no leading digit.
                        if (text in {"Governing Requirements",
                                     "Performance Requirements",
                                     "Deemed-to-Satisfy Provisions",
                                     "Introduction"}
                                or text.startswith("Introduction to ")):
                            flush()
                            current_subpart = text
                            current_prov_id = None
                            current_prov_title = None
                            current_subprov = None
                            continue
                        # Provision heading: "A1G1 Scope of NCC Volume One"
                        # or VIC override: "VIC H1D10 Flood hazard areas"
                        if m_prov:
                            flush()
                            vic_prefix = m_prov.group(1)  # "VIC" or None
                            base_id = m_prov.group(2)
                            current_prov_id = (
                                f"VIC {base_id}" if vic_prefix else base_id
                            )
                            current_prov_title = m_prov.group(3).strip()
                            current_subprov = None
                            continue
                        # Bold-but-not-heading — fall through as body text

                    # --- Subprovision marker ---
                    m_sub = RE_SUBPROV.match(text)
                    if m_sub:
                        flush()
                        current_subprov = f"({m_sub.group(1)})"
                        buffer_text = [m_sub.group(2)]
                        buffer_pages = [pdf_p]
                        continue

                    # --- Body line ---
                    buffer_text.append(text)
                    buffer_pages.append(pdf_p)

            # Flush at end of each PAGE_RANGES block
            flush()

    return chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = parse()
    raw_count = len(chunks)
    # Drop parent provisions with empty body (the body lives in their
    # subprovisions, e.g. H6P1's content is in H6P1(1), H6P1(2), H6P1(3)).
    chunks = [c for c in chunks if c.get("text", "").strip()]
    print(f"Produced {len(chunks)} chunks (filtered {raw_count - len(chunks)} empty parent provisions)\n")

    # Stats
    by_section = Counter(
        (c["section"] or "(no section)") for c in chunks
    )
    by_schedule = Counter(
        c["schedule"] for c in chunks if c.get("schedule")
    )
    by_part = Counter(
        c["part"] for c in chunks if c.get("part")
    )
    print(f"By section ({len(by_section)} groups):")
    for k, v in by_section.most_common():
        print(f"  {v:>4}  {k}")
    print(f"\nBy schedule ({len(by_schedule)} groups):")
    for k, v in by_schedule.most_common():
        print(f"  {v:>4}  {k}")
    print(f"\nBy part (top 15):")
    for k, v in by_part.most_common(15):
        print(f"  {v:>4}  {k}")
    print(f"\n  Chunks with cross-references:  "
          f"{sum(1 for c in chunks if c['cross_references'])}")
    print(f"  Chunks referencing AS/AS-NZS:  "
          f"{sum(1 for c in chunks if any('AS' in r for r in (c['cross_references'] or [])))}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
