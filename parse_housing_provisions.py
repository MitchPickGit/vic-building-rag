"""
Parser for the ABCB Housing Provisions Standard 2022 (Class 1 and 10).

The Housing Provisions sit one layer deeper than NCC Volume Two: NCC
deems a building compliant if it satisfies the Housing Provisions.
This is where the actual technical numbers live — riser heights, span
tables, R-values per climate zone, BAL distances, footing
classifications, etc.

Same licensing as NCC Vol 2: Creative Commons Attribution 4.0
International, © Commonwealth of Australia and the States and
Territories of Australia 2022, published by the Australian Building
Codes Board.

Structure:
  Section X (e.g. 2 Structure, 4 Footings and slabs, 11 Safe movement)
    Part X.Y (e.g. Part 4.2 Footings, slabs and associated elements)
      Provision X.Y.Z (e.g. 4.2.2 Site classification)
        Subprovisions (1), (2)
        Paragraphs (a), (b)

This parser ingests Sections 2-13 (body), and Schedule 10 (Victoria
overrides). Schedule 1 (Definitions) and Schedule 2 (Referenced
documents) are deferred — same column-extraction issue as NCC Vol 2's
definitions schedule.

Output: housing_provisions_chunks.jsonl — one chunk per provision /
subprovision.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

import pdfplumber

PDF_PATH = "docs/ncc2022-abcb-housing-provisions.pdf"
OUT_PATH = "housing_provisions_chunks.jsonl"

VERSION = "ABCB Housing Provisions Standard 2022 (1 May 2023)"
VERSION_DATE = "2023-05-01"
EDITION = "ABCB Housing Provisions 2022"
LICENCE = "CC BY 4.0 (Australian Building Codes Board)"

# In-scope page ranges (PDF pages, inclusive). Section body-starts
# detected empirically by scanning for 'X.1.1' provisions.
PAGE_RANGES = [
    ("Section 2",  20, 26,  "Section 2 Structure"),
    ("Section 3",  27, 41,  "Section 3 Site preparation"),
    ("Section 4",  42, 72,  "Section 4 Footings and slabs"),
    ("Section 5",  73, 106, "Section 5 Masonry"),
    ("Section 6", 107, 144, "Section 6 Framing"),
    ("Section 7", 145, 186, "Section 7 Roof and wall cladding"),
    ("Section 8", 187, 199, "Section 8 Glazing"),
    ("Section 9", 200, 239, "Section 9 Fire safety"),
    ("Section 10", 240, 287, "Section 10 Health and amenity"),
    ("Section 11", 288, 311, "Section 11 Safe movement and access"),
    ("Section 12", 312, 328, "Section 12 Ancillary provisions"),
    ("Section 13", 329, 451, "Section 13 Energy efficiency"),
    ("Schedule 10", 553, 575, "Schedule 10 Victoria"),
]

BODY_MIN_SIZE = 9.5
MARGIN_MAX_SIZE = 9.0
BOLD_FONT_SUBSTR = "Bold"

PART_HEADING_SIZE = 13.5     # 14pt bold for Part X.Y headings
PROVISION_HEADING_SIZE = 11.5  # 12pt bold for X.Y.Z provision headings

# Part heading: "Part 4.2 Footings, slabs and associated elements"
RE_PART = re.compile(
    r"^\s*(?:(VIC)\s+)?Part\s+(\d+\.\d+[A-Z]?)\s*(.+?)\s*$"
)
# Provision: "4.2.2 Site classification" or "4.2.2  Site classification" (one+ spaces).
# Schedule 10 Vic uses "VIC X.Y.Z Title".
RE_PROVISION = re.compile(
    r"^\s*(?:(VIC)\s+)?(\d+\.\d+\.\d+[A-Za-z]?)\s+(.+?)\s*$"
)
# Section heading: "9 Fire safety" — single digit + name, size 14 bold
RE_SECTION = re.compile(r"^\s*(\d{1,2})\s+([A-Z][a-zA-Z ,]+)\s*$")
# Schedule
RE_SCHEDULE = re.compile(r"^\s*Schedule\s+(\d+)\s+(.+?)\s*$")
# Subprovision (1), (a), (i)
RE_SUBPROV = re.compile(r"^\s*\((\d+[A-Z]*)\)\s*(.*)$")
# Cross-version tag like "[2019: 3.10.3.0]"
RE_VERSION_TAG = re.compile(r"\[(\d{4}):\s*[^\]]+\]|\[New for \d{4}\]")
# AS / NZ standards
RE_AS_REF = re.compile(
    r"\b(AS(?:/NZS)?\s+\d{3,4}(?:\.\d+)?(?:[-–]\d{4})?)\b",
    re.IGNORECASE,
)

# Running headers/footers — keep tight to avoid false positives.
RUNNING_HEADER_PATTERNS = [
    re.compile(r"^\s*ABCB Housing Provisions Standard.*Page\s+\d+\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    # A bare provision id "4.2.2" or "11.2.5" appearing as a continuation
    # header at the top of a page
    re.compile(r"^\s*\d+\.\d+\.\d+[A-Za-z]?\s*$"),
    # Section running banners — derived from the section names. Adding the
    # full set is verbose but reduces noise materially.
    re.compile(r"^\s*Front matter\s*$"),
    re.compile(r"^\s*Structure\s*$"),
    re.compile(r"^\s*Site preparation\s*$"),
    re.compile(r"^\s*Footings and slabs\s*$"),
    re.compile(r"^\s*Masonry\s*$"),
    re.compile(r"^\s*Framing\s*$"),
    re.compile(r"^\s*Roof and wall cladding\s*$"),
    re.compile(r"^\s*Glazing\s*$"),
    re.compile(r"^\s*Fire safety\s*$"),
    re.compile(r"^\s*Health and amenity\s*$"),
    re.compile(r"^\s*Safe movement and access\s*$"),
    re.compile(r"^\s*Ancillary provisions\s*$"),
    re.compile(r"^\s*Energy efficiency\s*$"),
    re.compile(r"^\s*Victoria\s*$"),
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
    return re.sub(r"\s+", " ", text).strip()


def extract_xrefs(text: str) -> list[str]:
    out = set()
    for m in RE_AS_REF.finditer(text):
        out.add(m.group(0).strip())
    # Internal HP refs e.g. "Part 4.2", "5.6.8", "Section 9"
    for m in re.finditer(r"\bPart\s+(\d+\.\d+)\b", text):
        out.add(m.group(0))
    for m in re.finditer(r"\b(\d+\.\d+\.\d+[A-Za-z]?)\b", text):
        out.add(m.group(0))
    for m in re.finditer(r"\bSection\s+(\d+)\b", text):
        out.add(m.group(0))
    # NCC back-references: "H1D4(2)" etc. — Housing Provisions points back
    # to NCC sometimes
    for m in re.finditer(r"\b([A-Z]\d+[A-Z]\d+(?:\(\d+\))?)\b", text):
        out.add(m.group(0))
    return sorted(out)


def parse():
    chunks: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        current_section: Optional[str] = None
        current_part: Optional[str] = None
        current_schedule: Optional[str] = None
        current_prov_id: Optional[str] = None
        current_prov_title: Optional[str] = None
        current_subprov: Optional[str] = None
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
            version_tags = RE_VERSION_TAG.findall(text)
            text = RE_VERSION_TAG.sub("", text).strip()
            text = re.sub(r"\s+", " ", text)

            citation = f"ABCB HP {current_prov_id}"
            if current_subprov:
                citation += current_subprov

            chunks.append({
                "doc_type": "housing_provisions",
                "authority_level": 4,    # Act=1, Regs=2, NCC=3, ABCB HP=4
                "version": VERSION,
                "version_date": VERSION_DATE,
                "edition": EDITION,
                "licence": LICENCE,
                "section": current_section,
                "part": current_part,
                "schedule": current_schedule,
                "section_number": current_prov_id,
                "section_title": current_prov_title,
                "subsection": current_subprov,
                "citation": citation,
                "text": text,
                "previous_edition_refs": version_tags or None,
                "penalty": None,
                "amendment_history": None,
                "cross_references": extract_xrefs(text),
                "pages": sorted(set(buffer_pages)),
            })
            buffer_text, buffer_pages = [], []

        for label, start, end, header_text in PAGE_RANGES:
            flush()
            if label.startswith("Section"):
                current_section = header_text
                current_schedule = None
            else:
                current_schedule = header_text
                current_section = None
            current_part = None
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

                    if ln["is_bold"]:
                        m_part = RE_PART.match(text)
                        m_prov = RE_PROVISION.match(text)
                        m_section = RE_SECTION.match(text)

                        # Part heading: "Part 4.2 Footings..."
                        if m_part and ln["size"] >= PART_HEADING_SIZE:
                            flush()
                            vic_prefix = m_part.group(1)
                            part_id = m_part.group(2)
                            part_title = m_part.group(3).strip()
                            current_part = (
                                f"VIC Part {part_id} {part_title}"
                                if vic_prefix
                                else f"Part {part_id} {part_title}"
                            )
                            current_prov_id = None
                            current_prov_title = None
                            current_subprov = None
                            continue
                        # Section heading: "9 Fire safety" — only treat as section
                        # if size 14+ AND we're at the top of a section's first
                        # body page. Otherwise this regex would gobble too much.
                        if (m_section and ln["size"] >= PART_HEADING_SIZE
                                and ln["x0"] < 100):
                            flush()
                            current_section = f"Section {m_section.group(1)} {m_section.group(2).strip()}"
                            current_part = None
                            current_prov_id = None
                            continue
                        # Provision heading "4.2.2 Site classification"
                        if m_prov and ln["size"] >= PROVISION_HEADING_SIZE:
                            flush()
                            vic_prefix = m_prov.group(1)
                            base_id = m_prov.group(2)
                            current_prov_id = (
                                f"VIC {base_id}" if vic_prefix else base_id
                            )
                            current_prov_title = m_prov.group(3).strip()
                            current_subprov = None
                            continue
                        # Bold subheadings inside provisions ("Table Notes",
                        # "Figure Notes", "Explanatory Information") fall
                        # through as body text.

                    m_sub = RE_SUBPROV.match(text)
                    if m_sub:
                        flush()
                        current_subprov = f"({m_sub.group(1)})"
                        buffer_text = [m_sub.group(2) or ""]
                        buffer_pages = [pdf_p]
                        continue

                    buffer_text.append(text)
                    buffer_pages.append(pdf_p)

            flush()
    return chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = parse()
    raw = len(chunks)
    chunks = [c for c in chunks if c.get("text", "").strip()]
    print(f"Produced {len(chunks)} chunks (filtered {raw - len(chunks)} empty parents)\n")

    by_section = Counter(c["section"] for c in chunks if c.get("section"))
    by_schedule = Counter(c["schedule"] for c in chunks if c.get("schedule"))
    print("By section:")
    for k, v in by_section.most_common():
        print(f"  {v:>4}  {k}")
    print("\nBy schedule:")
    for k, v in by_schedule.most_common():
        print(f"  {v:>4}  {k}")

    print(f"\n  Chunks with cross-references: "
          f"{sum(1 for c in chunks if c['cross_references'])}")
    print(f"  Chunks referencing AS:        "
          f"{sum(1 for c in chunks if any('AS' in r for r in (c['cross_references'] or [])))}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
