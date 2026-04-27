"""
Parser for Building Act 1993 (Vic) — Authorised Version 146.

Extracts the full body (Parts 1 through 15, including 11A and 12A) from the
authorised consolidated PDF. Schedules are not yet parsed (the Act has 11
schedules, mostly transitional/historical; can be added in a follow-up).

This was originally a Parts 3-5 prototype; it now mirrors the Regulations
parser's structure (sequential page walk + heading detection) so we don't
need to maintain a hand-curated PART_RANGES map.

Output: building_act_chunks.jsonl — one chunk per subsection (or per
section if no subsections).
"""

import json
import re
import sys
from collections import Counter, defaultdict

import pdfplumber

PDF_PATH = "docs/93-126aa146-authorised.pdf"
OUT_PATH = "building_act_chunks.jsonl"

# Body content (Parts 1..15 incl. 11A, 12A) runs printed pages 1..685.
# Printed page 686 is Schedule 1 onward (skipped for now).
BODY_PRINTED_RANGE = (1, 685)
PRINTED_TO_PDF_OFFSET = 30  # PDF page = printed page + 30

VERSION = "146"
VERSION_DATE = "2026-04-01"

# Font classification (same regime as Regs body pages)
BODY_MIN_SIZE = 11.5
MARGIN_MAX_SIZE = 9.0
BOLD_FONT_SUBSTR = "Bold"

PART_HEADING_SIZE = 15.0       # 16pt bold
DIVISION_HEADING_SIZE = 13.0   # 14pt bold
SECTION_HEADING_SIZE = 11.5    # 12pt bold

RE_PART       = re.compile(r"^\s*Part\s+(\d+[A-Z]*)\s*[—–-]\s*(.+?)\s*$")
RE_DIVISION   = re.compile(r"^\s*Division\s+(\d+[A-Z]*)\s*[—–-]\s*(.+?)\s*$")
RE_SUBDIV     = re.compile(r"^\s*Subdivision\s+(\d+[A-Z]*)\s*[—–-]\s*(.+?)\s*$")
RE_SECTION    = re.compile(r"^\s*(\d+[A-Z]*)\s+(.+?)\s*$")
RE_SUBSECTION = re.compile(r"^\s*\((\d+[A-Z]*)\)\s+(.*)$")

RE_PENALTY = re.compile(
    r"Penalty\s*:\s*(.+?)(?=(?:\s*\(\d+[A-Z]*\)\s)|$)", re.DOTALL
)

RE_XREF = re.compile(
    r"(?:sections?|s\.)\s+(\d+[A-Z]*(?:\(\d+[A-Z]*\))?(?:\([a-z]+\))?)"
    r"|Part\s+(\d+[A-Z]*)"
    r"|Division\s+(\d+[A-Z]*)"
    r"|Schedule\s+(\d+)"
    r"|clause\s+(\d+)",
    re.IGNORECASE,
)

# Running headers and footers stripped during parsing. Body lines are
# already filtered by size >= 11.5, so the size-10 page-top headers
# ("Building Act 1993" / "s 16") are removed automatically. The patterns
# below catch any text that slips through at body size.
HEADER_PATTERNS = [
    re.compile(r"^\s*Authorised by the Chief Parliamentary Counsel\s*$"),
    re.compile(r"^\s*Building Act 1993\s*$"),
    re.compile(r"^\s*No\.\s*126\s*of\s*1993\s*$"),
    re.compile(r"^\s*Authorised Version(\s+No\.\s*\d+)?\s*$"),
    re.compile(r"^\s*Authorised Version incorporating amendments as at\s*$"),
    re.compile(r"^\s*\d+\s+\w+\s+\d{4}\s*$"),  # e.g. "1 April 2026"
    re.compile(r"^\s*\d+\s*$"),                # bare page numbers
]

# PUA glyph translation. The authorised PDFs use Symbol-font glyphs in the
# Unicode private use area for a handful of characters — without this, "3·6 m"
# becomes "36 m" after whitespace normalisation. Same table as the Regs parser.
PUA_TRANSLATION = str.maketrans({
    "\uF0D7": "\u00B7",   # middle dot   ·
    "\uF0B4": "\u00D7",   # times sign   ×
    "\uF06F": "\u2610",   # ballot box   ☐  (form checkbox)
    "\uF0A8": "\u2610",   # ballot box   ☐  (form checkbox)
})


def normalize_text(text):
    text = text.translate(PUA_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chars_to_lines(chars, y_tolerance=3):
    """Group chars into lines by y-coordinate, top-to-bottom reading order."""
    lines = defaultdict(list)
    for ch in chars:
        key = round(ch["top"] / y_tolerance) * y_tolerance
        lines[key].append(ch)

    result = []
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

        result.append({
            "y": y,
            "x0": line_chars[0]["x0"],
            "text": text,
            "size": dominant_size,
            "font": dominant_font,
            "is_bold": is_bold,
        })
    return result


def is_header_line(text):
    if not text.strip():
        return True
    for pat in HEADER_PATTERNS:
        if pat.match(text):
            return True
    return False


def consolidate_margin_blocks(margin_lines, y_gap_threshold=20):
    if not margin_lines:
        return []
    blocks = []
    current_start = margin_lines[0]["y"]
    current_text = [margin_lines[0]["text"]]
    last_y = current_start
    for line in margin_lines[1:]:
        y, txt = line["y"], line["text"]
        if y - last_y > y_gap_threshold:
            blocks.append((current_start, last_y, " ".join(current_text).strip()))
            current_start = y
            current_text = [txt]
        else:
            current_text.append(txt)
        last_y = y
    blocks.append((current_start, last_y, " ".join(current_text).strip()))
    return blocks


def extract_xrefs(text):
    refs = set()
    for m in RE_XREF.finditer(text):
        matched = m.group(0).strip().rstrip(",.;:")
        matched = re.sub(r"^s\.\s*", "section ", matched, flags=re.IGNORECASE)
        refs.add(matched)
    return sorted(refs)


def merge_multiline_part_heading(body_lines):
    """Collapse consecutive bold lines at Part-heading size into one line."""
    merged = []
    i = 0
    while i < len(body_lines):
        ln = body_lines[i]
        if ln["is_bold"] and ln["size"] >= PART_HEADING_SIZE:
            combined = [ln["text"]]
            j = i + 1
            last_y = ln["y"]
            while j < len(body_lines):
                nxt = body_lines[j]
                if (nxt["is_bold"]
                        and nxt["size"] >= PART_HEADING_SIZE
                        and nxt["y"] - last_y < 25
                        and not RE_PART.match(nxt["text"])):
                    combined.append(nxt["text"])
                    last_y = nxt["y"]
                    j += 1
                else:
                    break
            merged_line = dict(ln)
            merged_line["text"] = " ".join(combined)
            merged.append(merged_line)
            i = j
        else:
            merged.append(ln)
            i += 1
    return merged


def merge_multiline_division_heading(body_lines):
    """Division headings often wrap to a second bold line (e.g.
    'Division 3A—Certificates of consent for' + 'owner-builders'). Without
    merging, the division metadata is truncated and downstream retrieval
    misses queries that depend on the full division name."""
    merged = []
    i = 0
    while i < len(body_lines):
        ln = body_lines[i]
        m_div = RE_DIVISION.match(ln["text"])
        is_div_heading = (
            ln["is_bold"]
            and DIVISION_HEADING_SIZE <= ln["size"] < PART_HEADING_SIZE
            and m_div is not None
        )
        if is_div_heading:
            combined = [ln["text"]]
            j = i + 1
            last_y = ln["y"]
            while j < len(body_lines):
                nxt = body_lines[j]
                if not (nxt["is_bold"]
                        and DIVISION_HEADING_SIZE <= nxt["size"] < PART_HEADING_SIZE
                        and nxt["y"] - last_y < 25
                        and not RE_PART.match(nxt["text"])
                        and not RE_DIVISION.match(nxt["text"])
                        and not RE_SUBDIV.match(nxt["text"])
                        and not RE_SECTION.match(nxt["text"])
                        and not RE_SUBSECTION.match(nxt["text"])):
                    break
                combined.append(nxt["text"])
                last_y = nxt["y"]
                j += 1
            if len(combined) > 1:
                merged_line = dict(ln)
                merged_line["text"] = " ".join(combined)
                merged.append(merged_line)
                i = j
                continue
        merged.append(ln)
        i += 1
    return merged


def merge_multiline_section_heading(body_lines):
    """Section titles sometimes wrap to a second bold line.
    Without this, the wrapped continuation leaks into the section's body."""
    merged = []
    i = 0
    while i < len(body_lines):
        ln = body_lines[i]
        m_sec = RE_SECTION.match(ln["text"])
        is_section_heading = (
            ln["is_bold"]
            and ln["size"] >= SECTION_HEADING_SIZE
            and ln["size"] < DIVISION_HEADING_SIZE
            and ln["x0"] < 160
            and m_sec is not None
        )
        if is_section_heading:
            combined = [ln["text"]]
            j = i + 1
            last_y = ln["y"]
            while j < len(body_lines):
                nxt = body_lines[j]
                if not (nxt["is_bold"]
                        and SECTION_HEADING_SIZE <= nxt["size"] < DIVISION_HEADING_SIZE
                        and nxt["y"] - last_y < 20
                        and not RE_PART.match(nxt["text"])
                        and not RE_DIVISION.match(nxt["text"])
                        and not RE_SUBDIV.match(nxt["text"])
                        and not RE_SECTION.match(nxt["text"])
                        and not RE_SUBSECTION.match(nxt["text"])):
                    break
                combined.append(nxt["text"])
                last_y = nxt["y"]
                j += 1
            if len(combined) > 1:
                merged_line = dict(ln)
                merged_line["text"] = " ".join(combined)
                merged.append(merged_line)
                i = j
                continue
        merged.append(ln)
        i += 1
    return merged


def parse_act():
    chunks = []

    with pdfplumber.open(PDF_PATH) as pdf:
        current_part = None
        current_division = None
        current_subdivision = None
        current_section_num = None
        current_section_title = None
        current_subsection = None

        buffer_text = []
        buffer_pages = []
        buffer_y_ranges = []
        page_margin_blocks = {}

        def flush():
            nonlocal buffer_text, buffer_pages, buffer_y_ranges
            if not buffer_text or current_section_num is None:
                buffer_text, buffer_pages, buffer_y_ranges = [], [], []
                return

            text = normalize_text(" ".join(buffer_text))
            if not text:
                buffer_text, buffer_pages, buffer_y_ranges = [], [], []
                return

            penalty = None
            pen_match = RE_PENALTY.search(text)
            if pen_match:
                penalty = pen_match.group(1).strip().rstrip(".")
                text = (text[: pen_match.start()] + text[pen_match.end():]).strip()

            amendments = []
            for (pg, ys, ye) in buffer_y_ranges:
                for b_start, b_end, b_text in page_margin_blocks.get(pg, []):
                    if not (b_end < ys - 5 or b_start > ye + 5):
                        if b_text not in amendments:
                            amendments.append(b_text)

            citation = f"Building Act 1993 (Vic) s {current_section_num}"
            if current_subsection:
                citation += current_subsection

            chunks.append({
                "doc_type": "act",
                "authority_level": 1,
                "act": "Building Act 1993 (Vic)",
                "version": VERSION,
                "version_date": VERSION_DATE,
                "part": current_part,
                "division": current_division,
                "subdivision": current_subdivision,
                "section_number": current_section_num,
                "section_title": normalize_text(current_section_title) if current_section_title else None,
                "subsection": current_subsection,
                "citation": citation,
                "text": text,
                "penalty": normalize_text(penalty) if penalty else None,
                "amendment_history": normalize_text(" | ".join(amendments)) if amendments else None,
                "cross_references": extract_xrefs(text),
                "pages": sorted(set(buffer_pages)),
            })
            buffer_text, buffer_pages, buffer_y_ranges = [], [], []

        start_printed, end_printed = BODY_PRINTED_RANGE
        for printed_p in range(start_printed, end_printed + 1):
            pdf_p = printed_p + PRINTED_TO_PDF_OFFSET
            if pdf_p > len(pdf.pages):
                continue
            page = pdf.pages[pdf_p - 1]

            body_chars = [c for c in page.chars if c["size"] >= BODY_MIN_SIZE]
            margin_chars = [c for c in page.chars if c["size"] <= MARGIN_MAX_SIZE]

            body_lines = chars_to_lines(body_chars)
            margin_lines = chars_to_lines(margin_chars)

            body_lines = [ln for ln in body_lines if not is_header_line(ln["text"])]
            body_lines = merge_multiline_part_heading(body_lines)
            body_lines = merge_multiline_division_heading(body_lines)
            body_lines = merge_multiline_section_heading(body_lines)

            margin_lines = [
                ln for ln in margin_lines
                if ln["size"] >= 7.5
                and not is_header_line(ln["text"])
            ]
            page_margin_blocks[pdf_p] = consolidate_margin_blocks(margin_lines)

            for ln in body_lines:
                text = ln["text"].strip()
                if not text:
                    continue

                if ln["is_bold"]:
                    m_part   = RE_PART.match(text)
                    m_div    = RE_DIVISION.match(text)
                    m_subdiv = RE_SUBDIV.match(text)
                    m_sec    = RE_SECTION.match(text)

                    if m_part and ln["size"] >= PART_HEADING_SIZE:
                        flush()
                        current_part = text
                        current_division = None
                        current_subdivision = None
                        current_section_num = None
                        current_section_title = None
                        current_subsection = None
                        continue
                    if m_div and ln["size"] >= DIVISION_HEADING_SIZE:
                        flush()
                        current_division = text
                        current_subdivision = None
                        continue
                    if m_subdiv:
                        flush()
                        current_subdivision = text
                        continue
                    if m_sec and ln["size"] >= SECTION_HEADING_SIZE and ln["x0"] < 160:
                        flush()
                        current_section_num = m_sec.group(1)
                        current_section_title = m_sec.group(2).strip()
                        current_subsection = None
                        continue
                    # Bold but not a heading → fall through as body text

                m_sub = RE_SUBSECTION.match(text)
                if m_sub:
                    flush()
                    current_subsection = f"({m_sub.group(1)})"
                    buffer_text = [m_sub.group(2)]
                    buffer_pages = [pdf_p]
                    buffer_y_ranges = [(pdf_p, ln["y"], ln["y"])]
                    continue

                buffer_text.append(text)
                buffer_pages.append(pdf_p)
                if buffer_y_ranges and buffer_y_ranges[-1][0] == pdf_p:
                    pg, ys, ye = buffer_y_ranges[-1]
                    buffer_y_ranges[-1] = (pg, ys, ln["y"])
                else:
                    buffer_y_ranges.append((pdf_p, ln["y"], ln["y"]))

        flush()

    return chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    chunks = parse_act()
    print(f"Produced {len(chunks)} chunks\n")

    by_part = Counter(c["part"] for c in chunks)
    for part, n in sorted(by_part.items(), key=lambda kv: kv[0] or ""):
        secs = sorted(
            set(c["section_number"] for c in chunks if c["part"] == part),
            key=lambda s: (int(re.match(r'\d+', s).group()), s),
        )
        print(f"  {part}: {n} chunks, {len(secs)} sections")
        head = secs[:25]
        print(f"    → {head}{' ...' if len(secs) > 25 else ''}")

    print(f"\n  Chunks with penalties:   {sum(1 for c in chunks if c['penalty'])}")
    print(f"  Chunks with amendments:  {sum(1 for c in chunks if c['amendment_history'])}")
    print(f"  Chunks with x-refs:      {sum(1 for c in chunks if c['cross_references'])}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
