"""
Parser for Building Regulations 2018 (Vic) — Authorised Version 028.

Adapted from parse_building_act.py. Main differences from the Act:

  1. "Regulation"/"Reg." instead of "Section"/"S." in citations and
     margin amendment notes.
  2. Running headers use "Building Regulations 2018" / "S.R. No. 38/2018"
     and a repeating "Part N—…" line at top of every page.
  3. Margin alternates side (right on odd pages, left on even pages).
     We use font-size separation only, so this is transparent.
  4. Schedule pages use a DIFFERENT size regime: body text is size 10
     (not 12 like the Parts). We detect the dominant body size per page
     and adapt the threshold.
  5. Schedule 3 is a 3-column exemption table. Each row (one exemption)
     becomes one chunk with structured fields in addition to body text.

Output: building_regs_chunks.jsonl — one chunk per subregulation
(plus one per Schedule 3 exemption item).
"""

import json
import re
import sys
from collections import Counter, defaultdict

import pdfplumber

PDF_PATH = "docs/18-38sra028-authorised.pdf"
OUT_PATH = "building_regs_chunks.jsonl"

# Printed page + PRINTED_TO_PDF_OFFSET == PDF page index (1-based)
PRINTED_TO_PDF_OFFSET = 16

# Body text (Parts 1..22, including 9A, 11A) runs printed pages 1..271.
# Printed p.272 onwards is Schedules (handled by separate sub-parser).
BODY_PRINTED_RANGE = (1, 271)

SCHEDULE_3_PRINTED = (275, 284)  # inclusive

# Font classification
BODY_MIN_SIZE_DEFAULT = 11.5   # body pages: text is 12
BODY_MIN_SIZE_SCHEDULE = 9.5   # schedule pages: text is 10
MARGIN_MAX_SIZE = 9.0          # for body pages; overridden on schedules
BOLD_FONT_SUBSTR = "Bold"

# Heading thresholds (use absolute sizes — the heading sizes don't change
# between body/schedule regimes).
PART_HEADING_SIZE = 15.0       # Part headings: 16pt bold
DIVISION_HEADING_SIZE = 13.0   # Division headings: 14pt bold
REG_HEADING_SIZE = 11.5        # Regulation heading: 12pt bold
SCHEDULE_HEADING_SIZE = 15.0   # Schedule heading: 16pt bold

# The source PDF uses Symbol-font Private Use Area glyphs for a handful of
# characters (middle dot in decimals, multiplication sign, form-field
# checkboxes). Map them back to their proper Unicode codepoints so chunk
# text renders correctly. Found by scanning the whole PDF for chars in
# U+E000-U+F8FF.
PUA_TRANSLATION = str.maketrans({
    "\uF0D7": "\u00B7",   # middle dot   ·  (used in "3·6 m")
    "\uF0B4": "\u00D7",   # times sign   ×
    "\uF06F": "\u2610",   # ballot box   ☐  (form checkbox)
    "\uF0A8": "\u2610",   # ballot box   ☐  (form checkbox)
})


def normalize_text(text):
    """Apply PUA translation and collapse whitespace."""
    text = text.translate(PUA_TRANSLATION)
    text = re.sub(r"\s+", " ", text).strip()
    return text


RE_PART       = re.compile(r"^\s*Part\s+(\d+[A-Z]*)\s*[—–-]\s*(.+?)\s*$")
RE_DIVISION   = re.compile(r"^\s*Division\s+(\d+[A-Z]*)\s*[—–-]\s*(.+?)\s*$")
RE_SUBDIV     = re.compile(r"^\s*Subdivision\s+(\d+[A-Z]*)\s*[—–-]\s*(.+?)\s*$")
RE_REG        = re.compile(r"^\s*(\d+[A-Z]*)\s+(.+?)\s*$")
RE_SUBREG     = re.compile(r"^\s*\((\d+[A-Z]*)\)\s+(.*)$")

# Regulations rarely carry penalties (they reference the Act for penalty units)
# but some do — preserve the Act's pattern just in case.
RE_PENALTY    = re.compile(r"Penalty\s*:\s*(.+?)(?=(?:\s*\(\d+[A-Z]*\)\s)|$)", re.DOTALL)

RE_XREF = re.compile(
    r"(?:regulations?|reg\.)\s+(\d+[A-Z]*(?:\(\d+[A-Z]*\))?(?:\([a-z]+\))?)"
    r"|(?:sections?|s\.)\s+(\d+[A-Z]*(?:\(\d+[A-Z]*\))?(?:\([a-z]+\))?)"
    r"|Part\s+(\d+[A-Z]*)"
    r"|Division\s+(\d+[A-Z]*)"
    r"|Schedule\s+(\d+)",
    re.IGNORECASE,
)

# Running headers and footers to strip from body.
HEADER_PATTERNS = [
    re.compile(r"^\s*Building Regulations 2018\s*$"),
    re.compile(r"^\s*S\.R\.\s*No\.\s*38/2018\s*$"),
    re.compile(r"^\s*Authorised Version(\s+No\.\s*\d+)?\s*$"),
    re.compile(r"^\s*Authorised Version incorporating amendments as at\s*$"),
    re.compile(r"^\s*\d+\s+\w+\s+\d{4}\s*$"),     # e.g. "26 November 2025"
    re.compile(r"^\s*Authorised by the Chief Parliamentary Counsel\s*$"),
    re.compile(r"^\s*\d+\s*$"),                    # bare page numbers
    # Running Part header — Part headings also appear at the top of every
    # body page (not just at the logical start). We drop it as header and
    # detect the real Part transition via size + page position.
    # Only treat as header if size < PART_HEADING_SIZE (running headers are
    # size 10; real heading is size 16). This is handled via size check in
    # the pipeline, not here — left as placeholder.
]


def printed_to_pdf(page_printed):
    return page_printed + PRINTED_TO_PDF_OFFSET


def pdf_to_printed(page_pdf):
    return page_pdf - PRINTED_TO_PDF_OFFSET


def chars_to_lines(chars, y_tolerance=3):
    """Group chars into lines by y-coordinate. Returns lines sorted top-down."""
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


def dominant_body_size(page):
    """Return 12.0 for Parts-style pages, 10.0 for Schedule-style pages."""
    sizes = [round(c["size"], 1) for c in page.chars if c["size"] >= 10]
    if not sizes:
        return 12.0
    return Counter(sizes).most_common(1)[0][0]


def consolidate_margin_blocks(margin_lines, y_gap_threshold=20):
    """Group nearby margin lines into blocks representing one amendment note."""
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
        refs.add(matched)
    return sorted(refs)


def merge_multiline_part_heading(body_lines):
    """Collapse consecutive bold lines at Part heading size into one."""
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


def merge_multiline_reg_heading(body_lines):
    """Regulation titles sometimes wrap to a second bold line (e.g. r 57:
    'Notice of imminent lapse of building permit—' + 'completion of work').
    Collapse those continuation lines into the heading so they don't bleed
    into body text."""
    merged = []
    i = 0
    while i < len(body_lines):
        ln = body_lines[i]
        m_reg = RE_REG.match(ln["text"])
        is_reg_heading = (
            ln["is_bold"]
            and ln["size"] >= REG_HEADING_SIZE
            and ln["size"] < DIVISION_HEADING_SIZE
            and ln["x0"] < 160
            and m_reg is not None
        )
        if is_reg_heading:
            combined = [ln["text"]]
            j = i + 1
            last_y = ln["y"]
            while j < len(body_lines):
                nxt = body_lines[j]
                # Continuation line must be: bold, same size band,
                # vertically close, NOT itself a new heading (no leading
                # digit, no Part/Division/Subdivision), and not a subreg
                # marker "(1)".
                if not (nxt["is_bold"]
                        and REG_HEADING_SIZE <= nxt["size"] < DIVISION_HEADING_SIZE
                        and nxt["y"] - last_y < 20
                        and not RE_PART.match(nxt["text"])
                        and not RE_DIVISION.match(nxt["text"])
                        and not RE_SUBDIV.match(nxt["text"])
                        and not RE_REG.match(nxt["text"])
                        and not RE_SUBREG.match(nxt["text"])):
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


# ---------------------------------------------------------------------------
# Parts parser (adapted Act parser)
# ---------------------------------------------------------------------------

def parse_parts(pdf):
    """Parse regulation body (Parts 1..N). Emits one chunk per subregulation."""
    chunks = []
    current_part = None
    current_division = None
    current_subdivision = None
    current_reg_num = None
    current_reg_title = None
    current_subreg = None

    buffer_text = []
    buffer_pages = []
    buffer_y_ranges = []
    page_margin_blocks = {}

    def flush():
        nonlocal buffer_text, buffer_pages, buffer_y_ranges
        if not buffer_text or current_reg_num is None:
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

        citation = f"Building Regulations 2018 (Vic) r {current_reg_num}"
        if current_subreg:
            citation += current_subreg

        chunks.append({
            "doc_type": "regulation",
            "authority_level": 2,
            "regulations": "Building Regulations 2018 (Vic)",
            "version": "028",
            "version_date": "2025-11-26",
            "part": current_part,
            "division": current_division,
            "subdivision": current_subdivision,
            "section_number": current_reg_num,       # schema-compatible name
            "section_title": normalize_text(current_reg_title) if current_reg_title else None,
            "subsection": current_subreg,            # schema-compatible name
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
        pdf_p = printed_to_pdf(printed_p)
        if pdf_p > len(pdf.pages):
            continue
        page = pdf.pages[pdf_p - 1]

        body_min = BODY_MIN_SIZE_DEFAULT
        body_chars = [c for c in page.chars if c["size"] >= body_min]
        margin_chars = [c for c in page.chars if c["size"] <= MARGIN_MAX_SIZE]

        body_lines = chars_to_lines(body_chars)
        margin_lines = chars_to_lines(margin_chars)

        body_lines = [
            ln for ln in body_lines
            if not is_header_line(ln["text"])
        ]
        body_lines = merge_multiline_part_heading(body_lines)
        body_lines = merge_multiline_reg_heading(body_lines)

        # Drop the footer line from margin extraction so it doesn't
        # pollute amendment attribution for the last reg on the page.
        margin_lines = [
            ln for ln in margin_lines
            if ln["size"] >= 7.5  # keep 8pt Arial-Bold amendments only
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
                m_reg    = RE_REG.match(text)

                if m_part and ln["size"] >= PART_HEADING_SIZE:
                    flush()
                    current_part = text
                    current_division = None
                    current_subdivision = None
                    current_reg_num = None
                    current_reg_title = None
                    current_subreg = None
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
                if m_reg and ln["size"] >= REG_HEADING_SIZE and ln["x0"] < 160:
                    # Only treat as reg heading if the line starts in the
                    # left column (x0 ~142). Avoids matching bold
                    # mid-sentence statute names.
                    flush()
                    current_reg_num = m_reg.group(1)
                    current_reg_title = m_reg.group(2).strip()
                    current_subreg = None
                    continue
                # Bold but not a heading — fall through as body text

            m_sub = RE_SUBREG.match(text)
            if m_sub:
                flush()
                current_subreg = f"({m_sub.group(1)})"
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

    flush()  # flush at end of body
    return chunks


# ---------------------------------------------------------------------------
# Schedule 3 parser — one chunk per exemption item
# ---------------------------------------------------------------------------

# Column x-ranges. The PDF cell boundaries are x=147-202 / 202-381 /
# 381-460 (from rect analysis on p.291). But col-2 text occasionally
# sits at x0≈201.8, within the col-1 cell by a fraction. We tighten
# col 1 to end at 170 — item numbers never extend past x0=160.
SCHED3_COL1_X = (140, 170)   # item number
SCHED3_COL2_X = (170, 381)   # description of building work
SCHED3_COL3_X = (381, 465)   # exempted regulations (col 3)

# Drop anything above y≈241: running headers + column headers.
# Row divider rects end at y≈240 on most pages, so 241 is safe.
SCHED3_BODY_Y_MIN = 241.0

# Body y-band on Schedule 3 pages. Above 241 is running headers +
# column-header rows; below 710 is the footer.
SCHED3_BODY_Y_MAX = 710.0


def parse_schedule_3(pdf):
    """Parse Schedule 3 as one chunk per exemption item."""
    chunks = []

    item_start_re = re.compile(r"^(\d+[A-Z]*)$")

    # Collect all body lines across Schedule 3 pages with their column.
    lines = []  # list of {pdf_p, y, col, text}
    start_printed, end_printed = SCHEDULE_3_PRINTED
    for printed_p in range(start_printed, end_printed + 1):
        pdf_p = printed_to_pdf(printed_p)
        if pdf_p > len(pdf.pages):
            continue
        page = pdf.pages[pdf_p - 1]

        # Body chars = anything inside the table's x-range and y-band,
        # regardless of size. The x-range [140, 465] excludes margin
        # amendments (left margin at x≈79, right margin at x≈468+).
        # Broad size acceptance keeps superscripts (e.g. "²" in "10 m²"
        # renders as a size-6.5 "2") which a strict size-10 filter drops.
        body_chars = [
            c for c in page.chars
            if SCHED3_COL1_X[0] <= c["x0"] < SCHED3_COL3_X[1]
            and SCHED3_BODY_Y_MIN < c["top"] < SCHED3_BODY_Y_MAX
        ]
        # Group into lines (preserve columns by x0)
        body_chars.sort(key=lambda c: (c["top"], c["x0"]))
        raw_lines = []
        cur = []
        for c in body_chars:
            if not cur or abs(c["top"] - cur[0]["top"]) < 3:
                cur.append(c)
            else:
                raw_lines.append(cur)
                cur = [c]
        if cur:
            raw_lines.append(cur)

        for ln in raw_lines:
            # Split line chars into column buckets
            ln.sort(key=lambda c: c["x0"])
            buckets = {1: [], 2: [], 3: []}
            for c in ln:
                if SCHED3_COL1_X[0] <= c["x0"] < SCHED3_COL1_X[1]:
                    buckets[1].append(c)
                elif SCHED3_COL2_X[0] <= c["x0"] < SCHED3_COL2_X[1]:
                    buckets[2].append(c)
                elif SCHED3_COL3_X[0] <= c["x0"] < SCHED3_COL3_X[1]:
                    buckets[3].append(c)
            for col_idx, col_chars in buckets.items():
                if not col_chars:
                    continue
                col_chars.sort(key=lambda c: c["x0"])
                text = ""
                prev_x1 = None
                for c in col_chars:
                    if prev_x1 is not None and c["x0"] - prev_x1 > 2:
                        text += " "
                    text += c["text"]
                    prev_x1 = c["x1"]
                text = text.strip()
                if text:
                    lines.append({
                        "pdf_p": pdf_p,
                        "y": ln[0]["top"],
                        "col": col_idx,
                        "text": text,
                    })

    # Sort by page then y then column to restore reading order
    lines.sort(key=lambda l: (l["pdf_p"], l["y"], l["col"]))

    # Walk lines, start a new item each time col1 contains a bare number
    current = None  # {"item": str, "col2": [str], "col3": [str], "pages": set, "y_start": ..., "y_end": ...}
    for line in lines:
        if line["col"] == 1 and item_start_re.fullmatch(line["text"].strip()):
            # New item begins — flush current
            if current:
                chunks.append(_finalise_sched3_item(current))
            current = {
                "item": line["text"].strip(),
                "col2": [],
                "col3": [],
                "pages": {line["pdf_p"]},
                "y_start": line["y"],
                "y_end": line["y"],
            }
            continue

        if current is None:
            # Text before the first item — skip
            continue

        current["pages"].add(line["pdf_p"])
        current["y_end"] = line["y"]
        if line["col"] == 2:
            current["col2"].append(line["text"])
        elif line["col"] == 3:
            current["col3"].append(line["text"])
        # col 1 continuations (e.g. multi-line item numbers) are rare — ignore

    if current:
        chunks.append(_finalise_sched3_item(current))

    return chunks


def _finalise_sched3_item(item):
    description = normalize_text(" ".join(item["col2"]))
    exempted = normalize_text(" ".join(item["col3"]))

    # Split out an inline "Note" block (legally informational, not part of
    # the operative exemption). Keeps the exemption text clean while still
    # preserving the note as a chunk field.
    note = None
    note_split = re.split(r"\s+Note\s+", description, maxsplit=1)
    if len(note_split) == 2:
        description, note = note_split
        description = description.rstrip()
        note = normalize_text(note)

    # Strip stray footnote markers (asterisks used to link to table-level
    # footnotes we don't otherwise capture).
    description = re.sub(r"\s*\*\s*\*+\s*", " ", description).strip()
    description = re.sub(r"\s+\*+\s*$", "", description).strip()
    if exempted:
        exempted = re.sub(r"\s*\*+\s*$", "", exempted).strip()

    citation = f"Building Regulations 2018 (Vic) Sch 3 item {item['item']}"

    return {
        "doc_type": "regulation_schedule",
        "authority_level": 2,
        "regulations": "Building Regulations 2018 (Vic)",
        "version": "028",
        "version_date": "2025-11-26",
        "part": None,
        "division": None,
        "subdivision": None,
        "schedule": "Schedule 3",
        "schedule_title": "Exemptions for building work and buildings",
        "section_number": f"Sch3-{item['item']}",  # schema-compatible
        "section_title": f"Schedule 3 item {item['item']}",
        "subsection": None,
        "item_number": item["item"],
        "citation": citation,
        "text": description,
        "exempted_regulations": exempted if exempted else None,
        "note": note,
        "penalty": None,
        "amendment_history": None,
        "cross_references": extract_xrefs(description),
        "pages": sorted(item["pages"]),
    }


def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        part_chunks = parse_parts(pdf)
        sched3_chunks = parse_schedule_3(pdf)

    print(f"Parts parser produced {len(part_chunks)} chunks")
    print(f"Schedule 3 parser produced {len(sched3_chunks)} chunks\n")

    by_part = Counter(c["part"] for c in part_chunks)
    for part, n in by_part.most_common():
        regs = sorted(
            set(c["section_number"] for c in part_chunks if c["part"] == part),
            key=lambda s: (int(re.match(r'\d+', s).group()), s),
        )
        print(f"  {part}: {n} chunks, {len(regs)} regulations")
        print(f"    → {regs[:30]}{' ...' if len(regs) > 30 else ''}")

    print(f"\n  Body chunks with amendments:  "
          f"{sum(1 for c in part_chunks if c['amendment_history'])}")
    print(f"  Body chunks with x-refs:      "
          f"{sum(1 for c in part_chunks if c['cross_references'])}")

    all_chunks = part_chunks + sched3_chunks
    print(f"\n  Schedule 3 items:")
    for c in sched3_chunks:
        desc = c["text"][:70]
        exempt = (c.get("exempted_regulations") or "")[:40]
        print(f"    item {c['item_number']:>4}: [{exempt:<40}] {desc}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} total chunks to {OUT_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
