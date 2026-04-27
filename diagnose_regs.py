"""
Diagnostic script for the Building Regulations 2018 PDF.
Run `.venv/Scripts/python.exe diagnose_regs.py` to print font / layout stats
for chosen pages. Extend as we encounter new quirks.

The parser itself lives in parse_building_regs.py (once we write it). This
script exists to answer "what does the document actually look like?" before
we commit to parsing rules.
"""

import sys
import pdfplumber
from collections import Counter

PDF_PATH = "docs/18-38sra028-authorised.pdf"

# Force UTF-8 on Windows so em-dashes don't crash prints
sys.stdout.reconfigure(encoding="utf-8")


def page_stats(page, label):
    """Print font-size histogram and x-coord distribution for a page."""
    chars = page.chars
    size_hist = Counter(round(c["size"], 1) for c in chars)
    font_hist = Counter(c["fontname"] for c in chars)

    print(f"\n=== {label} (PDF page {page.page_number}) ===")
    print(f"Total chars: {len(chars)}")
    print(f"Page width:  {page.width:.1f}")
    print(f"\n  Font-size histogram (size -> char count):")
    for size, count in sorted(size_hist.items()):
        bar = "#" * min(60, count // 20)
        print(f"    {size:>5}  {count:>5}  {bar}")

    print(f"\n  Font-name histogram (top 6):")
    for font, count in font_hist.most_common(6):
        print(f"    {font:<40}  {count:>5}")

    # Bucket x-coord for small-size chars (margin suspects) vs large
    margin_chars = [c for c in chars if c["size"] <= 9]
    body_chars = [c for c in chars if c["size"] >= 10]
    if margin_chars:
        mx = [c["x0"] for c in margin_chars]
        print(f"\n  Margin (<=9pt) chars: {len(margin_chars)}, x0 range: "
              f"{min(mx):.0f} - {max(mx):.0f}, median x0: {sorted(mx)[len(mx)//2]:.0f}")
    if body_chars:
        bx = [c["x0"] for c in body_chars]
        print(f"  Body  (>=10pt) chars: {len(body_chars)}, x0 range: "
              f"{min(bx):.0f} - {max(bx):.0f}, median x0: {sorted(bx)[len(bx)//2]:.0f}")


def group_into_lines(chars, y_tolerance=2.0):
    """Group chars into lines by y-coordinate, in top-to-bottom reading order."""
    if not chars:
        return []
    chars = sorted(chars, key=lambda c: (c["top"], c["x0"]))
    lines = []
    current_line = [chars[0]]
    for c in chars[1:]:
        if abs(c["top"] - current_line[0]["top"]) < y_tolerance:
            current_line.append(c)
        else:
            lines.append(current_line)
            current_line = [c]
    lines.append(current_line)
    return lines


def sample_lines(page, label, size_filter="body", n=30):
    """Print first N lines (in reading order) with font/size tags.
    size_filter: 'body' (>=10pt), 'margin' (<=9pt), or 'all'."""
    if size_filter == "body":
        chars = [c for c in page.chars if c["size"] >= 10]
    elif size_filter == "margin":
        chars = [c for c in page.chars if c["size"] <= 9]
    else:
        chars = list(page.chars)

    print(f"\n=== Line sample [{size_filter}]: {label} (PDF page {page.page_number}) ===")
    if not chars:
        print("  (no chars in filter)")
        return

    lines = group_into_lines(chars)
    for ln in lines[:n]:
        text = "".join(c["text"] for c in sorted(ln, key=lambda c: c["x0"]))
        sizes = sorted({round(c["size"], 1) for c in ln})
        fonts = sorted({c["fontname"].split("+")[-1] for c in ln})
        bold = any("Bold" in f or "Bd" in f for f in fonts)
        italic = any("Italic" in f or "It" in f for f in fonts)
        x0 = min(c["x0"] for c in ln)
        tag = f"size={sizes} bold={bold} italic={italic} x0={x0:.0f}"
        print(f"  [{tag:<48}] {text[:100]}")


if __name__ == "__main__":
    # Default: check a mid-body page, a page with margin amendments, and Schedule 3 start
    target_pages = [17, 20, 25, 50, 291]  # PDF page numbers
    if len(sys.argv) > 1:
        target_pages = [int(x) for x in sys.argv[1:]]

    pdf = pdfplumber.open(PDF_PATH)
    for pn in target_pages:
        p = pdf.pages[pn - 1]
        label = f"page_{pn}"
        page_stats(p, label)
        sample_lines(p, label, size_filter="body", n=30)
        sample_lines(p, label, size_filter="margin", n=15)
    pdf.close()
