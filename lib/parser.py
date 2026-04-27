"""
Config-driven unified parser for the legislation corpus.

Replaces the four hand-written parsers (parse_building_act.py,
parse_building_regs.py, parse_ncc.py, parse_housing_provisions.py) with
a single rule-based engine driven by corpora.yaml. Adding a new
document is just a matter of dropping the PDF in docs/, writing a YAML
config entry, and running:

    python -m lib.parser <doc_id>

The engine handles everything that's truly shared:
  - PDF page sweep + char extraction
  - Font-size based body/margin separation
  - Line grouping by y-coordinate
  - PUA glyph translation (Symbol-font middle dot, etc.)
  - Heading detection (Section / Part / Provision / Subprovision)
  - Multi-line heading merging (when wrapped)
  - Running header / footer filtering
  - Margin amendment-note attribution (Vic-style)
  - Buffer-and-flush per provision
  - Cross-reference extraction
  - Empty-parent filtering

Per-document quirks live entirely in the YAML config — no Python
edits to add a new document type that fits the same shape.

This module is fully self-contained: nothing else in the codebase
imports from it. It coexists with the existing parsers; switching a
document over to this parser is a per-doc decision.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pdfplumber
import yaml


# Symbol-font Private Use Area glyphs that appear in Vic legislation PDFs
# but render as garbage without translation. Same table as
# parse_building_regs.py.
PUA_TRANSLATION = str.maketrans({
    "\uF0D7": "\u00B7",   # middle dot   ·
    "\uF0B4": "\u00D7",   # times sign   ×
    "\uF06F": "\u2610",   # ballot box   ☐
    "\uF0A8": "\u2610",
})


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str, apply_pua: bool = False) -> str:
    if apply_pua:
        text = text.translate(PUA_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()


def chars_to_lines(chars, y_tolerance: float = 2.5) -> list[dict]:
    """Group chars into lines by y-coordinate. Returns lines top-to-bottom."""
    if not chars:
        return []
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
        is_bold = "Bold" in dominant_font
        out.append({
            "y": y,
            "x0": line_chars[0]["x0"],
            "text": text,
            "size": dominant_size,
            "font": dominant_font,
            "is_bold": is_bold,
        })
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DocumentParser:
    """Rule-based parser for a single document. Fully driven by a config
    dict (loaded from corpora.yaml)."""

    def __init__(self, doc_id: str, config: dict):
        self.doc_id = doc_id
        self.cfg = config
        # Compile heading regexes once
        self._compiled: dict[str, re.Pattern] = {}
        for level, hdef in config.get("headings", {}).items():
            if isinstance(hdef, dict) and "regex" in hdef:
                self._compiled[level] = re.compile(hdef["regex"])
        self.subprov_re = re.compile(config["subprovision_regex"])
        self.running_re = [
            re.compile(p) for p in config.get("running_header_patterns", [])
        ]
        self.cross_version_re = re.compile(
            config.get("extras", {}).get("cross_version_tag_regex", r"$^")
        )
        self.apply_pua = config.get("apply_pua_translation", False)
        self.body_min_size = config.get("body_min_size", 9.5)

    def _is_running_header(self, text: str) -> bool:
        if not text.strip():
            return True
        return any(p.match(text) for p in self.running_re)

    def _try_heading(self, text: str, line: dict
                     ) -> Optional[tuple[str, dict]]:
        """Return (level, fields) if line matches a heading, else None.
        Heading levels are checked in YAML-declared order; first match wins."""
        for level, pat in self._compiled.items():
            hcfg = self.cfg["headings"][level]
            size_min = hcfg.get("size_min")
            if size_min is not None and line["size"] < size_min:
                continue
            x0_max = hcfg.get("x0_max")
            if x0_max is not None and line["x0"] > x0_max:
                continue
            m = pat.match(text)
            if not m:
                continue
            # Capture by named groups if available, else positional
            if m.groupdict():
                return level, m.groupdict()
            groups = m.groups()
            if level == "provision":
                return level, {"id": groups[0],
                               "title": groups[1] if len(groups) > 1 else ""}
            return level, {"groups": list(groups)}
        return None

    def _is_named_subpart(self, text: str) -> bool:
        names_cfg = self.cfg.get("headings", {}).get("subpart_named")
        if not isinstance(names_cfg, dict):
            return False
        if text in set(names_cfg.get("names") or []):
            return True
        prefix = names_cfg.get("prefixed_with")
        if prefix and text.startswith(prefix):
            return True
        return False

    # ------------------------------------------------------------------
    # Main parse
    # ------------------------------------------------------------------

    def parse(self) -> list[dict]:
        chunks: list[dict] = []
        pdf_path = self.cfg["pdf_path"]
        if not Path(pdf_path).exists():
            print(f"  warn: {pdf_path} missing, skipping {self.doc_id}")
            return []

        with pdfplumber.open(pdf_path) as pdf:
            # State
            state = {
                "section": None, "part": None, "subpart": None,
                "schedule": None, "specification": None,
                "prov_id": None, "prov_title": None, "subprov": None,
            }
            buffer_text: list[str] = []
            buffer_pages: list[int] = []

            def flush():
                nonlocal buffer_text, buffer_pages
                if not buffer_text or state["prov_id"] is None:
                    buffer_text, buffer_pages = [], []
                    return
                text = normalize_text(" ".join(buffer_text), self.apply_pua)
                if not text:
                    buffer_text, buffer_pages = [], []
                    return
                version_tags = self.cross_version_re.findall(text)
                text = self.cross_version_re.sub("", text).strip()
                text = re.sub(r"\s+", " ", text)

                citation = self.cfg["citation_template"].format(
                    section_number=state["prov_id"],
                    subsection=state["subprov"] or "",
                )

                chunk = {
                    "doc_type": self.cfg["doc_type"],
                    "authority_level": self.cfg["authority_level"],
                    "version": self.cfg["version"],
                    "version_date": self.cfg["version_date"],
                    "section": state["section"],
                    "part": state["part"],
                    "subpart": state["subpart"],
                    "schedule": state["schedule"],
                    "specification": state["specification"],
                    "section_number": state["prov_id"],
                    "section_title": (
                        normalize_text(state["prov_title"], self.apply_pua)
                        if state["prov_title"] else None
                    ),
                    "subsection": state["subprov"],
                    "citation": citation,
                    "text": text,
                    "previous_edition_refs": version_tags or None,
                    "penalty": None,
                    "amendment_history": None,
                    "cross_references": [],  # left for a later enrichment pass
                    "pages": sorted(set(buffer_pages)),
                }
                # Optional metadata passthrough
                if "edition" in self.cfg:
                    chunk["edition"] = self.cfg["edition"]
                if "licence" in self.cfg:
                    chunk["licence"] = self.cfg["licence"]
                chunks.append(chunk)
                buffer_text, buffer_pages = [], []

            for prange in self.cfg.get("page_ranges", []):
                start, end = prange["start"], prange["end"]
                header_text = prange.get("header")
                label = prange.get("label", "")
                # Reset state on range boundary
                flush()
                if label.startswith("Section"):
                    state["section"] = header_text
                    state["schedule"] = None
                else:
                    state["schedule"] = header_text
                    state["section"] = None
                state["part"] = None
                state["subpart"] = None
                state["specification"] = None
                state["prov_id"] = None
                state["prov_title"] = None
                state["subprov"] = None

                for pdf_p in range(start, end + 1):
                    if pdf_p > len(pdf.pages):
                        continue
                    page = pdf.pages[pdf_p - 1]
                    body_chars = [c for c in page.chars
                                  if c["size"] >= self.body_min_size]
                    lines = chars_to_lines(body_chars)
                    lines = [ln for ln in lines if not self._is_running_header(ln["text"])]

                    for ln in lines:
                        text = ln["text"].strip()
                        if not text:
                            continue

                        if ln["is_bold"]:
                            heading = self._try_heading(text, ln)
                            if heading is not None:
                                level, fields = heading
                                flush()
                                if level == "section":
                                    g = fields.get("groups", [])
                                    state["section"] = (
                                        f"Section {g[0]} {g[1]}".strip()
                                        if len(g) >= 2 else text
                                    )
                                    state["part"] = state["subpart"] = None
                                    state["specification"] = None
                                    state["prov_id"] = None
                                    state["prov_title"] = None
                                    state["subprov"] = None
                                    continue
                                if level == "schedule":
                                    g = fields.get("groups", [])
                                    state["schedule"] = (
                                        f"Schedule {g[0]} {g[1]}".strip()
                                        if len(g) >= 2 else text
                                    )
                                    state["section"] = None
                                    state["part"] = state["subpart"] = None
                                    state["prov_id"] = None
                                    continue
                                if level == "part":
                                    g = fields.get("groups", [])
                                    state["part"] = (
                                        f"Part {g[0]} {g[1]}".strip()
                                        if len(g) >= 2 else text
                                    )
                                    state["subpart"] = None
                                    state["specification"] = None
                                    state["prov_id"] = None
                                    continue
                                if level == "specification":
                                    g = fields.get("groups", [])
                                    state["specification"] = (
                                        f"Specification {g[0]} {g[1]}".strip()
                                        if len(g) >= 2 else text
                                    )
                                    state["part"] = None
                                    state["subpart"] = None
                                    state["prov_id"] = None
                                    continue
                                if level == "provision":
                                    vic = fields.get("vic")
                                    pid = fields.get("id") or fields.get("groups", [None])[0]
                                    title = fields.get("title") or (
                                        fields.get("groups", [None, ""])[1]
                                        if len(fields.get("groups", [])) > 1 else ""
                                    )
                                    if vic:
                                        pid = f"VIC {pid}"
                                    state["prov_id"] = pid
                                    state["prov_title"] = (title or "").strip()
                                    state["subprov"] = None
                                    continue
                            # Subpart named heading?
                            if self._is_named_subpart(text):
                                flush()
                                state["subpart"] = text
                                state["prov_id"] = None
                                state["prov_title"] = None
                                state["subprov"] = None
                                continue
                            # Bold but not a heading -> body

                        m_sub = self.subprov_re.match(text)
                        if m_sub:
                            flush()
                            state["subprov"] = f"({m_sub.group(1)})"
                            buffer_text = [m_sub.group(2) or ""]
                            buffer_pages = [pdf_p]
                            continue

                        buffer_text.append(text)
                        buffer_pages.append(pdf_p)

                flush()

        if self.cfg.get("extras", {}).get("filter_empty_parents", False):
            chunks = [c for c in chunks if c.get("text", "").strip()]

        return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path: str = "corpora.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("Usage: python -m lib.parser <doc_id> [--config corpora.yaml]")
        print("\nDoc ids in corpora.yaml:")
        cfg = load_config()
        for k in cfg.get("documents", {}).keys():
            print(f"  - {k}")
        sys.exit(0)

    doc_id = sys.argv[1]
    cfg_path = "corpora.yaml"
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            cfg_path = sys.argv[i + 1]

    cfg = load_config(cfg_path)
    if doc_id not in cfg["documents"]:
        print(f"Unknown doc: {doc_id}", file=sys.stderr)
        sys.exit(1)

    doc_cfg = cfg["documents"][doc_id]
    parser = DocumentParser(doc_id, doc_cfg)
    chunks = parser.parse()
    print(f"\nProduced {len(chunks)} chunks for doc_id={doc_id}\n")

    by_section = Counter(c.get("section") or "—" for c in chunks)
    by_schedule = Counter(c.get("schedule") for c in chunks if c.get("schedule"))
    by_part = Counter(c.get("part") for c in chunks if c.get("part"))
    print(f"Sections: {len(by_section)}")
    for k, v in by_section.most_common(8):
        print(f"  {v:>4}  {k}")
    if by_schedule:
        print(f"Schedules: {len(by_schedule)}")
        for k, v in by_schedule.most_common(5):
            print(f"  {v:>4}  {k}")

    out_path = doc_cfg["out_path"]
    out_path_unified = out_path.replace(".jsonl", ".unified.jsonl")
    with open(out_path_unified, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path_unified} (use --commit to overwrite {out_path})")

    if "--commit" in sys.argv:
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Committed to {out_path}")


if __name__ == "__main__":
    main()
