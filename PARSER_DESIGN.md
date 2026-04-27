# Parser Design Notes

Read this before modifying `parse_building_act.py` or writing a new parser
for a different document.

## The core problem

Victorian legislation PDFs published by the Chief Parliamentary Counsel have a
two-column layout where the **main body** sits on the left and **amendment
history annotations** sit in a narrow right-hand column. A naïve text
extraction (e.g. `pdftotext`) interleaves these two streams, producing
output like:

```
A person must not carry out building work unless  S. 16
a building permit in relation to the work has     amended by
been issued and is in force under this Act.       No. 33/2010
Penalty: 500 penalty units...                     s. 3,
```

This is unusable for retrieval. The amendment text contaminates the body,
and the body fragments destroy the amendment provenance. This is the exact
problem that killed our previous attempt with GPT-based tools.

## The fix: font-size-based separation

On every page the body text renders at **size 12** (plus section headings at
12 bold, division headings at 14 bold, part headings at 16 bold). The
amendment annotations render at **size 8**. Size is a far sharper
discriminator than x-coordinate because body text sometimes extends into the
nominal margin area near the right edge of lines.

```python
body_chars   = [c for c in page.chars if c["size"] >= 10]
margin_chars = [c for c in page.chars if c["size"] <= 9]
```

This one rule removed ~95% of the parsing errors in the first iteration.

## Structural hierarchy detection

Every line is classified by font weight + size + regex match:

| Element | Font | Size | Regex |
|---------|------|------|-------|
| Part heading | Bold | ≥15 | `^Part \d+[A-Z]*—...` |
| Division heading | Bold | ≥13 | `^Division \d+[A-Z]*—...` |
| Subdivision heading | Bold | ≥11 | `^Subdivision \d+[A-Z]*—...` |
| Section heading | Bold | ≥11.5 | `^(\d+[A-Z]*) (.+)` |
| Subsection | Regular | 12 | `^\((\d+[A-Z]*)\) (.*)` |
| Paragraph | Regular | 12 | `^\(([a-z]+)\) (.*)` |
| Body | Regular | 12 | anything else |

**Critical:** the section-heading regex would match normal sentences starting
with a number (e.g. "10 working days after...") if we didn't require bold
font. Font weight is non-negotiable for this classifier.

## Multi-line Part headings

Some Part headings wrap. Part 5's heading is:
```
Part 5—Occupation of buildings and places
of public entertainment
```
Both lines are bold at size 16. `merge_multiline_part_heading()` collapses
consecutive bold-at-heading-size lines into one logical heading when they're
vertically close and the second line doesn't itself start with "Part N".

## Amendment note attribution

Margin notes are extracted separately and grouped into "blocks" by vertical
proximity (`consolidate_margin_blocks`). Each body chunk records the
(page, y_start, y_end) extent of its text. At flush time, we find every
margin block whose y-range overlaps the chunk's y-range and attach it as
`amendment_history`.

This works because amendment notes visually sit adjacent to the subsection
they annotate. If a future document puts notes in a different location, this
logic needs rethinking.

## Chunking strategy

One chunk per **subsection**, not per section and not per page. Rationale:
- Subsections are the natural unit of meaning in legislation
- Users ask questions that map to individual subsections ("what's the
  penalty in 16(1)?" not "tell me about section 16")
- Penalties attach to subsections, not sections, and must travel with the
  provision they punish
- Amendment history attaches to subsections, not sections

If a section has no subsections (rare in Parts 3-5 but common in Schedules),
the whole section becomes one chunk.

## Metadata emitted per chunk

```json
{
  "act": "Building Act 1993 (Vic)",
  "version": "141",
  "version_date": "2024-11-13",
  "part": "Part 3—Building permits",
  "division": "Division 1—Building permit required",
  "subdivision": null,
  "section_number": "16",
  "section_title": "Offences relating to carrying out building work",
  "subsection": "(1)",
  "citation": "Building Act 1993 (Vic) s 16(1)",
  "text": "A person must not carry out building work...",
  "penalty": "500 penalty units, in the case of a natural person; 2500 penalty units, in the case of a body corporate",
  "amendment_history": "S. 16 amended by No. 33/2010 s. 3, substituted by No. 15/2016 s. 44.",
  "cross_references": ["section 25AE", "Part 11"],
  "pages": [77]
}
```

The `citation` field is the formatted string that gets shown to users.

## Known limitations and TODOs

1. **Cross-reference extraction is regex-based.** It catches most explicit
   references ("section 30(1)", "Part 11", "Schedule 2") but misses
   narrative references ("the preceding section", "the offence in
   subsection (1)"). Post-processing the corpus with an LLM could fill
   these in.

2. **Cross-references aren't yet resolved to chunk IDs.** Each chunk
   records the text of its references but not which chunk they point to.
   Adding a post-processing step to build a reference graph is a clear
   next step (see `NEXT_STEPS.md`).

3. **Definitions (s 3) are not yet parsed.** Section 3 of the Act is a
   long list of defined terms with its own internal structure. Needs a
   bespoke parser variant before we can answer definition-based queries.

4. **Schedules are not yet parsed.** The Act has multiple Schedules with
   different structure (often tabular). These will need their own parser.

5. **Penalty parsing is brittle** for multi-penalty provisions that use
   formats like "Penalty: In the case of... In the case of..." — the
   current regex handles the standard two-penalty format but may mangle
   unusual formats. Test on the Regulations, which have more variety.

## Porting the parser to new documents

This parser is specifically tuned to documents published by the Victorian
Chief Parliamentary Counsel. Other Victorian Acts (Planning and Environment
Act 1987, Domestic Building Contracts Act 1995, Owners Corporations Act 2006)
use the same house style and the parser should work with minor tweaks.

The Building Regulations 2018 use a similar style but with differences:
- "Regulation N" instead of "Section N"
- Subregulations numbered like subsections
- Schedules are more prominent and often contain the key answers
- No penalty-unit column in most regulations

The NCC is **structurally different** and will need a new parser, not an
adaptation of this one.

---

## Building Regulations 2018 — quirks encountered

See `parse_building_regs.py`. Built by adapting the Act parser. Took ~1
session of diagnosis + code. Total output: **792 chunks** (770 body +
22 Schedule 3 items) across Parts 1–22 and Schedule 3.

### Two font-size regimes in one document

Body pages (Parts 1–22) follow the Act's regime: **body text size 12,
margin notes size 8**. Schedule pages (all 13 schedules) drop a size:
**body text size 10, margin notes size 9**. Any parser that hardcodes
a 12pt body threshold will emit empty chunks for schedules. The Regs
parser uses `BODY_MIN_SIZE_DEFAULT=11.5` for the parts pass and a
separate Schedule 3 sub-parser with its own size/x-band logic.

### Margin amendment notes alternate side by page parity

Odd pages (right-hand in a two-sided book) have the amendment column at
`x0 ≈ 468` (right margin). Even pages flip it to `x0 ≈ 79` (left
margin). Font-size-only separation is inherently side-agnostic, so no
special handling was required — the existing `consolidate_margin_blocks`
routine works unchanged. Just a trap to be aware of if you ever switch
to x-based margin detection.

### PUA (Private Use Area) glyphs replace standard symbols

The source PDF embeds a Symbol font mapped into the Unicode private
range. Four glyphs appear across the whole document and break naive
text extraction:

| PUA code | Intended | Used for |
|---|---|---|
| U+F0D7 | · (middle dot U+00B7) | decimal separator, e.g. "3·6 m" |
| U+F0B4 | × (multiplication U+00D7) | dimensions, e.g. "50 × 25 mm" |
| U+F06F | ☐ (ballot box U+2610) | form checkboxes (Schedule 4) |
| U+F0A8 | ☐ (ballot box U+2610) | form checkboxes (Schedule 5) |

Without translation, a "3·6 m" pergola height limit becomes "36 m" (!)
after whitespace normalisation — a serious content error. The parser
applies a `PUA_TRANSLATION` table via `str.translate()` inside
`normalize_text()` before any chunk emission.

Anything that parses Victorian authorised PDFs should add this
translation, or at least scan `[c for c in page.chars if 0xE000 <=
ord(c["text"]) <= 0xF8FF]` and surface a warning.

### Superscripts render as small-font characters (size ~6.5)

"10 m²" isn't stored with a Unicode ² — it's two characters: a size-10
"m" followed by a **size-6.5 "2"** at a slightly-higher y-coordinate.
A strict `size >= 10` filter drops it and corrupts the meaning.

The Schedule 3 sub-parser accepts all sizes within the table's x/y
band and relies on column x-bucketing instead of size for body/margin
separation. The Parts parser still uses a size-11.5 threshold because
superscripts are rare in body text (3 confirmed across all 271 body
pages) and the cost of a size-12-regime body page pulling in margin
noise is higher than the cost of a dropped superscript.

### Schedule 3 is a borderless 3-column table

Schedule 3 — *Exemptions for building work and buildings* — is the
homeowner-priority payload (permit-not-required list). Structure:

```
Col 1 (x 147–202)  |  Col 2 (x 202–381)       |  Col 3 (x 381–460)
Item #             |  Description of work      |  Exempted regulations
```

`pdfplumber.extract_tables()` finds no cells because the table has **no
drawn borders** — just thin horizontal `rect` fills at the page header,
column header, and page footer. Row boundaries between items are
implicit (determined by "the next col-1 line that starts with a
digit").

Parser approach:

1. Filter to chars with `140 ≤ x0 < 465` and `241 < top < 710`
   (inside the table body band, excluding margins and footer).
2. Group chars into lines by y-coordinate.
3. For each line, bucket its chars into col 1, col 2, col 3 by x0.
   **Column boundary is 170, not the rect cell boundary at 202**: col-2
   text occasionally renders with x0≈201.8, which under a 202 threshold
   gets misclassified as col 1. The item-number regex `^\d+[A-Z]*$`
   then stops matching because col 1 now contains the letter "A" too.
4. A line that has just a plain integer in col 1 marks the start of a
   new exemption item.
5. Concatenate col 2 (description) and col 3 (exempted regulations)
   separately into the item's chunk.
6. A trailing "Note" block in col 2 is split into its own `note` field
   so the operative exemption text stays clean.

Schedule 3 has **22 items** (numbered 1–8, 10–23; item 9 is missing from
the source — repealed). Each becomes one chunk.

### Multi-line regulation titles

Regulation titles wrap when long (e.g. r 57: "Notice of imminent lapse
of building permit— completion of work" wraps across two bold lines).
The Act parser didn't need this because Act section titles are
generally shorter. `merge_multiline_reg_heading()` handles it: after a
bold size-12 line matching `^\d+[A-Z]* ...`, merge any immediately
following bold size-12 lines (within 20 y-units, no leading digit, no
Part/Division/Subdivision prefix) into the heading.

Threshold is 20 y-units. Tried 15 first; missed r 22's 15.0-unit gap.

### Regulation-specific schema additions

The Regs chunks preserve the Act's schema (`section_number`,
`subsection`, etc.) so BM25 and evaluate.py work uniformly, and add:

- `doc_type`: `"regulation"` (body) or `"regulation_schedule"` (Sch 3)
- `authority_level`: `2` (Act is implicitly 1 — higher wins)
- `regulations`: `"Building Regulations 2018 (Vic)"`
- `version`: `"028"`, `version_date`: `"2025-11-26"` (Authorised v028,
  as at 26 Nov 2025)
- For Schedule 3 chunks: `schedule`, `schedule_title`, `item_number`,
  `exempted_regulations`, `note`

### Known limitations carried forward

- **Definitions (reg 5) are a single 11k-character mega-chunk.** Same
  issue as s 3 of the Act. Needs a bespoke per-term sub-parser as a
  later task.
- **Running-header re-detection on continuation pages.** A handful of
  regs (≈16 out of 352) produce a short fragment chunk in addition to
  the main content — the reg's title appears on subsequent pages as a
  running header and occasionally re-triggers the heading detector,
  emitting "reporting authorities" or similar 20-char fragments. Low
  impact: the real content is still captured in a separate chunk with
  the correct citation. Fix requires tracking "we've already seen this
  reg heading on this page-run" and suppressing duplicates.
- **BM25-only retrieval won't match natural-language queries.** The
  test "do I need a permit for a pergola?" doesn't retrieve Schedule 3
  item 16 at top-3 because "permit" and "need" are generic words. The
  query `"pergola exemption"` does retrieve it at rank 1. This is the
  motivation for Task 2 (hybrid retrieval with embeddings).
