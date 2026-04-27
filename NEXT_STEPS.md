# Next Steps

Prioritised roadmap for the project. Work top-down. Each task has explicit
acceptance criteria so you know when you're done.

---

## TASK 1 ✓ DONE (2026-04-23): Parsed the Building Regulations 2018

**Delivered:** `building_regs_chunks.jsonl` — 792 chunks (770 body across
Parts 1–22, 22 Schedule 3 items). Zero amendment-note contamination in
body text. Middle-dot, superscript and checkbox PUA glyphs normalised.
See `PARSER_DESIGN.md` "Building Regulations 2018 — quirks encountered"
for the full writeup.

Acceptance criteria:
- [x] `building_regs_chunks.jsonl` exists with at least 400 chunks (792)
- [x] Schedule 3 items are individual chunks (22 items: 1-8, 10-23)
- [x] Spot-check of 10 random chunks shows clean body text with no
      amendment-note contamination
- [x] Pergola exemption (Sch 3 item 16) retrievable from the corpus
      (BM25 rank 1 for query "pergola exemption"; natural-language
      phrasing will rely on Task 2 embeddings)
- [x] `PARSER_DESIGN.md` updated with Regulations quirks

Known limitations (carried forward):
- Reg 5 definitions are one 11k-char chunk — needs per-term splitter
  (same pattern as Act's s 3)
- ~16 short fragment chunks from running-header re-detection on
  continuation pages — cosmetic, real content is still captured

---

## TASK 2: Add embeddings + reranking to lift hit@3 above 80%

Current baseline is 53% hit@3 with BM25 only. This task is about closing
that gap.

**Approach:**
1. Add Voyage or OpenAI embeddings — embed the `text` field plus a
   structural prefix ("Part 3, Division 1, s 16(1): [text]")
2. Store embeddings in a simple sqlite-vec or duckdb index for now
   (don't bother with a vector DB service yet)
3. Implement hybrid retrieval: BM25 top 20 + vector top 20, merge with
   Reciprocal Rank Fusion (RRF, k=60)
4. Optionally add Cohere Rerank or a Claude Haiku-based reranker on the
   top 20

**Acceptance:** hit@3 ≥ 80% on the existing 30-question test set.
Update `evaluate.py` to report BM25-only vs hybrid vs hybrid+rerank so
the contribution of each layer is visible.

---

## TASK 3: End-to-end answer generation with strict citation

Wire up the retrieval to Claude and produce actual answers.

**Approach:**
1. For each query: retrieve top 10 chunks, pass to Claude with a system
   prompt that requires citing `section_number(subsection)` for every
   factual claim
2. Build two system prompts: `HOMEOWNER_MODE` (plain English, always
   recommends consulting a building surveyor) and `BUILDER_MODE`
   (technical, includes penalties and amendment history)
3. Hard constraint in the prompt: if retrieval returns no chunks with
   BM25 score > 5, respond "I don't have information on this in the
   Building Act. You may need to check the Regulations or consult a
   building surveyor." Test this on the two out-of-scope questions
   in `test_questions.json` (IDs 31 and 32).

**Acceptance:**
- Running all 30 in-scope test questions through the full pipeline
  produces answers that include at least one correct section citation
  per answer
- Running the 2 out-of-scope questions produces the "I don't have
  information" response, not a hallucinated citation
- A human-readable markdown report is produced comparing the answer
  to the expected facts for each test question

---

## TASK 4: Expand Act coverage to Parts 1, 6, 11

Once retrieval and answer generation are solid, widen the net:
- Part 1 (definitions) — critical for every other query to resolve terms
- Part 6 (private building surveyors) — high builder relevance
- Part 11 (registration of building practitioners) — high builder relevance

Part 1 needs a bespoke sub-parser for the definitions in s 3 because
each defined term is structured differently from a normal section.

---

## TASK 5: Cross-reference resolution

Post-process all chunks to link every `"section 16"` / `"regulation 233"`
/ `"Part 3"` reference to the actual target chunk ID. Store as a
`references` field: `[{"target_id": "...", "type": "section"}]`.

This enables "show me section 16 and everything it references" queries
and dramatically improves answer quality for questions that span
multiple provisions.

---

## TASK 6+: UI, NCC, deployment

Deferred until Tasks 1-3 are solid. Don't jump ahead.

---

## Working notes — read these before starting a session

- **Always read `PARSER_DESIGN.md` before touching the parser.** The
  font-size separation trick is non-obvious and easy to break.
- **Always run `evaluate.py` before and after retrieval changes.**
  Report the delta in the commit message.
- **Commit in small increments.** One task = one or two commits, not
  a sprawling mega-commit at the end of the session.
- **If you're tempted to rewrite something that's working, stop and
  ask the user.** The current parser is fragile in the way all PDF
  parsers are fragile. Incremental improvements are safer than
  clever refactors.
- **Never invent section numbers in generated answers.** This is a
  legal product. A hallucinated citation is worse than "I don't know".
