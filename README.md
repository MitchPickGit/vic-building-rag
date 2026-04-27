# Victorian Building Regulations RAG Prototype

A retrieval-augmented Q&A system for Victorian building legislation. The goal is a
web/app tool that lets homeowners and builders ask natural-language questions about
building regulations and get clear, correctly-cited answers.

## Current state

Proof-of-concept parser working for the Building Act 1993 (Vic), Parts 3, 4, 5.
274 chunks extracted with clean structure — section numbers, subsections, penalties,
amendment history, cross-references. BM25 retrieval baseline achieves 53% hit@3 on
a 30-question test set. Ready to extend.

## Target audience

1. **Homeowners** (owner-builders) renovating or building — need plain-English
   answers, always pointing them to verify with a registered building surveyor.
2. **Builders / developers / architects** navigating permits and compliance — need
   precise citations with section numbers, penalties, and cross-references.

The same corpus serves both, but the LLM prompt layer adapts tone and depth.

## Corpus plan

In scope for the MVP:
- Building Act 1993 (Vic) — done for Parts 3-5, extend to full Act
- Building Regulations 2018 (Vic) — **next target**
- NCC Volume 2 (residential) — after Regulations

Out of scope for MVP: Local planning schemes (79 LGAs, too much variance),
NCC Vol 1 (commercial), plumbing code.

## Architecture

```
source PDF
    ↓
[parser]  ← structural extraction with font-aware body/margin separation
    ↓
chunks.jsonl  ← one record per subsection with metadata
    ↓
[embedding + indexing]
    ↓
vector DB + keyword index
    ↓
[hybrid retrieval + reranker]
    ↓
[Claude API call with strict citation prompt]
    ↓
answer with section references
```

## Non-negotiable principles

1. **No hallucinated citations.** If the system can't find a matching section,
   it must say so. Never invent a section number. Test this adversarially.
2. **Authority hierarchy matters.** Act > Regulations > Codes > Guidelines.
   Chunks carry an `authority_level` field and prompts must respect it.
3. **Version everything.** Every chunk has `version` and `version_date`.
   Legislation changes; we need to know which version we're answering from.
4. **Structure preservation over chunk count.** A good chunking scheme with
   rich metadata beats a large number of dumb chunks. Always.
5. **Measure before improving.** `evaluate.py` is the source of truth for
   retrieval quality. Changes that don't improve hit@3/5 aren't worth shipping.

## Files in this repo

- `parse_building_act.py` — current parser, Parts 3-5
- `building_act_chunks.jsonl` — parsed output (274 chunks)
- `test_questions.json` — 32-question test set with expected citations
- `evaluate.py` — BM25 evaluation harness (hit@1, hit@3, hit@5)
- `NEXT_STEPS.md` — prioritised roadmap
- `PARSER_DESIGN.md` — how the parser works, gotchas encountered
- `docs/` — briefs, legislation URLs, notes

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required environment variables (add to `.env`, gitignored):
```
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=...   # or OPENAI_API_KEY if preferred
```

## Running what exists

```bash
# Re-parse the Act (writes building_act_chunks.jsonl)
python parse_building_act.py

# Evaluate current retrieval quality
python evaluate.py
```

## Working with Claude Code on this project

Read `NEXT_STEPS.md` to find the current priority task. Read `PARSER_DESIGN.md`
before touching the parser. Always run `evaluate.py` before and after any
retrieval-related change and include the before/after numbers in the commit
message.
