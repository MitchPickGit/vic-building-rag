# Project instructions for Claude Code

Read this first. It applies to every session.

## What this project is

A RAG system for Victorian building legislation. Parser → chunks → retrieval
→ Claude-generated answers with citations. See `README.md` for overview and
`NEXT_STEPS.md` for current priorities.

## Session workflow

1. **Read `NEXT_STEPS.md`** to find the current active task.
2. **Read `PARSER_DESIGN.md`** if the task involves parsing.
3. **Run `evaluate.py`** at the start of any retrieval-related work to
   capture the "before" baseline. Run again at the end for the "after".
4. **Make small commits.** One coherent change per commit with a clear
   message. Never a single mega-commit at the end.
5. **Update `NEXT_STEPS.md`** when you finish a task — mark it done and
   make sure the next task is clearly described.

## Code style

- Python 3.10+. Type hints on function signatures where the types aren't
  obvious.
- Parsers live at the project root (`parse_*.py`). Supporting modules
  in `lib/`. Tests in `tests/`.
- No framework dependencies unless necessary. `pdfplumber`, `anthropic`,
  `voyageai`, maybe `duckdb` for vector storage. Don't add LangChain or
  LlamaIndex — we want to understand every step of the pipeline.
- Prefer scripts that read/write JSONL files over in-memory pipelines
  during development. Makes debugging far easier.

## Non-negotiable constraints

1. **Never hallucinate citations.** If generated code, tests, or answers
   include section numbers, they must come from actual retrieved chunks.
   If a section reference can't be verified against the chunks file,
   the answer must say "I don't have that information" instead.
2. **Preserve the chunks schema.** `citation`, `section_number`,
   `subsection`, `text`, `penalty`, `amendment_history`,
   `cross_references`, `part`, `division`, `version`, `version_date`.
   If you add a new field, document it in `PARSER_DESIGN.md` and update
   any consumers.
3. **Measure before claiming improvement.** "This should be better"
   means nothing. `evaluate.py` numbers mean something.
4. **Ask before deleting chunks or rewriting working code.** The
   current parser is fragile. Treat it as load-bearing.

## When to ask the user vs proceed

Proceed without asking:
- Extending the parser to more pages within the same document
- Adding logging, tests, or documentation
- Fixing clear bugs with obvious correct fixes
- Running evaluations and reporting results

Ask the user first:
- Any change that deletes existing chunks
- Adding a new dependency
- Rewriting a parser (vs incremental fix)
- Anything affecting the chunks schema
- Switching vector DB or embedding provider
- Before sending many API calls that will incur significant cost

## Cost awareness

This project runs on the user's API credits. Before any operation that
will make more than ~20 Claude API calls or more than ~1000 embedding
calls, state the estimated cost and ask for confirmation.

Rough reference: embedding 274 chunks is ~$0.01. Running all 30 test
questions through Claude Sonnet is ~$0.30. Re-embedding the whole
corpus on every run is wasteful — cache embeddings in a local file.

## What success looks like for each phase

- **Phase 1 (now):** Parsers for Act + Regulations, clean chunks, 80%+
  hit@3 on test questions, end-to-end Claude answers with correct
  citations.
- **Phase 2:** NCC Vol 2 coverage, 100+ test questions, mode-switching
  homeowner/builder prompts, basic web UI.
- **Phase 3:** Cross-reference graph, version diffing when legislation
  updates, liability-aware response framing.

We are in early Phase 1. Don't skip ahead.
