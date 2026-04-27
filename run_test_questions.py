"""
Run the full RAG pipeline over the 30-question test set (plus the 2 out-of-scope
questions) and emit a human-readable markdown report.

Outputs:
  test_report.md        : per-question breakdown — answer, cited sections,
                          hallucination flags, expected sections from the
                          test set, retrieval diagnostics
  test_report.json      : same data in structured form, for later diffing

Cost: ~30 Claude Sonnet 4.6 calls + ~32 Voyage query embeddings. With
prompt caching the system prompt (~4K tokens) is only paid for once.
Estimated total: ~$0.15–0.30.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from lib.retrieval import VectorRetriever, load_all_chunks
from lib.answer import (
    answer_question,
    verify_citations,
    AnswerResult,
)
from evaluate import section_matches


QUESTIONS_PATH = "test_questions.json"
CORPUS_PATHS = ["building_act_chunks.jsonl", "building_regs_chunks.jsonl"]


def expected_section_found(result: AnswerResult, expected: list[str]) -> bool:
    """Did at least one expected section from the test set appear in any
    retrieved chunk? This is a retrieval check, independent of whether
    Claude happened to cite it."""
    for c in result.retrieved_chunks:
        if section_matches(c, expected):
            return True
    return False


def expected_section_cited(result: AnswerResult, expected: list[str]) -> bool:
    """Did the model cite at least one of the expected sections?"""
    for exp in expected:
        for cited in result.cited_sections:
            # Loose matching — 'cited' may include 'reg. ', 'Sch 3 item ', etc.
            if exp in cited:
                return True
    return False


def run_all(limit: int | None = None, mode: str = "homeowner") -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        qdata = json.load(f)
    questions = qdata["questions"]
    if limit:
        questions = questions[:limit]

    # Build retriever over the full corpus (Act + Regs) — the model can
    # draw on either document. Eval alignment will still use Act section
    # numbers since the test set's expected_sections are Act-based.
    chunks = load_all_chunks(CORPUS_PATHS)
    print(f"Loaded {len(chunks)} chunks from {CORPUS_PATHS}")
    retriever = VectorRetriever(chunks)

    records = []
    for i, q in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] Q{q['id']} [{q.get('difficulty', '?')}] {q['q']!r}")
        t0 = time.time()
        try:
            result = answer_question(q["q"], mode=mode, retriever=retriever)
        except Exception as e:
            print(f"  ERROR: {e!r}")
            records.append({
                "question_id": q["id"],
                "question": q["q"],
                "error": repr(e),
            })
            continue

        elapsed = time.time() - t0
        expected = q.get("expected_sections") or []
        ret_hit = expected_section_found(result, expected)
        cite_hit = expected_section_cited(result, expected)
        is_oos_in_test = q.get("out_of_scope", False)

        rec = {
            "question_id": q["id"],
            "question": q["q"],
            "difficulty": q.get("difficulty"),
            "expected_sections": expected,
            "is_oos_in_test": is_oos_in_test,
            "mode": result.mode,
            "answer": result.answer,
            "cited_sections": result.cited_sections,
            "confidence": result.confidence,
            "top_retrieval_score": result.top_retrieval_score,
            "oos_gated": result.out_of_scope_gated,
            "hallucinated_citations": result.hallucinated_citations,
            "retrieval_hit": ret_hit,
            "citation_hit": cite_hit,
            "elapsed_s": round(elapsed, 2),
            "usage": result.usage,
            "top_3_chunks": [
                f"{c['section_number']}{c.get('subsection') or ''}: {c.get('section_title', '')[:60]}"
                for c in result.retrieved_chunks[:3]
            ],
        }
        records.append(rec)

        status = "✓" if (cite_hit or is_oos_in_test and result.confidence == "out_of_scope") else "✗"
        print(f"  {status} conf={result.confidence} top_score={result.top_retrieval_score:.3f}"
              f" cited={result.cited_sections} expected={expected}"
              f" hallucinated={result.hallucinated_citations}")
        print(f"  took {elapsed:.2f}s")
    return records


def write_markdown_report(records: list[dict], path: str):
    in_scope = [r for r in records if not r.get("is_oos_in_test")]
    oos = [r for r in records if r.get("is_oos_in_test")]

    lines = ["# Test Run Report", ""]
    lines.append(f"Total questions: {len(records)}  ({len(in_scope)} in-scope, {len(oos)} OOS)")

    # Stats (ignore errored rows)
    valid = [r for r in records if "error" not in r]
    in_scope_valid = [r for r in valid if not r.get("is_oos_in_test")]

    retrieval_hit_count = sum(1 for r in in_scope_valid if r["retrieval_hit"])
    citation_hit_count = sum(1 for r in in_scope_valid if r["citation_hit"])
    hallucination_count = sum(1 for r in valid if r["hallucinated_citations"])
    oos_gated_correctly = sum(
        1 for r in oos if r["confidence"] == "out_of_scope"
    )
    total_input_tokens = sum((r.get("usage") or {}).get("input_tokens", 0) for r in valid)
    total_output_tokens = sum((r.get("usage") or {}).get("output_tokens", 0) for r in valid)
    total_cache_read = sum((r.get("usage") or {}).get("cache_read_input_tokens", 0) for r in valid)
    total_cache_create = sum((r.get("usage") or {}).get("cache_creation_input_tokens", 0) for r in valid)

    lines += [
        "",
        "## Summary",
        "",
        f"- **In-scope retrieval hit rate** (expected section in retrieved chunks): "
        f"**{retrieval_hit_count}/{len(in_scope_valid)}** "
        f"({100*retrieval_hit_count/max(len(in_scope_valid),1):.0f}%)",
        f"- **In-scope citation hit rate** (Claude cited an expected section): "
        f"**{citation_hit_count}/{len(in_scope_valid)}** "
        f"({100*citation_hit_count/max(len(in_scope_valid),1):.0f}%)",
        f"- **Hallucinated citations** (cited a section not in retrieved chunks): **{hallucination_count}**",
        f"- **OOS handled correctly** (confidence=out_of_scope on OOS questions): "
        f"**{oos_gated_correctly}/{len(oos)}**",
        "",
        "### Token usage",
        f"- input tokens: {total_input_tokens:,}",
        f"- output tokens: {total_output_tokens:,}",
        f"- cache reads: {total_cache_read:,}",
        f"- cache writes: {total_cache_create:,}",
        "",
        "---",
        "",
        "## In-scope questions",
        "",
    ]

    for r in in_scope:
        lines.append(f"### Q{r['question_id']} [{r.get('difficulty', '?')}] — {r['question']}")
        lines.append("")
        if "error" in r:
            lines.append(f"**ERROR:** `{r['error']}`")
            lines.append("")
            continue
        lines.append(f"- expected sections: `{r['expected_sections']}`")
        lines.append(f"- cited sections: `{r['cited_sections']}`")
        lines.append(f"- retrieval hit: **{r['retrieval_hit']}**   citation hit: **{r['citation_hit']}**")
        lines.append(f"- hallucinated citations: `{r['hallucinated_citations']}`")
        lines.append(f"- confidence: `{r['confidence']}`   top-1 score: `{r['top_retrieval_score']:.3f}`")
        lines.append(f"- retrieved top-3: `{r['top_3_chunks']}`")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append("> " + r["answer"].replace("\n", "\n> "))
        lines.append("")
        lines.append("---")
        lines.append("")

    lines += ["## Out-of-scope questions", ""]
    for r in oos:
        lines.append(f"### Q{r['question_id']} — {r['question']}")
        lines.append("")
        if "error" in r:
            lines.append(f"**ERROR:** `{r['error']}`")
            continue
        lines.append(f"- confidence: `{r['confidence']}`   top-1 score: `{r['top_retrieval_score']:.3f}`   "
                     f"oos_gated: `{r['oos_gated']}`")
        lines.append(f"- cited sections: `{r['cited_sections']}` (should be empty)")
        lines.append(f"- hallucinated citations: `{r['hallucinated_citations']}` (should be empty)")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append("> " + r["answer"].replace("\n", "\n> "))
        lines.append("")
        lines.append("---")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N questions (for debugging).")
    parser.add_argument("--mode", choices=["homeowner", "builder"], default="homeowner")
    parser.add_argument("--out", default="test_report.md")
    parser.add_argument("--json-out", default="test_report.json")
    args = parser.parse_args()

    records = run_all(limit=args.limit, mode=args.mode)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.json_out}")

    write_markdown_report(records, args.out)


if __name__ == "__main__":
    main()
