"""
NCC retrieval evaluation — RETRIEVAL ONLY, NO CLAUDE CALLS.

Mirrors what evaluate.py does for the Building Act, but for the NCC.
Pure embedding lookup + cross-encoder rerank against the full 3-doc
corpus (Act + Regs + NCC). Reports hit@1/3/5/10 specifically for NCC
provisions.

Cost: $0 in Anthropic, ~$0 in Voyage (queries are tiny — covered by
Voyage's free tier even after the embedding bill).
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from lib.retrieval import VectorRetriever, BM25Retriever, load_all_chunks
from lib.reranker import rerank_chunks


CHUNK_PATHS = [
    "building_act_chunks.jsonl",
    "building_regs_chunks.jsonl",
    "ncc_chunks.jsonl",
]
QUESTIONS_PATH = "ncc_test_questions.json"

CANDIDATE_POOL = 20      # vector top-K before rerank
FINAL_TOP_N    = 10      # reranked top-K we evaluate hit@1/3/5/10 against


def chunk_provision_id(c: dict) -> str | None:
    """Return the NCC provision id for a chunk, or None if it's not NCC."""
    if c.get("doc_type") != "ncc":
        return None
    sn = c.get("section_number") or ""
    sub = c.get("subsection") or ""
    return f"{sn}{sub}" if sub else sn


def matches_expected(provision_id: str, expected_list: list[str]) -> bool:
    """Did this provision_id match any of the expected provisions?
    Match if exact, or if expected is a parent (e.g. expected 'H4P2'
    matches retrieved 'H4P2(1)')."""
    pid = provision_id.strip()
    for exp in expected_list:
        e = exp.strip()
        if pid == e:
            return True
        # Parent match: expected 'H4P2' matches retrieved 'H4P2(1)' — strip subprov from pid
        # and compare. The parent ID is pid up to the first '('.
        bare = pid.split("(", 1)[0]
        if bare == e:
            return True
        # Also accept if pid is the parent of expected ('H6P1' retrieved when 'H6P1(1)' expected)
        bare_e = e.split("(", 1)[0]
        if pid == bare_e:
            return True
    return False


def evaluate(retrieve_fn, name: str, questions: list[dict]) -> dict:
    """Run `retrieve_fn(query) -> [(score, chunk), ...]` for each question
    and tally hit rates against expected NCC provisions."""
    results = []
    for q in questions:
        hits = retrieve_fn(q["q"])
        # Pull out the NCC provision IDs from the top results, in order
        ncc_ids = []
        for _, c in hits:
            pid = chunk_provision_id(c)
            ncc_ids.append(pid)  # None for non-NCC retrievals — kept for indexing parity

        # Did any expected provision appear at rank 1 / top-3 / top-5 / top-10?
        def hit_at(k: int) -> bool:
            for pid in ncc_ids[:k]:
                if pid and matches_expected(pid, q["expected_provisions"]):
                    return True
            return False

        first_hit_rank = None
        for rank, pid in enumerate(ncc_ids, start=1):
            if pid and matches_expected(pid, q["expected_provisions"]):
                first_hit_rank = rank
                break

        results.append({
            "id": q["id"],
            "question": q["q"],
            "topic": q.get("topic"),
            "difficulty": q.get("difficulty"),
            "expected": q["expected_provisions"],
            "hit_at_1": hit_at(1),
            "hit_at_3": hit_at(3),
            "hit_at_5": hit_at(5),
            "hit_at_10": hit_at(10),
            "first_hit_rank": first_hit_rank,
            "top_3": [
                f"{pid}" if pid else f"({c.get('citation', '?')})"
                for pid, (_, c) in zip(ncc_ids[:3], hits[:3])
            ],
        })
    n = len(results)
    return {
        "name": name,
        "total": n,
        "hit_at_1": sum(r["hit_at_1"] for r in results) / n,
        "hit_at_3": sum(r["hit_at_3"] for r in results) / n,
        "hit_at_5": sum(r["hit_at_5"] for r in results) / n,
        "hit_at_10": sum(r["hit_at_10"] for r in results) / n,
        "results": results,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    chunks = load_all_chunks(CHUNK_PATHS)
    print(f"Corpus: {len(chunks):,} chunks  "
          f"(Act + Regs + NCC)")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        qdata = json.load(f)
    questions = qdata["questions"]
    print(f"NCC test set: {len(questions)} questions\n")

    bm25   = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)

    def bm25_retrieve(q: str):
        return bm25.search(q, top_k=FINAL_TOP_N)

    def vector_retrieve(q: str):
        return vector.search(q, top_k=FINAL_TOP_N)

    def vector_plus_rerank(q: str):
        # Same shape as the deployed retrieval: vector top-K (large pool) →
        # cross-encoder rerank → top final_top_n.
        candidates = [c for _, c in vector.search(q, top_k=CANDIDATE_POOL)]
        return rerank_chunks(q, candidates, top_k=FINAL_TOP_N)

    reports = [
        evaluate(bm25_retrieve,        "BM25",                     questions),
        evaluate(vector_retrieve,      "Vector (voyage-3)",        questions),
        evaluate(vector_plus_rerank,   "Vector + rerank-2 (live)", questions),
    ]

    # ----------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------
    print("=" * 80)
    print("NCC RETRIEVAL EVALUATION  (no Claude calls — vector + rerank only)")
    print("=" * 80)
    for r in reports:
        n = r["total"]
        print(f"\n  {r['name']:<28}  "
              f"hit@1 {r['hit_at_1']:.0%}  "
              f"hit@3 {r['hit_at_3']:.0%}  "
              f"hit@5 {r['hit_at_5']:.0%}  "
              f"hit@10 {r['hit_at_10']:.0%}  "
              f"({int(r['hit_at_1']*n)}/{int(r['hit_at_3']*n)}/"
              f"{int(r['hit_at_5']*n)}/{int(r['hit_at_10']*n)} of {n})")

    # By difficulty
    print("\n" + "=" * 80)
    print("BY DIFFICULTY (live retrieval — vector + rerank)")
    print("=" * 80)
    live = reports[2]["results"]
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in live if r["difficulty"] == diff]
        if not subset:
            continue
        n = len(subset)
        h3 = sum(r["hit_at_3"] for r in subset) / n
        h5 = sum(r["hit_at_5"] for r in subset) / n
        print(f"  {diff:>6}  "
              f"hit@3 {h3:.0%}  hit@5 {h5:.0%}  ({n} qns)")

    # Misses
    print("\n" + "=" * 80)
    print("LIVE RETRIEVAL — misses at top-5")
    print("=" * 80)
    for r in live:
        if not r["hit_at_5"]:
            print(f"\n  {r['id']} [{r['difficulty']}] {r['question']}")
            print(f"    expected: {r['expected']}")
            print(f"    got top-3 NCC IDs: {[t for t in r['top_3'] if t]}")
            if r["first_hit_rank"]:
                print(f"    first hit rank: {r['first_hit_rank']}")
            else:
                print(f"    (expected not in top-{FINAL_TOP_N})")


if __name__ == "__main__":
    main()
