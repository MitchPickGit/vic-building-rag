"""
Evaluate retrieval quality on the 30-question Act test set.

Reports hit@1 / hit@3 / hit@5 for three configurations:
  - BM25-only      (lexical baseline)
  - Vector-only    (voyage-3 embeddings, cosine)
  - Hybrid (RRF)   (BM25 ∪ vector, fused by rank)

Also spots misses so you can see what's still slipping through.

The test set is about the Building Act 1993. To keep the eval honest, we
restrict the corpus to Act chunks only (not Act + Regulations) so a
"close-but-wrong" hit on a Regulation doesn't get counted as success.
"""

import json
import re
import sys

from lib.retrieval import BM25Retriever, VectorRetriever, HybridRetriever


ACT_CHUNKS_PATH = "building_act_chunks.jsonl"
QUESTIONS_PATH = "test_questions.json"


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def section_key(c):
    if c.get("subsection"):
        return f"{c['section_number']}{c['subsection']}"
    return c["section_number"]


def section_matches(chunk, expected_list):
    chunk_sec = chunk["section_number"]
    chunk_full = section_key(chunk)
    for exp in expected_list:
        if exp == chunk_sec or exp == chunk_full:
            return True
        m = re.match(r"^(\d+[A-Z]*)", exp)
        if m and m.group(1) == chunk_sec:
            return True
    return False


def evaluate_retriever(name, retriever, in_scope, out_scope):
    results = []
    for q in in_scope:
        hits = retriever.search(q["q"], top_k=10)
        top_chunks = [c for _, c in hits]
        hit1 = section_matches(top_chunks[0], q["expected_sections"]) if top_chunks else False
        hit3 = any(section_matches(c, q["expected_sections"]) for c in top_chunks[:3])
        hit5 = any(section_matches(c, q["expected_sections"]) for c in top_chunks[:5])

        first_hit_rank = None
        for rank, c in enumerate(top_chunks, start=1):
            if section_matches(c, q["expected_sections"]):
                first_hit_rank = rank
                break
        results.append({
            "q": q, "hit_at_1": hit1, "hit_at_3": hit3, "hit_at_5": hit5,
            "first_hit_rank": first_hit_rank,
            "top_3": [(section_key(c), c["section_title"]) for c in top_chunks[:3]],
        })
    total = len(results)
    h1 = sum(r["hit_at_1"] for r in results) / total
    h3 = sum(r["hit_at_3"] for r in results) / total
    h5 = sum(r["hit_at_5"] for r in results) / total

    # Out-of-scope: we want LOW retrieval scores so the answer-generation
    # layer can say "I don't know". Report the top-1 score of each.
    oos_scores = []
    for q in out_scope:
        hits = retriever.search(q["q"], top_k=1)
        if hits:
            oos_scores.append((q["id"], hits[0][0]))

    return {
        "name": name,
        "total": total,
        "hit_at_1": h1, "hit_at_3": h3, "hit_at_5": h5,
        "results": results,
        "oos_top1_scores": oos_scores,
    }


def print_summary(report):
    name = report["name"]
    n = report["total"]
    print(f"\n  {name:<14}  "
          f"hit@1 {report['hit_at_1']:.0%}  "
          f"hit@3 {report['hit_at_3']:.0%}  "
          f"hit@5 {report['hit_at_5']:.0%}  "
          f"({int(report['hit_at_1']*n)}/{int(report['hit_at_3']*n)}/{int(report['hit_at_5']*n)} of {n})")


def print_misses(report, label):
    print(f"\n--- {label}: misses at top-3 ---")
    for r in report["results"]:
        if not r["hit_at_3"]:
            q = r["q"]
            print(f"\n  Q{q['id']} [{q['difficulty']}] {q['q']}")
            print(f"    expected: {q['expected_sections']}")
            print(f"    got top-3: {[t[0] for t in r['top_3']]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    chunks = load_chunks(ACT_CHUNKS_PATH)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        qdata = json.load(f)
    questions = qdata["questions"]
    in_scope = [q for q in questions if not q.get("out_of_scope")]
    out_scope = [q for q in questions if q.get("out_of_scope")]

    print(f"Corpus: {len(chunks)} Act chunks")
    print(f"Test set: {len(in_scope)} in-scope, {len(out_scope)} out-of-scope")

    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)
    hybrid = HybridRetriever(bm25, vector)

    reports = [
        evaluate_retriever("BM25", bm25, in_scope, out_scope),
        evaluate_retriever("Vector", vector, in_scope, out_scope),
        evaluate_retriever("Hybrid (RRF)", hybrid, in_scope, out_scope),
    ]

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)
    for r in reports:
        print_summary(r)

    print("\n" + "=" * 80)
    print("BY DIFFICULTY (hit@3)")
    print("=" * 80)
    print(f"\n  {'':<14}{'easy':>10}{'medium':>10}{'hard':>10}")
    for r in reports:
        cells = [f"  {r['name']:<14}"]
        for diff in ["easy", "medium", "hard"]:
            subset = [x for x in r["results"] if x["q"]["difficulty"] == diff]
            if subset:
                rate = sum(x["hit_at_3"] for x in subset) / len(subset)
                cells.append(f"{rate:>9.0%} ")
            else:
                cells.append(f"{'—':>9} ")
        print("".join(cells))

    # Show hybrid misses for debugging
    hybrid_report = reports[2]
    print_misses(hybrid_report, "Hybrid (RRF)")

    # Out-of-scope scores
    print("\n" + "=" * 80)
    print("OUT-OF-SCOPE TOP-1 SCORES (lower is better: gate for \"I don't know\")")
    print("=" * 80)
    for r in reports:
        print(f"\n  {r['name']}:")
        for qid, score in r["oos_top1_scores"]:
            print(f"    Q{qid}: top-1 score = {score:.3f}")


if __name__ == "__main__":
    main()
