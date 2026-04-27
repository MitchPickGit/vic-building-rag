"""
Retrieval primitives for the legislation corpus.

Three retrievers:
  - BM25Retriever        : lexical, no external dependencies
  - VectorRetriever      : dense, loads the embeddings.npy cache
  - HybridRetriever      : BM25 + vector fused with Reciprocal Rank Fusion

Each retriever's `.search(query, top_k)` returns a list of
(score, chunk) tuples ordered best-first. The corpus is passed in at
construction, which lets callers filter (e.g. to Act-only chunks for
evaluation against the existing Act-only test set).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_all_chunks(paths: Iterable[str]) -> list[dict]:
    chunks = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        docs = []
        for c in chunks:
            search_text = " ".join(filter(None, [
                c.get("section_title"),
                c.get("division"),
                c.get("text"),
                c.get("penalty"),
                c.get("note"),
                c.get("exempted_regulations"),
                # NCC-specific structural fields
                c.get("part"),
                c.get("subpart"),
                c.get("specification"),
                c.get("schedule"),
            ]))
            docs.append(tokenize(search_text))
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.df = Counter()
        for d in docs:
            for t in set(d):
                self.df[t] += 1

    def _score(self, qt: list[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        tf = Counter(doc)
        dl = len(doc)
        s = 0.0
        for q in qt:
            if q not in self.df:
                continue
            idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
            f = tf.get(q, 0)
            s += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, dict]]:
        qt = tokenize(query)
        scores = [(self._score(qt, i), i) for i in range(self.N)]
        scores.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scores[:top_k]]


# ---------------------------------------------------------------------------
# Vector (Voyage)
# ---------------------------------------------------------------------------

class VectorRetriever:
    """Dense retriever backed by the embeddings.npy + embeddings_cache.jsonl
    produced by embed_chunks.py. Queries are embedded live via the Voyage
    client (cheap — a query is ~30 tokens)."""

    def __init__(self, chunks: list[dict],
                 embeddings_path: str = "embeddings.npy",
                 cache_path: str = "embeddings_cache.jsonl",
                 voyage_model: str = "voyage-3"):
        load_dotenv(override=True)
        import voyageai

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY missing — put it in .env")

        self._voyage = voyageai.Client(api_key=api_key)
        self._model = voyage_model

        matrix = np.load(embeddings_path)
        with open(cache_path, encoding="utf-8") as f:
            cache_entries = [json.loads(line) for line in f]
        assert matrix.shape[0] == len(cache_entries), (
            "embeddings.npy and embeddings_cache.jsonl out of sync"
        )
        cache_row_by_citation = {
            e["citation"]: i for i, e in enumerate(cache_entries)
        }

        # Align the matrix with the caller's chunk list order. Some chunks
        # may lack an embedding (shouldn't happen if you re-run embed_chunks
        # after editing the corpus), in which case we log and skip.
        self.chunks = []
        rows = []
        missing = 0
        for c in chunks:
            row = cache_row_by_citation.get(c["citation"])
            if row is None:
                missing += 1
                continue
            self.chunks.append(c)
            rows.append(matrix[row])
        if missing:
            print(f"  warn: {missing} chunks have no embedding "
                  "(run embed_chunks.py)")
        self.matrix = np.stack(rows) if rows else np.zeros((0, matrix.shape[1]))
        # Pre-normalise for cosine similarity via dot product
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix_norm = self.matrix / norms

    def embed_query(self, query: str) -> np.ndarray:
        resp = self._voyage.embed(
            [query], model=self._model, input_type="query"
        )
        v = np.asarray(resp.embeddings[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, dict]]:
        if self.matrix_norm.shape[0] == 0:
            return []
        qv = self.embed_query(query)
        sims = self.matrix_norm @ qv
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.chunks[i]) for i in idx]


# ---------------------------------------------------------------------------
# Hybrid (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """BM25 + vector, fused via Reciprocal Rank Fusion (RRF).

    RRF score for a chunk = sum over retrievers of 1/(k + rank_in_retriever).
    Chunks appearing high in either list win; k=60 is the standard value
    from the original RRF paper (Cormack et al. 2009) and is also what
    Elastic's default hybrid ranker uses.
    """

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever,
                 k_rrf: int = 60, fusion_pool: int = 20):
        self.bm25 = bm25
        self.vector = vector
        self.k_rrf = k_rrf
        self.fusion_pool = fusion_pool

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, dict]]:
        bm25_hits = self.bm25.search(query, top_k=self.fusion_pool)
        vec_hits = self.vector.search(query, top_k=self.fusion_pool)

        # RRF score = sum(1 / (k + rank)) across retrievers.
        # Key each hit by citation (stable across runs).
        rrf = {}
        chunks_by_cit = {}
        for rank, (_, c) in enumerate(bm25_hits, start=1):
            cit = c["citation"]
            rrf[cit] = rrf.get(cit, 0.0) + 1.0 / (self.k_rrf + rank)
            chunks_by_cit[cit] = c
        for rank, (_, c) in enumerate(vec_hits, start=1):
            cit = c["citation"]
            rrf[cit] = rrf.get(cit, 0.0) + 1.0 / (self.k_rrf + rank)
            chunks_by_cit[cit] = c

        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
        return [(score, chunks_by_cit[cit]) for cit, score in ranked[:top_k]]
