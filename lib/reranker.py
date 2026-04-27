"""
Cross-encoder reranker for the retrieval stack.

Vector retrieval is fast and broad but treats query+document as separate
embeddings (bi-encoder). It misses fine-grained relevance signals — e.g.
which of two similar provisions actually answers the user's question. A
cross-encoder reranker reads (query, document) jointly and produces a
sharper relevance score.

We use Voyage's rerank-2 because the project already has a Voyage key
and billing set up. Cohere Rerank or any other cross-encoder would slot
into the same place.

Pipeline position:
    user question
        ↓
    rewrite into N queries (Haiku)
        ↓
    vector search top ~25 unique candidates (per query, then union)
        ↓
    rerank against the original user question  ← THIS MODULE
        ↓
    top 12 → Claude
"""

from __future__ import annotations

import os
from typing import Optional

import voyageai
from dotenv import load_dotenv


load_dotenv(override=True)

RERANK_MODEL = "rerank-2"


def _chunk_to_rerank_doc(c: dict) -> str:
    """Format a chunk as a single string for the reranker input. Include
    the bits a cross-encoder needs to judge relevance: citation header,
    section title, body, and the optional note/exempted_regs fields."""
    parts = [c["citation"]]
    if c.get("section_title"):
        parts.append(c["section_title"])
    if c.get("part"):
        parts.append(c["part"])
    if c.get("text"):
        parts.append(c["text"])
    if c.get("note"):
        parts.append(f"Note: {c['note']}")
    if c.get("exempted_regulations"):
        parts.append(f"Exempted: {c['exempted_regulations']}")
    return "\n".join(parts)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int,
    model: str = RERANK_MODEL,
    client: Optional[voyageai.Client] = None,
) -> list[tuple[float, dict]]:
    """Rerank `chunks` against `query` using a cross-encoder.

    Returns (relevance_score, chunk) pairs ordered best-first, length up
    to top_k.
    """
    if not chunks:
        return []
    if top_k >= len(chunks):
        # Reranker would just sort the same set — cheaper to skip the API call
        # if we wanted, but the rerank order is still useful (different from
        # vector order). Keep the call.
        pass

    if client is None:
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY missing — put it in .env")
        client = voyageai.Client()

    documents = [_chunk_to_rerank_doc(c) for c in chunks]
    resp = client.rerank(
        query=query,
        documents=documents,
        model=model,
        top_k=min(top_k, len(chunks)),
        truncation=True,
    )

    # resp.results is a list of objects with .index and .relevance_score
    out: list[tuple[float, dict]] = []
    for r in resp.results:
        out.append((float(r.relevance_score), chunks[r.index]))
    return out
