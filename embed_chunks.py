"""
Embed all chunks (Act + Regulations) using Voyage AI and cache locally.

Writes two files:
  - embeddings_cache.jsonl  : chunk_id -> JSON metadata + structural prefix
  - embeddings.npy          : float32 matrix, row i = embedding for chunk i

Re-running is idempotent: chunks whose hashed (structural_prefix + text)
already appear in the cache are skipped. Only new/changed chunks hit
the API.

Embedding input per chunk is the structural prefix + text, e.g.:

    Building Act 1993 (Vic) s 16(1) — Offences relating to carrying
    out building work — Part 3, Division 1: A person must not carry
    out building work unless a building permit …

Prefix gives the embedder anchor terms ("Part 3", "occupancy permit",
section title) even when the clause body uses very different wording.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import voyageai


load_dotenv()

CHUNK_FILES = [
    "building_act_chunks.jsonl",
    "building_regs_chunks.jsonl",
]
CACHE_JSONL = "embeddings_cache.jsonl"
EMBEDDINGS_NPY = "embeddings.npy"

# voyage-3 is the general-purpose model. 1024-dim, ~$0.06/M tokens or
# 50M free. voyage-3-large is higher quality but costs more and we
# don't need it for 1000 chunks.
VOYAGE_MODEL = "voyage-3"
EMBED_INPUT_TYPE = "document"
QUERY_INPUT_TYPE = "query"
BATCH_SIZE = 64  # Voyage accepts up to 128 inputs per request

EMBEDDING_DIM = 1024  # for voyage-3


def load_chunks():
    chunks = []
    for path in CHUNK_FILES:
        if not Path(path).exists():
            print(f"  warn: {path} missing — skipping", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                chunks.append(c)
    return chunks


def structural_prefix(c):
    """Compact header string that anchors the embedding with metadata
    the body text might not mention explicitly (Part, Division, title)."""
    parts = []
    parts.append(c["citation"])
    if c.get("section_title"):
        parts.append(c["section_title"])
    ctx = []
    if c.get("part"):
        ctx.append(c["part"])
    if c.get("division"):
        ctx.append(c["division"])
    if c.get("subdivision"):
        ctx.append(c["subdivision"])
    if c.get("schedule"):
        ctx.append(c["schedule"])
    if ctx:
        parts.append(" | ".join(ctx))
    return " — ".join(parts)


def embed_input(c):
    """Full string passed to the embedding API for one chunk."""
    lines = [structural_prefix(c)]
    if c.get("text"):
        lines.append(c["text"])
    if c.get("note"):
        lines.append(f"Note: {c['note']}")
    if c.get("exempted_regulations"):
        lines.append(f"Exempted: {c['exempted_regulations']}")
    return "\n".join(lines)


def chunk_id(c):
    """Stable identifier for a chunk across runs."""
    return c["citation"]


def cache_key(text):
    """Hash the embedding input so we can detect when to re-embed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_cache():
    """Return {chunk_id: {"key": hash, "row": i}} and the existing matrix."""
    index = {}
    matrix = None
    if Path(CACHE_JSONL).exists() and Path(EMBEDDINGS_NPY).exists():
        with open(CACHE_JSONL, encoding="utf-8") as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                index[entry["chunk_id"]] = {"key": entry["key"], "row": i}
        matrix = np.load(EMBEDDINGS_NPY)
        assert matrix.shape[0] == len(index), (
            f"cache desync: {matrix.shape[0]} vectors vs {len(index)} ids"
        )
    return index, matrix


def embed_batch(client, texts):
    """Call Voyage with retry on transient failures."""
    for attempt in range(3):
        try:
            resp = client.embed(
                texts,
                model=VOYAGE_MODEL,
                input_type=EMBED_INPUT_TYPE,
            )
            return resp.embeddings
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"  warn: {e!r}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set. Add it to .env.", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(CHUNK_FILES)} files")

    existing_index, existing_matrix = load_cache()
    print(f"Cache has {len(existing_index)} entries")

    # Plan: build the new ordered chunk list, figure out which need embedding.
    to_embed = []          # list of (idx, input_text)
    new_entries = []       # list of dicts matching chunks list order
    for i, c in enumerate(chunks):
        cid = chunk_id(c)
        inp = embed_input(c)
        key = cache_key(inp)
        entry = {"chunk_id": cid, "key": key, "citation": c["citation"]}
        new_entries.append(entry)
        cached = existing_index.get(cid)
        if not cached or cached["key"] != key:
            to_embed.append((i, inp))

    print(f"  {len(to_embed)} chunks need embedding, "
          f"{len(chunks) - len(to_embed)} cache hits")

    if not to_embed:
        print("Nothing to do — cache is up to date.")
        return

    # Allocate final matrix, fill cache hits first
    matrix = np.zeros((len(chunks), EMBEDDING_DIM), dtype=np.float32)
    for i, c in enumerate(chunks):
        cid = chunk_id(c)
        cached = existing_index.get(cid)
        if cached and cached["key"] == new_entries[i]["key"]:
            matrix[i] = existing_matrix[cached["row"]]

    client = voyageai.Client(api_key=api_key)

    # Embed in batches, filling in the remaining rows
    t0 = time.time()
    for batch_start in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[batch_start : batch_start + BATCH_SIZE]
        indices = [i for i, _ in batch]
        texts = [t for _, t in batch]
        vecs = embed_batch(client, texts)
        for idx, vec in zip(indices, vecs):
            matrix[idx] = np.asarray(vec, dtype=np.float32)
        done = batch_start + len(batch)
        elapsed = time.time() - t0
        print(f"  embedded {done}/{len(to_embed)} "
              f"(elapsed {elapsed:.1f}s)")

    # Save atomically
    np.save(EMBEDDINGS_NPY, matrix)
    with open(CACHE_JSONL, "w", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {matrix.shape[0]} vectors (dim {matrix.shape[1]}) "
          f"→ {EMBEDDINGS_NPY}")
    print(f"Wrote cache manifest → {CACHE_JSONL}")


if __name__ == "__main__":
    main()
