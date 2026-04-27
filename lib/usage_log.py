"""
Lightweight SQLite-backed query log + feedback collection.

The point: turn builder testing from vibes into data. Every question
gets logged with retrieval/answer metadata; every answer gets a
👍/👎/comment widget that wires back to the same database. An admin
view lets you inspect or export the whole table.

Caveat: Streamlit Cloud's filesystem is ephemeral — the SQLite file
is wiped on container restart (which happens after idle timeout, on
redeploy, or randomly). For long-running data collection, swap this
out for a hosted DB (Supabase / Neon / a Google Sheet). For prototype
testing with a handful of builders the in-container DB is fine if you
download it periodically via the admin view.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DB_PATH = "usage_log.db"
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def _connect():
    """Connection context manager with thread-safe write lock. SQLite
    handles single-process concurrency fine; the lock here is belt-
    and-braces against Streamlit's worker threads."""
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db():
    """Create tables if they don't exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                session_id TEXT,
                mode TEXT,
                question TEXT NOT NULL,
                cited_sections TEXT,
                hallucinated_count INTEGER DEFAULT 0,
                hallucinated_citations TEXT,
                confidence TEXT,
                top_retrieval_score REAL,
                top_rerank_score REAL,
                query_shape TEXT,
                rewritten_queries TEXT,
                retrieved_chunk_count INTEGER,
                elapsed_s REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_input_tokens INTEGER,
                cache_creation_input_tokens INTEGER,
                error TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_log_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                FOREIGN KEY (query_log_id) REFERENCES query_log (id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_ts ON query_log(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_log_id)"
        )


def log_query(
    *,
    session_id: Optional[str],
    mode: str,
    question: str,
    cited_sections: list[str] | None = None,
    hallucinated_citations: list[str] | None = None,
    confidence: Optional[str] = None,
    top_retrieval_score: Optional[float] = None,
    top_rerank_score: Optional[float] = None,
    query_shape: Optional[str] = None,
    rewritten_queries: list[str] | None = None,
    retrieved_chunk_count: Optional[int] = None,
    elapsed_s: Optional[float] = None,
    usage: Optional[dict] = None,
    error: Optional[str] = None,
) -> int:
    """Insert a query record and return its row id (use this id when
    logging feedback for the same query)."""
    init_db()
    usage = usage or {}
    cited_str = json.dumps(cited_sections or [])
    hallu_str = json.dumps(hallucinated_citations or [])
    rew_str = json.dumps(rewritten_queries or [])
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO query_log (
                ts, session_id, mode, question,
                cited_sections, hallucinated_count, hallucinated_citations,
                confidence, top_retrieval_score, top_rerank_score,
                query_shape, rewritten_queries, retrieved_chunk_count,
                elapsed_s, input_tokens, output_tokens,
                cache_read_input_tokens, cache_creation_input_tokens, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_iso(),
                session_id,
                mode,
                question,
                cited_str,
                len(hallucinated_citations or []),
                hallu_str,
                confidence,
                top_retrieval_score,
                top_rerank_score,
                query_shape,
                rew_str,
                retrieved_chunk_count,
                elapsed_s,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("cache_read_input_tokens"),
                usage.get("cache_creation_input_tokens"),
                error,
            ),
        )
        return int(cur.lastrowid)


def log_feedback(query_log_id: int, rating: str, comment: Optional[str] = None) -> None:
    """Record a thumbs-up/down + optional comment for a logged query."""
    if rating not in ("up", "down"):
        raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback (query_log_id, ts, rating, comment)
            VALUES (?, ?, ?, ?)
            """,
            (query_log_id, _now_iso(), rating, comment or None),
        )


def fetch_recent(limit: int = 100) -> list[dict]:
    """Return the most recent N query records, joined with their
    feedback (if any)."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT q.*, GROUP_CONCAT(f.rating || ':' || COALESCE(f.comment, ''),
                                     '|') AS feedback
            FROM query_log q
            LEFT JOIN feedback f ON f.query_log_id = q.id
            GROUP BY q.id
            ORDER BY q.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def fetch_summary() -> dict:
    """Quick stats for the admin view header."""
    init_db()
    with _connect() as conn:
        total_queries = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        total_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        ups = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        downs = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
        hallucinations = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE hallucinated_count > 0"
        ).fetchone()[0]
        errored = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE error IS NOT NULL"
        ).fetchone()[0]
        # Token aggregates
        agg = conn.execute(
            """
            SELECT
                SUM(input_tokens) AS in_tok,
                SUM(output_tokens) AS out_tok,
                SUM(cache_read_input_tokens) AS cr_tok
            FROM query_log
            """
        ).fetchone()
    return {
        "total_queries": total_queries,
        "total_feedback": total_feedback,
        "thumbs_up": ups,
        "thumbs_down": downs,
        "queries_with_hallucination": hallucinations,
        "errored_queries": errored,
        "input_tokens": agg["in_tok"] or 0,
        "output_tokens": agg["out_tok"] or 0,
        "cache_read_tokens": agg["cr_tok"] or 0,
    }


def export_jsonl() -> str:
    """Return all queries + feedback as JSONL (one JSON object per line).
    Used by the admin 'Download all' button."""
    init_db()
    with _connect() as conn:
        queries = conn.execute(
            "SELECT * FROM query_log ORDER BY id"
        ).fetchall()
        feedback_rows = conn.execute(
            "SELECT * FROM feedback ORDER BY id"
        ).fetchall()
    fb_by_query: dict[int, list[dict]] = {}
    for f in feedback_rows:
        fb_by_query.setdefault(f["query_log_id"], []).append(dict(f))
    out_lines = []
    for q in queries:
        record = dict(q)
        # Decode embedded JSON columns
        for k in ("cited_sections", "hallucinated_citations", "rewritten_queries"):
            try:
                record[k] = json.loads(record[k]) if record.get(k) else []
            except Exception:
                pass
        record["feedback"] = fb_by_query.get(q["id"], [])
        out_lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(out_lines)
