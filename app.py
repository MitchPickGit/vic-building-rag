"""
Streamlit chat UI for the Victorian Building Legislation RAG.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. https://share.streamlit.io → New app → point at repo + app.py
    3. Add VOYAGE_API_KEY and ANTHROPIC_API_KEY in the Secrets panel
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime

import streamlit as st


# Streamlit secrets get bridged into env vars BEFORE we import the libs that
# read os.environ. Locally, .env still works via python-dotenv.
def _bridge_streamlit_secrets():
    try:
        for key in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        # No secrets file locally — rely on .env / env vars
        pass


_bridge_streamlit_secrets()

from lib.retrieval import VectorRetriever, load_all_chunks
from lib.answer import answer_question, AnswerTruncated
from lib.citation_graph import CitationGraph
from lib.permit_classifier import classify as classify_permits
from lib import usage_log


# ---------------------------------------------------------------------------
# Tunables (Stage 1 hardening)
# ---------------------------------------------------------------------------

# Max questions a single browser session can ask per hour. Belt-and-braces
# against accidental burn from one user sitting on a bug.
RATE_LIMIT_PER_HOUR = 30


def _get_secret(key: str) -> str | None:
    """Read a key from Streamlit secrets if available, else env."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


# ---------------------------------------------------------------------------
# Auth gate — single shared password
# ---------------------------------------------------------------------------

def _check_password() -> bool:
    """Block access until the visitor enters APP_PASSWORD. If no
    APP_PASSWORD is configured, the gate is open (useful locally)."""
    expected = _get_secret("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True

    st.markdown("# 🏗️ Victorian Building Legislation Q&A")
    st.info(
        "This is a private prototype shared with a small group of testers. "
        "Enter the access password to continue. (If you don't have one, "
        "ask the person who shared this URL with you.)"
    )
    pw = st.text_input("Access password", type="password",
                       label_visibility="collapsed",
                       placeholder="Access password")
    if pw and pw == expected:
        st.session_state.auth_ok = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    st.stop()


_check_password()


CORPUS_PATHS = [
    "building_act_chunks.jsonl",
    "building_regs_chunks.jsonl",
    "ncc_chunks.jsonl",
    "housing_provisions_chunks.jsonl",
]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vic Building Legislation Q&A",
    page_icon="🏗️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading legislation corpus...")
def get_retriever():
    chunks = load_all_chunks(CORPUS_PATHS)
    return VectorRetriever(chunks), chunks


@st.cache_resource(show_spinner="Loading citation graph...")
def get_citation_graph():
    try:
        return CitationGraph()
    except FileNotFoundError:
        # If the SQLite DB is missing on Streamlit Cloud, just disable the
        # graph features rather than crash the app.
        return None


retriever, all_chunks = get_retriever()
citation_graph = get_citation_graph()


# Stable per-browser-session id used for rate-limit + log correlation
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
# Per-session timestamp ring buffer for the rate limit
if "query_timestamps" not in st.session_state:
    st.session_state.query_timestamps = []


def _rate_limit_ok() -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds). Allowed = whether the user
    is under their hourly quota right now."""
    now = time.time()
    window = 3600
    recent = [t for t in st.session_state.query_timestamps if now - t < window]
    st.session_state.query_timestamps = recent
    if len(recent) >= RATE_LIMIT_PER_HOUR:
        oldest_recent = min(recent)
        return False, int(window - (now - oldest_recent))
    return True, 0


# ---------------------------------------------------------------------------
# Admin view — gated by a separate password from the main app
# ---------------------------------------------------------------------------

def _show_admin_view_if_requested() -> bool:
    """If the URL has ?admin=1 and the visitor enters ADMIN_PASSWORD, show
    the admin dashboard instead of the chat UI. Returns True if shown
    (caller should st.stop() afterwards)."""
    if st.query_params.get("admin") != "1":
        return False
    expected = _get_secret("ADMIN_PASSWORD") or _get_secret("APP_PASSWORD")
    if not expected:
        st.error("ADMIN_PASSWORD not configured.")
        return True
    if not st.session_state.get("admin_ok"):
        pw = st.text_input("Admin password", type="password")
        if pw == expected:
            st.session_state.admin_ok = True
            st.rerun()
        elif pw:
            st.error("Incorrect")
        return True

    st.title("📊 Admin — query log + feedback")
    st.caption(f"Session: `{st.session_state.session_id}`")

    summary = usage_log.fetch_summary()
    cols = st.columns(4)
    cols[0].metric("Total queries", summary["total_queries"])
    cols[1].metric("Feedback received", summary["total_feedback"])
    cols[2].metric("👍 / 👎",
                   f"{summary['thumbs_up']} / {summary['thumbs_down']}")
    cols[3].metric("Hallucination flags", summary["queries_with_hallucination"])

    cols2 = st.columns(3)
    cols2[0].metric("Input tokens", f"{summary['input_tokens']:,}")
    cols2[1].metric("Output tokens", f"{summary['output_tokens']:,}")
    cols2[2].metric("Cache reads", f"{summary['cache_read_tokens']:,}")

    st.divider()
    st.markdown("### 📥 Export")
    jsonl = usage_log.export_jsonl()
    st.download_button(
        "Download all queries + feedback (JSONL)",
        data=jsonl,
        file_name=f"usage_log_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl",
        mime="application/json",
        use_container_width=True,
    )
    st.caption(
        "⚠️ Streamlit Cloud's filesystem is ephemeral — download "
        "regularly. Container restart wipes the SQLite db."
    )

    st.divider()
    st.markdown("### Recent queries")
    recent = usage_log.fetch_recent(limit=50)
    if not recent:
        st.info("No queries logged yet.")
    for r in recent:
        title = (r.get("question") or "")[:80]
        with st.expander(
            f"Q{r['id']:>4}  [{r.get('mode','?')}]  {title}"
            f"  — {r.get('confidence','?')}"
            f"  ({r.get('top_rerank_score') or 0:.2f})"
        ):
            cited = r.get("cited_sections")
            try:
                cited = json.loads(cited) if cited else []
            except Exception:
                pass
            st.write(f"**Cited:** {cited}")
            hallu = r.get("hallucinated_citations")
            try:
                hallu = json.loads(hallu) if hallu else []
            except Exception:
                pass
            if hallu:
                st.error(f"⚠️ Hallucinated: {hallu}")
            st.caption(
                f"shape={r.get('query_shape')}  "
                f"cosine={r.get('top_retrieval_score') or 0:.3f}  "
                f"rerank={r.get('top_rerank_score') or 0:.3f}  "
                f"chunks={r.get('retrieved_chunk_count')}  "
                f"elapsed={r.get('elapsed_s') or 0:.1f}s  "
                f"in_tok={r.get('input_tokens')}  "
                f"out_tok={r.get('output_tokens')}"
            )
            if r.get("feedback"):
                st.write(f"**Feedback:** `{r['feedback']}`")
            if r.get("error"):
                st.error(f"Error: {r['error']}")
    return True


if _show_admin_view_if_requested():
    st.stop()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🏗️ Victorian Building Legislation Q&A")
st.caption(
    "Prototype Q&A over the Building Act 1993 (Vic) and Building Regulations 2018 (Vic). "
    f"Corpus: {len(all_chunks):,} indexed provisions."
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Who are you?",
        options=["homeowner", "builder", "surveyor"],
        format_func=lambda m: {
            "homeowner": "🏠 Homeowner — work done through a builder",
            "builder": "🔨 Builder / Owner-builder — concise, practical",
            "surveyor": "📐 Building Surveyor — full detail, deeper analysis",
        }[m],
        index=0,
        help=(
            "**Homeowner**: plain-English answers, redirects to surveyor / "
            "council / owners corp where relevant.\n\n"
            "**Builder / Owner-builder**: bottom-line answer + actionable "
            "steps + the detail, in that order. Covers both working builders "
            "and homeowners doing the work under an owner-builder cert. "
            "Faster (medium thinking effort) — designed to set expectations or "
            "to learn before talking to a surveyor.\n\n"
            "**Building Surveyor**: full technical detail, complete call-up "
            "chain, penalty tables, distinguishes Act/Regs/NCC/HP layers "
            "explicitly. Higher thinking effort — slower but more thorough. "
            "Use when you actually want to read the legislation."
        ),
    )

    st.divider()
    # Export conversation — useful for builders who want to send an answer
    # to a colleague or save an interesting thread before resetting.
    if st.session_state.get("history"):
        # Strip retrieved_chunks (large) before exporting; keep the answer
        # text + cited sections + key meta. Keeps the JSON readable.
        export_payload = {
            "session_id": st.session_state.get("session_id"),
            "exported_at_utc": datetime.utcnow().isoformat() + "Z",
            "mode": mode,
            "turns": [
                {
                    "role": t["role"],
                    "content": t["content"],
                    "cited_sections": (t.get("meta") or {}).get("cited_sections"),
                    "confidence": (t.get("meta") or {}).get("confidence"),
                    "elapsed_s": (t.get("meta") or {}).get("elapsed_s"),
                }
                for t in st.session_state.history
            ],
        }
        st.download_button(
            "💾 Export conversation",
            data=json.dumps(export_payload, indent=2, ensure_ascii=False),
            file_name=f"vic-building-rag_{st.session_state.get('session_id','')}_"
                      f"{datetime.utcnow():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
        )

    if st.button("🗑️ Reset conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("**Coverage**")
    st.caption("• Building Act 1993 — Authorised v146 (1 April 2026)")
    st.caption("• Building Regulations 2018 — Authorised v028 (26 November 2025)")
    st.caption("• Schedule 3 (exempt building work) — broken out per item")
    st.caption("• **NCC 2022 Volume Two** — Class 1 + 10 buildings (residential), incl. Vic Schedule 10 overrides")
    st.caption("• **ABCB Housing Provisions Standard 2022** — the technical detail that NCC Vol 2 defers to (stair geometry, R-values per climate zone, footing classifications, BAL distances, span tables)")
    st.caption("Schedules 1, 2, 4-13 of the Regulations and other NCC volumes are not yet indexed.")

    st.divider()
    st.caption("**About this prototype**")
    st.caption(
        "Answers come from indexed excerpts only. Every cited section number is "
        "verified against the retrieved passages — the system is designed never "
        "to invent a citation. If the answer isn't in the corpus, it will say so."
    )
    st.caption(
        "**Not legal advice.** Verify any answer against legislation.vic.gov.au "
        "or with a registered building surveyor before acting."
    )


# ---------------------------------------------------------------------------
# Render helper (defined before history replay below)
# ---------------------------------------------------------------------------


def _render_feedback_widget(query_log_id: int, key_suffix: str = "") -> None:
    """Lightweight 👍/👎 + comment widget. Renders inline. Calls
    usage_log.log_feedback when the user submits.

    Idempotent in session: once feedback is given for a query_log_id,
    we show a thank-you instead of the buttons (so users don't double-
    submit on rerender)."""
    submitted_key = f"fb_submitted_{query_log_id}"
    if st.session_state.get(submitted_key):
        st.caption(f"✓ Thanks for the feedback on this answer.")
        return

    st.caption("**Was this answer useful?**")
    col_up, col_down = st.columns(2)
    rating: str | None = None
    if col_up.button("👍 Helpful", key=f"up_{query_log_id}_{key_suffix}",
                     use_container_width=True):
        rating = "up"
    if col_down.button("👎 Not helpful", key=f"down_{query_log_id}_{key_suffix}",
                       use_container_width=True):
        rating = "down"

    # If a rating was clicked, capture an optional comment and log.
    if rating:
        st.session_state[f"fb_pending_{query_log_id}"] = rating
        st.rerun()

    pending = st.session_state.get(f"fb_pending_{query_log_id}")
    if pending:
        with st.form(key=f"fb_form_{query_log_id}_{key_suffix}",
                     clear_on_submit=True):
            st.caption(
                f"You picked **{'👍 Helpful' if pending == 'up' else '👎 Not helpful'}**. "
                f"Optional: tell us why (what was right or wrong about the answer)."
            )
            comment = st.text_area("Comment (optional)",
                                   key=f"fb_comment_{query_log_id}_{key_suffix}",
                                   label_visibility="collapsed")
            cols = st.columns(2)
            sent = cols[0].form_submit_button("Submit", use_container_width=True)
            cancel = cols[1].form_submit_button("Cancel", use_container_width=True)
            if sent:
                try:
                    usage_log.log_feedback(query_log_id, pending, comment or None)
                    st.session_state[submitted_key] = True
                    del st.session_state[f"fb_pending_{query_log_id}"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save feedback: {e!r}")
            elif cancel:
                del st.session_state[f"fb_pending_{query_log_id}"]
                st.rerun()


def _render_permit_notices(notices):
    """Render the dual-permit warning + per-regime notice cards."""
    st.warning(
        "⚠️ **Heads up — this question may also involve regulation outside "
        "the Building Act.** The answer below covers building-permit "
        "requirements only. The following parallel regimes may also apply:"
    )
    for n in notices:
        # `n` may be a PermitNotice dataclass (live) or a dict (replayed
        # from session_state). Handle both.
        headline = n.headline if hasattr(n, "headline") else n["headline"]
        body = n.body if hasattr(n, "body") else n["body"]
        with st.container(border=True):
            st.markdown(f"**{headline}**")
            st.caption(body)


def _render_meta(meta: dict):
    """Render the post-answer metadata (cited sections, retrieved chunks, etc.)."""
    is_clarification = meta.get("confidence") == "needs_clarification"

    cols = st.columns(3)
    cols[0].metric("Confidence", meta["confidence"])
    cols[1].metric(
        "Cited sections" if not is_clarification else "Citations",
        len(meta["cited_sections"]) if not is_clarification else "—",
    )
    cols[2].metric("Top retrieval score", f"{meta['top_retrieval_score']:.2f}")

    if meta["hallucinated_citations"]:
        st.error(
            f"⚠️ The model produced {len(meta['hallucinated_citations'])} citation(s) "
            f"that don't appear in the retrieved passages: "
            f"{', '.join(meta['hallucinated_citations'])}. "
            f"This is a known failure mode the system is designed to catch — do not "
            f"rely on those particular references."
        )

    if meta["cited_sections"]:
        with st.expander("📚 Cited provisions"):
            for sec in meta["cited_sections"]:
                st.markdown(f"- `{sec}`")
            st.caption(
                "Verify on the official source: "
                "[Building Act 1993](https://www.legislation.vic.gov.au/in-force/acts/building-act-1993/) "
                "• [Building Regulations 2018]"
                "(https://www.legislation.vic.gov.au/in-force/statutory-rules/building-regulations-2018)"
            )

    if meta.get("rewritten_queries") and len(meta["rewritten_queries"]) > 1:
        with st.expander(f"🔁 Search queries used ({len(meta['rewritten_queries'])})"):
            st.caption(
                "Your question was rewritten into multiple retrieval queries to "
                "improve coverage of the legislation. The model only sees the "
                "chunks retrieved by these queries — never the queries themselves."
            )
            for q in meta["rewritten_queries"]:
                st.markdown(f"- `{q}`")

    if meta.get("retrieved_chunks"):
        with st.expander(f"🔍 View {len(meta['retrieved_chunks'])} retrieved passages"):
            for i, c in enumerate(meta["retrieved_chunks"], 1):
                cite = c["citation"]
                title = c.get("section_title") or ""
                st.markdown(f"**{i}. {cite}** — {title}")
                if c.get("part"):
                    st.caption(f"{c['part']}")
                st.markdown(
                    f"> {c.get('text', '')[:500]}"
                    f"{'…' if len(c.get('text', '')) > 500 else ''}"
                )
                if c.get("note"):
                    st.caption(f"**Note:** {c['note'][:300]}")
                if c.get("penalty"):
                    st.caption(f"**Penalty:** {c['penalty']}")
                if c.get("amendment_history"):
                    with st.popover("Amendment history"):
                        st.caption(c["amendment_history"])
                st.divider()


# Chat history sits in session state so prior turns stay visible.
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, meta?}

# Replay history
for turn_idx, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        # Permit notices render at the TOP of an assistant turn, before
        # the answer text — so the warning is visible before the answer.
        if turn["role"] == "assistant" and turn.get("meta", {}).get("permit_notices"):
            _render_permit_notices(turn["meta"]["permit_notices"])
        st.markdown(turn["content"])
        if "meta" in turn:
            _render_meta(turn["meta"])
        # Replay-time feedback widget — needs a unique key per turn so
        # repeated rerenders don't collide.
        if turn["role"] == "assistant":
            qlid = (turn.get("meta") or {}).get("query_log_id")
            conf = (turn.get("meta") or {}).get("confidence")
            if qlid is not None and conf != "needs_clarification":
                _render_feedback_widget(qlid, key_suffix=f"replay_{turn_idx}")


# Suggested starter questions (only show when conversation is empty)
starter_clicked = None
if not st.session_state.history:
    st.markdown("**Try asking:**")
    starter = st.columns(2)
    starters = [
        "Do I need a permit for a pergola under 3.6m?",
        "What's the penalty for building without a permit?",
        "Can a building surveyor refuse to issue a permit?",
        "How do I register as a building practitioner?",
    ]
    for i, q in enumerate(starters):
        with starter[i % 2]:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                starter_clicked = q


# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask a question about the Building Act or Regulations...")
if not prompt and starter_clicked:
    prompt = starter_clicked

if prompt:
    # Rate limit check — kick in BEFORE rendering the user turn so a
    # blocked user doesn't see their question echoed (cleaner UX).
    allowed, retry_after = _rate_limit_ok()
    if not allowed:
        mins = max(1, retry_after // 60)
        st.warning(
            f"⏱️ You've reached this session's rate limit "
            f"({RATE_LIMIT_PER_HOUR} questions per hour). "
            f"Try again in about {mins} minute{'s' if mins != 1 else ''}."
        )
        st.stop()
    st.session_state.query_timestamps.append(time.time())

    # Render the user turn immediately (before the slow LLM call)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dual-permit classifier runs against the new question. Notices are
    # attached to the assistant turn's meta so they replay with the
    # transcript when the user scrolls back through history.
    permit_notices = classify_permits(prompt)

    # Build the conversation history that gets passed into answer_question.
    # Strip our local meta — only role + content go to the API.
    api_history = [
        {"role": t["role"], "content": t["content"]}
        for t in st.session_state.history[:-1]  # exclude the just-appended user turn
        if t.get("role") in ("user", "assistant")
    ]

    # Render assistant turn with spinner
    with st.chat_message("assistant"):
        # Show the dual-permit notice ABOVE the answer so the user
        # can't miss it. Same rendering path as on history replay.
        if permit_notices:
            _render_permit_notices(permit_notices)

        result = None
        elapsed = 0.0
        error_msg: str | None = None
        truncated = False
        with st.spinner("Searching legislation and drafting answer…"):
            try:
                t0 = time.time()
                result = answer_question(
                    prompt,
                    mode=mode,
                    retriever=retriever,
                    history=api_history,
                    graph=citation_graph,
                )
                elapsed = time.time() - t0
            except AnswerTruncated as e:
                error_msg = repr(e)
                truncated = True
                elapsed = time.time() - t0
                st.warning(
                    "⚠️ **The answer was cut off mid-response.** This usually "
                    "means the question is complex enough that the model ran "
                    "out of room to write the full reply.\n\n"
                    "**What to try**: ask a more focused version of the "
                    "question, or split it into two questions. For example, "
                    "instead of *'do I need a permit AND what are the "
                    "exemptions AND what does the council need?'*, ask just "
                    "the first part and follow up with the rest."
                )
                st.caption(f"Internal: {e}")
            except Exception as e:
                error_msg = repr(e)
                st.error(f"Something went wrong: `{e!r}`")

        # Always log the query attempt — even errored ones — so the admin
        # can see operational issues, not just successful runs.
        try:
            query_log_id = usage_log.log_query(
                session_id=st.session_state.session_id,
                mode=mode,
                question=prompt,
                cited_sections=(result.cited_sections if result else None),
                hallucinated_citations=(result.hallucinated_citations if result else None),
                confidence=(result.confidence if result else None),
                top_retrieval_score=(result.top_retrieval_score if result else None),
                top_rerank_score=(result.top_rerank_score if result else None),
                query_shape=(result.query_shape if result else None),
                rewritten_queries=(result.rewritten_queries if result else None),
                retrieved_chunk_count=(len(result.retrieved_chunks) if result else None),
                elapsed_s=elapsed,
                usage=(result.usage if result else None),
                error=error_msg,
            )
        except Exception as log_err:
            # Logging failures must never block the user. Surface in dev,
            # silent in prod.
            print(f"  warn: usage_log failed: {log_err!r}")
            query_log_id = None

        if result is not None:
            # Visually distinguish clarifying questions from full answers
            if result.confidence == "needs_clarification":
                st.info(f"💡 **I need a bit more detail to answer this accurately:**\n\n{result.answer}")
            else:
                st.markdown(result.answer)

            meta = {
                "confidence": result.confidence,
                "cited_sections": result.cited_sections,
                "hallucinated_citations": result.hallucinated_citations,
                "top_retrieval_score": result.top_retrieval_score,
                "retrieved_chunks": result.retrieved_chunks,
                "rewritten_queries": result.rewritten_queries,
                "elapsed_s": elapsed,
                "query_log_id": query_log_id,
                "permit_notices": [
                    {"kind": n.kind, "headline": n.headline, "body": n.body}
                    for n in permit_notices
                ],
            }
            _render_meta(meta)
            st.caption(f"⏱️ Generated in {elapsed:.1f}s")

            # Feedback widget — only shown for non-clarifying answers since
            # clarifying questions don't have an answer to rate yet.
            if (query_log_id is not None
                    and result.confidence != "needs_clarification"):
                _render_feedback_widget(query_log_id, key_suffix=f"new_{query_log_id}")

            st.session_state.history.append({
                "role": "assistant",
                "content": result.answer,
                "meta": meta,
            })


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "This is a research prototype. Answers are generated by Claude Sonnet 4.6 "
    "from indexed excerpts of Victorian building legislation and the National "
    "Construction Code 2022 Volume Two. Citations are verified against "
    "retrieved passages but the prototype may still produce errors. "
    "**Verify all answers against the official sources before acting.**"
)
st.caption(
    "**Sources & attribution.** The Building Act 1993 (Vic) and Building "
    "Regulations 2018 (Vic) are reproduced from the authorised versions "
    "published at [legislation.vic.gov.au](https://www.legislation.vic.gov.au). "
    "The National Construction Code 2022 Volume Two and the ABCB Housing "
    "Provisions Standard 2022 are reproduced under the "
    "[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/), "
    "© Commonwealth of Australia and the States and Territories of Australia 2022, "
    "published by the Australian Building Codes Board. These ABCB texts have "
    "been extracted, chunked, and re-presented for question-answering — view "
    "the official versions at [ncc.abcb.gov.au](https://ncc.abcb.gov.au)."
)
st.caption(
    "_This tool is not produced by, affiliated with, or endorsed by the "
    "Australian Building Codes Board, the Victorian Building Authority, or "
    "any government body. Not legal or professional advice._"
)
