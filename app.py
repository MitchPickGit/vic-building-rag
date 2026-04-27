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

import os
import time

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
from lib.answer import answer_question
from lib.citation_graph import CitationGraph
from lib.permit_classifier import classify as classify_permits


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
        options=["homeowner", "owner-builder", "builder"],
        format_func=lambda m: {
            "homeowner": "🏠 Homeowner — building work through a builder",
            "owner-builder": "🛠️ Owner-builder — doing the work yourself",
            "builder": "🔨 Builder / Surveyor — registered practitioner",
        }[m],
        index=0,
        help=(
            "**Homeowner**: plain-English answers, redirects to surveyor / council / "
            "owners corp where relevant.  \n"
            "**Owner-builder**: procedural step-by-step, emphasising certificate of "
            "consent, insurance, timing, and warranty obligations.  \n"
            "**Builder**: technical detail with full citations, penalties, "
            "amendment history, and cross-references."
        ),
    )

    st.divider()
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
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        # Permit notices render at the TOP of an assistant turn, before
        # the answer text — so the warning is visible before the answer.
        if turn["role"] == "assistant" and turn.get("meta", {}).get("permit_notices"):
            _render_permit_notices(turn["meta"]["permit_notices"])
        st.markdown(turn["content"])
        if "meta" in turn:
            _render_meta(turn["meta"])


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
            except Exception as e:
                st.error(f"Something went wrong: `{e!r}`")
                result = None

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
                "permit_notices": [
                    {"kind": n.kind, "headline": n.headline, "body": n.body}
                    for n in permit_notices
                ],
            }
            _render_meta(meta)
            st.caption(f"⏱️ Generated in {elapsed:.1f}s")

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
