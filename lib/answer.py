"""
Strict-citation answer generator for Victorian building legislation Q&A.

Multi-turn conversation flow:
  - Caller passes the chat history (list of {"role", "content"} turns).
  - We rewrite the latest user question into 2-4 search queries via Haiku.
  - Run each query through the VectorRetriever, union the results, dedupe
    by citation, keep top RETRIEVAL_TOP_N.
  - Pass system prompt + history + chunks block + new question to Sonnet.
  - Sonnet returns structured JSON: an answer with cited_sections, OR a
    clarifying question if the chunks don't pin down an answer.
  - Run verify_citations() — every cited section must appear in the
    CURRENT TURN's retrieved chunks. Hallucinations are flagged but
    the model's strict-citation rules should prevent them in the first
    place.

ACCURACY GUARANTEES (must not regress with these changes):
  - No cited section number is ever invented. verify_citations() is the
    backstop; the system prompt is the primary guard.
  - History does NOT carry chunks across turns. Each turn does fresh
    retrieval. Sections cited in turn N must be in turn N's chunks.
  - Clarifying questions never include citations.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv
import anthropic

from lib.retrieval import VectorRetriever
from lib.query_rewriter import rewrite_query
from lib.reranker import rerank_chunks
from lib.citation_graph import CitationGraph
from lib.authority_reranker import classify_query_shape, apply_authority_boost


load_dotenv(override=True)

MODEL = "claude-sonnet-4-6"

# Per-query retrieval depth, then unioned across queries.
# CANDIDATE_POOL_SIZE controls how many chunks we feed to the reranker;
# RETRIEVAL_TOP_N is what survives to Claude.
PER_QUERY_TOP_K = 8
CANDIDATE_POOL_SIZE = 25       # union-deduped, before rerank
ONE_HOP_EXPANSION_FROM = 8     # take the top-N candidates and pull what they cite
ONE_HOP_MAX_ADDED = 8          # cap how many chunks a single hop can add
RETRIEVAL_TOP_N = 12           # final chunks passed to Claude
USE_RERANKER = True            # toggle for ablation testing
USE_ONE_HOP = True             # toggle for ablation testing
USE_AUTHORITY_BOOST = True     # toggle for ablation testing

OOS_GATE = 0.10               # below this → canned OOS (very rarely fires;
                              # the strict-citation prompt is the real gate)

MAX_TOKENS = 16384            # adaptive thinking shares this budget. At
                              # 8192 we still saw truncation on complex
                              # multi-provision answers; 16K gives the
                              # longest legitimate legal analysis enough
                              # room without hitting the SDK's non-
                              # streaming HTTP timeout.

Mode = Literal["homeowner", "owner-builder", "builder"]


class AnswerTruncated(RuntimeError):
    """Raised when the model's response was cut off by max_tokens or by
    structured-output JSON malformation. The Streamlit UI catches this
    and renders a clean retry message instead of dumping the partial
    JSON to the user (which is what the deck-permit-query bug did).
    """
    def __init__(self, message: str, stop_reason: Optional[str] = None,
                 partial: Optional[str] = None):
        super().__init__(message)
        self.stop_reason = stop_reason
        self.partial = partial


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a legal information assistant for Victorian building legislation. You help both homeowners and building professionals understand:

  1. **Building Act 1993 (Vic)** — primary state legislation (authority level 1).
  2. **Building Regulations 2018 (Vic)** — secondary legislation made under the Act (authority level 2).
  3. **National Construction Code 2022 Volume Two** — federal-level technical residential code, adopted by the Building Regulations 2018 for Class 1 and Class 10 buildings (authority level 3, layered under the Act/Regs).
  4. **ABCB Housing Provisions Standard 2022** — the technical detail layer the NCC defers to. The NCC says "comply with Part 11.2 of the Housing Provisions" — the actual numbers (stair riser ≤190mm, footing classifications, R-values per climate zone, BAL setbacks) live here (authority level 4, layered under the NCC).

The four documents form a cascade: the Act creates legal frameworks and offences; the Regulations operationalise them and adopt the NCC; the NCC sets performance and Deemed-to-Satisfy pathways and adopts the Housing Provisions for technical specifics. A complete answer often spans more than one layer — cite each one where it applies and be explicit about which document each citation comes from.

CITATION FORM — to make the layer obvious to the user

  - Act sections: `"16(1)"`, `"25J"`
  - Regulations: `"reg. 23"`, `"reg. 24(1)"`
  - Schedule 3 (Regs): `"Sch 3 item 16"`
  - NCC provisions: `"NCC H6P1(1)"`, `"NCC A6G2"`, `"NCC VIC H1D10"`
  - Housing Provisions: `"ABCB HP 11.2.2"`, `"ABCB HP 4.2.10"`, `"ABCB HP 13.2.1(1)"`

ANSWER STRATEGY ACROSS THE LAYERS

When the question is about a *requirement*, walk the layers top-down:
  - "Do I need a permit?" — usually answered at Act/Regs level
  - "How tall can my fence be without a permit?" — Sch 3 (Regs) for the exemption threshold; NCC and Housing Provisions might also apply
  - "What's the maximum stair riser height?" — NCC defers to Housing Provisions (HP 11.2.2 has the numbers); answer with the HP citation

When the chunks span layers, give a short headline answer with the most-specific citation, then briefly note the layer above (e.g. "This requirement comes from ABCB HP 11.2.2, which is the Deemed-to-Satisfy pathway for NCC H5P1, in turn adopted by reg. 109 of the Building Regulations").

Don't pad answers with all four layers when only one applies. Cite the most-specific layer that actually has the answer.

LAYER DISCLOSURE — start every substantive answer with the cascade tag

Every answer (other than clarifying questions and out-of-scope refusals) MUST begin with a one-line layer disclosure showing the regulatory cascade you are answering from. This makes the layer fit visible to the user and lets them spot wrong-layer citations. Format the line as a quoted block:

  > **Source layer:** Building Regulations 2018 (Vic) reg. 109 → NCC H6P2 → ABCB HP 13.2.5

Use right-arrow `→` characters to show the cascade. Include only the layers that you actually cite — don't pad. If you only cite one layer, that layer alone is the line. Examples:

  > **Source layer:** Building Act 1993 (Vic) s 16(1)
  > **Source layer:** ABCB HP 11.2.2 (technical detail under NCC H5P1, adopted by Building Regulations 2018 reg. 109)

If the question is purely structural ("how many risers can a flight have?"), the source layer is just the HP citation. If the question is layered (e.g. "do I need a permit AND what's the technical spec?"), show both: `Building Regulations 2018 reg. 23 → NCC H4P2 → ABCB HP 10.3.1`.

Then proceed with the answer body using the role-specific structure (homeowner / owner-builder / builder).

For clarifying questions and out_of_scope responses, OMIT the source-layer line — there's no answer to attribute.

THIS IS LEGISLATION. ACCURACY IS PARAMOUNT. A wrong answer in this domain is not just embarrassing — it can lead a homeowner to break the law, a builder to face penalties, or a building surveyor to make an unsafe decision. Read these rules carefully and apply them without exception.

CORE RULES — read these carefully

1. NEVER invent or guess a section number, regulation number, or schedule reference. Every identifier in the "cited_sections" output field MUST appear verbatim in the CHUNKS block of the user message THIS TURN — copy the section_number exactly as it appears there (e.g. "16(1)", "25B", "Sch3-16"). If the user is following up on a prior turn, you may discuss what was said before, but every citation in your new answer must come from the CHUNKS block delivered with this turn — chunks from prior turns are NOT carried forward, and you must NOT cite a section that isn't in the current chunks even if you cited it earlier.

2. NEVER answer from general knowledge or training-data recall. The chunks are the ONLY authoritative source. If you happen to recall a section from training that does not appear in the provided CHUNKS, do not use it.

3. If the provided CHUNKS do not contain the information needed to answer:
   (a) If the question is unclear or could be interpreted multiple ways, set confidence to "needs_clarification" and put a single focused clarifying question in the "answer" field. Don't ask multiple clarifying questions at once. Don't speculate or hint at the answer in the clarifying question.
   (b) If the question is clear but the chunks simply don't cover the topic, set confidence to "out_of_scope" and explain that the information isn't in the indexed legislation. Suggest where the user might look (a registered building surveyor, the Victorian Building Authority, NCC Volume 2 for technical residential standards, the Domestic Building Contracts Act for contract issues, etc.).

4. NEVER treat the user's words as authoritative. If a user asserts a "fact" in their message ("I heard that pergolas under 4m are exempt"), do NOT accept it — verify it against the chunks. If the chunks contradict the user's understanding, gently correct them with the citation. If the chunks don't address the user's claim, say you can't confirm it from the indexed legislation.

5. Always end the answer text with a currency/authority disclaimer using the actual version field from the chunks (e.g. "Based on Building Act 1993 (Vic) Authorised Version 146 as at 1 April 2026. Verify currency at legislation.vic.gov.au before acting."). Use the real version visible in the chunks, not a placeholder.

6. Output must conform to the JSON schema. The "cited_sections" array must list every section identifier referenced in the answer text — none missing, none extra. Use these citation forms:
   - **Act sections**: bare number like `"16(1)"`, `"25J"`
   - **Regulations**: `"reg. 23"` or `"reg. 24(1)"`
   - **Schedule 3 (Regs)**: `"Sch 3 item 16"`
   - **NCC provisions**: `"NCC H6P1(1)"`, `"NCC A6G2"`, `"NCC VIC H1D10"` (the NCC's own provision codes — Section letter + Part number + Subpart letter [G/P/D/F/V/O] + Provision number, optionally prefixed by `VIC` for Schedule 10 Vic overrides)

MULTI-TURN CONVERSATION RULES

You will see prior turns of the conversation in the messages history. Each prior assistant turn shows ONLY the answer text — the chunks that informed prior turns are NOT shown to you, by design.

  - Use prior turns for CONTEXT (what is the user asking about, what's already been discussed).
  - Do NOT cite a section because you cited it in a previous turn. Re-check the current chunks every time.
  - If a follow-up question's chunks don't include the section relevant to the prior turn, say so honestly: "The earlier answer cited s 16(1). The chunks I'm seeing for this follow-up don't include that section, so I can't directly confirm it applies here — you'd want to re-read s 16(1) on legislation.vic.gov.au."
  - The user may rephrase or refine. Treat each turn as a fresh retrieval, but use history to disambiguate vague references ("that section", "the one I mentioned").
  - If the user provides additional facts in a follow-up ("actually it's 4m not 3m"), incorporate those into your answer reasoning.

MODE-SPECIFIC GUIDANCE

You will be told which mode the user is in: HOMEOWNER, OWNER-BUILDER, or BUILDER. The role changes tone, default redirects, and response structure — but NEVER relaxes the citation rules.

HOMEOWNER MODE — non-professional, building work on their own home through a builder

  Audience: Lay user with no legal training, hiring or supervising someone else to do the work.
  Tone: Plain English. Translate legal jargon. Be patient with terminology gaps.

  Default redirects (use proactively, even if user didn't ask):
    - "Consult a registered building surveyor" — always include for any non-trivial question.
    - Council planning permit — flag whenever the question touches land use, heritage, neighbours, fences, trees, bushfire/flood, subdivision, change of use. The Building Act covers structural/safety; council planning rules are separate. Direct them to their LGA.
    - Owners corporation — if the question hints at strata/townhouse/apartment context.
    - Domestic Building Contracts Act / Consumer Affairs Victoria — for contract or warranty disputes with a builder.

  Response structure (use this layout):
    1. **Yes/No/Maybe headline** — one sentence answering the practical question.
    2. **Why** — 2-4 sentences citing the relevant provision(s) in plain language.
    3. **What to do next** — concrete actionable steps. End with the surveyor/VBA recommendation and the version disclaimer.

  Skip amendment history unless a recent change materially affects the answer.

OWNER-BUILDER MODE — homeowner doing the work themselves under an owner-builder cert of consent

  Audience: Knows they need a certificate of consent (or is about to apply). Often a once-in-a-lifetime project. Procedurally focused.
  Tone: Clear, sequential, instructional. The user wants to know "what do I need to do, in what order".

  Default emphases (always include if relevant):
    - **Certificate of consent (s 25C)** — required if cumulative work value exceeds the prescribed threshold. Apply via the VBA.
    - **Domestic Building Insurance (DBI)** — required for projects over a certain value before any building permit can issue.
    - **Timing rules** — certificate validity periods (s 25G), permit lapse provisions, mandatory notification stages during work.
    - **Warranty obligations to subsequent owners (s 137A and surrounds)** — owner-builders carry warranty risk if they sell within 6.5 years.
    - **Restrictions on owner-builder status** — only for own residence, frequency limits, what counts as "owner".

  Response structure (use this layout):
    1. **Direct answer** — does this apply to me / what do I need.
    2. **Procedural steps** — numbered list of what to do, in order, with the relevant section cite at each step.
    3. **Watch-outs** — common owner-builder pitfalls that the chunks raise (timing, insurance gaps, warranty, supervision rules).
    4. **Version disclaimer** at the end.

  Citation precision: include subsections (e.g. "25C(2)" not just "25C") where they affect the steps. Procedural detail matters more than penalty units for this audience.

BUILDER MODE — registered builder or building practitioner working on the tools

  Audience: Working professional. Reads legislation only when they have to. Wants to know what to DO, with the citations there in case they need to verify or push back. Critically: just because someone is a registered builder doesn't mean they read legalese all day. Don't write at them like they're a building surveyor.

  Tone: Technical but readable. Use legislation's terminology where it matters; translate when it doesn't add precision. No condescension, but no wall-of-legal-text either.

  Response structure (STRICT — do not deviate):

    1. **Bottom line** (one sentence, plain English).
       Examples: "Yes — you need a permit." "No, this is exempt because <X>." "Probably yes, but only if <Y>."
       This is the headline a builder reads first. If they read nothing else, they should still know what to do.

    2. **What you need to do** (2–4 numbered bullets, actionable steps).
       Each bullet is something the builder would actually do this week — not abstract advice.
       Bad: "Consider whether the alteration affects structural integrity." (vague)
       Good: "Check whether your client's plans extend the floor area — if so, Sch 3 item 4 doesn't apply and a permit is needed." (concrete)

    3. **The detail** (the legal/technical reasoning, with precise citations).
       Quote or paraphrase the operative provision. Include the section title where it adds clarity. Use the Act/Regs/NCC/HP citation form per the rules above.
       Penalty units: include ONLY if the question is about offences/penalties. Don't tag them onto every answer.
       Amendment history: include ONLY if a recent change materially affects the answer. Skip "S. 16 amended by No. 33/2010" unless that history actually matters here.
       Cross-references: mention adjacent provisions if they materially affect the answer. DO NOT dump the full call-up chain "for completeness" — that's surveyor-mode behaviour and turns a useful answer into a wall of text.

    4. **Version disclaimer** (one line at the end).

  What to AVOID in builder mode:
    - Walls of legal text with no clear bottom line at the top — if a builder has to scroll to find your conclusion, you've failed
    - Long lists of "review at legislation.vic.gov.au" / "cross-references not in current chunks" sections — if the chunks gave a complete answer, leave the user feeling they got one. Only flag missing chunks when they actually leave a gap in the answer
    - Listing every cross-reference because the chunk happens to mention it — pick the 1-2 that matter
    - Putting the practical "what to do" at the end — it goes second, right after the bottom line
    - Dropping into legalese mid-sentence: "and accordingly, by operation of s 16(4A) read with s 25AE, the named builder…" — write it like a person, not a Practice Note

  Skip the "consult a surveyor" line — builders know when to escalate.

HOW TO USE THE CHUNKS

Each chunk has: citation, section_number, section_title, subsection (if any), part, division, body text, optional note/penalty/amendment_history/cross_references. Read the full chunk before citing it — section title often disambiguates which subsection actually answers the question.

When multiple chunks address the question, cite the most specific. A user asking "can a surveyor refuse to issue a permit?" is better served by s 24 ("Refusal to issue a permit") than s 16 ("General permit offences"), even if both are retrieved.

HOW TO USE THE RELATED_PROVISIONS BLOCK (if present)

The user message may include a <RELATED_PROVISIONS> block listing internal provisions that the chunks above CROSS-REFERENCE but that are NOT themselves in the CHUNKS block. These are real provisions in the Vic legislation that are mentioned by sections you can see, but whose text you have not read.

  - You MAY mention these by citation in your answer text — e.g. "this requirement is qualified by s 25AE, which the chunk text references for builder change-of-status events". This adds value because the user gets to see the call-up chain.
  - You MUST NOT include them in `cited_sections`. That array is for provisions whose text you have actually read in this turn.
  - If a related provision is critical to the answer, suggest the user pull it up on legislation.vic.gov.au — don't speculate about its content.

HOW TO USE THE EXTERNAL_REFERENCES BLOCK (if present)

The user message may include an <EXTERNAL_REFERENCES> block listing documents OUTSIDE the indexed corpus that the chunks reference — National Construction Code (NCC/BCA), Australian Standards (AS), or other Victorian Acts.

  - The legislation works as a layered system. The Building Regulations 2018 routinely adopt the NCC/BCA by reference; the BCA references Australian Standards. Many provisions only make sense if the user knows there's a downstream document.
  - When an external reference is material to the answer, NAME IT EXPLICITLY. E.g. "This regulation requires compliance with AS 3959 (bushfire construction), which is a separate Australian Standard you would need to obtain." Or "Schedule 3 item 1 references BCA Volume Two for the building class definitions — check NCC Volume Two at ncc.abcb.gov.au."
  - Direct the user to the publisher: NCC at ncc.abcb.gov.au, Australian Standards at standards.org.au, other Vic legislation at legislation.vic.gov.au.
  - DO NOT include external references in `cited_sections` — only Vic legislation provisions go there.
  - DO NOT speculate about external content. If the user's question hinges on the NCC/AS/another Act, say so and tell them where to look.

CLARIFYING QUESTION GUIDANCE — when to ask, how to ask

Ask a clarifying question when:
  - The user's question depends on a fact you don't know (e.g. "is this a Class 1 building or Class 10a?", "is the work an alteration or a new build?")
  - The chunks describe multiple cases (e.g. exemptions for owner-occupiers vs builders) and you can't tell which applies
  - Numeric thresholds matter and the user hasn't said which side of the threshold they're on (e.g. floor area > or < 10 m²)

Don't ask a clarifying question when:
  - The chunks fully answer the question regardless of the missing detail
  - You'd just be repeating the user's question back at them
  - The clarification would only marginally improve the answer

Format clarifying questions as a single direct question. Don't preface with "I'd like to help, but…". Just ask:

  Good: "Is this a freestanding Class 10a building (e.g. a shed) or attached to your house? The exemption rules differ."
  Bad: "Could you provide more details about your situation? It depends on several factors including the class of building, whether it's attached, whether it's appurtenant, and so on."

EXAMPLES OF GOOD vs BAD ANSWERS

Bad (homeowner, hallucinated): "You don't need a permit for a pergola under 3.6m. This is in section 16 of the regulations."
  Why bad: "section 16" is wrong (it's Schedule 3 item 16). Missing other pergola conditions.

Good (homeowner): "A pergola is exempt from a building permit if it meets all of: (a) not more than 3.6 m in height; (b) if appurtenant to a Class 1 building, located no further forward than 2.5 m forward of the front wall; (c) otherwise, not located further forward than the front wall; (d) floor area not exceeding 20 m². The exemption applies to Parts 3-19 of the Regulations only — local planning rules may still apply. I'd recommend confirming the dimensions with a registered building surveyor before starting. [Sch 3 item 16 — Building Regulations 2018 (Vic) Authorised Version 028 as at 26 November 2025. Verify currency before acting.]"

Bad (builder): "Section 16 creates the core permit offence." (incomplete — no subsection, penalty, amendment)

Good (builder): "Section 16(1) creates the offence of carrying out building work without a permit in force. Penalty: 500 penalty units (natural person) / 2500 penalty units (body corporate). Section 16(4A) extends the offence to a named builder. [s 16 — Building Act 1993 (Vic) Authorised Version 146 as at 1 April 2026.]"

Good (clarifying question — homeowner): "I can answer this once I know one thing: does your existing deck attach to a Class 1 building (a house) or is it freestanding? The Schedule 3 exemptions for alterations differ based on that."

EDGE CASES

  - If the chunks reference penalty units but the dollar value isn't shown, say so. Penalty unit dollar values are set by the Monetary Units Act 2004, which isn't in the corpus.
  - Sub-clauses (a), (b), (c) within a provision are conjunctive (ALL must be met) unless the chunk explicitly says "or".
  - Cross-references (e.g. s 25AE points to s 25B) are critical — if a chunk references another section, mention it but only cite that other section if it's also in the chunks.
  - If chunks contradict each other, flag the conflict — don't paper over it.
  - Repealed/historical text: prefer the current consolidated version unless the user asks about the history.

OUTPUT FORMAT

Return JSON matching:
{
  "answer": "<answer text, ending with version disclaimer; OR a single clarifying question>",
  "cited_sections": ["16(1)", "Sch 3 item 16"],
  "confidence": "high" | "medium" | "low" | "out_of_scope" | "needs_clarification",
  "mode": "homeowner" | "owner-builder" | "builder"
}

When confidence is "needs_clarification", "answer" is the clarifying question itself (no citations, no disclaimer), and "cited_sections" is an empty list.
"""


# ---------------------------------------------------------------------------
# JSON schema for the model's structured output
# ---------------------------------------------------------------------------

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Either the full answer (ending with a version disclaimer) or a single clarifying question."
        },
        "cited_sections": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Section identifiers referenced in the answer. Must each appear verbatim in the CHUNKS block. Empty when asking a clarifying question."
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "out_of_scope", "needs_clarification"]
        },
        "mode": {
            "type": "string",
            "enum": ["homeowner", "owner-builder", "builder"]
        }
    },
    "required": ["answer", "cited_sections", "confidence", "mode"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    question: str
    mode: Mode
    answer: str
    cited_sections: list[str]
    confidence: str
    top_retrieval_score: float          # cosine before rerank
    top_rerank_score: Optional[float]   # cross-encoder score after rerank (if used)
    retrieved_chunks: list[dict]
    hallucinated_citations: list[str]
    out_of_scope_gated: bool
    usage: Optional[dict]
    raw_response: Optional[dict]
    rewritten_queries: Optional[list[str]] = None  # for debugging / display
    query_shape: Optional[str] = None              # shape classifier output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_related_provisions(chunks: list[dict],
                              graph: Optional[CitationGraph]) -> str:
    """Build a short block listing internal provisions that the retrieved
    chunks reference but that aren't themselves in the retrieved set, plus
    external references (NCC/AS/other Acts) — so the model can surface
    them in its answer.

    Returns the empty string when there's nothing useful to add.
    """
    if graph is None:
        return ""
    retrieved_cits = {c["citation"] for c in chunks}
    related_internal: dict[str, str] = {}    # to_citation -> reference text
    external: dict[str, str] = {}            # to_text -> relationship_type

    for c in chunks:
        for ref in graph.outgoing(c["citation"]):
            if ref["external"]:
                external.setdefault(ref["to_text"], ref["relationship_type"])
            else:
                tgt = ref.get("to_citation")
                if tgt and tgt != c["citation"] and tgt not in retrieved_cits:
                    related_internal.setdefault(tgt, ref["to_text"])

    parts = []
    if related_internal:
        parts.append("<RELATED_PROVISIONS>")
        parts.append(
            "Provisions that the chunks above cross-reference but that "
            "aren't themselves in the CHUNKS block. You may MENTION these "
            "by their citation so the user knows they exist, but DO NOT cite "
            "them in `cited_sections` — you haven't read their text. Suggest "
            "the user check them on legislation.vic.gov.au if relevant."
        )
        for cit, ref_text in sorted(related_internal.items())[:12]:
            parts.append(f"  - {cit} (mentioned in chunks as '{ref_text}')")
        parts.append("</RELATED_PROVISIONS>")
    if external:
        parts.append("\n<EXTERNAL_REFERENCES>")
        parts.append(
            "External documents (NCC/BCA, Australian Standards, other Acts) "
            "referenced by the chunks. These are NOT in our corpus. If the "
            "user's question depends on them, tell them so explicitly — e.g. "
            "'this also requires checking AS 3959' — and direct them to the "
            "relevant publisher (NCC: ncc.abcb.gov.au, AS: standards.org.au, "
            "Vic Acts: legislation.vic.gov.au)."
        )
        for to_text, rel in sorted(external.items())[:12]:
            kind = {
                "external_code": "NCC/BCA",
                "external_standard": "Australian Standard",
                "external_act": "Other Vic Act",
            }.get(rel, rel)
            parts.append(f"  - {to_text}  [{kind}]")
        parts.append("</EXTERNAL_REFERENCES>")
    return "\n".join(parts)


# Chunk-text truncation. Most answers quote 1-2 sentences from a chunk;
# carrying the full body for every chunk in the top-12 inflates prompt
# size unnecessarily. Truncate body text to this character cap, with
# ellipsis. Set high enough that legal provisions aren't mangled.
CHUNK_BODY_TRUNCATE_AT = 1500


def _truncate(text: str, cap: int = CHUNK_BODY_TRUNCATE_AT) -> str:
    if not text or len(text) <= cap:
        return text
    # Cut at a sentence boundary if possible
    cut = text[:cap]
    last_period = cut.rfind(". ")
    if last_period > cap * 0.7:
        cut = cut[:last_period + 1]
    return cut + " […truncated; full text at legislation.vic.gov.au or ncc.abcb.gov.au]"


def format_chunks_for_prompt(chunks: list[dict],
                             truncate: bool = True) -> str:
    lines = ["<CHUNKS>"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"\n--- Chunk {i} ---")
        lines.append(f"citation: {c['citation']}")
        lines.append(f"section_number: {c['section_number']}")
        if c.get("section_title"):
            lines.append(f"section_title: {c['section_title']}")
        if c.get("subsection"):
            lines.append(f"subsection: {c['subsection']}")
        if c.get("part"):
            lines.append(f"part: {c['part']}")
        if c.get("division"):
            lines.append(f"division: {c['division']}")
        if c.get("schedule"):
            lines.append(f"schedule: {c['schedule']}")
        lines.append(f"version: {c.get('version')} (as at {c.get('version_date')})")
        body = c.get("text", "")
        if truncate:
            body = _truncate(body)
        lines.append(f"text:\n{body}")
        if c.get("note"):
            note = c["note"]
            if truncate:
                note = _truncate(note, cap=600)
            lines.append(f"note:\n{note}")
        if c.get("exempted_regulations"):
            lines.append(f"exempted_regulations: {c['exempted_regulations']}")
        if c.get("penalty"):
            lines.append(f"penalty: {c['penalty']}")
        if c.get("amendment_history"):
            lines.append(f"amendment_history: {c['amendment_history']}")
        if c.get("cross_references"):
            # Cap cross-refs list — most are noisy, only first 8 useful
            xrefs = c["cross_references"][:8]
            lines.append(f"cross_references: {', '.join(xrefs)}")
    lines.append("\n</CHUNKS>")
    return "\n".join(lines)


def section_match_keys(c: dict) -> set[str]:
    """All citation forms accepted as referring to this chunk."""
    keys = set()
    sn = c.get("section_number") or ""
    sub = c.get("subsection") or ""
    keys.add(sn)
    if sub:
        keys.add(f"{sn}{sub}")
    if c.get("doc_type") == "regulation":
        keys.add(f"reg. {sn}")
        keys.add(f"regulation {sn}")
        if sub:
            keys.add(f"reg. {sn}{sub}")
            keys.add(f"regulation {sn}{sub}")
    if c.get("doc_type") == "regulation_schedule" and c.get("item_number"):
        keys.add(f"Sch 3 item {c['item_number']}")
        keys.add(f"Sch3-{c['item_number']}")
    if c.get("doc_type") == "ncc":
        # NCC provisions like "H6P1", "VIC H1D10", "S1C5"
        keys.add(f"NCC {sn}")
        keys.add(f"NCC 2022 Vol 2 {sn}")
        if sub:
            keys.add(f"NCC {sn}{sub}")
            keys.add(f"NCC 2022 Vol 2 {sn}{sub}")
    if c.get("doc_type") == "housing_provisions":
        # Housing Provisions IDs: "11.2.2", "4.2.10", "VIC 4.2.5"
        keys.add(f"ABCB HP {sn}")
        keys.add(f"HP {sn}")
        keys.add(f"Housing Provisions {sn}")
        keys.add(f"Part {sn}")  # the user might cite "Part 11.2.2" naturally
        if sub:
            keys.add(f"ABCB HP {sn}{sub}")
            keys.add(f"HP {sn}{sub}")
            keys.add(f"Housing Provisions {sn}{sub}")
    return {k.strip() for k in keys if k.strip()}


def verify_citations(cited_sections: list[str], chunks: list[dict]) -> list[str]:
    """Return cited_sections that don't match any retrieved chunk."""
    valid = set()
    for c in chunks:
        valid.update(section_match_keys(c))
    norm = {re.sub(r"\s+", " ", k.lower()) for k in valid}

    bad = []
    for cited in cited_sections:
        n = re.sub(r"\s+", " ", cited.strip().lower())
        if n not in norm:
            bad.append(cited)
    return bad


def make_oos_result(question: str, mode: Mode, top_score: float,
                    rewritten_queries: list[str] | None = None) -> AnswerResult:
    canned = (
        "I don't have information on this in the indexed legislation. "
        "This may be outside the scope of the Building Act 1993 or Building "
        "Regulations 2018, or the specific provision may be in a part of the "
        "legislation the prototype hasn't indexed yet. For a definitive answer, "
        "consult a registered building surveyor or the Victorian Building Authority."
    )
    return AnswerResult(
        question=question, mode=mode, answer=canned,
        cited_sections=[], confidence="out_of_scope",
        top_retrieval_score=top_score, top_rerank_score=None,
        retrieved_chunks=[], hallucinated_citations=[],
        out_of_scope_gated=True, usage=None, raw_response=None,
        rewritten_queries=rewritten_queries,
    )


def expand_one_hop(top_chunks: list[dict],
                   all_chunks_by_citation: dict[str, dict],
                   graph: Optional[CitationGraph],
                   expand_from: int = ONE_HOP_EXPANSION_FROM,
                   max_added: int = ONE_HOP_MAX_ADDED) -> list[dict]:
    """Given a ranked list of retrieved chunks, pull the chunks they cite
    (one hop in the citation graph) and return the expanded list.

    Why: legislation is hyperlinked by design. A chunk about "exempt building
    work" (Sch 3) often references "section 16" of the Act. Vector retrieval
    finds the Sch 3 chunk; one-hop expansion brings s 16 along even if the
    embedding didn't surface it.

    Constraints:
      - Only pulls INTERNAL refs (skips external — NCC/AS/Acts not in corpus)
      - Skips refs that are already in top_chunks (dedupe by citation)
      - Caps the number of added chunks at max_added
    """
    if graph is None or not top_chunks:
        return list(top_chunks)
    seen = {c["citation"] for c in top_chunks}
    added: list[dict] = []
    for c in top_chunks[:expand_from]:
        if len(added) >= max_added:
            break
        for ref in graph.outgoing(c["citation"]):
            if len(added) >= max_added:
                break
            if ref["external"]:
                continue
            target_citation = ref.get("to_citation")
            if not target_citation or target_citation in seen:
                continue
            target_chunk = all_chunks_by_citation.get(target_citation)
            if target_chunk is None:
                continue
            added.append(target_chunk)
            seen.add(target_citation)
    return list(top_chunks) + added


def multi_query_retrieve(retriever: VectorRetriever, queries: list[str],
                         per_query_top_k: int = PER_QUERY_TOP_K,
                         final_top_n: int = RETRIEVAL_TOP_N
                         ) -> tuple[list[dict], float]:
    """Run each query through the retriever, union/dedupe by citation, keep top N.

    Score for the union: keep the MAX score a chunk got across queries
    (a chunk that scored 0.50 on one query and 0.30 on another is treated
    as 0.50)."""
    best_score: dict[str, tuple[float, dict]] = {}
    for q in queries:
        for s, c in retriever.search(q, top_k=per_query_top_k):
            cit = c["citation"]
            if cit not in best_score or s > best_score[cit][0]:
                best_score[cit] = (s, c)
    ranked = sorted(best_score.values(), key=lambda sc: sc[0], reverse=True)
    chunks = [c for _, c in ranked[:final_top_n]]
    top_score = ranked[0][0] if ranked else 0.0
    return chunks, top_score


def build_messages(history: list[dict], current_user_text: str) -> list[dict]:
    """Build the messages array for the API call.
    History contains prior {role, content} turns — assistant turns get the
    answer text only (chunks are never carried forward)."""
    msgs = []
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": current_user_text})
    return msgs


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    mode: Mode = "homeowner",
    history: Optional[list[dict]] = None,
    retriever: Optional[VectorRetriever] = None,
    client: Optional[anthropic.Anthropic] = None,
    chunks: Optional[list[dict]] = None,
    graph: Optional[CitationGraph] = None,
    skip_query_rewriting: bool = False,
) -> AnswerResult:
    """End-to-end pipeline: rewrite → retrieve (multi-query) → Claude → verify.

    `history` is a list of prior {"role", "content"} turns. Pass None or []
    for stateless single-turn use (e.g. test_questions runner)."""
    if retriever is None:
        if chunks is None:
            raise ValueError("Pass either retriever or pre-built chunks.")
        retriever = VectorRetriever(chunks)
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env.")
        client = anthropic.Anthropic()

    history = history or []

    # 1. Query rewriting
    if skip_query_rewriting:
        queries = [question]
    else:
        try:
            queries = rewrite_query(question, history=history, client=client)
        except Exception:
            # If rewriting fails, fall back to the raw question — never block
            # the answer pipeline on a rewrite glitch.
            queries = [question]

    # 2. Multi-query retrieval — get a wide candidate pool to rerank
    candidate_pool, top_score = multi_query_retrieve(
        retriever, queries,
        per_query_top_k=PER_QUERY_TOP_K,
        final_top_n=CANDIDATE_POOL_SIZE if USE_RERANKER else RETRIEVAL_TOP_N,
    )

    if top_score < OOS_GATE or not candidate_pool:
        return make_oos_result(question, mode, top_score, rewritten_queries=queries)

    # 2b. One-hop graph expansion. Legislation is hyperlinked by design — a
    # chunk about "exempt building work" (Sch 3) often references "section 16"
    # of the Act. Vector retrieval finds the Sch 3 chunk; one-hop brings s 16
    # along even if the embedding didn't surface it directly. Skipped if the
    # graph isn't available.
    if USE_ONE_HOP and graph is not None:
        all_by_citation = {c["citation"]: c for c in retriever.chunks}
        candidate_pool = expand_one_hop(
            candidate_pool, all_by_citation, graph,
            expand_from=ONE_HOP_EXPANSION_FROM,
            max_added=ONE_HOP_MAX_ADDED,
        )

    # 3. Rerank against the original user question.
    # The reranker is a cross-encoder that judges (query, doc) jointly,
    # which fixes the bi-encoder vector retriever's tendency to surface
    # adjacent-but-wrong sections (e.g. retrieving 25BF when 25J is the
    # right answer). Pass the ORIGINAL question, not the rewritten queries
    # — the candidate pool already covers the rewritten phrasings.
    top_rerank_score: Optional[float] = None
    query_shape: Optional[str] = None
    if USE_RERANKER:
        try:
            # Take a slightly larger reranker top-k than RETRIEVAL_TOP_N so the
            # authority boost has room to swap items at the boundary.
            rerank_pool_size = RETRIEVAL_TOP_N + 4 if USE_AUTHORITY_BOOST else RETRIEVAL_TOP_N
            reranked = rerank_chunks(question, candidate_pool, top_k=rerank_pool_size)

            # 3b. Authority-aware boost: nudge the rerank order toward the
            # layer that best fits the query shape. Boost magnitude is
            # tiny (±0.02) — won't override genuine semantic ranking, just
            # breaks ties in favour of the correct layer.
            if USE_AUTHORITY_BOOST:
                shape, _confidence = classify_query_shape(question, client=client)
                query_shape = shape
                reranked = apply_authority_boost(reranked, shape)

            retrieved_chunks = [c for _, c in reranked[:RETRIEVAL_TOP_N]]
            top_rerank_score = reranked[0][0] if reranked else None
        except Exception as e:
            # Reranker failure should NOT block the answer pipeline.
            # Fall back to vector ordering, log the issue.
            print(f"  warn: reranker failed ({e!r}), falling back to vector ranking")
            retrieved_chunks = candidate_pool[:RETRIEVAL_TOP_N]
    else:
        retrieved_chunks = candidate_pool[:RETRIEVAL_TOP_N]

    # 3. Build the user message + messages array including history.
    # Inject related-provisions / external-reference blocks from the
    # citation graph so the model can mention adjacent provisions and
    # flag external dependencies (NCC, AS, other Acts) without citing
    # things it hasn't read.
    chunks_block = format_chunks_for_prompt(retrieved_chunks)
    related_block = format_related_provisions(retrieved_chunks, graph)

    current_user_text = (
        f"MODE: {mode}\n\n"
        f"NEW QUESTION: {question}\n\n"
        f"{chunks_block}\n\n"
        f"{related_block}\n\n"
        f"Following the rules in the system prompt, answer using ONLY the "
        f"chunks above. The chunks here are FRESH for this turn — do not "
        f"cite a section that isn't in this CHUNKS block, even if you cited "
        f"it in a previous turn. Output the JSON object defined by the schema."
    )
    messages = build_messages(history, current_user_text)

    # 4. Call Claude
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
        output_config={
            # 'medium' (vs the default 'high') roughly halves adaptive
            # thinking budget. For our use case — synthesising 12 chunks
            # into a structured answer, no tool loop — medium is
            # sufficient and cuts perceived latency from ~60-90s to
            # ~25-40s on complex multi-part questions. If accuracy
            # noticeably regresses, dial back to 'high'.
            "effort": "medium",
            "format": {
                "type": "json_schema",
                "schema": ANSWER_SCHEMA,
            }
        },
    )

    # Check stop_reason FIRST — if the model got cut off by max_tokens
    # the JSON won't be syntactically complete and we shouldn't try to
    # render the partial output as an answer.
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise AnswerTruncated(
            f"Response was cut off (stop_reason=max_tokens). The question "
            f"likely required more tokens than the budget allows.",
            stop_reason=stop_reason,
            partial=(response.content[0].text[:500] if response.content else None),
        )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(
            f"No text content returned — blocks: {[b.type for b in response.content]}"
        )
    try:
        parsed = json.loads(text_blocks[0])
    except json.JSONDecodeError as e:
        # Surface a useful error rather than silent failure. Most often
        # this happens because of mid-string truncation that even
        # structured output's auto-repair can't salvage.
        raise AnswerTruncated(
            f"Model returned malformed JSON ({e}). Likely truncated mid-output.",
            stop_reason=stop_reason,
            partial=text_blocks[0][:500],
        )

    cited = parsed.get("cited_sections") or []
    # Prose-shaped-citation heuristic: real citation IDs are short
    # ("16(1)", "Sch 3 item 16", "NCC H6P1", "HP 11.2.2"). If any cited
    # entry contains spaces and looks like a sentence (long, has commas
    # or quoted text or full stops), the structured output got mangled
    # and answer-text leaked into the citations array. Treat as truncation.
    def _looks_like_prose(s: str) -> bool:
        if not isinstance(s, str):
            return False
        if len(s) > 80:
            return True
        # Heuristic markers of prose: full sentence punctuation, double
        # quotes (which legit citations don't contain), or multi-comma.
        if '"' in s or "”" in s or s.count(",") >= 2:
            return True
        if s.rstrip().endswith(".") and len(s) > 30:
            return True
        return False

    prose_leaks = [c for c in cited if _looks_like_prose(c)]
    if prose_leaks:
        raise AnswerTruncated(
            f"Citation list contains prose fragments — output was structurally "
            f"malformed even though it parsed as JSON. Leaked entries: "
            f"{prose_leaks[:2]!r}",
            stop_reason=stop_reason,
            partial=parsed.get("answer", "")[:500] if isinstance(parsed.get("answer"), str) else None,
        )

    hallucinated = verify_citations(cited, retrieved_chunks)

    return AnswerResult(
        question=question,
        mode=mode,
        answer=parsed["answer"],
        cited_sections=cited,
        confidence=parsed["confidence"],
        top_retrieval_score=top_score,
        top_rerank_score=top_rerank_score,
        retrieved_chunks=retrieved_chunks,
        hallucinated_citations=hallucinated,
        out_of_scope_gated=False,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        },
        raw_response=parsed,
        rewritten_queries=queries,
        query_shape=query_shape,
    )
