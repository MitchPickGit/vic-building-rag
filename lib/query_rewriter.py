"""
Haiku-based query rewriter.

Takes a user question (and optionally prior conversation context) and
returns 2-4 retrieval-optimized search queries. We then run each query
against the vector index and union the results.

This exists because users phrase questions colloquially ("can I rip up my
deck?") while the legislation is formal ("alteration to existing
building"). The vector model does some of this lifting, but multiple
phrasings dramatically lift recall on long-tail queries.

This module is read-only against the corpus and CANNOT introduce
hallucinated citations — its only job is to produce search queries that
get fed back into the vector index. The downstream answer pipeline still
verifies every cited section against the retrieved chunks.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv


load_dotenv(override=True)

REWRITER_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 512

REWRITER_SYSTEM = """You are a search query rewriter for a Q&A system over Victorian building legislation (Building Act 1993 and Building Regulations 2018).

Your job is to convert a user's natural-language question into 2-4 short search queries that will surface the relevant legislative provisions from a vector index.

You do NOT answer the question. You do NOT speculate about the law. Your only output is search queries.

GOOD QUERIES:
- Use the legislation's terminology (e.g. "alteration to existing building", "exempt from building permit", "occupancy permit", "owner-builder certificate of consent", "private building surveyor")
- Are 3-12 words long
- Cover different angles of the question (e.g. for "do I need a permit for X", produce one query about exemptions and one about general permit requirements)
- If the user mentions specific dimensions, include numeric thresholds where relevant
- Include the relevant Part/area (permits, occupancy, demolition, registration, etc.) when inferable

BAD QUERIES:
- Just echoing the user's natural-language question word-for-word
- Generic words like "building" or "Victoria" alone
- Made-up section numbers
- Speculative legal conclusions

EXAMPLES

User: "Do I need a permit to change an existing deck. It's over 1.8m at one end. I want to replace boards, new stairs etc"
Output:
{
  "queries": [
    "alteration to existing building exempt from building permit",
    "repair renewal maintenance part of existing building",
    "deck balcony stair construction permit required"
  ]
}

User: "What's the penalty for building without a permit?"
Output:
{
  "queries": [
    "offence carry out building work without permit penalty units",
    "indictable offence person in business of building knowingly without permit"
  ]
}

User: "How do I become a registered builder?"
Output:
{
  "queries": [
    "application for registration as building practitioner",
    "qualifications and experience required for registration",
    "registration of building practitioners process"
  ]
}

User: "Can I move into the house before getting an occupancy permit?"
Output:
{
  "queries": [
    "occupancy permit must be obtained before occupation",
    "occupation of building in accordance with occupancy permit",
    "temporary occupation before occupancy permit issued"
  ]
}

If the conversation has prior turns and the user is following up, treat their new message in context. E.g. if they previously asked about pergolas and now ask "what about a 4m one?", rewrite as queries about pergola height limits and the exemption schedule.

OUTPUT FORMAT
Return JSON: {"queries": ["query 1", "query 2", ...]}. 2-4 queries.
"""


REWRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}


def rewrite_query(
    question: str,
    history: Optional[list[dict]] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> list[str]:
    """Return a list of 2-4 retrieval queries for `question`.

    `history` is the conversation so far — list of {"role": "user"|"assistant",
    "content": str}. We include only the last 4 turns to keep the prompt small.
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic()

    user_lines = []
    if history:
        recent = history[-4:]
        user_lines.append("PRIOR CONVERSATION:")
        for turn in recent:
            role = turn.get("role", "user").upper()
            content = (turn.get("content") or "").strip()
            # Truncate very long prior assistant messages — we just want context
            if len(content) > 600:
                content = content[:600] + "…"
            user_lines.append(f"{role}: {content}")
        user_lines.append("")
    user_lines.append(f"NEW QUESTION: {question}")
    user_lines.append("")
    user_lines.append("Return 2-4 search queries as JSON.")

    response = client.messages.create(
        model=REWRITER_MODEL,
        max_tokens=MAX_TOKENS,
        system=REWRITER_SYSTEM,
        messages=[{"role": "user", "content": "\n".join(user_lines)}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": REWRITER_SCHEMA,
            }
        },
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        # Fallback: just use the raw question
        return [question]
    try:
        parsed = json.loads(text_blocks[0])
        queries = parsed.get("queries") or []
    except json.JSONDecodeError:
        return [question]

    # Always include the raw question too — useful when the rewriter is overconfident
    queries = list(dict.fromkeys([question, *queries]))  # dedupe preserving order
    return queries[:5]  # cap at 5 total
