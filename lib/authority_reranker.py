"""
Authority-aware reranker.

The cross-encoder reranker treats all chunks equally. But the same query
can have answers at different layers of the regulatory cascade — and the
correct layer is usually predictable from the question shape:

  "Do I need a permit?"      → Act / Regulations (legal obligations)
  "What's the riser height?" → Housing Provisions (technical detail)
  "What's exempt?"           → Regulations Schedule 3 (specific exemptions)
  "Does X apply to me?"      → Act definitions / regulations (scope)

This module classifies a query into a layer-preference profile and
applies a small score boost (or penalty) to chunks based on their
authority_level. Conservative by design — boost magnitudes are tiny
relative to rerank scores so we don't override genuinely-correct
retrievals; we just nudge ties toward the layer most likely to match
the user's intent.

The classifier is a small Haiku call (~50 tokens in, ~30 out =
fractions of a cent). For ablation testing the whole module can be
disabled via USE_AUTHORITY_BOOST below.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Literal

import anthropic
from dotenv import load_dotenv


load_dotenv(override=True)

CLASSIFIER_MODEL = "claude-haiku-4-5"
CLASSIFIER_MAX_TOKENS = 256

# Boost magnitude. Voyage rerank scores typically span 0.0–1.0. A boost
# of ±0.02 is small enough to break ties but won't override a clear
# semantic mismatch. Tune empirically against the test set.
LAYER_BOOST_MAGNITUDE = 0.02

# Layer codes
PRINCIPLED  = "principled"      # legal obligations, offences, definitions, scope, appeals
TECHNICAL   = "technical"       # numbers, dimensions, materials, climate zones, specs
EXEMPTION   = "exemption"       # what's exempt / what doesn't need a permit
CLASSIFICATION = "classification"  # what class is this building, who counts as an owner-builder
PROCEDURAL  = "procedural"      # how to apply, who issues, timing
MIXED       = "mixed"           # spans layers; no boost applied

QueryShape = Literal["principled", "technical", "exemption",
                     "classification", "procedural", "mixed"]


# Per-shape preferred layers. Higher = more preferred.
# Layers: 1 = Act, 2 = Regs, 2.5 = Sch 3 items (regulation_schedule),
#         3 = NCC, 4 = Housing Provisions
SHAPE_PREFERENCES: dict[str, dict[float, float]] = {
    PRINCIPLED:     {1: +1.0, 2: +0.5, 3: -0.3, 4: -0.5},
    TECHNICAL:      {1: -0.5, 2: -0.2, 3: +0.4, 4: +1.0},
    EXEMPTION:      {1: +0.0, 2: +0.5, 2.5: +1.0, 3: -0.2, 4: -0.5},
    CLASSIFICATION: {1: +0.3, 2: +0.0, 3: +1.0, 4: -0.3},
    PROCEDURAL:     {1: +0.5, 2: +1.0, 3: -0.3, 4: -0.5},
    MIXED:          {1: +0.0, 2: +0.0, 3: +0.0, 4: +0.0},
}


CLASSIFIER_SYSTEM = """You classify questions about Victorian building legislation into one of six query shapes. Output JSON only.

The Vic building regulatory stack:
  - Building Act 1993 / Building Regulations 2018 — legal framework, offences, exemptions, application processes
  - NCC Volume Two — performance and Deemed-to-Satisfy compliance pathways for residential buildings
  - ABCB Housing Provisions — the actual technical numbers (riser heights, R-values, footing classifications, BAL distances)

Map the user's question to the shape that best describes what answers it:

  - "principled" — legal obligations, offences, penalties, definitions, scope of duties, appeal rights, who is liable
       Examples: "What's the penalty for building without a permit?" "Can I appeal?" "Who is the owner under the Act?"

  - "technical" — concrete numbers, dimensions, materials, climate zones, specifications, span tables
       Examples: "What's the max stair riser?" "What R-value insulation in zone 6?" "How deep do footings need to be?"

  - "exemption" — what's exempt from a permit / what doesn't need one
       Examples: "Do I need a permit for a pergola?" "Is a small shed exempt?"

  - "classification" — what class of building is this, who is an owner-builder, what use applies
       Examples: "Is my granny flat a Class 1 or Class 10?" "Am I an owner-builder?"

  - "procedural" — how to apply, who issues, timing, who notifies whom
       Examples: "Who do I apply to for a permit?" "When does my permit lapse?" "Who is told when a permit issues?"

  - "mixed" — clearly spans multiple shapes / no single layer fits
       Examples: "Do I need a permit and what does compliance look like for a deck?"

Output: {"shape": "<one of the six values>", "confidence": "high"|"medium"|"low"}
"""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "shape": {
            "type": "string",
            "enum": [PRINCIPLED, TECHNICAL, EXEMPTION,
                     CLASSIFICATION, PROCEDURAL, MIXED],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["shape", "confidence"],
    "additionalProperties": False,
}


def classify_query_shape(question: str,
                         client: Optional[anthropic.Anthropic] = None
                         ) -> tuple[QueryShape, str]:
    """Return (shape, confidence). Falls back to ('mixed', 'low') on any
    error so this is never a blocker for the answer pipeline."""
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return MIXED, "low"  # type: ignore
        client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            system=CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": question}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": CLASSIFIER_SCHEMA,
                }
            },
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        if not text_blocks:
            return MIXED, "low"  # type: ignore
        parsed = json.loads(text_blocks[0])
        return parsed["shape"], parsed["confidence"]  # type: ignore
    except Exception:
        return MIXED, "low"  # type: ignore


def chunk_layer(c: dict) -> float:
    """Return the authority layer of a chunk for boosting purposes.
    Schedule 3 items (regulation_schedule) are at 2.5 because they're a
    regulation-level provision but specifically about exemptions."""
    doc_type = c.get("doc_type", "")
    if doc_type == "act":
        return 1.0
    if doc_type == "regulation":
        return 2.0
    if doc_type == "regulation_schedule":
        return 2.5
    if doc_type == "ncc":
        return 3.0
    if doc_type == "housing_provisions":
        return 4.0
    # Unknown — fall back to authority_level metadata if available
    al = c.get("authority_level")
    if isinstance(al, (int, float)):
        return float(al)
    return 2.0


def apply_authority_boost(reranked: list[tuple[float, dict]],
                          shape: QueryShape,
                          magnitude: float = LAYER_BOOST_MAGNITUDE,
                          ) -> list[tuple[float, dict]]:
    """Apply a small layer-aware boost to reranked scores and re-sort.
    For shape `mixed` (or unknown shape) returns the input unchanged."""
    prefs = SHAPE_PREFERENCES.get(shape)
    if not prefs:
        return reranked
    # Normalise prefs: each value in [-1.0, +1.0] mapped to ±magnitude
    boosted = []
    for score, c in reranked:
        layer = chunk_layer(c)
        # Find the closest declared preference layer
        delta = prefs.get(layer, 0.0)
        # If exact key not found, try truncated to int (Sch 3 = 2.5 falls
        # back to 2 if 2.5 not declared)
        if delta == 0.0 and int(layer) in prefs:
            delta = prefs[int(layer)] * 0.5  # half-credit for inexact match
        boosted.append((score + delta * magnitude, c))
    boosted.sort(key=lambda sc: sc[0], reverse=True)
    return boosted
