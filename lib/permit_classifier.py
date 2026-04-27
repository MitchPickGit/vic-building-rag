"""
Lightweight keyword classifier that flags when a building question may
also implicate a separate regulatory regime — typically a planning permit
or owners corporation consent — that lives outside the Building Act /
Building Regulations.

This is a UI/UX layer, not a content layer. It does NOT change what
Claude cites or how the answer reads. It surfaces a notice above the
answer telling the user "this question may also involve [X]," directing
them to the correct authority.

The motivation: many homeowner questions about "can I build a fence /
plant a tree / convert my garage / do this near a heritage building"
have a building-permit answer (which we provide) AND a planning-permit
answer (which we don't — that's the council's local planning scheme).
A user who only sees the building-permit answer can mistakenly think
they're cleared to act, when in fact council planning consent is also
required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PermitNotice:
    kind: str             # short code, e.g. "planning_permit", "oc_consent"
    headline: str         # one-line headline for the UI
    body: str             # 2–3 sentence explanation


# Triggers under 4 chars are abbreviations (FO, BAL, LSIO, BMO) and must
# match only as whole words, not as substrings — otherwise 'fo' fires on
# 'for', 'bal' fires on 'balance', etc. Anything 4+ chars uses substring
# match which is more forgiving for natural-language phrasing.
_SHORT_TRIGGER_THRESHOLD = 4


def _trigger_matches(trigger: str, lower_text: str) -> bool:
    if len(trigger) < _SHORT_TRIGGER_THRESHOLD or " " not in trigger and len(trigger) <= _SHORT_TRIGGER_THRESHOLD:
        return re.search(rf"\b{re.escape(trigger)}\b", lower_text) is not None
    return trigger in lower_text


# Each rule: a set of trigger phrases (case-insensitive matching, word
# boundaries on short abbreviations) and the notice to surface when any
# phrase matches the question text. Multiple notices can fire per question.
RULES: list[tuple[set[str], PermitNotice]] = [
    (
        {
            "heritage", "heritage overlay", "historic building",
            "heritage register",
        },
        PermitNotice(
            kind="planning_permit_heritage",
            headline="Heritage controls may also apply",
            body=(
                "If the property is in a heritage overlay or on the Victorian "
                "Heritage Register, council planning consent is required *in "
                "addition to* a building permit. The Building Act covers "
                "structural and safety requirements only — heritage approval "
                "is governed by the local planning scheme and (for state-listed "
                "places) the Heritage Act 2017. Check your council's planning "
                "scheme and Heritage Victoria before relying on this answer."
            ),
        ),
    ),
    (
        {
            "overlay", "zone", "zoning", "planning scheme",
            "neighbourhood character",
        },
        PermitNotice(
            kind="planning_permit_overlay",
            headline="Council planning rules may apply",
            body=(
                "Many works that don't need a building permit (or that need "
                "a building permit *and* something else) trigger planning "
                "controls — overlays, zoning, design and development "
                "requirements — under your local council's planning scheme. "
                "The Building Act is silent on these. Check your council's "
                "planning department or the relevant planning scheme on "
                "planning.vic.gov.au."
            ),
        ),
    ),
    (
        {
            "neighbour", "neighbours", "boundary fence", "fence height",
            "common boundary", "dividing fence",
        },
        PermitNotice(
            kind="fences_act",
            headline="Fence disputes may sit under the Fences Act 1968",
            body=(
                "Disputes between neighbours about boundary fences — cost, "
                "height, who pays for repair — are governed by the Fences Act "
                "1968 (Vic), not the Building Act. The Building Regulations "
                "set permit-exemption conditions for fence construction; the "
                "Fences Act sets the rules for who is liable to contribute. "
                "Both can apply at the same time. Consumer Affairs Victoria "
                "and the Dispute Settlement Centre of Victoria are the usual "
                "first ports of call."
            ),
        ),
    ),
    (
        {
            "tree", "vegetation", "vegetation removal", "significant tree",
            "tree protection",
        },
        PermitNotice(
            kind="planning_permit_vegetation",
            headline="Vegetation / tree controls are a planning matter",
            body=(
                "Removing, lopping or building near trees or other "
                "vegetation is regulated by your council's planning scheme "
                "(often via a Vegetation Protection Overlay or a Significant "
                "Landscape Overlay), not the Building Act. A separate "
                "planning permit may be required even if no building permit "
                "is needed. Check with your council before any works affecting "
                "trees."
            ),
        ),
    ),
    (
        {
            "bushfire", "bushfire-prone", "bushfire prone", "bal",
            "bushfire attack level", "bushfire management overlay", "bmo",
        },
        PermitNotice(
            kind="bushfire",
            headline="Bushfire controls span multiple regimes",
            body=(
                "Building work in a designated bushfire-prone area must "
                "comply with the technical requirements of AS 3959 (a separate "
                "Australian Standard) AND, if the land is in a Bushfire "
                "Management Overlay, a planning permit from council. The "
                "Building Regulations 2018 reference these — but the technical "
                "spec lives in AS 3959 (purchase from Standards Australia) "
                "and the BMO consent comes from your council."
            ),
        ),
    ),
    (
        {
            "flood", "flooding", "land liable to flooding", "flood overlay",
            "lsio", "fo", "land subject to inundation",
        },
        PermitNotice(
            kind="flood_planning",
            headline="Flood overlays are a planning-scheme matter",
            body=(
                "If the property is in a Land Subject to Inundation Overlay "
                "(LSIO), Floodway Overlay (FO), or similar, council planning "
                "consent is typically required separately from a building "
                "permit. The Building Regulations reference flood-prone "
                "land conditions — but the trigger and process for a flood "
                "overlay live in the council's planning scheme, not the "
                "Building Act. Check both."
            ),
        ),
    ),
    (
        {
            "body corporate", "owners corporation", "owners corp",
            "owner corp", "strata", "common property",
        },
        PermitNotice(
            kind="oc_consent",
            headline="Owners corporation consent may also be required",
            body=(
                "If the building is part of an owners corporation (strata / "
                "body corporate), changes to common property or works that "
                "affect other lots typically require the OC's written consent "
                "*in addition to* any building permit. This is governed by "
                "the Owners Corporations Act 2006, not the Building Act. "
                "Check your OC rules and any registered plan of subdivision "
                "before proceeding."
            ),
        ),
    ),
    (
        {
            "subdivide", "subdivision", "lot boundary", "consolidate lots",
            "re-subdivide",
        },
        PermitNotice(
            kind="planning_permit_subdivision",
            headline="Subdivision is governed separately",
            body=(
                "Subdividing land or consolidating lots is regulated by the "
                "Subdivision Act 1988 and the local planning scheme — not "
                "the Building Act. A planning permit and a certified plan of "
                "subdivision are required. Speak to a licensed land surveyor "
                "and your council's planning department."
            ),
        ),
    ),
    (
        {
            "change of use", "change in use", "change use",
            "convert garage", "convert my garage", "convert the garage",
            "convert shed", "convert my shed", "granny flat",
            "office to residential", "residential to commercial",
            "use of building",
        },
        PermitNotice(
            kind="planning_permit_change_of_use",
            headline="Change of use is a planning matter",
            body=(
                "Changing the use of a building (e.g. residential to "
                "commercial, or vice versa, or a habitable conversion of a "
                "Class 10 garage/shed) typically requires a planning permit "
                "from council *as well as* any building permit. The Building "
                "Regulations cover the building-classification side; the use "
                "side is the council's planning scheme."
            ),
        ),
    ),
]


def classify(question: str) -> list[PermitNotice]:
    """Return the list of permit notices triggered by `question`.
    Multiple notices can fire on a single question (e.g. heritage AND
    bushfire). Order is preserved as the rules list above.
    """
    if not question:
        return []
    lower = question.lower()
    fired: list[PermitNotice] = []
    seen_kinds: set[str] = set()
    for triggers, notice in RULES:
        for t in triggers:
            if _trigger_matches(t, lower):
                if notice.kind not in seen_kinds:
                    fired.append(notice)
                    seen_kinds.add(notice.kind)
                break
    return fired
