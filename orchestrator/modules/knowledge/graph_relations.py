"""
Knowledge Graph relations: the controlled vocabulary and the rules an edge passes
==================================================================================

LLM extraction (documents + agent reports) used to emit a free-text relation per
edge, so a workspace graph accrued thousands of singleton relation strings: the
legend flooded and the graph-neighbors ``relation_filter`` tool was unusable (no
caller could guess the exact phrase). Every LLM-extracted relation is snapped to
the bounded set below, and the model's original phrase stays on the edge as
``relation_label`` for display. Deterministic mappers (shopify / agents /
blueprints) own their own clean relations and never pass through here.

F312 (night 9): the Harbourline graph carried "October 2026 Box produces Priya"
(from "Card copy for the box is with Priya"), "Guji blocks swaps" (from "swap the
Guji"), and eight "Supplies:" lines filed as ``produces``, four of them pointing
from the coffee to the importer. The vocabulary had no supplier, customer, owner
or substitute relation, so the model forced those facts into the nearest wrong
one, and nothing checked an edge once it had a canonical name. This module now:

* names those relations, with the direction each one reads in, for the prompt;
* turns a passive or whole-to-part phrasing ("contains", "supplied by") round,
  so "Harbour Blend contains Brazil Cerrado" becomes Brazil Cerrado ``part_of``
  Harbour Blend instead of the reverse;
* refuses a relation the endpoints' declared types rule out (a person is never
  produced), turning the edge round where only its direction was wrong;
* refuses a claim of effect (blocks, causes, triggers, enables, mitigates) that
  the document's own words do not make.

A refused relation keeps the link as ``related_to``, so traversal still finds it,
and the refused name stays on the edge as ``relation_refused``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# ---------------------------------------------------------------------------
# Vocabulary: each relation reads source -> target, as glossed for the prompt.
# ---------------------------------------------------------------------------

RELATION_GLOSSES: dict[str, str] = {
    "uses": "A uses B",
    "part_of": "A is a part, component or item of the whole B: each coffee of a blend, each bag of a box",
    "member_of": "A belongs to the group or team B",
    "depends_on": "A needs B",
    "produces": "A makes or outputs B; a person is never produced",
    "supplies": "the supplier A provides B",
    "buys": "the customer A buys, orders or takes B",
    "responsible_for": "the person or team A owns, handles or holds B",
    "substitutes_for": "A is used instead of B",
    "causes": "A brings about B",
    "enables": "A makes B possible",
    "blocks": "A stops or prevents B",
    "mitigates": "A reduces or fixes B",
    "measures": "A measures or tracks B",
    "governed_by": "A is constrained by the rule B",
    "precedes": "A happens before B",
    "triggers": "A sets off B",
    "has_property": "A has the attribute or value B",
    "references": "A mentions B",
    "related_to": "any other link",
}
CANONICAL_RELATIONS: tuple[str, ...] = tuple(RELATION_GLOSSES)
_CANONICAL_SET = frozenset(CANONICAL_RELATIONS)
FALLBACK_RELATION = "related_to"
ALLOWED_RELATIONS_PROMPT = "; ".join(f"{rel} ({gloss})" for rel, gloss in RELATION_GLOSSES.items())

# Exact slugified phrase -> canonical, read in the SAME direction. Deterministic, no LLM cost.
_RELATION_SYNONYMS: dict[str, str] = {
    "used_as": "uses", "using": "uses", "used": "uses",
    "utilizes": "uses", "consumes": "uses", "leverages": "uses",
    "belongs_to": "part_of", "is_part_of": "part_of", "contained_in": "part_of",
    "component_of": "part_of", "half_of": "part_of", "ingredient_of": "part_of",
    "is_a": "part_of", "type_of": "part_of", "located_in": "part_of",
    "instance_of": "member_of", "member": "member_of", "part_of_team": "member_of",
    "reassigns": "member_of",
    "requires": "depends_on", "needs": "depends_on", "needing": "depends_on",
    "depends": "depends_on", "contingent_on": "depends_on", "backed_by": "depends_on",
    "is_blind_without": "depends_on",
    "produced": "produces", "produced_output": "produces", "generates": "produces",
    "creates": "produces", "outputs": "produces", "returns": "produces",
    "returned_agent": "produces",
    "supply": "supplies", "supplied": "supplies", "provides": "supplies", "delivers": "supplies",
    "sells": "supplies",
    "buy": "buys", "bought": "buys", "purchases": "buys", "orders": "buys", "takes": "buys",
    "customer_of": "buys", "subscribes_to": "buys", "stocks": "buys",
    "owns": "responsible_for", "owner_of": "responsible_for", "manages": "responsible_for",
    "runs": "responsible_for", "handles": "responsible_for", "in_charge_of": "responsible_for",
    "maintains": "responsible_for",
    "replaces": "substitutes_for", "substitutes": "substitutes_for", "swapped_for": "substitutes_for",
    "instead_of": "substitutes_for", "substitute_for": "substitutes_for", "alternative_to": "substitutes_for",
    "leads_to": "causes", "resulting_in": "causes", "results_in": "causes", "resulted_in_issue": "causes",
    "allows": "enables", "supports": "enables", "aims_to_achieve": "enables",
    "prevents": "blocks", "prevents_all": "blocks", "stops": "blocks", "restricts": "blocks",
    "restricted_to": "blocks",
    "reduces": "mitigates", "resolves": "mitigates", "fixes": "mitigates", "to_fix": "mitigates",
    "addresses": "mitigates",
    "tracks": "measures", "tracks_metric": "measures", "quantifies": "measures", "has_impact": "measures",
    "constrained_by": "governed_by", "regulated_by": "governed_by", "controlled_by": "governed_by",
    "before": "precedes", "followed_by": "precedes", "then": "precedes", "next": "precedes",
    "at_step": "precedes",
    "invokes": "triggers", "calls": "triggers", "fires": "triggers", "feed": "triggers", "feeds": "triggers",
    "has": "has_property", "is": "has_property", "has_status": "has_property",
    "is_unavailable": "has_property", "is_rated_as": "has_property",
    "described_as": "has_property", "is_described_as": "has_property",
    "has_description": "has_property", "is_a_value_of": "has_property",
    "has_tags": "has_property", "has_summary": "has_property",
    "mentions": "references", "refers_to": "references", "about": "references",
    "is_about": "references", "contrasts_with": "references",
    "referencing": "references", "relates_to": "related_to",
}

# Exact phrases that read target -> source: "A contains B" is "B part_of A".
_INVERSE_SYNONYMS: dict[str, str] = {
    "used_by": "uses",
    "has_part": "part_of", "includes": "part_of", "contains": "part_of", "consists_of": "part_of",
    "made_of": "part_of", "made_up_of": "part_of", "comprises": "part_of", "composed_of": "part_of",
    "has_component": "part_of", "has_member": "member_of",
    "required_by": "depends_on", "needed_by": "depends_on",
    "produced_by": "produces", "generated_by": "produces", "created_by": "produces",
    "supplied_by": "supplies", "provided_by": "supplies", "sourced_from": "supplies",
    "bought_from": "supplies", "purchased_from": "supplies", "comes_from": "supplies",
    "bought_by": "buys", "purchased_by": "buys", "ordered_by": "buys", "sold_to": "buys",
    "sells_to": "buys", "taken_by": "buys",
    "owned_by": "responsible_for", "managed_by": "responsible_for", "run_by": "responsible_for",
    "handled_by": "responsible_for", "is_with": "responsible_for", "maintained_by": "responsible_for",
    "assigned_to": "responsible_for",
    "replaced_by": "substitutes_for", "substituted_by": "substitutes_for",
    "caused_by": "causes", "is_caused_by": "causes", "results_from": "causes",
    "resulted_from": "causes", "due_to": "causes",
    "enabled_by": "enables", "blocked_by": "blocks", "mitigated_by": "mitigates",
    "measured_by": "measures", "governs": "governed_by", "determines": "governed_by",
    "after": "precedes", "stopped_after": "precedes", "triggered_by": "triggers",
}

# Ordered substring heuristics for phrases not matched exactly (first hit wins):
# (stem, canonical, active phrasing reads target -> source). Broad stems ("use")
# come last so specific ones ("caus" -> causes) win first.
_RELATION_KEYWORDS: tuple[tuple[str, str, bool], ...] = (
    ("depend", "depends_on", False), ("requir", "depends_on", False),
    ("caus", "causes", False),
    ("produc", "produces", False), ("generat", "produces", False), ("creat", "produces", False),
    ("suppl", "supplies", False), ("purchas", "buys", False),
    ("substitut", "substitutes_for", False), ("replac", "substitutes_for", False),
    ("swap", "substitutes_for", False), ("responsib", "responsible_for", False),
    ("manag", "responsible_for", False),
    ("trigger", "triggers", False), ("invok", "triggers", False),
    ("enabl", "enables", False),
    ("prevent", "blocks", False), ("block", "blocks", False),
    ("mitigat", "mitigates", False), ("resolv", "mitigates", False),
    ("measur", "measures", False), ("metric", "measures", False),
    ("govern", "governed_by", True), ("constrain", "governed_by", True), ("regulat", "governed_by", True),
    ("member", "member_of", False),
    ("belong", "part_of", False), ("includ", "part_of", True), ("contain", "part_of", True),
    ("part", "part_of", False),
    ("precede", "precedes", False), ("before", "precedes", False), ("after", "precedes", True),
    ("mention", "references", False), ("referenc", "references", False), ("contrast", "references", False),
    ("propert", "has_property", False), ("status", "has_property", False), ("attribut", "has_property", False),
    ("utili", "uses", False), ("use", "uses", False),
)
# A whole-to-part stem read with "in" ("included in") already points part -> whole.
_IN_PASSIVE_STEMS = frozenset({"includ", "contain"})


def _slug(raw: str) -> str:
    return _NON_ALNUM.sub("_", raw.lower()).strip("_")


def _keyword_relation(slug: str) -> Optional[tuple[str, bool]]:
    """``(canonical, inverted)`` from the first matching stem; a ``by`` phrasing turns it."""
    tokens = set(slug.split("_"))
    for needle, canon, reads_backwards in _RELATION_KEYWORDS:
        if needle not in slug:
            continue
        passive = "by" in tokens or (needle in _IN_PASSIVE_STEMS and "in" in tokens)
        return canon, reads_backwards != passive
    return None


def resolve_relation(raw: str | None) -> tuple[str, str, bool]:
    """Map a free-text relation to ``(canonical, original_label, inverted)``.

    Deterministic and LLM-free: exact canonical -> same-direction synonym ->
    reversed synonym -> substring heuristic -> ``related_to``. ``inverted`` says
    the phrase reads target -> source (F312: "Harbour Blend contains Brazil
    Cerrado" is Brazil Cerrado ``part_of`` Harbour Blend), so the caller swaps
    the endpoints. The original phrase is always kept as the label.
    """
    original = (raw or "").strip() or FALLBACK_RELATION
    slug = _slug(original)
    if slug in _CANONICAL_SET:
        return slug, original, False
    if slug in _RELATION_SYNONYMS:
        return _RELATION_SYNONYMS[slug], original, False
    if slug in _INVERSE_SYNONYMS:
        return _INVERSE_SYNONYMS[slug], original, True
    keyword = _keyword_relation(slug)
    if keyword is not None:
        return keyword[0], original, keyword[1]
    return FALLBACK_RELATION, original, False


def canonicalize_relation(raw: str | None) -> tuple[str, str]:
    """Map a free-text relation to ``(canonical, original_label)`` (direction ignored)."""
    canonical, original, _inverted = resolve_relation(raw)
    return canonical, original


# ---------------------------------------------------------------------------
# Edge rules: endpoint types and the document's own words.
# ---------------------------------------------------------------------------

PERSON_TYPE = "person"
ORGANIZATION_TYPE = "organization"
_AGENT_TYPES = frozenset({PERSON_TYPE, ORGANIZATION_TYPE})
# Declared types that are certainly not a supplier, customer or owner. "entity"
# is the bucket for any other named thing, so it is never ruled out.
_NON_AGENT_TYPES = frozenset({
    "product", "concept", "process", "metric", "rule", "action", "outcome", "issue",
})
# The relations a person may be the target of; every other one rules it out.
_PERSON_TARGET_RELATIONS = frozenset({
    "related_to", "references", "depends_on", "governed_by", "supplies", "responsible_for",
})
# Relations whose target can never be an organization ("a coffee produces its importer").
_NO_ORGANIZATION_TARGET = frozenset({"produces", "has_property"})
# Relations whose source must be a person or organization, when its type is known.
# Only these are turned round: for them the direction is the one thing that can be wrong.
_AGENT_SOURCE_RELATIONS = frozenset({"supplies", "buys", "responsible_for"})

# Claims of effect need the document's words: a cue word in the edge's label.
# A cue matches a word exactly, or as its stem when the cue has 4+ letters.
_MIN_STEM_LENGTH = 4
_CLAIM_CUES: dict[str, tuple[str, ...]] = {
    "blocks": (
        "block", "prevent", "stop", "halt", "cannot", "can't", "won't", "don't", "not", "no",
        "never", "without", "unless", "until", "restrict", "forbid", "ban", "deny", "denied",
        "refus", "delay", "late", "hold", "held",
    ),
    "causes": (
        "caus", "because", "due", "lead", "leads", "led", "result", "so", "therefore", "hence",
        "means", "mean", "if", "when", "since", "reason", "driv", "make", "makes", "made", "effect",
    ),
    "triggers": (
        "trigger", "if", "when", "whenever", "once", "after", "fire", "fires", "start", "kick",
        "prompt", "invok", "call", "calls",
    ),
    "enables": (
        "enabl", "allow", "let", "lets", "permit", "support", "help", "helps", "possible",
        "unlock", "can", "so", "means",
    ),
    "mitigates": (
        "mitigat", "reduc", "resolv", "fix", "address", "lower", "cut", "avoid", "offset",
        "compensat", "solv", "cover", "protect", "prevent", "limit",
    ),
}


def _types_rule_out(relation: str, source_type: Optional[str], target_type: Optional[str]) -> bool:
    """True when the endpoints' declared types make ``relation`` impossible."""
    if target_type == PERSON_TYPE and relation not in _PERSON_TARGET_RELATIONS:
        return True
    if target_type == ORGANIZATION_TYPE and relation in _NO_ORGANIZATION_TARGET:
        return True
    return relation in _AGENT_SOURCE_RELATIONS and source_type in _NON_AGENT_TYPES


def _label_makes_claim(relation: str, label: str) -> bool:
    """True when ``label`` names ``relation`` itself, carries a cue for it, or nothing needs one."""
    cues = _CLAIM_CUES.get(relation)
    if cues is None or canonicalize_relation(label)[0] == relation:
        return True
    words = _WORD.findall(label.lower().replace("’", "'"))
    return any(
        word == cue or (len(cue) >= _MIN_STEM_LENGTH and word.startswith(cue))
        for word in words for cue in cues
    )


def _turned_round(edge: dict[str, Any]) -> dict[str, Any]:
    source, target = edge["source"], edge["target"]
    return {**edge, "source": target, "target": source, "_src": target, "_tgt": source,
            "direction_repaired": True}


def _refused(edge: dict[str, Any]) -> dict[str, Any]:
    return {**edge, "relation": FALLBACK_RELATION, "relation_refused": edge["relation"]}


def enforce_relation_rules(edge: dict[str, Any], node_types: Mapping[str, str]) -> dict[str, Any]:
    """The edge as the graph keeps it: as extracted, turned round, or refused.

    F312 (night 9): "October 2026 Box produces Priya" and "Guji blocks swaps"
    reached the graph because nothing checked an edge after its relation was
    named. ``node_types`` maps the extraction's declared node ids to their
    ``file_type``; an endpoint it does not declare is unconstrained. A refused
    edge keeps its link as ``related_to``, so traversal still reaches it.
    """
    relation = edge["relation"]
    source_type = node_types.get(edge["source"])
    target_type = node_types.get(edge["target"])
    if _types_rule_out(relation, source_type, target_type):
        if relation in _AGENT_SOURCE_RELATIONS and not _types_rule_out(relation, target_type, source_type):
            return _turned_round(edge)
        return _refused(edge)
    if not _label_makes_claim(relation, edge.get("relation_label") or relation):
        return _refused(edge)
    return edge


def log_relation_repairs(edges: list[dict[str, Any]], source_file: str) -> None:
    """One line per extraction naming how many edges were turned round or refused."""
    turned = sum(1 for edge in edges if edge.get("direction_repaired"))
    refused = [edge for edge in edges if edge.get("relation_refused")]
    if not turned and not refused:
        return
    logger.info(
        "graph extraction %s: %d edge(s) turned round, %d relation(s) refused (%s)",
        source_file, turned, len(refused),
        ", ".join(sorted({f"{e['relation_refused']}: {e['source']}->{e['target']}" for e in refused})),
    )
