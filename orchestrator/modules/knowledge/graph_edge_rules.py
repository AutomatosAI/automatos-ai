"""
Knowledge Graph edge rules: what each end of an edge is, and what may join them
================================================================================

F312 (night 9): the Harbourline graph carried "October 2026 Box produces Priya"
and "Guji blocks swaps": nothing checked an extracted edge once its relation had
a canonical name. The first fix checked edges against the node types the model
declared.

F312 (build 14 rebuild): that left "Meridian Bean Traders causes Harbour Blend"
("If Meridian are late the blend is the problem") and "Lantern Kitchen
has_property Harbour Blend" ("12 kg a week": a café buying the blend). A
declared type can't be trusted to be there or to be specific, so each node's
KIND is also read from the role it plays in the document's structural edges:
the source of ``supplies`` is a supplier, the source of ``buys`` a customer,
either end of ``part_of`` and the target of ``supplies`` or ``buys`` a product.
A specific declared type (person, organization, product) wins; a role vote
refines "organization" into supplier or customer. Then, per relation:

* a person is never produced, and an organization never produced either;
* supplies / buys / responsible_for need a person or organization as source,
  and are turned round when only their direction is wrong;
* ``has_property`` never points at a product or organization: a customer with
  a quantity ("12 kg a week") buys it, anything else is refused;
* causes / blocks / triggers and part_of never join an organization and a product;
* a claim of effect (blocks, causes, triggers, enables, mitigates) needs the
  document's own words.

Nothing is dropped: a refused edge keeps the link as ``related_to`` with the
refused name in ``relation_refused``, so traversal still reaches it.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Optional

from modules.knowledge.graph_relations import FALLBACK_RELATION, canonicalize_relation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------

PERSON = "person"
ORGANIZATION = "organization"
SUPPLIER = "supplier"
CUSTOMER = "customer"
PRODUCT = "product"
NODE_KIND_ATTR = "kind"

_ORGANIZATION_KINDS = frozenset({ORGANIZATION, SUPPLIER, CUSTOMER})
_AGENT_KINDS = _ORGANIZATION_KINDS | {PERSON}
_KNOWN_KINDS = _AGENT_KINDS | {PRODUCT}
_SPECIFIC_DECLARED: dict[str, str] = {PERSON: PERSON, ORGANIZATION: ORGANIZATION, PRODUCT: PRODUCT}
# Declared types that are certainly not a supplier, customer or owner. "concept"
# and "entity" are catch-alls, so they rule nothing out.
_NON_AGENT_KINDS = frozenset({PRODUCT, "process", "metric", "rule", "action", "outcome", "issue"})
# The kinds a buyer may be: a café, a person, or a thing nothing says is not one.
_GENERIC_KINDS = frozenset({"concept", "entity"})
_BUYER_KINDS = frozenset({PERSON, ORGANIZATION, CUSTOMER}) | _GENERIC_KINDS
# The role each end of a structural edge votes its node into.
_ROLE_KINDS: dict[str, tuple[str, str]] = {
    "supplies": (SUPPLIER, PRODUCT),
    "buys": (CUSTOMER, PRODUCT),
    "part_of": (PRODUCT, PRODUCT),
}

# ---------------------------------------------------------------------------
# Relation rules
# ---------------------------------------------------------------------------

# The relations a person may be the target of; every other one rules it out.
_PERSON_TARGET_RELATIONS = frozenset({
    "related_to", "references", "depends_on", "governed_by", "supplies", "responsible_for",
})
# Relations whose target is never an organization ("a coffee produces its importer").
_NO_ORGANIZATION_TARGET = frozenset({"produces"})
# Relations whose source must be a person or organization. Only these are turned
# round: for them the direction is the one thing that can be wrong.
_AGENT_SOURCE_RELATIONS = frozenset({"supplies", "buys", "responsible_for"})
# Relations that never join an organization and a product, whichever way round.
_NO_ORGANIZATION_PRODUCT = frozenset({"causes", "blocks", "triggers", "part_of"})
_PROPERTY_RELATION = "has_property"
_PURCHASE_RELATION = "buys"
# "12 kg a week", "4 bags", "60kg": a number with a unit of goods.
_QUANTITY = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kg|g|grams?|kilos?|bags?|boxes?|sacks?|units?|cases?|lbs?|litres?|l)\b",
    re.IGNORECASE,
)

# Claims of effect need the document's words: a cue word in the edge's label.
# A cue matches a word exactly, or as its stem when the cue has 4+ letters.
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
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


def _majority(votes: Optional[Counter]) -> Optional[str]:
    """The kind most votes name, or ``None`` when there are none or a tie."""
    if not votes:
        return None
    ranked = votes.most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _kind_of(declared: Optional[str], votes: Optional[Counter]) -> Optional[str]:
    """A specific declared type wins; a role vote refines an organization."""
    specific = _SPECIFIC_DECLARED.get(declared or "")
    voted = _majority(votes)
    if specific == ORGANIZATION and voted in _ORGANIZATION_KINDS:
        return voted
    return specific or voted or declared


def infer_node_kinds(nodes: Iterable[Mapping[str, Any]], edges: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each node's kind in one extraction: declared type and structural roles.

    F312 (build 14): the declared types alone left "Meridian causes Harbour
    Blend" standing; Meridian is the source of ``supplies`` edges, so it is a
    supplier whatever type the model wrote on it.
    """
    declared = {str(n["id"]): str(n.get("file_type") or "").strip().lower() for n in nodes}
    votes: dict[str, Counter] = {}
    for edge in edges:
        roles = _ROLE_KINDS.get(edge["relation"])
        if roles is None:
            continue
        votes.setdefault(edge["source"], Counter())[roles[0]] += 1
        votes.setdefault(edge["target"], Counter())[roles[1]] += 1
    kinds = {node_id: _kind_of(declared.get(node_id), votes.get(node_id)) for node_id in {*declared, *votes}}
    return {node_id: kind for node_id, kind in kinds.items() if kind}


def _joins_organization_and_product(source_kind: Optional[str], target_kind: Optional[str]) -> bool:
    pair = {source_kind, target_kind}
    return PRODUCT in pair and bool(pair & _ORGANIZATION_KINDS)


def _kinds_rule_out(relation: str, source_kind: Optional[str], target_kind: Optional[str]) -> bool:
    """True when the endpoints' kinds make ``relation`` impossible."""
    if target_kind == PERSON and relation not in _PERSON_TARGET_RELATIONS:
        return True
    if target_kind in _ORGANIZATION_KINDS and relation in _NO_ORGANIZATION_TARGET:
        return True
    if relation in _AGENT_SOURCE_RELATIONS and source_kind in _NON_AGENT_KINDS:
        return True
    return relation in _NO_ORGANIZATION_PRODUCT and _joins_organization_and_product(source_kind, target_kind)


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


def _refused(edge: dict[str, Any], replacement: str = FALLBACK_RELATION) -> dict[str, Any]:
    return {**edge, "relation": replacement, "relation_refused": edge["relation"]}


def _checked_property(edge: dict[str, Any], source_kind: Optional[str], target_kind: Optional[str]) -> dict[str, Any]:
    """A ``has_property`` edge onto a product or organization: a purchase, or refused.

    F312 (build 14): "Lantern Kitchen has_property Harbour Blend" from "12 kg a
    week". A thing's property is never a product or a company; a buyer with a
    quantity of a product is a customer of it.
    """
    label = str(edge.get("relation_label") or "")
    if target_kind == PRODUCT and (source_kind is None or source_kind in _BUYER_KINDS) and _QUANTITY.search(label):
        return _refused(edge, _PURCHASE_RELATION)
    return _refused(edge)


def enforce_relation_rules(edge: dict[str, Any], kinds: Mapping[str, str]) -> dict[str, Any]:
    """The edge as the graph keeps it: as extracted, turned round, rewritten or refused.

    ``kinds`` maps the extraction's node ids to their kind (``infer_node_kinds``);
    an endpoint it does not know is unconstrained. A refused edge keeps its link
    as ``related_to``, so traversal still reaches it.
    """
    relation = edge["relation"]
    source_kind, target_kind = kinds.get(edge["source"]), kinds.get(edge["target"])
    if relation == _PROPERTY_RELATION and target_kind in _KNOWN_KINDS:
        return _checked_property(edge, source_kind, target_kind)
    if _kinds_rule_out(relation, source_kind, target_kind):
        if relation in _AGENT_SOURCE_RELATIONS and not _kinds_rule_out(relation, target_kind, source_kind):
            return _turned_round(edge)
        return _refused(edge)
    if not _label_makes_claim(relation, edge.get("relation_label") or relation):
        return _refused(edge)
    return edge


def check_extracted_graph(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], source_file: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One extraction's nodes stamped with their kind, and its edges checked.

    The kind goes on the node as ``kind`` because the graph build keeps only its
    own short list of ``file_type`` values and files every other one as "concept"
    (build 14: 148 of 149 nodes), so the stored graph would otherwise lose it.
    """
    kinds = infer_node_kinds(nodes, edges)
    stamped = [
        {**node, NODE_KIND_ATTR: kinds[node["id"]]} if kinds.get(node["id"]) in _KNOWN_KINDS else node
        for node in nodes
    ]
    checked = [enforce_relation_rules(edge, kinds) for edge in edges]
    log_relation_repairs(checked, source_file)
    return stamped, checked


def log_relation_repairs(edges: list[dict[str, Any]], source_file: str) -> None:
    """One line per extraction naming how many edges were turned round or refused."""
    turned = sum(1 for edge in edges if edge.get("direction_repaired"))
    refused = [edge for edge in edges if edge.get("relation_refused")]
    if not turned and not refused:
        return
    logger.info(
        "graph extraction %s: %d edge(s) turned round, %d relation(s) refused or rewritten (%s)",
        source_file, turned, len(refused),
        ", ".join(sorted({
            f"{e['relation_refused']}->{e['relation']}: {e['source']}->{e['target']}" for e in refused
        })),
    )
