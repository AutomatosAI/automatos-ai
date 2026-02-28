"""
Entity Extraction Module
=========================

Extracts entities (people, organizations, concepts, topics, technologies)
from document chunks and builds co-occurrence relationships between them.

Stores results in document_entities and document_relationships tables
for graph-based retrieval (US-019).

Uses heuristic/regex-based extraction -- no external ML dependencies.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from itertools import combinations

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technology / domain patterns
# ---------------------------------------------------------------------------

_TECHNOLOGY_TERMS: set = {
    # AI / ML
    "machine learning", "deep learning", "neural network", "neural networks",
    "natural language processing", "nlp", "computer vision", "reinforcement learning",
    "transformer", "transformers", "large language model", "large language models",
    "llm", "llms", "generative ai", "gen ai", "rag",
    "retrieval augmented generation", "fine-tuning", "fine tuning",
    "embedding", "embeddings", "vector database", "vector search",
    "prompt engineering", "langchain", "llamaindex",
    # Cloud & infra
    "kubernetes", "k8s", "docker", "terraform", "ansible", "jenkins",
    "aws", "azure", "gcp", "google cloud", "amazon web services",
    "lambda", "ec2", "s3", "cloudformation", "ecs", "eks",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "pinecone", "weaviate", "qdrant", "milvus", "chromadb", "pgvector",
    # Languages & frameworks
    "python", "javascript", "typescript", "java", "golang", "rust",
    "react", "nextjs", "next.js", "fastapi", "flask", "django",
    "node.js", "nodejs", "express", "spring boot",
    # Protocols & patterns
    "api", "rest", "rest api", "graphql", "grpc", "websocket", "oauth",
    "jwt", "microservices", "event-driven", "ci/cd", "cicd",
    "devops", "mlops", "gitops",
    # Data
    "apache kafka", "kafka", "spark", "apache spark", "airflow",
    "dbt", "snowflake", "bigquery", "databricks", "pandas", "numpy",
}

# Build a regex that matches any technology term (longest-first to avoid partial matches)
_sorted_tech = sorted(_TECHNOLOGY_TERMS, key=len, reverse=True)
_TECH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _sorted_tech) + r")\b",
    re.IGNORECASE,
)

# Pattern for capitalised multi-word names (potential PERSON or ORG)
# Matches sequences of 2-5 capitalised words, optionally joined by
# connector words like "of", "the", "and".
_PROPER_NOUN_PATTERN = re.compile(
    r"\b([A-Z][a-z]+"
    r"(?:\s+(?:of|the|and|&|for)\s+|\s+)"
    r"(?:[A-Z][a-z]+)"
    r"(?:(?:\s+(?:of|the|and|&|for)\s+|\s+)(?:[A-Z][a-z]+)){0,3})\b"
)

# Single capitalised word that could be an acronym/org (2-6 uppercase letters)
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6})\b")

# Common English words that look like proper nouns at sentence start
_FALSE_POSITIVE_NAMES: set = {
    "The", "This", "That", "These", "Those", "There", "Here",
    "When", "Where", "What", "Which", "While", "However", "Moreover",
    "Furthermore", "Additionally", "Although", "Because", "Since",
    "After", "Before", "During", "Between", "Through", "Within",
    "About", "Under", "Above", "Below", "Also", "Each", "Every",
    "Many", "Most", "Some", "Several", "Various", "Other",
    "First", "Second", "Third", "Next", "Last", "New", "Old",
    "Based", "Using", "Given", "According", "Including", "Following",
    "Note", "Figure", "Table", "Section", "Chapter", "Example",
}

# Common acronyms that are NOT entities
_FALSE_POSITIVE_ACRONYMS: set = {
    "IT", "US", "UK", "EU", "UN", "ID", "OR", "IF", "AS", "BY",
    "TO", "IN", "ON", "AT", "OF", "AN", "IS", "AM", "PM", "VS",
    "NO", "DO", "SO", "UP", "WE", "HE", "ME", "BE",
}

# Concept trigger patterns (abstract nouns, domain ideas)
_CONCEPT_SUFFIXES = re.compile(
    r"\b([a-z]+-?(?:tion|sion|ment|ness|ity|ance|ence|ism|ology|ics|ing))\b",
    re.IGNORECASE,
)

_CONCEPT_PHRASES = re.compile(
    r"\b((?:data\s+)?(?:governance|pipeline|quality|security|privacy|compliance|"
    r"architecture|strategy|framework|methodology|workflow|integration|"
    r"optimization|scalability|reliability|observability|monitoring|"
    r"authentication|authorization|encryption))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_entities(text_content: str) -> List[Dict]:
    """
    Extract entities from a text chunk using regex and heuristic patterns.

    Returns a list of dicts, each containing:
        - name (str): normalised entity name
        - entity_type (str): PERSON | ORG | TECHNOLOGY | CONCEPT | TOPIC
        - confidence (float): 0.0 - 1.0

    No external ML dependencies required.
    """
    if not text_content or not text_content.strip():
        return []

    entities: Dict[Tuple[str, str], float] = {}  # (name, type) -> best confidence

    # --- 1. Technology terms ---
    for match in _TECH_PATTERN.finditer(text_content):
        name = _normalise_tech(match.group(0))
        key = (name, "TECHNOLOGY")
        entities[key] = max(entities.get(key, 0.0), 0.90)

    # --- 2. Proper noun phrases (potential PERSON / ORG) ---
    for match in _PROPER_NOUN_PATTERN.finditer(text_content):
        raw = match.group(0).strip()
        if raw.split()[0] in _FALSE_POSITIVE_NAMES:
            continue
        # Skip if it's already captured as technology
        raw_lower = raw.lower()
        if raw_lower in _TECHNOLOGY_TERMS or _normalise_tech(raw_lower) in _TECHNOLOGY_TERMS:
            continue
        # Skip if every word in the phrase is a known tech term (e.g. "Microsoft Azure")
        raw_words = raw_lower.split()
        if len(raw_words) <= 3 and all(
            w in _TECHNOLOGY_TERMS or _normalise_tech(w) in _TECHNOLOGY_TERMS
            for w in raw_words
        ):
            continue
        # Heuristic: if contains Corp, Inc, Ltd, LLC, Co, Group → ORG
        if re.search(r"\b(Corp|Inc|Ltd|LLC|Co|Group|Foundation|Institute|University|Lab|Labs)\b", raw):
            etype = "ORG"
            conf = 0.80
        elif any(w in _TECHNOLOGY_TERMS or _normalise_tech(w) in _TECHNOLOGY_TERMS
                 for w in raw_words):
            # Contains a tech term word → likely company/product combo (e.g. "Microsoft Azure")
            etype = "ORG"
            conf = 0.65
        else:
            # Default to PERSON for 2-word proper nouns, ORG for 3+
            etype = "PERSON" if len(raw.split()) <= 2 else "ORG"
            conf = 0.60
        key = (raw, etype)
        entities[key] = max(entities.get(key, 0.0), conf)

    # --- 3. Acronyms (potential ORG / TECHNOLOGY) ---
    for match in _ACRONYM_PATTERN.finditer(text_content):
        acr = match.group(0)
        if acr in _FALSE_POSITIVE_ACRONYMS:
            continue
        # Already captured as technology?
        if acr.lower() in _TECHNOLOGY_TERMS:
            key = (_normalise_tech(acr), "TECHNOLOGY")
            entities[key] = max(entities.get(key, 0.0), 0.85)
        else:
            key = (acr, "ORG")
            entities[key] = max(entities.get(key, 0.0), 0.50)

    # --- 4. Domain concepts ---
    for match in _CONCEPT_PHRASES.finditer(text_content):
        name = match.group(0).strip().lower()
        key = (name, "CONCEPT")
        entities[key] = max(entities.get(key, 0.0), 0.70)

    # --- 5. Abstract-noun concepts (suffix-based) ---
    for match in _CONCEPT_SUFFIXES.finditer(text_content):
        word = match.group(0).strip().lower()
        if len(word) < 6:
            continue
        # Skip very common words that aren't meaningful concepts
        if word in {"something", "nothing", "everything", "anything"}:
            continue
        key = (word, "CONCEPT")
        entities[key] = max(entities.get(key, 0.0), 0.40)

    # Build output list, sorted by confidence descending
    result = [
        {"name": name, "entity_type": etype, "confidence": round(conf, 2)}
        for (name, etype), conf in sorted(entities.items(), key=lambda x: -x[1])
    ]

    logger.debug("Extracted %d entities from text (%d chars)", len(result), len(text_content))
    return result


def _normalise_tech(raw: str) -> str:
    """Normalise a technology term to its canonical form."""
    lowered = raw.strip().lower()
    # Map common aliases
    aliases = {
        "k8s": "kubernetes",
        "postgres": "postgresql",
        "node.js": "nodejs",
        "next.js": "nextjs",
        "gen ai": "generative ai",
        "ci/cd": "cicd",
    }
    return aliases.get(lowered, lowered)


# ---------------------------------------------------------------------------
# Relationship building
# ---------------------------------------------------------------------------

def build_relationships(
    entities: List[Dict],
    document_id: int,
    chunk_index: int,
) -> List[Dict]:
    """
    Build co-occurrence relationships between entities found in the same chunk.

    Every pair of entities that appear in the same chunk gets a "co_occurs_in"
    relationship.  Weight is inversely proportional to the total number of
    entities (more entities in a chunk = weaker individual links).

    Returns a list of dicts, each containing:
        - source_entity: dict with name/entity_type
        - target_entity: dict with name/entity_type
        - relationship_type: str
        - document_id: int
        - chunk_index: int
        - weight: float
    """
    if len(entities) < 2:
        return []

    # Weight: fewer co-occurrences = stronger signal
    pair_count = len(entities) * (len(entities) - 1) / 2
    base_weight = min(1.0, 3.0 / pair_count) if pair_count > 0 else 1.0

    relationships: List[Dict] = []
    for e1, e2 in combinations(entities, 2):
        # Boost weight when both entities are high confidence
        conf_boost = (e1["confidence"] + e2["confidence"]) / 2.0
        weight = round(min(1.0, base_weight * conf_boost * 1.5), 4)

        relationships.append({
            "source_entity": {"name": e1["name"], "entity_type": e1["entity_type"]},
            "target_entity": {"name": e2["name"], "entity_type": e2["entity_type"]},
            "relationship_type": "co_occurs_in",
            "document_id": document_id,
            "chunk_index": chunk_index,
            "weight": weight,
        })

    logger.debug(
        "Built %d relationships for document_id=%d chunk=%d",
        len(relationships), document_id, chunk_index,
    )
    return relationships


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_entities_and_relationships(
    db_session,
    document_id: int,
    entities: List[Dict],
    relationships: List[Dict],
    workspace_id: Optional[str] = None,
) -> Dict:
    """
    Persist extracted entities and relationships to the database.

    Uses INSERT ... ON CONFLICT DO UPDATE to handle re-processing of the same
    document gracefully (idempotent upserts).

    Args:
        db_session: SQLAlchemy Session
        document_id: ID of the source document
        entities: Output of extract_entities()
        relationships: Output of build_relationships()
        workspace_id: Optional workspace UUID string

    Returns:
        Summary dict with counts of inserted/updated entities and relationships.
    """
    if not entities:
        return {"entities_upserted": 0, "relationships_upserted": 0}

    entity_count = 0
    relationship_count = 0

    try:
        # --- Upsert entities ---
        upsert_entity_sql = text("""
            INSERT INTO document_entities
                (document_id, name, entity_type, confidence, mention_count, workspace_id)
            VALUES
                (:document_id, :name, :entity_type, :confidence, 1, :workspace_id)
            ON CONFLICT (document_id, name, entity_type) DO UPDATE SET
                confidence = GREATEST(document_entities.confidence, EXCLUDED.confidence),
                mention_count = document_entities.mention_count + 1
            RETURNING id
        """)

        # Map (name, entity_type) -> database id for relationship insertion
        entity_id_map: Dict[Tuple[str, str], int] = {}

        for entity in entities:
            result = db_session.execute(upsert_entity_sql, {
                "document_id": document_id,
                "name": entity["name"],
                "entity_type": entity["entity_type"],
                "confidence": entity["confidence"],
                "workspace_id": workspace_id,
            })
            row = result.fetchone()
            if row:
                entity_id_map[(entity["name"], entity["entity_type"])] = row.id
                entity_count += 1

        # --- Upsert relationships ---
        upsert_rel_sql = text("""
            INSERT INTO document_relationships
                (source_entity_id, target_entity_id, relationship_type, weight,
                 document_id, metadata, workspace_id)
            VALUES
                (:source_entity_id, :target_entity_id, :relationship_type, :weight,
                 :document_id, :metadata, :workspace_id)
            ON CONFLICT DO NOTHING
            RETURNING id
        """)

        for rel in relationships:
            src_key = (rel["source_entity"]["name"], rel["source_entity"]["entity_type"])
            tgt_key = (rel["target_entity"]["name"], rel["target_entity"]["entity_type"])

            src_id = entity_id_map.get(src_key)
            tgt_id = entity_id_map.get(tgt_key)

            if src_id is None or tgt_id is None:
                logger.debug(
                    "Skipping relationship %s -> %s: entity ID not found",
                    src_key, tgt_key,
                )
                continue

            metadata_json = json.dumps({
                "chunk_index": rel.get("chunk_index"),
                "relationship_type_detail": rel["relationship_type"],
            })

            result = db_session.execute(upsert_rel_sql, {
                "source_entity_id": src_id,
                "target_entity_id": tgt_id,
                "relationship_type": rel["relationship_type"],
                "weight": rel["weight"],
                "document_id": rel["document_id"],
                "metadata": metadata_json,
                "workspace_id": workspace_id,
            })
            row = result.fetchone()
            if row:
                relationship_count += 1

        db_session.commit()

        logger.info(
            "Stored %d entities and %d relationships for document_id=%d",
            entity_count, relationship_count, document_id,
        )

    except Exception as e:
        db_session.rollback()
        logger.error("Error storing entities/relationships for document_id=%d: %s", document_id, e)
        raise

    return {
        "entities_upserted": entity_count,
        "relationships_upserted": relationship_count,
    }


# ---------------------------------------------------------------------------
# Convenience: full pipeline for a single chunk
# ---------------------------------------------------------------------------

def process_chunk_entities(
    db_session,
    document_id: int,
    chunk_index: int,
    chunk_text: str,
    workspace_id: Optional[str] = None,
) -> Dict:
    """
    End-to-end: extract entities from a chunk, build relationships, and store.

    Args:
        db_session: SQLAlchemy Session
        document_id: Document ID
        chunk_index: Index of the chunk within the document
        chunk_text: Raw text of the chunk
        workspace_id: Optional workspace UUID

    Returns:
        Summary dict from store_entities_and_relationships.
    """
    entities = extract_entities(chunk_text)
    relationships = build_relationships(entities, document_id, chunk_index)
    return store_entities_and_relationships(
        db_session, document_id, entities, relationships, workspace_id
    )
