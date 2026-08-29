"""
Entity Extraction Service for Knowledge Graph
Extracts entities and relationships from text using NER + LLM
Part of PRD-21: Hybrid RAG Integration
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    entity_name: str
    entity_type: str  # person, technology, concept, organization, location, event
    canonical_name: str
    description: Optional[str] = None
    confidence: float = 1.0
    mention_context: Optional[str] = None
    position: Optional[int] = None


@dataclass
class ExtractedRelationship:
    """Represents a relationship between entities"""
    from_entity: str
    to_entity: str
    relationship_type: str  # is_part_of, uses, created_by, related_to, improves, depends_on, alternative_to
    strength: float = 1.0
    evidence: Optional[str] = None


class EntityExtractor:
    """Extract entities and relationships from text"""
    
    def __init__(self):
        """Initialize entity extractor.

        Deliberately constructs NO LLM client here. ``AsyncOpenAI()`` raises
        ``OpenAIError: Missing credentials`` at construction when
        ``OPENAI_API_KEY`` is unset — and this platform runs on OpenRouter, so
        that raise fired on EVERY document ingestion (2026-08-29). The caller
        catches it and logs a warning, so ingestion appeared to succeed while
        entity extraction was silently disabled and the knowledge graph
        quietly degraded. Same class as the #645 Harbourline failure: never
        demand a vendor key that may not exist.

        The manager is built lazily on first use and resolves provider, model
        and key through the platform's own routing (workspace key first,
        OpenRouter fallback), so no vendor is hard-coded here.
        """
        self._llm_manager = None

    def _manager(self):
        """The shared LLM manager, created on first use."""
        if self._llm_manager is None:
            from core.llm import create_llm_manager

            self._llm_manager = create_llm_manager(
                service_name="entity_extraction",
                request_type="entity_extraction",
            )
        return self._llm_manager

    @staticmethod
    def _content_of(response) -> str:
        """Text of an LLMManager response, whatever shape it arrives in."""
        content = getattr(response, "content", None)
        return (content if content is not None else str(response)).strip()
        
    async def extract_entities(
        self, 
        text: str, 
        use_llm: bool = True,
        max_entities: int = 50
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text
        
        Args:
            text: Text to extract entities from
            use_llm: Whether to use LLM for extraction (vs regex only)
            max_entities: Maximum number of entities to extract
            
        Returns:
            List of ExtractedEntity objects
        """
        if not text or len(text.strip()) < 20:
            return []
        
        entities = []
        
        # Method 1: Fast regex-based extraction for common patterns
        regex_entities = self._extract_with_regex(text)
        entities.extend(regex_entities)
        
        # Method 2: LLM-based extraction for better accuracy
        if use_llm and len(text) < 8000:  # Only use LLM for reasonable text sizes
            try:
                llm_entities = await self._extract_with_llm(text, max_entities)
                entities.extend(llm_entities)
            except Exception as e:
                logger.error(f"LLM entity extraction failed: {e}")
        
        # Deduplicate and normalize
        entities = self._deduplicate_entities(entities)
        
        # Limit results
        entities = entities[:max_entities]
        
        logger.info(f"Extracted {len(entities)} entities from {len(text)} chars")
        return entities
    
    def _extract_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Fast regex-based entity extraction for common patterns"""
        entities = []
        
        # Pattern 1: Technology names (capitalized, often with numbers/versions)
        tech_pattern = r'\b([A-Z][a-z]*(?:[-][A-Z][a-z]*)*(?:\s+[A-Z][a-z]*)*(?:\s+\d+)?(?:\.\d+)*)\b'
        for match in re.finditer(tech_pattern, text):
            name = match.group(1)
            # Filter out common words
            if len(name) > 2 and name not in ['The', 'This', 'That', 'These', 'Those', 'When', 'Where']:
                entities.append(ExtractedEntity(
                    entity_name=name,
                    entity_type="technology",
                    canonical_name=name.lower().strip(),
                    confidence=0.6,  # Lower confidence for regex
                    position=match.start()
                ))
        
        # Pattern 2: Acronyms (2-5 capital letters)
        acronym_pattern = r'\b([A-Z]{2,5})\b'
        for match in re.finditer(acronym_pattern, text):
            name = match.group(1)
            if name not in ['USA', 'UK', 'US', 'EU', 'UN']:  # Filter common non-tech acronyms
                entities.append(ExtractedEntity(
                    entity_name=name,
                    entity_type="concept",
                    canonical_name=name.lower(),
                    confidence=0.5,
                    position=match.start()
                ))
        
        return entities
    
    async def _extract_with_llm(self, text: str, max_entities: int = 30) -> List[ExtractedEntity]:
        """LLM-based entity extraction for better accuracy"""
        
        # Truncate text if too long
        text_sample = text[:6000] if len(text) > 6000 else text
        
        prompt = f"""Extract named entities from this text. Focus on:
- Technologies (software, frameworks, tools, AI models)
- Concepts (algorithms, methodologies, techniques)
- Organizations (companies, institutions)
- People (researchers, developers, leaders)
- Products (specific products or services)

Return ONLY a JSON array with this exact format:
[
  {{"name": "GPT-4", "type": "technology", "description": "OpenAI language model"}},
  {{"name": "Neural Networks", "type": "concept", "description": "Machine learning architecture"}},
  {{"name": "OpenAI", "type": "organization", "description": "AI research company"}}
]

Text to analyze:
{text_sample}

Return ONLY the JSON array, no other text."""

        try:
            response = await self._manager().generate_response(
                messages=[{"role": "user", "content": prompt}]
            )

            content = self._content_of(response)
            
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            entities_data = json.loads(content)
            
            entities = []
            for entity_dict in entities_data[:max_entities]:
                entities.append(ExtractedEntity(
                    entity_name=entity_dict.get("name", ""),
                    entity_type=entity_dict.get("type", "concept"),
                    canonical_name=entity_dict.get("name", "").lower().strip(),
                    description=entity_dict.get("description"),
                    confidence=0.9  # Higher confidence for LLM
                ))
            
            return entities
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM entity response: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM entity extraction error: {e}")
            return []
    
    async def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        max_relationships: int = 30
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships between entities
        
        Args:
            text: Source text
            entities: List of entities found in text
            max_relationships: Maximum relationships to extract
            
        Returns:
            List of ExtractedRelationship objects
        """
        if len(entities) < 2:
            return []
        
        # Truncate text if too long
        text_sample = text[:4000] if len(text) > 4000 else text
        
        # Get entity names for LLM
        entity_names = [e.entity_name for e in entities[:20]]  # Limit to avoid token limits
        
        prompt = f"""Analyze relationships between these entities found in the text:
{', '.join(entity_names)}

Return ONLY a JSON array with this exact format:
[
  {{"from": "Neural Networks", "to": "Machine Learning", "type": "is_part_of", "evidence": "Neural networks are a type of machine learning..."}},
  {{"from": "GPT-4", "to": "OpenAI", "type": "created_by", "evidence": "GPT-4 was developed by OpenAI..."}}
]

Relationship types to use:
- is_part_of: Entity A is a subset or component of Entity B
- uses: Entity A uses or depends on Entity B
- created_by: Entity A was created/developed by Entity B
- improves: Entity A improves or enhances Entity B
- related_to: General relationship
- alternative_to: Entity A is an alternative to Entity B
- depends_on: Entity A requires Entity B

Text:
{text_sample}

Return ONLY the JSON array, no other text."""

        try:
            response = await self._manager().generate_response(
                messages=[{"role": "user", "content": prompt}]
            )

            content = self._content_of(response)
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            relationships_data = json.loads(content)
            
            relationships = []
            for rel_dict in relationships_data[:max_relationships]:
                relationships.append(ExtractedRelationship(
                    from_entity=rel_dict.get("from", ""),
                    to_entity=rel_dict.get("to", ""),
                    relationship_type=rel_dict.get("type", "related_to"),
                    strength=0.8,  # Default strength
                    evidence=rel_dict.get("evidence", "")[:500]  # Limit evidence length
                ))
            
            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM relationship response: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM relationship extraction error: {e}")
            return []
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Deduplicate entities by canonical name, keeping highest confidence
        """
        seen = {}
        for entity in entities:
            canonical = entity.canonical_name
            if canonical not in seen or entity.confidence > seen[canonical].confidence:
                seen[canonical] = entity
        
        return list(seen.values())
    
    def _get_context(self, text: str, position: int, window: int = 150) -> str:
        """Get surrounding context for an entity mention"""
        if position is None:
            return ""
        
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]
        
        # Clean up context
        context = context.strip()
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context


# Helper functions for database operations (sync versions for psycopg2 cursor)
async def create_or_get_entity(
    cursor,
    entity_name: str,
    entity_type: str,
    canonical_name: str,
    description: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    workspace_id: Optional[str] = None
) -> int:
    """
    Create entity if it doesn't exist, or return existing entity ID

    Returns:
        entity_id
    """
    # Try to get existing entity
    cursor.execute("""
        SELECT id FROM kb_entities
        WHERE LOWER(canonical_name) = LOWER(%s)
        LIMIT 1
    """, [canonical_name])

    result = cursor.fetchone()

    if result:
        # Update mention count
        cursor.execute(
            "UPDATE kb_entities SET mention_count = mention_count + 1, updated_at = NOW() WHERE id = %s",
            [result[0]]
        )
        return result[0]

    # Create new entity
    cursor.execute("""
        INSERT INTO kb_entities (entity_name, entity_type, canonical_name, description, embedding, mention_count, workspace_id)
        VALUES (%s, %s, %s, %s, %s, 1, %s)
        RETURNING id
    """, [entity_name, entity_type, canonical_name, description, embedding, workspace_id])

    result = cursor.fetchone()
    return result[0]


async def create_entity_mention(
    cursor,
    knowledge_item_id: int,
    entity_id: int,
    mention_context: Optional[str] = None,
    confidence: float = 1.0,
    position: Optional[int] = None,
    extraction_method: str = "llm"
):
    """Create a link between entity and knowledge item"""
    cursor.execute("""
        INSERT INTO knowledge_entity_mentions 
        (knowledge_item_id, entity_id, mention_context, confidence, position_in_source, extraction_method)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (knowledge_item_id, entity_id, position_in_source) DO NOTHING
    """, [knowledge_item_id, entity_id, mention_context, confidence, position or 0, extraction_method])


async def create_entity_relationship(
    cursor,
    from_entity_name: str,
    to_entity_name: str,
    relationship_type: str,
    strength: float = 1.0,
    evidence_source_id: Optional[int] = None,
    evidence_text: Optional[str] = None,
    workspace_id: Optional[str] = None
):
    """Create relationship between two entities"""

    # Get entity IDs
    cursor.execute("SELECT id FROM kb_entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1", [from_entity_name.lower()])
    from_result = cursor.fetchone()

    cursor.execute("SELECT id FROM kb_entities WHERE LOWER(canonical_name) = LOWER(%s) LIMIT 1", [to_entity_name.lower()])
    to_result = cursor.fetchone()

    if not from_result or not to_result:
        logger.warning(f"Could not create relationship: entities not found ({from_entity_name} -> {to_entity_name})")
        return

    from_entity_id = from_result[0]
    to_entity_id = to_result[0]

    # Create relationship
    cursor.execute("""
        INSERT INTO entity_relationships
        (from_entity_id, to_entity_id, relationship_type, strength, evidence_source_id, evidence_text, workspace_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (from_entity_id, to_entity_id, relationship_type)
        DO UPDATE SET
            strength = GREATEST(entity_relationships.strength, EXCLUDED.strength),
            evidence_text = COALESCE(EXCLUDED.evidence_text, entity_relationships.evidence_text),
            updated_at = NOW()
    """, [from_entity_id, to_entity_id, relationship_type, strength, evidence_source_id, evidence_text, workspace_id])

