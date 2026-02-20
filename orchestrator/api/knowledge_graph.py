"""
Knowledge Graph API Endpoints
Provides entity search, relationship queries, and graph visualization data
Part of PRD-21: Hybrid RAG Integration
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["🧠 Knowledge Graph"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class Entity(BaseModel):
    id: int
    entity_name: str
    entity_type: str
    canonical_name: str
    description: Optional[str] = None
    mention_count: int = 0
    importance_score: float = 0.0
    created_at: datetime


class EntityRelationship(BaseModel):
    id: int
    from_entity_id: int
    from_entity_name: str
    to_entity_id: int
    to_entity_name: str
    relationship_type: str
    strength: float
    evidence_text: Optional[str] = None


class GraphNode(BaseModel):
    id: int
    label: str
    type: str
    importance: float
    mention_count: int


class GraphEdge(BaseModel):
    source: int
    target: int
    label: str
    strength: float


class KnowledgeGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    center_entity: Optional[Entity] = None
    metadata: Dict[str, Any] = {}


class EntitySearchResult(BaseModel):
    entity: Entity
    knowledge_items_count: int
    relationships_count: int
    related_entities: List[Entity] = []


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/entities", response_model=List[Entity])
async def list_entities(
    db: Session = Depends(get_db),
    entity_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    sort_by: str = Query("importance", regex="^(importance|mentions|name)$")
):
    """
    List all entities with optional filtering
    
    Args:
        entity_type: Filter by entity type (technology, concept, person, etc.)
        limit: Maximum results to return
        offset: Pagination offset
        sort_by: Sort field (importance, mentions, name)
    """
    try:
        # Build query
        where_clause = "WHERE entity_type = :entity_type" if entity_type else ""
        
        order_map = {
            "importance": "importance_score DESC",
            "mentions": "mention_count DESC",
            "name": "entity_name ASC"
        }
        order_clause = order_map.get(sort_by, "importance_score DESC")
        
        query_text = f"""
            SELECT id, entity_name, entity_type, canonical_name, description, 
                   mention_count, importance_score, created_at
            FROM kb_entities
            {where_clause}
            ORDER BY {order_clause}
            LIMIT :limit OFFSET :offset
        """
        
        params = {"limit": limit, "offset": offset}
        if entity_type:
            params["entity_type"] = entity_type
        
        result = db.execute(text(query_text), params)
        rows = result.fetchall()
        
        return [Entity(
            id=row[0],
            entity_name=row[1],
            entity_type=row[2],
            canonical_name=row[3],
            description=row[4],
            mention_count=row[5],
            importance_score=row[6],
            created_at=row[7]
        ) for row in rows]
        
    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/entities/search")
async def search_entities(
    db: Session = Depends(get_db),
    query: str = Query(..., min_length=2),
    limit: int = Query(20, le=100)
):
    """
    Search entities by name (full-text search)
    
    Args:
        query: Search query
        limit: Maximum results
    """
    try:
        search_query = text("""
            SELECT id, entity_name, entity_type, canonical_name, description, 
                   mention_count, importance_score, created_at
            FROM kb_entities
            WHERE entity_name ILIKE :search_pattern
               OR description ILIKE :search_pattern
            ORDER BY importance_score DESC, mention_count DESC
            LIMIT :limit
        """)
        
        result = db.execute(search_query, {"search_pattern": f"%{query}%", "limit": limit})
        rows = result.fetchall()
        
        return [Entity(
            id=row[0],
            entity_name=row[1],
            entity_type=row[2],
            canonical_name=row[3],
            description=row[4],
            mention_count=row[5],
            importance_score=row[6],
            created_at=row[7]
        ) for row in rows]
    
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/entities/{entity_id}", response_model=EntitySearchResult)
async def get_entity_details(
    entity_id: int,
    include_related: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about an entity

    Args:
        entity_id: Entity ID
        include_related: Whether to include related entities
    """
    try:
        # Get entity
        entity_row = db.execute(text("""
            SELECT id, entity_name, entity_type, canonical_name, description,
                   mention_count, importance_score, created_at
            FROM kb_entities
            WHERE id = :entity_id
        """), {"entity_id": entity_id}).fetchone()

        if not entity_row:
            raise HTTPException(status_code=404, detail="Entity not found")

        entity = Entity(
            id=entity_row[0], entity_name=entity_row[1], entity_type=entity_row[2],
            canonical_name=entity_row[3], description=entity_row[4],
            mention_count=entity_row[5], importance_score=entity_row[6],
            created_at=entity_row[7]
        )

        # Count knowledge items mentioning this entity
        count_row = db.execute(text("""
            SELECT COUNT(DISTINCT knowledge_item_id) as count
            FROM knowledge_entity_mentions
            WHERE entity_id = :entity_id
        """), {"entity_id": entity_id}).fetchone()
        knowledge_items_count = count_row[0] if count_row else 0

        # Count relationships
        rel_count_row = db.execute(text("""
            SELECT COUNT(*) as count
            FROM entity_relationships
            WHERE from_entity_id = :entity_id OR to_entity_id = :entity_id
        """), {"entity_id": entity_id}).fetchone()
        relationships_count = rel_count_row[0] if rel_count_row else 0

        # Get related entities
        related_entities = []
        if include_related:
            related_rows = db.execute(text("""
                SELECT DISTINCT e.id, e.entity_name, e.entity_type, e.canonical_name,
                       e.description, e.mention_count, e.importance_score, e.created_at
                FROM kb_entities e
                JOIN entity_relationships er ON
                    (er.from_entity_id = :entity_id AND er.to_entity_id = e.id) OR
                    (er.to_entity_id = :entity_id AND er.from_entity_id = e.id)
                ORDER BY e.importance_score DESC
                LIMIT 10
            """), {"entity_id": entity_id}).fetchall()
            related_entities = [Entity(
                id=row[0], entity_name=row[1], entity_type=row[2],
                canonical_name=row[3], description=row[4],
                mention_count=row[5], importance_score=row[6],
                created_at=row[7]
            ) for row in related_rows]

        return EntitySearchResult(
            entity=entity,
            knowledge_items_count=knowledge_items_count,
            relationships_count=relationships_count,
            related_entities=related_entities
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/entities/{entity_id}/graph", response_model=KnowledgeGraphResponse)
async def get_entity_graph(
    entity_id: int,
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    """
    Get graph visualization data for an entity

    Args:
        entity_id: Center entity ID
        depth: How many relationship hops to include
        max_nodes: Maximum nodes to return

    Returns:
        Graph structure with nodes and edges for visualization
    """
    try:
        # Get center entity
        entity_row = db.execute(text("""
            SELECT id, entity_name, entity_type, canonical_name, description,
                   mention_count, importance_score, created_at
            FROM kb_entities
            WHERE id = :entity_id
        """), {"entity_id": entity_id}).fetchone()

        if not entity_row:
            raise HTTPException(status_code=404, detail="Entity not found")

        center_entity = Entity(
            id=entity_row[0], entity_name=entity_row[1], entity_type=entity_row[2],
            canonical_name=entity_row[3], description=entity_row[4],
            mention_count=entity_row[5], importance_score=entity_row[6],
            created_at=entity_row[7]
        )

        # Get related entities with BFS traversal
        graph_rows = db.execute(text("""
            WITH RECURSIVE entity_graph AS (
                SELECT
                    CASE WHEN from_entity_id = :entity_id THEN to_entity_id ELSE from_entity_id END as entity_id,
                    from_entity_id, to_entity_id, relationship_type, strength, 1 as hop_count
                FROM entity_relationships
                WHERE from_entity_id = :entity_id OR to_entity_id = :entity_id
                UNION
                SELECT
                    CASE WHEN er.from_entity_id = eg.entity_id THEN er.to_entity_id ELSE er.from_entity_id END,
                    er.from_entity_id, er.to_entity_id, er.relationship_type, er.strength, eg.hop_count + 1
                FROM entity_relationships er
                INNER JOIN entity_graph eg ON (er.from_entity_id = eg.entity_id OR er.to_entity_id = eg.entity_id)
                WHERE eg.hop_count < :depth
            )
            SELECT DISTINCT eg.entity_id, eg.from_entity_id, eg.to_entity_id,
                   eg.relationship_type, eg.strength, e.entity_name, e.entity_type,
                   e.importance_score, e.mention_count
            FROM entity_graph eg
            JOIN kb_entities e ON eg.entity_id = e.id
            ORDER BY eg.hop_count ASC, e.importance_score DESC
            LIMIT :max_nodes
        """), {"entity_id": entity_id, "depth": depth, "max_nodes": max_nodes}).fetchall()

        # Build nodes and edges
        nodes = [GraphNode(
            id=center_entity.id, label=center_entity.entity_name,
            type=center_entity.entity_type, importance=center_entity.importance_score,
            mention_count=center_entity.mention_count
        )]

        seen_nodes = {center_entity.id}
        edges = []

        for row in graph_rows:
            if row[0] not in seen_nodes:
                nodes.append(GraphNode(
                    id=row[0], label=row[5], type=row[6],
                    importance=row[7], mention_count=row[8]
                ))
                seen_nodes.add(row[0])

            edges.append(GraphEdge(
                source=row[1], target=row[2], label=row[3], strength=row[4]
            ))

        return KnowledgeGraphResponse(
            nodes=nodes, edges=edges, center_entity=center_entity,
            metadata={"depth": depth, "total_nodes": len(nodes), "total_edges": len(edges)}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity graph: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/entities/{entity_id}/documents")
async def get_entity_documents(
    entity_id: int,
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db)
):
    """
    Get all knowledge items (documents, tables, formulas, etc.) mentioning this entity

    Args:
        entity_id: Entity ID
        limit: Maximum results
    """
    try:
        results = db.execute(text("""
            SELECT
                ki.id, ki.title, ki.content, ki.summary, ki.quality_score,
                ki.created_at, kt.display_name as kb_type,
                em.mention_context, em.confidence
            FROM knowledge_items ki
            JOIN knowledge_entity_mentions em ON ki.id = em.knowledge_item_id
            JOIN kb_types kt ON ki.kb_type_id = kt.id
            WHERE em.entity_id = :entity_id
            ORDER BY ki.quality_score DESC, ki.created_at DESC
            LIMIT :limit
        """), {"entity_id": entity_id, "limit": limit}).fetchall()

        return [
            {
                "id": row[0], "title": row[1], "content": row[2],
                "summary": row[3], "quality_score": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "kb_type": row[6], "mention_context": row[7],
                "confidence": row[8]
            }
            for row in results
        ]

    except Exception as e:
        logger.error(f"Error getting entity documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/entities/find-connection")
async def find_connection(
    entity_a: str = Query(..., description="First entity name"),
    entity_b: str = Query(..., description="Second entity name"),
    max_hops: int = Query(3, ge=1, le=5),
    db: Session = Depends(get_db)
):
    """
    Find connection path between two entities

    Args:
        entity_a: First entity name
        entity_b: Second entity name
        max_hops: Maximum relationship hops to search

    Returns:
        Shortest path and supporting evidence
    """
    try:
        # Get entity IDs
        entity_a_row = db.execute(text(
            "SELECT id, entity_name FROM kb_entities WHERE LOWER(entity_name) = LOWER(:name) LIMIT 1"
        ), {"name": entity_a}).fetchone()
        entity_b_row = db.execute(text(
            "SELECT id, entity_name FROM kb_entities WHERE LOWER(entity_name) = LOWER(:name) LIMIT 1"
        ), {"name": entity_b}).fetchone()

        if not entity_a_row or not entity_b_row:
            raise HTTPException(status_code=404, detail="One or both entities not found")

        entity_a_id = entity_a_row[0]
        entity_b_id = entity_b_row[0]

        # Find shortest path using recursive CTE
        path_row = db.execute(text("""
            WITH RECURSIVE entity_path AS (
                SELECT from_entity_id, to_entity_id, relationship_type, evidence_text,
                       1 as depth, ARRAY[from_entity_id, to_entity_id] as path,
                       ARRAY[relationship_type] as relationship_path
                FROM entity_relationships
                WHERE from_entity_id = :entity_a_id
                UNION
                SELECT er.from_entity_id, er.to_entity_id, er.relationship_type,
                       er.evidence_text, ep.depth + 1,
                       ep.path || er.to_entity_id,
                       ep.relationship_path || er.relationship_type
                FROM entity_relationships er
                INNER JOIN entity_path ep ON er.from_entity_id = ep.to_entity_id
                WHERE ep.depth < :max_hops
                  AND NOT er.to_entity_id = ANY(ep.path)
            )
            SELECT * FROM entity_path WHERE to_entity_id = :entity_b_id
            ORDER BY depth ASC LIMIT 1
        """), {"entity_a_id": entity_a_id, "entity_b_id": entity_b_id, "max_hops": max_hops}).fetchone()

        if not path_row:
            return {
                "found": False,
                "message": f"No connection found between '{entity_a}' and '{entity_b}' within {max_hops} hops"
            }

        # path_row: (from_entity_id, to_entity_id, relationship_type, evidence_text, depth, path, relationship_path)
        depth = path_row[4]
        path_entity_ids = path_row[5]
        relationship_path = path_row[6]

        # Get entity names for path
        entities_rows = db.execute(text(
            "SELECT id, entity_name FROM kb_entities WHERE id = ANY(:ids)"
        ), {"ids": path_entity_ids}).fetchall()
        entity_names = {row[0]: row[1] for row in entities_rows}

        # Build path explanation
        path_steps = []
        for i in range(len(relationship_path)):
            path_steps.append({
                "from": entity_names.get(path_entity_ids[i], "Unknown"),
                "to": entity_names.get(path_entity_ids[i + 1], "Unknown"),
                "relationship": relationship_path[i]
            })

        return {
            "found": True,
            "depth": depth,
            "path": path_steps,
            "summary": " -> ".join([entity_names.get(eid, "Unknown") for eid in path_entity_ids])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding connection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/entities")
async def get_entity_stats(db: Session = Depends(get_db)):
    """Get statistics about entities in the knowledge graph"""
    try:
        # Get overview stats
        stats_query = text("""
            SELECT 
                COUNT(*) as total_entities,
                COUNT(DISTINCT entity_type) as unique_types,
                (SELECT COUNT(*) FROM entity_relationships) as total_relationships,
                COALESCE(AVG(mention_count), 0) as avg_mentions_per_entity,
                COALESCE(MAX(importance_score), 0) as max_importance
            FROM kb_entities
        """)
        
        stats_row = db.execute(stats_query).fetchone()
        
        # Entity type breakdown
        type_breakdown_query = text("""
            SELECT entity_type, COUNT(*) as count
            FROM kb_entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        type_breakdown = db.execute(type_breakdown_query).fetchall()
        
        # Top entities
        top_entities_query = text("""
            SELECT entity_name, entity_type, mention_count, importance_score
            FROM kb_entities
            ORDER BY importance_score DESC
            LIMIT 10
        """)
        top_entities = db.execute(top_entities_query).fetchall()
        
        return {
            "overview": {
                "total_entities": stats_row[0],
                "unique_types": stats_row[1],
                "total_relationships": stats_row[2],
                "avg_mentions_per_entity": float(stats_row[3]),
                "max_importance": float(stats_row[4])
            },
            "by_type": [{"entity_type": row[0], "count": row[1]} for row in type_breakdown],
            "top_entities": [
                {
                    "entity_name": row[0],
                    "entity_type": row[1],
                    "mention_count": row[2],
                    "importance_score": float(row[3])
                } for row in top_entities
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting entity stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

