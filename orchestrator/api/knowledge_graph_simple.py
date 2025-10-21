"""
Simplified Knowledge Graph API - Just the essentials that work
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["🧠 Knowledge Graph"])


# Models
class Entity(BaseModel):
    id: int
    entity_name: str
    entity_type: str
    mention_count: int
    importance_score: float


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


# Stats endpoint
@router.get("/stats/entities")
async def get_entity_stats(db: Session = Depends(get_db)):
    """Get entity statistics"""
    try:
        stats = db.execute(text("""
            SELECT 
                COUNT(*) as total_entities,
                COUNT(DISTINCT entity_type) as unique_types,
                (SELECT COUNT(*) FROM entity_relationships) as total_relationships
            FROM kb_entities
        """)).fetchone()
        
        types = db.execute(text("""
            SELECT entity_type, COUNT(*) as count
            FROM kb_entities
            GROUP BY entity_type
        """)).fetchall()
        
        top = db.execute(text("""
            SELECT entity_name, entity_type, mention_count, importance_score
            FROM kb_entities
            ORDER BY importance_score DESC
            LIMIT 10
        """)).fetchall()
        
        return {
            "overview": {
                "total_entities": stats[0] or 0,
                "unique_types": stats[1] or 0,
                "total_relationships": stats[2] or 0,
                "avg_mentions_per_entity": 0.0,
                "max_importance": 0.0
            },
            "by_type": [{"entity_type": row[0], "count": row[1]} for row in types],
            "top_entities": [
                {
                    "entity_name": row[0],
                    "entity_type": row[1],
                    "mention_count": row[2],
                    "importance_score": float(row[3] or 0)
                } for row in top
            ]
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "overview": {"total_entities": 0, "unique_types": 0, "total_relationships": 0},
            "by_type": [],
            "top_entities": []
        }


# Search entities
@router.get("/entities/search")
async def search_entities(
    db: Session = Depends(get_db),
    query: str = Query(..., min_length=1),
    limit: int = Query(20, le=100)
):
    """Search entities"""
    try:
        result = db.execute(text("""
            SELECT id, entity_name, entity_type, canonical_name, description, 
                   mention_count, importance_score, created_at
            FROM kb_entities
            WHERE entity_name ILIKE :pattern OR COALESCE(description, '') ILIKE :pattern
            ORDER BY importance_score DESC
            LIMIT :limit
        """), {"pattern": f"%{query}%", "limit": limit})
        
        rows = result.fetchall()
        
        return [
            {
                "id": row[0],
                "entity_name": row[1],
                "entity_type": row[2],
                "canonical_name": row[3],
                "description": row[4],
                "mention_count": row[5],
                "importance_score": float(row[6] or 0),
                "created_at": row[7]
            } for row in rows
        ]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


# Get entity graph
@router.get("/entities/{entity_id}/graph")
async def get_entity_graph(
    entity_id: int,
    db: Session = Depends(get_db),
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(50, le=200)
):
    """Get graph visualization data"""
    try:
        # Get center entity
        entity = db.execute(text("""
            SELECT id, entity_name, entity_type, importance_score, mention_count
            FROM kb_entities WHERE id = :id
        """), {"id": entity_id}).fetchone()
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Get related entities and relationships
        relationships = db.execute(text("""
            SELECT 
                er.from_entity_id,
                er.to_entity_id,
                er.relationship_type,
                er.strength,
                e1.entity_name as from_name,
                e1.entity_type as from_type,
                e1.importance_score as from_importance,
                e1.mention_count as from_mentions,
                e2.entity_name as to_name,
                e2.entity_type as to_type,
                e2.importance_score as to_importance,
                e2.mention_count as to_mentions
            FROM entity_relationships er
            JOIN kb_entities e1 ON er.from_entity_id = e1.id
            JOIN kb_entities e2 ON er.to_entity_id = e2.id
            WHERE er.from_entity_id = :eid OR er.to_entity_id = :eid
            LIMIT :max_nodes
        """), {"eid": entity_id, "max_nodes": max_nodes}).fetchall()
        
        # Build nodes and edges
        nodes_dict = {
            entity_id: GraphNode(
                id=entity[0],
                label=entity[1],
                type=entity[2],
                importance=float(entity[3] or 0),
                mention_count=entity[4] or 0
            )
        }
        
        edges = []
        
        for rel in relationships:
            # Add from node
            if rel[0] not in nodes_dict:
                nodes_dict[rel[0]] = GraphNode(
                    id=rel[0],
                    label=rel[4],
                    type=rel[5],
                    importance=float(rel[6] or 0),
                    mention_count=rel[7] or 0
                )
            
            # Add to node
            if rel[1] not in nodes_dict:
                nodes_dict[rel[1]] = GraphNode(
                    id=rel[1],
                    label=rel[8],
                    type=rel[9],
                    importance=float(rel[10] or 0),
                    mention_count=rel[11] or 0
                )
            
            # Add edge
            edges.append(GraphEdge(
                source=rel[0],
                target=rel[1],
                label=rel[2],
                strength=float(rel[3] or 1.0)
            ))
        
        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges,
            "center_entity": {
                "id": entity[0],
                "entity_name": entity[1],
                "entity_type": entity[2]
            },
            "metadata": {
                "depth": depth,
                "total_nodes": len(nodes_dict),
                "total_edges": len(edges)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


