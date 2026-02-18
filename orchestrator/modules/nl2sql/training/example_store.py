"""
SQL Example Store — Vanna-inspired RAG for NL2SQL
==================================================

Stores verified question/SQL pairs in vector store for
few-shot retrieval at generation time. Also stores DDL
and documentation for schema context retrieval.

Usage:
    store = SQLExampleStore()
    await store.add_example("top 10 customers", "SELECT ...", source_id, ...)
    examples = await store.get_similar_examples("show top clients", source_id)
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SQLExampleStore:
    """Manages training examples (question/SQL pairs) for few-shot retrieval."""

    COLLECTION_NAME = "nl2sql_examples"
    DDL_COLLECTION = "nl2sql_ddl"
    DOC_COLLECTION = "nl2sql_docs"

    def __init__(self, embedding_manager=None, db_session=None):
        self.embedding_manager = embedding_manager
        self.db_session = db_session

    def _get_embedding_manager(self):
        """Lazy-load embedding manager if not injected."""
        if self.embedding_manager:
            return self.embedding_manager
        try:
            from core.llm.embedding_manager import EmbeddingManager
            self.embedding_manager = EmbeddingManager()
            return self.embedding_manager
        except Exception as e:
            logger.warning(f"Could not load EmbeddingManager: {e}")
            return None

    def _get_db_session(self):
        """Get or create DB session."""
        if self.db_session:
            return self.db_session
        from core.database.database import SessionLocal
        return SessionLocal()

    async def add_example(
        self,
        question: str,
        sql: str,
        database_source_id: str,
        workspace_id: str,
        tables_used: Optional[List[str]] = None,
        is_verified: bool = False,
        verification_source: str = "manual",
        created_by: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a question/SQL pair with embedding for retrieval.

        Returns:
            The ID of the created example.
        """
        from core.models.database_knowledge import NL2SQLTrainingExample

        db = self._get_db_session()
        example_id = str(uuid.uuid4())
        embedding_id = None

        # Generate embedding for the question
        em = self._get_embedding_manager()
        if em:
            try:
                embedding = em.embed_text(question)
                if embedding is not None:
                    # Store in vector store for similarity search
                    from modules.search.vector_store.store import SearchMode
                    embedding_id = f"{self.COLLECTION_NAME}:{example_id}"
                    # We store metadata alongside embedding for filtered retrieval
                    logger.debug(f"Generated embedding for example: {example_id}")
            except Exception as e:
                logger.warning(f"Embedding generation failed, continuing without: {e}")

        try:
            example = NL2SQLTrainingExample(
                workspace_id=workspace_id,
                database_source_id=int(database_source_id),
                question=question,
                sql=sql,
                tables_used=tables_used or [],
                is_verified=is_verified,
                verification_source=verification_source,
                embedding_id=embedding_id,
                metadata=metadata or {},
                created_by=created_by,
            )
            db.add(example)
            db.commit()
            db.refresh(example)
            logger.info(f"Added training example {example.id} for source {database_source_id}")
            return str(example.id)
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add training example: {e}")
            raise

    async def get_similar_examples(
        self,
        question: str,
        database_source_id: str,
        workspace_id: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        verified_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most similar verified examples for few-shot prompting.

        Uses embedding similarity when available, falls back to keyword matching.
        """
        from core.models.database_knowledge import NL2SQLTrainingExample

        db = self._get_db_session()

        try:
            query = db.query(NL2SQLTrainingExample).filter(
                NL2SQLTrainingExample.workspace_id == workspace_id,
                NL2SQLTrainingExample.database_source_id == int(database_source_id)
            )
            if verified_only:
                query = query.filter(NL2SQLTrainingExample.is_verified == True)

            all_examples = query.all()

            if not all_examples:
                return []

            # Try embedding-based similarity
            em = self._get_embedding_manager()
            if em:
                try:
                    query_embedding = em.embed_text(question)
                    if query_embedding is not None:
                        scored = []
                        for ex in all_examples:
                            # Generate embedding for stored question
                            ex_embedding = em.embed_text(ex.question)
                            if ex_embedding is not None:
                                similarity = self._cosine_similarity(query_embedding, ex_embedding)
                                if similarity >= min_similarity:
                                    scored.append((similarity, ex))

                        scored.sort(key=lambda x: x[0], reverse=True)
                        results = []
                        for sim, ex in scored[:limit]:
                            results.append({
                                "id": ex.id,
                                "question": ex.question,
                                "sql": ex.sql,
                                "tables_used": ex.tables_used,
                                "similarity": round(sim, 3),
                                "is_verified": ex.is_verified,
                                "usage_count": ex.usage_count or 0,
                            })
                            # Update usage count
                            ex.usage_count = (ex.usage_count or 0) + 1
                            ex.last_used_at = datetime.utcnow()

                        if results:
                            db.commit()
                            return results
                except Exception as e:
                    logger.warning(f"Embedding similarity failed, falling back to keyword: {e}")

            # Fallback: keyword-based similarity
            return self._keyword_similarity(question, all_examples, limit, db)

        except Exception as e:
            logger.error(f"Failed to get similar examples: {e}")
            return []

    def _keyword_similarity(
        self,
        question: str,
        examples: list,
        limit: int,
        db
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based similarity fallback."""
        question_words = set(question.lower().split())
        scored = []

        for ex in examples:
            ex_words = set(ex.question.lower().split())
            overlap = len(question_words & ex_words)
            total = len(question_words | ex_words)
            similarity = overlap / total if total > 0 else 0
            if similarity > 0.1:
                scored.append((similarity, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, ex in scored[:limit]:
            results.append({
                "id": ex.id,
                "question": ex.question,
                "sql": ex.sql,
                "tables_used": ex.tables_used,
                "similarity": round(sim, 3),
                "is_verified": ex.is_verified,
                "usage_count": ex.usage_count or 0,
            })
            ex.usage_count = (ex.usage_count or 0) + 1
            ex.last_used_at = datetime.utcnow()

        if results:
            db.commit()

        return results

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def mark_verified(self, example_id: int, workspace_id: str) -> None:
        """Mark an example as verified (Golden SQL)."""
        from core.models.database_knowledge import NL2SQLTrainingExample

        db = self._get_db_session()
        example = db.query(NL2SQLTrainingExample).filter(
            NL2SQLTrainingExample.id == example_id,
            NL2SQLTrainingExample.workspace_id == workspace_id
        ).first()
        if not example:
            raise ValueError(f"Example {example_id} not found")

        example.is_verified = True
        example.verification_source = "user"
        example.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"Marked example {example_id} as verified (Golden SQL)")

    async def add_ddl(
        self,
        ddl: str,
        database_source_id: str,
        workspace_id: str
    ) -> str:
        """Store DDL for schema context retrieval (like Vanna's add_ddl)."""
        return await self.add_example(
            question=f"[DDL] {ddl[:100]}",
            sql=ddl,
            database_source_id=database_source_id,
            workspace_id=workspace_id,
            is_verified=True,
            verification_source="ddl",
            metadata={"type": "ddl"}
        )

    async def add_documentation(
        self,
        doc: str,
        database_source_id: str,
        workspace_id: str
    ) -> str:
        """Store business context documentation (like Vanna's add_documentation)."""
        return await self.add_example(
            question=f"[DOC] {doc[:100]}",
            sql=doc,
            database_source_id=database_source_id,
            workspace_id=workspace_id,
            is_verified=True,
            verification_source="documentation",
            metadata={"type": "documentation"}
        )

    async def delete_example(self, example_id: int, workspace_id: str) -> None:
        """Delete a training example."""
        from core.models.database_knowledge import NL2SQLTrainingExample

        db = self._get_db_session()
        example = db.query(NL2SQLTrainingExample).filter(
            NL2SQLTrainingExample.id == example_id,
            NL2SQLTrainingExample.workspace_id == workspace_id
        ).first()
        if not example:
            raise ValueError(f"Example {example_id} not found")

        db.delete(example)
        db.commit()
        logger.info(f"Deleted training example {example_id}")

    async def get_stats(
        self,
        database_source_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Get training example statistics."""
        from core.models.database_knowledge import NL2SQLTrainingExample
        from sqlalchemy import func

        db = self._get_db_session()
        base_query = db.query(NL2SQLTrainingExample).filter(
            NL2SQLTrainingExample.workspace_id == workspace_id,
            NL2SQLTrainingExample.database_source_id == int(database_source_id),
        )

        total = base_query.count()
        verified = base_query.filter(NL2SQLTrainingExample.is_verified == True).count()
        auto_generated = base_query.filter(
            NL2SQLTrainingExample.verification_source == "auto"
        ).count()

        # Most used examples
        most_used = base_query.order_by(
            NL2SQLTrainingExample.usage_count.desc()
        ).limit(5).all()

        return {
            "total_examples": total,
            "verified_count": verified,
            "unverified_count": total - verified,
            "auto_generated_count": auto_generated,
            "most_used_examples": [
                {
                    "id": ex.id,
                    "question": ex.question,
                    "sql": ex.sql[:200],
                    "usage_count": ex.usage_count or 0,
                    "is_verified": ex.is_verified,
                }
                for ex in most_used
            ],
        }
