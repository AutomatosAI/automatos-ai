"""
Playbook Miner
==============

Mines workflow execution patterns to identify reusable playbooks.
"""

from __future__ import annotations

from typing import Optional, List, Dict
from sqlalchemy.orm import Session
import json


class PlaybookMiner:
    """
    Mines workflow patterns from execution history to identify 
    common successful patterns that can be turned into reusable playbooks.
    """
    
    def __init__(self, db: Session, tenant_id: Optional[str] = None):
        self.db = db
        self.tenant_id = tenant_id

    def _fetch_sequences(self, limit: int = 1000) -> List[List[str]]:
        """Fetch action sequences from execution history."""
        # TODO: Implement real sequence fetching from run_events table
        # Currently returns example sequences for demo purposes
        return [
            ["retrieve", "assemble_context", "tool:web_search"],
            ["retrieve", "assemble_context", "tool:db_query"],
            ["retrieve", "assemble_context", "tool:web_search"],
            ["retrieve", "assemble_context", "tool:web_search"],
        ]

    def mine(self, min_support: int = 5, max_len: int = 6) -> List[Dict]:
        """
        Mine patterns from execution sequences.
        
        Args:
            min_support: Minimum occurrence count for a pattern to be included
            max_len: Maximum pattern length to consider
            
        Returns:
            List of pattern dicts with 'pattern' and 'support' keys
        """
        counts: Dict[str, int] = {}
        for seq in self._fetch_sequences():
            key = ",".join(seq)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"pattern": k.split(","), "support": v}
            for k, v in counts.items()
            if v >= min_support
        ]

    def persist_top(self, top_k: int = 20, min_support: int = 5, name_prefix: str = "auto") -> List[Dict]:
        """
        Mine and persist the top K patterns as playbooks.
        
        Args:
            top_k: Number of top patterns to persist
            min_support: Minimum support threshold
            name_prefix: Prefix for auto-generated playbook names
            
        Returns:
            List of created playbook patterns
        """
        rows = self.mine(min_support=min_support)[:top_k]
        created = []
        for idx, r in enumerate(rows, start=1):
            self.db.execute(
                """
                INSERT INTO playbooks (id, name, tenant_id, pattern, support)
                VALUES (gen_random_uuid(), :name, :tenant_id, :pattern::jsonb, :support)
                """,
                {
                    "name": f"{name_prefix}-{idx}",
                    "tenant_id": self.tenant_id,
                    "pattern": json.dumps(r["pattern"]),
                    "support": r["support"],
                },
            )
            created.append(r)
        self.db.commit()
        return created



