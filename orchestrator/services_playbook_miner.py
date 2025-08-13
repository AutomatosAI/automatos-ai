from __future__ import annotations

from typing import Optional, List, Dict
from sqlalchemy.orm import Session


class PlaybookMiner:
    def __init__(self, db: Session, tenant_id: Optional[str] = None):
        self.db = db
        self.tenant_id = tenant_id

    def _fetch_sequences(self, limit: int = 1000) -> List[List[str]]:
        # Minimal stub: in a real system, fetch from run_events
        # Here we return a few fake sequences for demo purposes
        return [
            ["retrieve", "assemble_context", "tool:web_search"],
            ["retrieve", "assemble_context", "tool:db_query"],
            ["retrieve", "assemble_context", "tool:web_search"],
            ["retrieve", "assemble_context", "tool:web_search"],
        ]

    def mine(self, min_support: int = 5, max_len: int = 6) -> List[Dict]:
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
                    "pattern": __import__("json").dumps(r["pattern"]),
                    "support": r["support"],
                },
            )
            created.append(r)
        self.db.commit()
        return created


