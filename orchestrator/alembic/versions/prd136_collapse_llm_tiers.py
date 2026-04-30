"""PRD-136 — Collapse 12 LLM silos to 3 tiers (Auto / System / Embeddings).

Migrates existing system_settings rows from the legacy per-service categories
into the three canonical tiers, preserving any user-customized values along
the way.

Mapping:
  codegraph, chatbot, complexity_assessor, coordination,
  knowledge_graph, memory_management        →  system_llm
  general (embedding_*, vector_store_*, chunk_*, rag_rerank_*) → embeddings

Rules:
  - Per-tier user-customized value: first non-default-equal wins, logged.
  - Embedding key prefix `embedding_` is stripped on move (e.g.
    `embedding_model` → `embeddings.model`).
  - Two non-LLM feature flags survive the collapse and move to `general`:
      coordination.consistency_check_enabled →
        general.coordinator_consistency_check_enabled
      memory_management.store_max_chars →
        general.memory_store_max_chars
  - All other rows in retired categories are deleted.

Idempotent: safe to re-run. If new tier rows already exist, only fills gaps.

Standalone migration — `down_revision = None` so it can run in any order
relative to schema migrations. Pure data move.
"""

import logging

from alembic import op
from sqlalchemy import text

revision = "prd136_collapse_llm_tiers"
down_revision = None  # standalone — pure data migration
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


# Legacy LLM categories that fold into `system_llm`. Keys map old → new.
# `None` means "drop, no canonical equivalent in the new schema".
SYSTEM_LLM_KEY_MAP = {
    # generic LLM dials — keep canonical names
    "provider": "provider",
    "model": "model",
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "timeout_seconds": "timeout_seconds",
    "retry_attempts": "max_retries",
    "max_retries": "max_retries",
    # legacy per-service max-tokens variants → canonical
    "extraction_max_tokens": "max_tokens",
    "planner_max_tokens": "max_tokens",
    "task_max_tokens": "max_tokens",
}

LEGACY_LLM_CATEGORIES = (
    "codegraph",
    "chatbot",
    "complexity_assessor",
    "coordination",
    "knowledge_graph",
    "memory_management",
)

# Embedding-shaped keys living in `general` that move to `embeddings`.
GENERAL_TO_EMBEDDINGS = {
    "embedding_provider": "provider",
    "embedding_model": "model",
    "openai_embedding_model": "model",  # legacy alias
    "embedding_max_seq_length": "max_seq_length",
    "embedding_cache_dir": "cache_dir",
    "vector_store_type": "vector_store_type",
    "vector_store_dimensions": "dimensions",
    "chunk_size": "chunk_size",
    "chunk_overlap": "chunk_overlap",
    "rag_rerank_enabled": "rerank_enabled",
    "rag_rerank_model": "rerank_model",
}

# Non-LLM feature flags rescued from retired categories → general.
RESCUE_TO_GENERAL = {
    ("coordination", "consistency_check_enabled"): "coordinator_consistency_check_enabled",
    ("memory_management", "store_max_chars"): "memory_store_max_chars",
}


def _move_setting(conn, old_category: str, old_key: str, new_category: str, new_key: str) -> None:
    """Move a row to the new (category, key) target.

    Behavior:
    - If the source row is missing, no-op.
    - If a target row already exists with a non-default user value, keep target,
      delete source.
    - Otherwise: copy source.value to target (creating target if needed),
      then delete source.
    """
    src = conn.execute(text("""
        SELECT id, value, default_value
        FROM system_settings
        WHERE category = :cat AND key = :key
    """), {"cat": old_category, "key": old_key}).first()
    if not src:
        return

    src_id, src_value, src_default = src
    src_user_customized = (src_value is not None and src_value != src_default)

    tgt = conn.execute(text("""
        SELECT id, value, default_value
        FROM system_settings
        WHERE category = :cat AND key = :key
    """), {"cat": new_category, "key": new_key}).first()

    if tgt:
        tgt_id, tgt_value, tgt_default = tgt
        tgt_user_customized = (tgt_value is not None and tgt_value != tgt_default)
        # Prefer existing target if user customized; otherwise overwrite when source was customized.
        if not tgt_user_customized and src_user_customized:
            conn.execute(text("""
                UPDATE system_settings SET value = :v WHERE id = :id
            """), {"v": src_value, "id": tgt_id})
            logger.info(
                "PRD-136 migrate: %s.%s='%s' → %s.%s (user value preserved)",
                old_category, old_key, src_value, new_category, new_key,
            )
        else:
            logger.info(
                "PRD-136 migrate: %s.%s dropped (target %s.%s already set)",
                old_category, old_key, new_category, new_key,
            )
    else:
        # Create target row carrying source value forward.
        conn.execute(text("""
            INSERT INTO system_settings
                (category, key, value, default_value, value_type, description, created_by)
            VALUES (:cat, :key, :value, :default_value, 'string', NULL, 'prd136_migration')
        """), {
            "cat": new_category,
            "key": new_key,
            "value": src_value,
            "default_value": src_default,
        })
        logger.info(
            "PRD-136 migrate: %s.%s='%s' → created %s.%s",
            old_category, old_key, src_value, new_category, new_key,
        )

    conn.execute(text("DELETE FROM system_settings WHERE id = :id"), {"id": src_id})


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Rescue non-LLM feature flags into `general` first.
    for (old_cat, old_key), new_key in RESCUE_TO_GENERAL.items():
        _move_setting(conn, old_cat, old_key, "general", new_key)

    # 2. Collapse legacy LLM categories into `system_llm`.
    for legacy_cat in LEGACY_LLM_CATEGORIES:
        rows = conn.execute(text("""
            SELECT key FROM system_settings WHERE category = :cat
        """), {"cat": legacy_cat}).fetchall()
        for (legacy_key,) in rows:
            new_key = SYSTEM_LLM_KEY_MAP.get(legacy_key)
            if new_key is None:
                # Unmapped key — drop with a log.
                conn.execute(text("""
                    DELETE FROM system_settings WHERE category = :cat AND key = :key
                """), {"cat": legacy_cat, "key": legacy_key})
                logger.info(
                    "PRD-136 migrate: dropped %s.%s (no canonical equivalent)",
                    legacy_cat, legacy_key,
                )
                continue
            _move_setting(conn, legacy_cat, legacy_key, "system_llm", new_key)

    # 3. Lift embedding-shaped keys out of `general` → `embeddings`.
    for old_key, new_key in GENERAL_TO_EMBEDDINGS.items():
        _move_setting(conn, "general", old_key, "embeddings", new_key)

    # 4. Final sweep: any stragglers in retired categories get deleted.
    conn.execute(text("""
        DELETE FROM system_settings
        WHERE category IN ('codegraph','chatbot','complexity_assessor',
                           'coordination','knowledge_graph','memory_management')
    """))


def downgrade() -> None:
    """No-op — this is a forward-only data migration.

    The legacy categories are deliberately deleted (PRD-136 §5 — "no
    backwards-compat shims"). Restoring them would require re-running the
    pre-PRD-136 seed and is not supported. To roll back, restore from a DB
    snapshot taken before this migration ran.
    """
    raise NotImplementedError(
        "PRD-136 collapse is forward-only. Restore from snapshot to roll back."
    )
