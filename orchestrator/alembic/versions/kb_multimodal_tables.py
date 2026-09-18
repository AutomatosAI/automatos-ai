"""kb_tables / kb_formulas / kb_images — the multimodal knowledge tables, back on every path.

The Knowledge Bases page's Multimodal tab (``api/knowledge_multimodal.py``), the
agents' table/formula search tools (``modules/rag/services/multimodal_knowledge_tools.py``)
and the document ingester's table extraction (``modules/rag/ingestion/manager.py``) all
read and write three tables that NO migration and NO model ever declared. They lived
only in the hand-maintained ``init_complete_schema.sql``, which PRD-209 retired on
2026-08-29 when fresh databases moved to the CI-proven create_all + raw-DDL path — a
path that carried ``kb_types`` and ``knowledge_items`` across but not these three.
Since then every fresh install 500s on the Multimodal tab
(``relation "kb_tables" does not exist``) and silently skips table extraction.

Production still has them from the original init (verified 2026-09-18: kb_types 9 rows,
kb_tables 862, kb_formulas 134 901, kb_images 0), so every statement here is
``IF NOT EXISTS`` / ``ON CONFLICT DO NOTHING`` and the upgrade is a no-op there — the
idempotent house style of ``users_last_sign_in_column``. ``kb_types`` is also seeded
with the nine reference rows production carries, because the Multimodal tab lists
types from that table and a fresh install had none.

The fresh-clone boot path gets the same DDL in ``scripts/init_test_db.py`` (it stamps
this head without running it); ``tests/test_kb_multimodal_tables.py`` holds the two
copies together.

Revision ID: kb_multimodal_tables
Revises: tool_execution_logs_workspace_user_idx
Create Date: 2026-09-18
"""
from alembic import op
from sqlalchemy import text

revision = "kb_multimodal_tables"
down_revision = "tool_execution_logs_workspace_user_idx"
branch_labels = None
depends_on = None

KB_TABLES = """CREATE TABLE IF NOT EXISTS kb_tables (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    headers JSONB NOT NULL,
    data_types JSONB,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    markdown_representation TEXT,
    csv_data TEXT,
    json_data JSONB,
    has_header_row BOOLEAN DEFAULT true,
    is_numeric BOOLEAN DEFAULT false,
    has_totals BOOLEAN DEFAULT false,
    caption TEXT,
    footnotes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    workspace_id UUID REFERENCES workspaces(id)
)"""

KB_FORMULAS = """CREATE TABLE IF NOT EXISTS kb_formulas (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    latex TEXT NOT NULL,
    mathml TEXT,
    ascii_math TEXT,
    variables JSONB,
    operators JSONB,
    complexity_level VARCHAR(50),
    formula_type VARCHAR(100),
    domain VARCHAR(100),
    rendered_svg TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    workspace_id UUID REFERENCES workspaces(id)
)"""

# ``__VISUAL_EMBEDDING__`` is a pgvector column where the extension exists and
# nothing on stock postgres, matching ``scripts/init_test_db._with_embedding``.
KB_IMAGES = """CREATE TABLE IF NOT EXISTS kb_images (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    width INTEGER,
    height INTEGER,
    format VARCHAR(50),
    file_size_bytes INTEGER,
    description TEXT,
    caption TEXT,
    alt_text TEXT,
    detected_objects JSONB,
    detected_text TEXT,
    image_data BYTEA,
    thumbnail_data BYTEA,
    storage_path VARCHAR(500),
    __VISUAL_EMBEDDING__
    created_at TIMESTAMP DEFAULT NOW(),
    workspace_id UUID REFERENCES workspaces(id)
)"""

KB_TYPES_SEED = """INSERT INTO kb_types (type_name, display_name, description, icon, processor_class, storage_strategy,
                      supports_embedding, supports_search, supports_relationships, enabled)
VALUES
    ('document', 'Documents', 'Text documents (PDF, DOCX, MD, TXT)', 'file-text', 'DocumentProcessor', 'pgvector', true, true, false, true),
    ('codegraph', 'Code Graph', 'Source code symbols and relationships', 'code', 'CodeGraphProcessor', 'pgvector', true, true, true, true),
    ('table', 'Tables', 'Structured tabular data from documents', 'table', 'TableProcessor', 'hybrid', true, true, false, true),
    ('image', 'Images', 'Images and visual content with descriptions', 'image', 'ImageProcessor', 'hybrid', true, true, false, true),
    ('formula', 'Formulas', 'Mathematical formulas and equations (LaTeX)', 'function', 'FormulaProcessor', 'pgvector', true, true, false, true),
    ('diagram', 'Diagrams', 'Diagrams and flowcharts', 'diagram', 'DiagramProcessor', 'hybrid', true, true, true, true),
    ('knowledge_graph', 'Knowledge Graph', 'Concept relationships and knowledge nodes', 'network', 'KnowledgeGraphProcessor', 'pgvector', true, true, true, true),
    ('memory', 'Agent Memory', 'Agent experiences and learned patterns', 'brain', 'MemoryProcessor', 'pgvector', true, true, true, true),
    ('entity', 'Entity', 'Extracted entities from documents (people, concepts, technologies)', NULL, 'EntityProcessor', NULL, true, true, false, true)
ON CONFLICT (type_name) DO NOTHING"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_kb_tables_knowledge_item ON kb_tables(knowledge_item_id)",
    "CREATE INDEX IF NOT EXISTS idx_kb_tables_size ON kb_tables(row_count, column_count)",
    "CREATE INDEX IF NOT EXISTS idx_kb_formulas_knowledge_item ON kb_formulas(knowledge_item_id)",
    "CREATE INDEX IF NOT EXISTS idx_kb_formulas_type ON kb_formulas(formula_type)",
    "CREATE INDEX IF NOT EXISTS idx_kb_formulas_domain ON kb_formulas(domain)",
    "CREATE INDEX IF NOT EXISTS idx_kb_images_knowledge_item ON kb_images(knowledge_item_id)",
    "CREATE INDEX IF NOT EXISTS idx_kb_images_format ON kb_images(format)"
]

VISUAL_EMBEDDING_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_kb_images_visual_embedding ON kb_images USING ivfflat (visual_embedding vector_cosine_ops) WITH (lists = 100)"
)


def _has_vector(conn) -> bool:
    return bool(conn.execute(text("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector')")).scalar())


def upgrade() -> None:
    conn = op.get_bind()
    has_vector = _has_vector(conn)
    op.execute(KB_TABLES)
    op.execute(KB_FORMULAS)
    op.execute(KB_IMAGES.replace("__VISUAL_EMBEDDING__", "visual_embedding vector(512)," if has_vector else ""))
    for stmt in INDEXES:
        op.execute(stmt)
    if has_vector:
        op.execute(VISUAL_EMBEDDING_INDEX)
    op.execute(KB_TYPES_SEED)


def downgrade() -> None:
    # Deliberately a no-op: production's rows in these tables predate this
    # migration, which only ever recreates what the retired init SQL built.
    pass
