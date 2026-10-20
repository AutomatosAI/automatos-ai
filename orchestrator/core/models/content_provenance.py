"""Content provenance: one row per marked file Automatos generated (PRD-264 EUAI-10).

Article 50(2) of the AI Act asks that generated content be detectable as AI-generated,
by the people who receive it. A marked file's SHA-256 is recorded here when it is
produced, so the public detection route (``GET /api/public/provenance/{sha256}``)
can answer "made by Automatos AI: yes or no" for a file it is shown.

A row holds the file's hash, its media kind, the C2PA manifest id when the file is
signed, the generator (toolkit or model name) and when it was made. ``workspace_id``
is there for tenancy and erasure only: the public answer never reads it. No prompt,
user, file name or path is stored (Guidelines para. 94). Written only through
``services.content_provenance``. The Alembic revision ``prd264_content_provenance``
declares the same names.
"""
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID

from core.database.base import Base

# The media kinds a marked file can be. ck_content_provenance_kind allows exactly these.
PROVENANCE_KINDS = ("video", "image", "audio", "document")
SHA256_HEX_LENGTH = 64
KIND_MAX_LENGTH = 16
GENERATOR_MAX_LENGTH = 120
MANIFEST_ID_MAX_LENGTH = 255

UNIQUE_FILE_CONSTRAINT = "uq_content_provenance_ws_sha256"
SHA256_INDEX = "ix_content_provenance_sha256"


class ContentProvenance(Base):
    """One marked file: its SHA-256, kind, manifest id, generator and creation time.

    One row per workspace and hash (``uq_content_provenance_ws_sha256``), so writing
    the same file twice keeps the first row and its date. ``ix_content_provenance_sha256``
    serves the public lookup, which matches on the hash alone.
    """

    __tablename__ = "content_provenance"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    sha256 = Column(String(SHA256_HEX_LENGTH), nullable=False)
    kind = Column(String(KIND_MAX_LENGTH), nullable=False)
    manifest_id = Column(String(MANIFEST_ID_MAX_LENGTH), nullable=True)
    generator = Column(String(GENERATOR_MAX_LENGTH), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("workspace_id", "sha256", name=UNIQUE_FILE_CONSTRAINT),
        Index(SHA256_INDEX, "sha256"),
        CheckConstraint(
            "kind IN ('video', 'image', 'audio', 'document')",
            name="ck_content_provenance_kind",
        ),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="ck_content_provenance_sha256"),
        {"extend_existing": True},
    )
