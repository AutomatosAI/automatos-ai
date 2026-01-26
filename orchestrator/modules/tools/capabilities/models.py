"""
Composio Action Metadata Model
==============================

SQLAlchemy model for storing classified action metadata.
Enables capability-based filtering at runtime without API calls.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Boolean, Float, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.sql import func
import uuid

from core.models.core import Base


class ComposioActionMetadata(Base):
    """
    Stores LLM-classified capability metadata for Composio actions.

    This table enables:
    - Sync-time classification (classify once, cache forever)
    - Runtime capability filtering (local DB, no API calls)
    - Deterministic action eligibility based on intent
    - Separation of semantic capabilities from safety policies
    """
    __tablename__ = 'composio_action_metadata'
    __table_args__ = (
        # Composite index for runtime filtering
        Index('ix_composio_action_metadata_runtime_filter', 'app_id', 'destructive'),
        # Index for sync operations
        Index('ix_composio_action_metadata_sync', 'synced_at', 'manually_overridden'),
        {'extend_existing': True}
    )

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Identity - unique action identification
    app_id = Column(String(255), nullable=False, index=True,
                    comment='Composio app identifier (e.g., "slack", "github")')
    action_id = Column(String(512), nullable=False, unique=True,
                       comment='Full Composio action ID')
    action_name_normalized = Column(String(255), nullable=True,
                                    comment='LLM-friendly normalized name')

    # SEMANTIC CAPABILITIES (what the action can do)
    capabilities = Column(ARRAY(String), nullable=False, default=[],
                          comment='Capability tags like ["message.send"]')
    intent_keywords = Column(ARRAY(String), nullable=True, default=[],
                             comment='Keywords users might say')

    # SAFETY POLICY (separate from capabilities)
    risk_level = Column(String(50), nullable=False, default='safe',
                        comment='Risk classification: safe, cautious, dangerous')
    destructive = Column(Boolean, nullable=False, default=False,
                         comment='True for delete/remove/revoke actions')
    requires_confirmation = Column(Boolean, nullable=False, default=False,
                                   comment='True for actions needing confirmation')

    # Parameter metadata
    required_params = Column(JSONB, nullable=True,
                             comment='Required parameters list')
    optional_params = Column(JSONB, nullable=True,
                             comment='Optional parameters list')

    # Descriptions
    description_short = Column(String(255), nullable=True,
                               comment='Short description, <100 chars')
    description_original = Column(Text, nullable=True,
                                  comment='Original description from Composio')

    # Classification tracking
    classification_method = Column(String(50), nullable=False, default='heuristic',
                                   comment='How classified: heuristic or llm')
    classification_confidence = Column(Float, nullable=True,
                                        comment='Confidence score 0-1')
    classification_model = Column(String(100), nullable=True,
                                  comment='LLM model used if applicable')
    manually_overridden = Column(Boolean, nullable=False, default=False,
                                 comment='True if admin corrected classification')

    # Sync tracking
    composio_hash = Column(String(64), nullable=True,
                           comment='Hash of action spec for change detection')
    synced_at = Column(DateTime, nullable=True,
                       comment='Last sync from Composio')
    classified_at = Column(DateTime, nullable=True,
                           comment='When classification was performed')

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(),
                        onupdate=func.now())

    def __repr__(self) -> str:
        return f"<ComposioActionMetadata(action_id='{self.action_id}', caps={self.capabilities})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': str(self.id),
            'app_id': self.app_id,
            'action_id': self.action_id,
            'action_name_normalized': self.action_name_normalized,
            'capabilities': self.capabilities or [],
            'intent_keywords': self.intent_keywords or [],
            'risk_level': self.risk_level,
            'destructive': self.destructive,
            'requires_confirmation': self.requires_confirmation,
            'required_params': self.required_params,
            'optional_params': self.optional_params,
            'description_short': self.description_short,
            'classification_method': self.classification_method,
            'classification_confidence': self.classification_confidence,
            'manually_overridden': self.manually_overridden,
            'synced_at': self.synced_at.isoformat() if self.synced_at else None,
        }

    def matches_capabilities(self, required_caps: List[str]) -> bool:
        """Check if this action matches any of the required capabilities."""
        if not self.capabilities:
            return False
        return bool(set(self.capabilities) & set(required_caps))

    def is_safe_for_intent(self, intent: str, allow_destructive: bool = False) -> bool:
        """
        Check if this action is safe to use for a given intent.

        Args:
            intent: The user's intent text
            allow_destructive: Whether destructive actions are allowed

        Returns:
            True if action is safe to use
        """
        if self.destructive and not allow_destructive:
            # Check if intent explicitly mentions destructive action
            destructive_keywords = ['delete', 'remove', 'archive', 'revoke', 'destroy']
            intent_lower = intent.lower()
            if not any(kw in intent_lower for kw in destructive_keywords):
                return False

        return True

    @classmethod
    def from_classification(
        cls,
        app_id: str,
        action_id: str,
        classification: 'ActionClassification',
        description_original: Optional[str] = None,
        composio_hash: Optional[str] = None,
    ) -> 'ComposioActionMetadata':
        """Create metadata from a classification result."""
        return cls(
            app_id=app_id,
            action_id=action_id,
            action_name_normalized=classification.normalized_name,
            capabilities=classification.capabilities,
            intent_keywords=classification.intent_keywords,
            risk_level=classification.risk_level,
            destructive=classification.destructive,
            requires_confirmation=classification.requires_confirmation,
            required_params=classification.required_params,
            optional_params=classification.optional_params,
            description_short=classification.short_description,
            description_original=description_original,
            classification_method=classification.method,
            classification_confidence=classification.confidence,
            classification_model=classification.model,
            composio_hash=composio_hash,
            synced_at=datetime.utcnow(),
            classified_at=datetime.utcnow(),
        )
