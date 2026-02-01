"""
Base ingestor interface (PRD-50: Universal Orchestrator Router).

All channel-specific ingestors inherit from BaseIngestor and implement
the `ingest()` method to normalise channel-specific payloads into a
uniform RequestEnvelope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models.routing import RequestEnvelope


class BaseIngestor(ABC):
    """Abstract base class for all channel ingestors."""

    @abstractmethod
    def ingest(self, **kwargs: Any) -> RequestEnvelope:
        """Transform channel-specific input into a RequestEnvelope.

        Subclasses must implement this method.
        """
        ...
