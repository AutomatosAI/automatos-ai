"""
Token count estimator.

Fast path: character-based (4 chars ≈ 1 token).
Precise path: tiktoken (if available), with fallback to fast estimate.
"""

import logging

logger = logging.getLogger(__name__)


class TokenEstimator:
    """Estimates token counts for text content."""

    def estimate(self, text: str) -> int:
        """Fast estimate: len(text) / 4."""
        if not text:
            return 0
        return len(text) // 4

    def precise(self, text: str, model: str = "gpt-4") -> int:
        """Precise estimate using tiktoken if available, else fallback."""
        if not text:
            return 0
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            logger.debug("tiktoken unavailable, falling back to char estimate")
            return self.estimate(text)
