"""Secure error handling utilities to prevent information leakage."""
import logging
from fastapi import HTTPException


def raise_safe_error(
    status_code: int,
    message: str,
    exception: Exception,
    logger: logging.Logger
) -> None:
    """Log the real exception server-side and raise a generic HTTPException."""
    logger.error(f"{message}: {exception}", exc_info=True)
    raise HTTPException(status_code=status_code, detail=message)
