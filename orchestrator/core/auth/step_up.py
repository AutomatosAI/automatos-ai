"""PRD-261 A-07c — step-up: a fresh sign-in factor on sensitive routes.

``require_step_up(max_age_minutes)`` is a Depends factory, layered after the
shared hybrid dependency like ``require_workspace_permission``. It refuses a
request unless the caller's first sign-in factor was verified within the last
``max_age_minutes`` minutes, so code running in the page as the user (an
installed theme) cannot finish these routes silently.

Edition gate (``AUTH_EDITION``, never the caller's role):

==========  ==================================================================
edition     behaviour
==========  ==================================================================
``local``   passes: the edition has one anonymous owner and no sign-in to
            repeat (PRD-175 trusted single-user posture).
``saas``    a ``clerk`` context passes only when its verified token's ``fva``
            claim says the first factor is fresher than the max age. Every
            other context (``api_key``, ``sdk_key``, ``anonymous``) is refused:
            none of them has a person who can sign in again.
==========  ==================================================================

The ``fva`` claim ("factor verification age") is Clerk's session-token claim
``[first_factor_minutes, second_factor_minutes]``, ``-1`` for a factor never
verified. The installed Clerk SDK reads it from the session claims
(``@clerk/shared`` 3.44.0 ``jwtPayloadParser.js:40``) and decides with
``afterMinutes > factorAge`` (``authorization.js:87-97``); this module makes the
same decision on the token ``core/auth/clerk.py`` verified, which
``core/auth/hybrid.py`` keeps as ``ctx.user.raw_claims``. The ages are measured
when the token was issued, so the time since ``iat`` is added before comparing.

The refusal (plan §2.2): HTTP 403 whose JSON body carries Clerk's
``reverificationError({level: 'first_factor', afterMinutes: N})`` ``clerk_error``
at the TOP level, so the frontend's ``useReverification`` recognises it
(``isReverificationHint``). A dependency cannot return a response, so it raises
``StepUpRequired`` and ``step_up_exception_handler`` renders that body; the app
registers the handler once with ``install_step_up_handler(app)``. Without the
handler the exception is still an ``HTTPException``: a plain 403, never a pass.

Residual window, stated plainly: any sign-in within the max age satisfies the
check. For the first ``max_age_minutes`` minutes after a sign-in, code in the
page can complete these routes without a prompt. Narrowing it is Gerard's
D1-e: the smallest max age, or a per-action server challenge (the backend
records ``issued_at`` and accepts only an ``fva`` newer than it).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid

logger = logging.getLogger(__name__)

# Gerard D1-a/e (18 Oct 2026): ten minutes. B2-wire passes
# config.THEME_STEP_UP_MAX_AGE_MINUTES; this is the fallback default.
DEFAULT_MAX_AGE_MINUTES = 10

# Clerk's level names (@clerk/shared authorization.js ALLOWED_LEVELS).
STEP_UP_LEVEL = "first_factor"

# @clerk/shared authorization-errors.js: REVERIFICATION_REASON and the error type.
REVERIFICATION_REASON = "reverification-error"
REVERIFICATION_ERROR_TYPE = "forbidden"

# fva value for a factor that was never verified in this session.
FACTOR_NOT_VERIFIED = -1

STEP_UP_CODE = "step_up_required"
STEP_UP_UNAVAILABLE_CODE = "step_up_unavailable"

# Marker stamped on every step-up dependency (its max age, in minutes), so the
# wire story's test can find exactly the routes that carry it on the route table.
STEP_UP_MARKER_ATTR = "_step_up_max_age_minutes"

_CLERK_AUTH_TYPE = "clerk"
_LOCAL_EDITION = "local"
_SECONDS_PER_MINUTE = 60.0


class StepUpRequired(HTTPException):
    """403 that carries Clerk's reverification body (rendered by the handler)."""

    def __init__(self, content: Dict[str, Any]) -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=content["detail"])
        self.content = content


def reverification_error(max_age_minutes: int) -> Dict[str, Any]:
    """Clerk's ``reverificationError({ level, afterMinutes })`` body, as JSON."""
    return {
        "clerk_error": {
            "type": REVERIFICATION_ERROR_TYPE,
            "reason": REVERIFICATION_REASON,
            "metadata": {
                "reverification": {"level": STEP_UP_LEVEL, "afterMinutes": max_age_minutes},
            },
        }
    }


def step_up_body(max_age_minutes: int) -> Dict[str, Any]:
    """The 403 body: Clerk's top-level ``clerk_error`` plus a plain message and code."""
    body = reverification_error(max_age_minutes)
    return {
        **body,
        "code": STEP_UP_CODE,
        "detail": (
            "Confirm it's you to continue: this action needs a sign-in "
            f"from the last {max_age_minutes} minutes."
        ),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _first_factor_age(claims: Any) -> Optional[int]:
    """Minutes since the first factor was verified when the token was issued, or
    None when the claim is missing, malformed or says "never verified"."""
    if not isinstance(claims, dict):
        return None
    fva = claims.get("fva")
    if not isinstance(fva, (list, tuple)) or len(fva) != 2:
        return None
    if not all(_is_int(age) for age in fva):
        return None
    first = fva[0]
    if first == FACTOR_NOT_VERIFIED or first < 0:
        return None
    return first


def _minutes_since_issued(claims: Dict[str, Any], now: float) -> float:
    """Minutes between the token's ``iat`` and ``now`` (0 for a missing or future iat)."""
    issued_at = claims.get("iat")
    if not isinstance(issued_at, (int, float)) or isinstance(issued_at, bool):
        return 0.0
    if not math.isfinite(issued_at):
        return 0.0
    return max(0.0, (now - float(issued_at)) / _SECONDS_PER_MINUTE)


def first_factor_is_fresh(claims: Any, max_age_minutes: int, now: float) -> bool:
    """True when the first factor was verified less than ``max_age_minutes`` ago
    (Clerk's ``afterMinutes > factorAge``, with the token's age added)."""
    age = _first_factor_age(claims)
    if age is None:
        return False
    return age + _minutes_since_issued(claims, now) < max_age_minutes


def _is_local_edition() -> bool:
    return str(getattr(config, "AUTH_EDITION", "") or "").strip().lower() == _LOCAL_EDITION


def _refuse_without_person(auth_type: Any) -> HTTPException:
    logger.warning("step-up refused: %s context cannot sign in again", auth_type)
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "code": STEP_UP_UNAVAILABLE_CODE,
            "message": "This action needs a person signed in to Automatos; keys cannot confirm it.",
        },
    )


def check_step_up(ctx: Any, max_age_minutes: int, now: float) -> None:
    """Raise unless ``ctx`` satisfies step-up in the current edition."""
    if _is_local_edition():
        return
    auth_type = getattr(ctx, "auth_type", None)
    if auth_type != _CLERK_AUTH_TYPE:
        raise _refuse_without_person(auth_type)
    claims = getattr(getattr(ctx, "user", None), "raw_claims", None)
    if first_factor_is_fresh(claims, max_age_minutes, now):
        return
    if not isinstance(claims, dict) or "fva" not in claims:
        # R5: the Clerk instance's session token carries no fva claim. Logged so
        # an operator can tell "never fresh" from "stale".
        logger.warning("step-up refused: the Clerk session token has no fva claim")
    raise StepUpRequired(step_up_body(max_age_minutes))


def _validated_max_age(max_age_minutes: Any) -> int:
    # Clerk's isValidMaxAge: a number > 0. Whole minutes keep the hint exact.
    if not _is_int(max_age_minutes) or max_age_minutes <= 0:
        raise ValueError(f"step-up max age must be a whole number of minutes above 0, got {max_age_minutes!r}")
    return max_age_minutes


def _now() -> float:
    return time.time()


def require_step_up(max_age_minutes: int = DEFAULT_MAX_AGE_MINUTES) -> Callable:
    """Dependency factory: 403 unless the caller signed in within ``max_age_minutes``.

    Apply it next to the route's permission gate, e.g.
    ``dependencies=[Depends(require_workspace_permission("workspace:manage")),
    Depends(require_step_up(config.THEME_STEP_UP_MAX_AGE_MINUTES))]``.
    """
    max_age = _validated_max_age(max_age_minutes)

    async def _dep(ctx: RequestContext = Depends(get_request_context_hybrid)) -> RequestContext:
        check_step_up(ctx, max_age, _now())
        return ctx

    setattr(_dep, STEP_UP_MARKER_ATTR, max_age)
    _dep.__name__ = f"require_step_up_{max_age}m"
    return _dep


async def step_up_exception_handler(request: Request, exc: StepUpRequired) -> JSONResponse:
    """Render ``StepUpRequired`` with ``clerk_error`` at the top level (plan §2.2)."""
    return JSONResponse(status_code=exc.status_code, content=exc.content)


def install_step_up_handler(app: FastAPI) -> None:
    """Register the step-up handler on ``app`` (B2-wire: once, in main.py)."""
    app.add_exception_handler(StepUpRequired, step_up_exception_handler)
