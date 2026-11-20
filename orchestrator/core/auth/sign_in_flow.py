"""PRD-277 US-402, US-404, US-408 (§5.2, §5.4, §5.13): from a verified identity to a session.

The routes (``api/auth_oidc.py``) call only ``start_sign_in`` and ``finish_sign_in``, which use
the identity method registered in ``REGISTRY`` (OIDC, ``core.auth.oidc_sign_in``) and never a
provider directly. ``finish_sign_in`` runs, in this order:

1. **The identity method's ``finish``**: the callback checked and the identity verified.
2. **The access rule** (``apply_access_rule``): US-403's one call site. It lets everyone through
   until that story fills it.
3. **The person**, found by ``(issuer, subject)`` or created (``people.link_person``), with
   ``email_verified`` exactly as the verified ID token states it (absent is false).
4. **A deactivated person** is refused here, before anything is granted.
5. **First sign-in only**: ``ENTERPRISE_DEFAULT_ROLE`` in the default workspace, unless the person
   already has a membership there; then the first-admin rule (``apply_first_admin_rule``), US-405's
   one call site, which grants nothing until that story fills it. A repeat sign-in grants nothing
   and changes no row (PRD-277 §12, entry 8).
6. **The hooks** (``REGISTRY.hooks()``), in registration order, each handed ``HookPeople`` for this
   person and ``SignedIn``, never the database session. One may refuse (``SignInRefusedByHook``) or
   set memberships (viewer, editor or admin) through it. Any other exception is a failed sign-in:
   rolled back, audited as the access rule's refusal, and answered ``sign_in_failed``.
7. **Commit**: each membership a hook set with its ``member:role_changed`` row, then a first
   sign-in with its ``auth:identity_linked`` row (naming the default role it granted), a repeat one
   plainly. A failed row rolls the person back and nobody gets in.
8. **The session** (``sessions.issue_session``), committed with its ``auth:sign_in`` row.

Every refusal is a ``SignInRefusal`` whose ``code`` is all the person is shown:

* ``sign_in_expired``: the callback belongs to no live sign-in of this browser. Logged, not audited
  (nobody is known, and anyone can send such a request).
* ``sign_in_refused``: the token did not verify, the provider answered with an error, the email
  belongs to another person, the person is deactivated, or a hook refused. One code for all, so
  the answer never tells whether an address is known to the install. Audited as
  ``auth:sign_in_refused`` with its reason; anything pending is rolled back first.
* ``identity_provider_unavailable``: the provider could not be reached or publishes settings sign-in
  cannot use. Logged, not audited.
* ``sign_in_busy``: ``OIDC_PENDING_SIGN_INS_MAX`` sign-ins are under way; try again shortly. Logged.
* ``sign_in_failed``: the sign-in could not be stored or recorded, or a hook failed. Logged; nothing
  was kept. A hook's failure is audited too.

Log lines name the step and the rule, never a code, token, state, nonce, verifier or secret.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.auth import people, session_audit, sessions
from core.auth.oidc_http import ProviderUnreachable
from core.auth.oidc_pending import PendingSignInFailed, SignInsBusy
from core.auth.oidc_sign_in import METHOD_NAME, OidcSignIn
from core.auth.sign_in_methods import (
    Callback,
    HookPeople,
    InvalidSignIn,
    SignedIn,
    SignInExpired,
    SignInRefusedByHook,
    SignInRegistry,
    Started,
    VerifiedIdentity,
)
from core.auth.user_link import IdentityLinkRefused, LinkedPerson
from core.edition_start_up import DEFAULT_ROLES, NO_DEFAULT_ROLE

logger = logging.getLogger(__name__)

REGISTRY = SignInRegistry({METHOD_NAME: OidcSignIn()})

EXPIRED = "sign_in_expired"
REFUSED = "sign_in_refused"
UNAVAILABLE = "identity_provider_unavailable"
BUSY = "sign_in_busy"
FAILED = "sign_in_failed"
REFUSAL_CODES = (EXPIRED, REFUSED, UNAVAILABLE, BUSY, FAILED)
# The roles a hook may set: a workspace role below owner, as ENTERPRISE_DEFAULT_ROLE.
HOOK_ROLES = tuple(role for role in DEFAULT_ROLES if role != NO_DEFAULT_ROLE)

_WORKSPACE_EXISTS = text("SELECT 1 FROM workspaces WHERE id = CAST(:ws AS uuid)")


class SignInRefusal(Exception):
    """A sign-in that ends without a session; ``code`` (one of ``REFUSAL_CODES``) is what the person sees."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Browser:
    """Where the sign-in came from: the ``audit_logs`` fields of its rows."""

    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass(frozen=True)
class SignInDone:
    """The new session and the path on this install the person goes to."""

    session: sessions.IssuedSession = field(repr=False)
    return_to: str


def apply_access_rule(identity: VerifiedIdentity) -> None:
    """US-403's call site: who may get in beyond the provider's own assignment. Lets everyone in today."""
    return None


def apply_first_admin_rule(db: Session, identity: VerifiedIdentity, linked: LinkedPerson) -> None:
    """US-405's call site, on a first sign-in only: ``ENTERPRISE_BOOTSTRAP_ADMINS``. Grants nothing today."""
    return None


def start_sign_in(db: Session, return_to: str, method: str = METHOD_NAME) -> Started:
    """Where to send the browser to sign in; ``SignInRefusal`` when the provider cannot be used."""
    failure: Optional[str] = None
    try:
        return REGISTRY.method(method).start(db, return_to)
    except ProviderUnreachable:
        failure = UNAVAILABLE
    except SignInsBusy:
        failure = BUSY
    except PendingSignInFailed:
        failure = FAILED
    logger.warning("A sign-in could not start: %s", failure)
    raise SignInRefusal(failure)


def _refuse(db: Session, reason: str, browser: Browser, user_id: Optional[int] = None,
            code: str = REFUSED) -> SignInRefusal:
    """Roll back what is pending, record ``auth:sign_in_refused`` with ``reason``, and say ``code``."""
    db.rollback()
    session_audit.record_sign_in_refused(db, reason=reason, user_id=user_id, ip_address=browser.ip_address,
                                         user_agent=browser.user_agent)
    return SignInRefusal(code)


def _finished(db: Session, callback: Callback, browser: Browser, method: str):
    """The identity method's verified answer, or the refusal its exception means."""
    failure: Optional[str] = None
    try:
        return REGISTRY.method(method).finish(db, callback)
    except SignInExpired as expired:
        failure, why = EXPIRED, str(expired)
    except InvalidSignIn as invalid:
        failure, why = REFUSED, str(invalid)
    except ProviderUnreachable:
        failure, why = UNAVAILABLE, "the identity provider could not be used"
    except PendingSignInFailed:
        failure, why = FAILED, "the sign-in's state could not be read"
    logger.warning("Sign-in refused (%s): %s", failure, why)
    if failure == REFUSED:
        raise _refuse(db, session_audit.REFUSED_INVALID_TOKEN, browser)
    raise SignInRefusal(failure)


def _linked(db: Session, identity: VerifiedIdentity, browser: Browser) -> LinkedPerson:
    """The person the identity signs in as; an email held by another person is refused."""
    reason: Optional[str] = None
    try:
        return people.link_person(db, issuer=identity.issuer, subject=identity.subject,
                                  email_verified=identity.email_verified, email=identity.email,
                                  name=identity.name)
    except IdentityLinkRefused:
        reason = session_audit.REFUSED_EMAIL_CONFLICT
    except (ValueError, TypeError):
        reason = session_audit.REFUSED_INVALID_TOKEN
    logger.warning("Sign-in refused: the identity could not be linked (%s)", reason)
    raise _refuse(db, reason, browser)


def _grant_default_role(db: Session, user_id: int) -> Optional[str]:
    """``ENTERPRISE_DEFAULT_ROLE`` in the default workspace, when it is a role and the workspace exists."""
    role, workspace_id = config.ENTERPRISE_DEFAULT_ROLE, session_audit.audit_workspace()
    if role == NO_DEFAULT_ROLE or workspace_id is None:
        return None
    if db.execute(_WORKSPACE_EXISTS, {"ws": workspace_id}).fetchone() is None:
        logger.warning("ENTERPRISE_DEFAULT_ROLE not granted: the default workspace does not exist")
        return None
    return role if people.set_membership(db, user_id, workspace_id, role, if_absent=True) else None


Membership = Tuple[str, str, str]  # (workspace_id, role, the hook that set it)


def _hook_people(db: Session, user_id: int, hook: str, recorded: List[Membership]) -> HookPeople:
    """``HookPeople`` for one hook: memberships of this person only, each one recorded for its audit row."""
    def set_membership(workspace_id: str, role: str) -> bool:
        if role not in HOOK_ROLES:
            raise ValueError(f"a hook sets one of {', '.join(HOOK_ROLES)}")
        written = people.set_membership(db, user_id, workspace_id, role)
        if written:
            recorded.append((str(workspace_id), role, hook))
        return written

    return HookPeople(set_membership=set_membership)


def _run_hooks(db: Session, signed_in: SignedIn, browser: Browser) -> Tuple[Membership, ...]:
    """Every hook, in order; the memberships they set. A refusal or any failure ends the sign-in."""
    recorded: List[Membership] = []
    # A person this sign-in created is rolled back with it, so a refusal's row names nobody.
    known = None if signed_in.created else signed_in.user_id
    for name, hook in REGISTRY.hooks():
        try:
            hook(_hook_people(db, signed_in.user_id, name, recorded), signed_in)
        except SignInRefusedByHook as refused:
            logger.warning("Sign-in refused by hook %s: %s", name, refused)
            raise _refuse(db, session_audit.REFUSED_ACCESS_RULE, browser, known) from None
        except Exception:
            logger.exception("Sign-in hook %s failed; the sign-in was rolled back and refused", name)
            raise _refuse(db, session_audit.REFUSED_ACCESS_RULE, browser, known, code=FAILED) from None
    return tuple(recorded)


def _commit_plainly(db: Session) -> None:
    try:
        db.commit()
    except SQLAlchemyError as error:
        failure = type(error).__name__
    else:
        return
    db.rollback()
    logger.error("A repeat sign-in could not be committed (%s)", failure)
    raise SignInRefusal(FAILED)


def _commit_person(db: Session, linked: LinkedPerson, default_role: Optional[str],
                   memberships: Tuple[Membership, ...], browser: Browser) -> None:
    """Commit the person with a row for each membership a hook set, and a first sign-in's link row."""
    for workspace_id, role, hook in memberships:
        session_audit.stage_membership_set(db, user_id=linked.user_id, workspace_id=workspace_id, role=role,
                                           hook=hook, ip_address=browser.ip_address,
                                           user_agent=browser.user_agent)
    if not linked.created:
        _commit_plainly(db)
        return
    try:
        session_audit.record_identity_linked(db, user_id=linked.user_id, details={"default_role": default_role},
                                             ip_address=browser.ip_address, user_agent=browser.user_agent)
    except session_audit.SessionAuditFailed:
        raise SignInRefusal(FAILED) from None


def _issue(db: Session, identity: VerifiedIdentity, user_id: int, browser: Browser) -> sessions.IssuedSession:
    failure: Optional[str] = None
    try:
        return sessions.issue_session(db, user_id=user_id, auth_time=identity.auth_time,
                                      sid=identity.idp_session_id, ip_address=browser.ip_address,
                                      user_agent=browser.user_agent)
    except sessions.SignInRefused:
        failure = REFUSED
    except (session_audit.SessionAuditFailed, sessions.SignInFailed, sessions.SessionsNotIssued):
        failure = FAILED
    logger.warning("users.id=%s: no session was issued (%s)", user_id, failure)
    raise SignInRefusal(failure)


def finish_sign_in(db: Session, callback: Callback, browser: Browser, method: str = METHOD_NAME) -> SignInDone:
    """The callback, to a session: every step in this module's docstring, or ``SignInRefusal``."""
    finished = _finished(db, callback, browser, method)
    identity = finished.identity
    apply_access_rule(identity)
    linked = _linked(db, identity, browser)
    if not linked.created and not people.person_is_active(db, linked.user_id):
        logger.warning("users.id=%s: sign-in refused, the person is deactivated", linked.user_id)
        raise _refuse(db, session_audit.REFUSED_DEACTIVATED, browser, linked.user_id)
    default_role = None
    if linked.created:
        default_role = _grant_default_role(db, linked.user_id)
        apply_first_admin_rule(db, identity, linked)
    memberships = _run_hooks(db, SignedIn(user_id=linked.user_id, created=linked.created, identity=identity),
                             browser)
    _commit_person(db, linked, default_role, memberships, browser)
    issued = _issue(db, identity, linked.user_id, browser)
    return SignInDone(session=issued, return_to=finished.return_to)


__all__ = [
    "BUSY",
    "EXPIRED",
    "FAILED",
    "HOOK_ROLES",
    "REFUSAL_CODES",
    "REFUSED",
    "REGISTRY",
    "UNAVAILABLE",
    "Browser",
    "SignInDone",
    "SignInRefusal",
    "apply_access_rule",
    "apply_first_admin_rule",
    "finish_sign_in",
    "start_sign_in",
]
