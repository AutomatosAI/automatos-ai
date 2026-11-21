"""PRD-277 US-402, US-404, US-408 (§5.2, §5.4, §5.13): from a verified identity to a session.

The routes (``api/auth_oidc.py``) call only ``start_sign_in`` and ``finish_sign_in``, which use
the identity method registered in ``REGISTRY`` (OIDC, ``core.auth.oidc_sign_in``) and never a
provider directly. ``finish_sign_in`` runs, in this order:

0. **The default workspace** every row of the sign-in is recorded in is made, with no owner, when
   it is missing, and committed (``ensure_audit_workspace``), so even the first refusal is audited.
1. **The identity method's ``finish``**: the callback checked and the identity verified.
2. **The access rule** (``apply_access_rule``, US-403): every ``OIDC_REQUIRED_CLAIMS`` rule must
   hold for the verified token's claims (``core.auth.access_rule``). A refusal is audited
   (``access_rule``) and answered ``sign_in_not_allowed``, before any person is found, created or
   linked, so a refused person leaves no row.
3. **A step-up** (US-406, a sign-in started with ``step_up=1``) ends here: the asking session's
   ``auth_time`` is refreshed or the step-up refused (``core.auth.step_up_sign_in``), and no person
   is created or linked and no session issued. Before any first-sign-in step.
4. **The person**, found by ``(issuer, subject)`` or created (``people.link_person``), with
   ``email_verified`` exactly as the verified ID token states it (absent is false).
5. **A deactivated person** is refused here, before anything is granted.
6. **First sign-in only**: ``ENTERPRISE_DEFAULT_ROLE`` in the default workspace, unless the person
   already has a membership there. Then the first-admin rule (``apply_first_admin_rule``, US-405,
   ``core.auth.first_admin``): a person ``ENTERPRISE_BOOTSTRAP_ADMINS`` names becomes an install
   admin and a member of the default workspace (its owner when it has none, else an admin),
   audited: at their first sign-in, or at any sign-in while the install has no active install
   admin. Otherwise a repeat sign-in grants nothing and changes no row (PRD-277 §12, entry 8).
7. **The hooks** (``REGISTRY.hooks()``), in registration order, each handed ``HookPeople`` for this
   person and ``SignedIn``, never the database session. One may refuse (``SignInRefusedByHook``) or
   set memberships (viewer, editor or admin) through it. Any other exception is a failed sign-in:
   rolled back, audited as the access rule's refusal, and answered ``sign_in_failed``.
8. **Commit**: each membership a hook set with its ``member:role_changed`` row, then a first
   sign-in with its ``auth:identity_linked`` row (naming the default role it granted), a repeat one
   plainly, each with any ``auth:bootstrap_admin`` row. A failed row rolls the person back and
   nobody gets in.
9. **The session** (``sessions.issue_session``), committed with its ``auth:sign_in`` row.

Every refusal is a ``SignInRefusal`` whose ``code`` is all the person is shown:

* ``sign_in_expired``: the callback belongs to no live sign-in of this browser: unknown, used, out of
  time, or ended to make room for newer ones (``core.auth.oidc_pending``). Logged, not audited
  (nobody is known, and anyone can send such a request).
* ``sign_in_refused``: the token did not verify, the provider answered with an error, the email
  belongs to another person, the person is deactivated, or a hook refused. One code for all, so
  the answer never tells whether an address is known to the install. Audited as
  ``auth:sign_in_refused`` with its reason; anything pending is rolled back first.
* ``sign_in_not_allowed``: the person signed in at the provider, but the access rule does not let
  them in. Audited as ``auth:sign_in_refused`` with reason ``access_rule``. It tells the person only
  that this install does not admit their account, which they already know they hold.
* ``step_up_refused``: a step-up could not start (no live session in this browser, or a request
  sent from another site) or did not confirm the session's own person with a fresh sign-in.
  Audited as ``auth:step_up`` when it got as far as the callback.
* ``step_up_no_auth_time``: the provider's ID token said nothing of when the person signed in
  (``auth_time``), so a step-up cannot be confirmed; the provider must be set up to send it.
  Audited as ``auth:step_up`` with reason ``no_auth_time``.
* ``identity_provider_unavailable``: the provider could not be reached or publishes settings sign-in
  cannot use. Logged, not audited.
* ``sign_in_failed``: the sign-in could not be stored or recorded, or a hook failed. Logged; nothing
  was kept. A hook's failure is audited too.

Log lines name the step and the rule, never a code, token, state, nonce, verifier or secret.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.auth import access_rule, first_admin, people, session_audit, sessions, step_up_sign_in
from core.auth.oidc_http import ProviderUnreachable
from core.auth.oidc_pending import PendingSignInFailed
from core.auth.oidc_sign_in import METHOD_NAME, OidcSignIn
from core.auth.sign_in_methods import (
    Callback,
    Finished,
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
from core.seeds.seed_default_workspace import ensure_default_workspace

logger = logging.getLogger(__name__)

REGISTRY = SignInRegistry({METHOD_NAME: OidcSignIn()})

EXPIRED = "sign_in_expired"
REFUSED = "sign_in_refused"
NOT_ALLOWED = "sign_in_not_allowed"
STEP_UP_REFUSED = "step_up_refused"
STEP_UP_NO_AUTH_TIME = "step_up_no_auth_time"
UNAVAILABLE = "identity_provider_unavailable"
FAILED = "sign_in_failed"
REFUSAL_CODES = (EXPIRED, REFUSED, NOT_ALLOWED, STEP_UP_REFUSED, STEP_UP_NO_AUTH_TIME, UNAVAILABLE, FAILED)
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
    """The new session and the path on this install the person goes to.

    ``session`` is None after a step-up: the browser's own session was refreshed, none was issued.
    """

    session: Optional[sessions.IssuedSession] = field(repr=False)
    return_to: str


def _rules_hold(identity: VerifiedIdentity) -> bool:
    """Whether every ``OIDC_REQUIRED_CLAIMS`` rule holds for the verified claims. Start-up refuses a
    malformed value, so one here was changed since: it fails closed, and says so in the log."""
    try:
        rules = access_rule.parse_rules(config.OIDC_REQUIRED_CLAIMS)
    except access_rule.MalformedRule:
        logger.error("OIDC_REQUIRED_CLAIMS cannot be read; every sign-in is refused until it is fixed")
        return False
    return access_rule.every_rule_holds(rules, identity.claims)


def apply_access_rule(db: Session, identity: VerifiedIdentity, browser: Browser) -> None:
    """US-403: refuse, audited (``access_rule``), unless every rule holds for the verified claims.

    Runs before any person is found, created or linked, so the refusal names nobody.
    """
    if _rules_hold(identity):
        return
    logger.warning("Sign-in refused: the access rule does not let this account in")
    raise _refuse(db, session_audit.REFUSED_ACCESS_RULE, browser, code=NOT_ALLOWED)


def apply_first_admin_rule(db: Session, identity: VerifiedIdentity, linked: LinkedPerson,
                           browser: Browser) -> Optional[str]:
    """US-405: ``ENTERPRISE_BOOTSTRAP_ADMINS`` (``core.auth.first_admin``): at a first sign-in, or at
    any sign-in while the install has no active install admin."""
    return first_admin.grant_if_listed(db, identity, linked.user_id, created=linked.created,
                                       ip_address=browser.ip_address, user_agent=browser.user_agent)


def ensure_audit_workspace(db: Session) -> None:
    """The default workspace, made (with no owner) when it is missing and committed, before anything
    of this sign-in is audited in it; an existing row is never changed. A failure is logged as an
    error and the sign-in goes on: its rows then fail as any unwritable row does."""
    workspace_id = session_audit.audit_workspace()
    if workspace_id is None:
        return
    try:
        ensure_default_workspace(db, workspace_id)
        db.commit()
    except SQLAlchemyError as error:
        db.rollback()
        logger.error("The default workspace could not be made before sign-in (%s)", type(error).__name__)


def _started(db: Session, method: str, return_to: str, client_address: str,
             step_up_session_id: Optional[UUID]) -> Started:
    chosen = REGISTRY.method(method)
    if step_up_session_id is None:
        return chosen.start(db, return_to, client_address)
    return chosen.start(db, return_to, client_address, step_up_session_id=step_up_session_id)


def start_sign_in(db: Session, return_to: str, client_address: str, method: str = METHOD_NAME,
                  step_up_session_id: Optional[UUID] = None) -> Started:
    """Where to send the browser at ``client_address`` to sign in; ``SignInRefusal`` when the provider
    cannot be used or the sign-in cannot be stored. ``step_up_session_id`` starts a step-up for that
    session instead (``start_step_up``)."""
    failure: Optional[str] = None
    try:
        return _started(db, method, return_to, client_address, step_up_session_id)
    except ProviderUnreachable:
        failure = UNAVAILABLE
    except PendingSignInFailed:
        failure = FAILED
    logger.warning("A sign-in could not start: %s", failure)
    raise SignInRefusal(failure)


def start_step_up(db: Session, session_token: Optional[str], return_to: str, client_address: str) -> Started:
    """US-406: a sign-in that confirms the live session ``session_token`` names; ``SignInRefusal``
    (``step_up_refused``) when there is none, or its person is deactivated."""
    try:
        session_id = step_up_sign_in.session_asking(db, session_token)
    except step_up_sign_in.StepUpRefused as refused:
        logger.warning("A step-up could not start: %s", refused)
        raise SignInRefusal(STEP_UP_REFUSED) from None
    return start_sign_in(db, return_to, client_address, step_up_session_id=session_id)


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


def _step_up(db: Session, finished: Finished, browser: Browser) -> SignInDone:
    """US-406: the asking session refreshed, and no session issued; else ``SignInRefusal``."""
    failure: Optional[str] = None
    try:
        step_up_sign_in.confirm_step_up(db, finished, ip_address=browser.ip_address,
                                        user_agent=browser.user_agent)
    except step_up_sign_in.StepUpRefused as refused:
        failure = STEP_UP_NO_AUTH_TIME if str(refused) == session_audit.STEP_UP_NO_AUTH_TIME else STEP_UP_REFUSED
    except (session_audit.SessionAuditFailed, SQLAlchemyError) as error:
        db.rollback()
        logger.error("A step-up could not be recorded (%s)", type(error).__name__)
        failure = FAILED
    if failure is not None:
        raise SignInRefusal(failure)
    return SignInDone(session=None, return_to=finished.return_to)


def finish_sign_in(db: Session, callback: Callback, browser: Browser, method: str = METHOD_NAME) -> SignInDone:
    """The callback, to a session: every step in this module's docstring, or ``SignInRefusal``."""
    ensure_audit_workspace(db)
    finished = _finished(db, callback, browser, method)
    identity = finished.identity
    apply_access_rule(db, identity, browser)
    if finished.step_up_session_id is not None:
        return _step_up(db, finished, browser)
    linked = _linked(db, identity, browser)
    if not linked.created and not people.person_is_active(db, linked.user_id):
        logger.warning("users.id=%s: sign-in refused, the person is deactivated", linked.user_id)
        raise _refuse(db, session_audit.REFUSED_DEACTIVATED, browser, linked.user_id)
    default_role = _grant_default_role(db, linked.user_id) if linked.created else None
    apply_first_admin_rule(db, identity, linked, browser)
    memberships = _run_hooks(db, SignedIn(user_id=linked.user_id, created=linked.created, identity=identity),
                             browser)
    _commit_person(db, linked, default_role, memberships, browser)
    issued = _issue(db, identity, linked.user_id, browser)
    return SignInDone(session=issued, return_to=finished.return_to)


__all__ = [
    "EXPIRED",
    "FAILED",
    "HOOK_ROLES",
    "NOT_ALLOWED",
    "REFUSAL_CODES",
    "REFUSED",
    "REGISTRY",
    "STEP_UP_NO_AUTH_TIME",
    "STEP_UP_REFUSED",
    "UNAVAILABLE",
    "Browser",
    "SignInDone",
    "SignInRefusal",
    "apply_access_rule",
    "apply_first_admin_rule",
    "ensure_audit_workspace",
    "finish_sign_in",
    "start_sign_in",
    "start_step_up",
]
