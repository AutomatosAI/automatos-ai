"""PRD-277 US-408 (§5.13, FR-27): one interface for proving who someone is, and one hook after it.

Sign-in is written against this module, so another way of proving identity can be added without
touching the code that issues sessions or checks permissions.

* **The interface.** An ``IdentityMethod`` has a ``start`` that sends the person to sign in (where
  to send the browser, and the value the browser must carry back) and a ``finish`` that takes the
  callback and returns a ``VerifiedIdentity``: issuer, subject, email and whether the provider
  verified it, name, when the person signed in, the provider's session id, and the claims. A
  ``finish`` refuses with one of three exceptions, and only these: ``SignInExpired`` (the callback
  belongs to no live sign-in of this browser), ``InvalidSignIn`` (what came back did not prove who
  the person is) and ``oidc_http.ProviderUnreachable`` (the provider could not be reached). OIDC is
  the one implementation in core (``core.auth.oidc_sign_in``).
* **The hook.** A ``SignInHook`` runs after the identity is verified and the person is found or
  created, and before the session is issued (``core.auth.sign_in_flow``). It is handed
  ``HookPeople`` (the one thing it may change: this person's workspace memberships, through
  ``core.auth.people``) and ``SignedIn`` (who signed in, and their verified identity), never the
  database session, so it cannot commit or reach another table. It may refuse the sign-in
  (``SignInRefusedByHook``). Any other exception from it is a failed sign-in: rolled back, audited
  and refused. Each membership it sets is audited when the sign-in commits. With nothing
  registered, sign-in is exactly as PRD-277 describes.
* **Registration closes at start-up** (``SignInRegistry.close``, called when the boot finishes, and
  by the first sign-in, whichever is first). After that, registering is ``RegistrationClosed``;
  registering a name twice is ``AlreadyRegistered``.

``safe_return_path`` is the one rule for where a person goes after signing in: a path on the
install's own address, never an address from the request.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy.orm import Session

from core.auth.phone_sign_in import PhoneStart
from core.models.oidc_pending_sign_ins import RETURN_TO_MAX

DEFAULT_RETURN_PATH = "/"


class SignInExpired(Exception):
    """The callback names no live sign-in this browser started (unknown, used, expired or unbound)."""


class InvalidSignIn(Exception):
    """What the provider sent back does not prove who the person is. Its text is for the log only."""


class PhoneSignInFailed(Exception):
    """US-603: a phone's sign-in that failed after its start was found, so its answer goes to the app.

    ``failure`` is the ``InvalidSignIn`` or ``oidc_http.ProviderUnreachable`` the method would have
    raised; ``phone`` is what the phone's start asked for. Its text is for the log only.
    """

    def __init__(self, failure: Exception, phone: PhoneStart) -> None:
        super().__init__(str(failure))
        self.failure = failure
        self.phone = phone


class SignInRefusedByHook(Exception):
    """A hook refused this sign-in. Its text is for the log only: the person sees the plain refusal."""


class RegistrationClosed(RuntimeError):
    """Something was registered after start-up finished."""


class AlreadyRegistered(ValueError):
    """The same name was registered twice."""


@dataclass(frozen=True)
class VerifiedIdentity:
    """Who the person is, as the identity method verified it. ``claims`` is read-only.

    ``subject`` is the person's id, the one value they are found and linked by. ``standard_sub`` is
    never an id: it is the token's ``sub``, carried only when the subject is read from another claim
    (``OIDC_CLAIM_SUBJECT``), so a first sign-in can be refused for a person already linked under
    ``sub`` (``user_link.link_identity``'s ``linked_before_as``). None otherwise.
    """

    issuer: str
    subject: str
    email: Optional[str]
    email_verified: bool
    name: Optional[str]
    auth_time: datetime
    idp_session_id: Optional[str]
    claims: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    standard_sub: Optional[str] = None


@dataclass(frozen=True)
class Started:
    """Where to send the browser, and the value it must carry back to the callback."""

    redirect_to: str = field(repr=False)
    browser_binding: str = field(repr=False)


@dataclass(frozen=True)
class Callback:
    """The callback as the identity method reads it: its query, and the value the browser carried."""

    params: Mapping[str, Tuple[str, ...]] = field(repr=False)
    browser_binding: Optional[str] = field(default=None, repr=False)


@dataclass(frozen=True)
class Finished:
    """A verified identity, and the path the person goes to next.

    ``started_at`` is when the sign-in started. For a step-up (US-406), ``step_up_session_id`` is the
    session that asked; for a step-up and for a phone's sign-in (US-603) the identity's ``auth_time``
    must not be older than ``started_at``. A method that offers no step-up leaves
    ``step_up_session_id`` None, and its callbacks are ordinary sign-ins.
    """

    identity: VerifiedIdentity
    return_to: str
    step_up_session_id: Optional[UUID] = None
    started_at: Optional[datetime] = None
    # US-603: what a phone's start asked for. The sign-in then ends with a code for the app, no session.
    phone: Optional[PhoneStart] = None


@dataclass(frozen=True)
class SignedIn:
    """What a hook is told: the person's ``users.id``, whether this sign-in created them, and who they are."""

    user_id: int
    created: bool
    identity: VerifiedIdentity


class IdentityMethod(Protocol):
    """One way of proving who someone is (§5.13)."""

    def start(self, db: Session, return_to: str, client_address: str) -> Started:
        """Begin a sign-in that returns to ``return_to``, for the browser at ``client_address`` (the
        trusted client address, ``api.credentials.get_client_ip``); where to send the browser.

        A method that offers step-up (US-406) also takes ``step_up_session_id``, the live session
        asking for it: it then makes the person prove who they are again, whatever session the
        provider holds, and its ``finish`` hands the id back. Sign-in passes it only for a step-up.
        A method that offers phone sign-in (US-603) also takes ``phone``, keeps it with the sign-in,
        and its ``finish`` hands it back (``Finished.phone``), or raises ``PhoneSignInFailed`` for a
        failure once it is known; sign-in passes it only for a phone's start.
        """

    def finish(self, db: Session, callback: Callback) -> Finished:
        """The verified identity the callback proves, or one of the three refusals."""


@dataclass(frozen=True)
class HookPeople:
    """What a hook may change, for the person signing in only.

    ``set_membership(workspace_id, role)`` gives the person ``role`` (viewer, editor or admin) in that
    workspace, adding the membership or changing it, and returns whether a row was written. It is
    committed, and audited, with the sign-in, or rolled back with it.
    """

    set_membership: Callable[[str, str], bool] = field(repr=False)


SignInHook = Callable[[HookPeople, SignedIn], None]


def safe_return_path(value: Any) -> str:
    """``value`` when it is a path on the install's own address, else ``DEFAULT_RETURN_PATH``.

    A path starts with one ``/`` and has no scheme, host, backslash, control character or space,
    and fits ``RETURN_TO_MAX``; its fragment is dropped. ``//host``, ``/\\host``, ``https://…`` and
    ``javascript:`` are never a path, so no request can choose an address off the install.
    """
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        return DEFAULT_RETURN_PATH
    if len(value) > RETURN_TO_MAX or "\\" in value or any(ch.isspace() or not ch.isprintable() for ch in value):
        return DEFAULT_RETURN_PATH
    try:
        parts = urlsplit(value)
    except ValueError:
        return DEFAULT_RETURN_PATH
    if parts.scheme or parts.netloc or not parts.path.startswith("/"):
        return DEFAULT_RETURN_PATH
    return parts.path + (f"?{parts.query}" if parts.query else "")


class SignInRegistry:
    """The identity methods and hooks sign-in uses; closed for registration once start-up finishes."""

    def __init__(self, methods: Mapping[str, IdentityMethod]) -> None:
        self._lock = threading.Lock()
        self._methods: Mapping[str, IdentityMethod] = MappingProxyType(dict(methods))
        self._hooks: Tuple[Tuple[str, SignInHook], ...] = ()
        self._closed = False

    def _open_for(self, name: str, taken: Any) -> None:
        if self._closed:
            raise RegistrationClosed(f"sign-in registration closed at start-up; {name!r} came too late")
        if not isinstance(name, str) or not name or name in taken:
            raise AlreadyRegistered(f"{name!r} is already registered for sign-in")

    def register_method(self, name: str, method: IdentityMethod) -> None:
        """Add an identity method under ``name``, before start-up finishes."""
        with self._lock:
            self._open_for(name, self._methods)
            self._methods = MappingProxyType({**self._methods, name: method})

    def register_hook(self, name: str, hook: SignInHook) -> None:
        """Add a hook under ``name``; hooks run in the order they were registered."""
        with self._lock:
            self._open_for(name, {taken for taken, _ in self._hooks})
            self._hooks = (*self._hooks, (name, hook))

    def close(self) -> None:
        """Refuse every registration from now on (idempotent)."""
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def method(self, name: str) -> IdentityMethod:
        """The identity method ``name``; sign-in has begun, so registration closes. ``KeyError`` if none."""
        self.close()
        return self._methods[name]

    def hooks(self) -> Tuple[Tuple[str, SignInHook], ...]:
        """Every hook, in registration order."""
        return self._hooks


__all__ = [
    "DEFAULT_RETURN_PATH",
    "AlreadyRegistered",
    "Callback",
    "Finished",
    "HookPeople",
    "IdentityMethod",
    "InvalidSignIn",
    "PhoneSignInFailed",
    "RegistrationClosed",
    "SignInExpired",
    "SignInHook",
    "SignInRefusedByHook",
    "SignInRegistry",
    "SignedIn",
    "Started",
    "VerifiedIdentity",
    "safe_return_path",
]
