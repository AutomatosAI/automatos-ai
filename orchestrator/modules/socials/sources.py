"""PRD-251 S1.4 (D7): a claim's source, resolved in the caller's workspace.

A post binds each claim (a variable marked ``claim: true``) to a source,
``{kind, ref, as_of}``. The service checks the shape; this module checks that
the source is really there, in the caller's workspace. The API runs it when a
post is saved (the sources the save adds or changes) and again when it is
approved (every source: one deleted since reads as unsourced).

* ``deliverable``: ``ref`` is the id of an item in Deliverables Outputs
  (``v_workspace_outputs``, the view ``services/deliverable_service.py`` reads):
  a file, report or blog post that is not deleted.
* ``report``: ``ref`` is the id of an ``agent_reports`` row that is not deleted.
  The table has no ORM model, so it is read with raw SQL, the way
  ``services/report_service.py`` reads it.
* ``document``: ``ref`` is the id of a knowledge-base document (``documents``).
* ``metric``: a figure at a timestamp. ``ref`` names the metric and ``as_of``
  is when it was read. Its value is the one the workspace's latest report at or
  before ``as_of`` carries in ``metrics`` (the structured metrics an agent
  submits with ``platform_submit_report``). A metric needs ``as_of``, never one
  in the future. Metrics come from reports because they are the figures agents
  report and every member sees in Deliverables; the LLM Analytics page holds
  the workspace's AI spend and is for admins only (``api/llm_analytics.py``).
* ``url``: ``ref`` is an absolute http(s) address. It is checked for shape and
  never fetched: the platform does not call addresses users type (SSRF), and a
  page that is down at approval time is still the claim's source.

Every lookup is filtered by the caller's workspace, so another workspace's
source reads as "not found" and nothing about it is revealed. A malformed ref
is refused before any query, so an id Postgres cannot parse never reaches it.

``search`` offers candidates per kind for the composer's source picker
(``GET /api/socials/sources``), in the same shape ``resolve`` returns.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlsplit
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import func, or_

from core.models.core import Document
from modules.socials import service

# The source picker (GET /api/socials/sources): results per kind, and q's length.
SEARCH_DEFAULT_LIMIT = 10
SEARCH_MAX_LIMIT = 50
SEARCH_QUERY_MAX_CHARS = 200
# How many of the newest matching reports one metric lookup or search reads.
METRIC_SCAN_REPORTS = 50
REF_MAX_CHARS = 2048
URL_SCHEMES = frozenset({"http", "https"})
# documents.id is a Postgres INTEGER.
DOCUMENT_ID_MAX = 2**31 - 1
LIKE_ESCAPE = "\\"


class SourceNotResolved(Exception):
    """Why one source does not resolve, in words for the post's author."""


class SourcesNotFound(service.SocialsError):
    """A save binds claims to sources that are not in the caller's workspace."""

    def __init__(self, unresolved: Mapping[str, str]):
        self.unresolved = dict(sorted(unresolved.items()))
        super().__init__(
            "these sources could not be found in this workspace: "
            + "; ".join(f"{name} ({why})" for name, why in self.unresolved.items())
        )


@dataclass(frozen=True)
class ResolvedSource:
    """A source that exists, or a search candidate: what the picker shows."""

    kind: str
    ref: str
    title: str
    # What it is: the Deliverable's type, the report's summary, the report a metric was read from.
    detail: Optional[str] = None
    # When it was made, or when a metric was read; the composer stores it as the source's as_of.
    as_of: Optional[str] = None
    # A metric's figure.
    value: Any = None
    # The report a metric was read from.
    report_id: Optional[str] = None
    # A Deliverable's preview link, for the picker's thumbnail.
    preview_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── parsing: every check here runs before any query ─────────────────────────
def _workspace(workspace_id: Any) -> UUID:
    return workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id))


def _uuid(ref: str, what: str) -> str:
    try:
        return str(UUID(ref))
    except (TypeError, ValueError):
        raise SourceNotResolved(f"{ref!r} is not a {what} id") from None


def _document_id(ref: str) -> int:
    if not (ref.isascii() and ref.isdigit()) or not 0 < int(ref) <= DOCUMENT_ID_MAX:
        raise SourceNotResolved(f"{ref!r} is not a document id")
    return int(ref)


def _url(ref: str) -> str:
    if any(ch.isspace() for ch in ref):
        raise SourceNotResolved("a URL cannot contain spaces")
    try:
        parts = urlsplit(ref)
        host, _port = parts.hostname, parts.port
    except ValueError:
        raise SourceNotResolved(f"{ref!r} is not a URL") from None
    if parts.scheme.lower() not in URL_SCHEMES or not host:
        raise SourceNotResolved("a URL source must be an http or https address")
    if parts.username is not None or parts.password is not None:
        raise SourceNotResolved("a URL source cannot carry a user name or password")
    return ref


def _as_of(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise SourceNotResolved(f"as_of {value!r} is not an ISO date-time") from None
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    """A row's timestamp as ISO 8601 in UTC (SQLite hands timestamps back as text)."""
    if value is None:
        return None
    when = datetime.fromisoformat(value) if isinstance(value, str) else value
    return (when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)).isoformat()


def report_metrics(raw: Any) -> Dict[str, Any]:
    """``agent_reports.metrics``: a dict on Postgres (JSONB), text on SQLite."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)) and raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_figure(value: Any) -> bool:
    """One number or one piece of text: something a claim can quote."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, str) and bool(value.strip())


def _escape_like(text: str) -> str:
    return text.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2).replace("%", LIKE_ESCAPE + "%").replace("_", LIKE_ESCAPE + "_")


def _contains(text: str) -> str:
    """A LIKE pattern matching ``text`` anywhere, case folded, wildcards literal."""
    return f"%{_escape_like(text.lower())}%"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── the queries (bound parameters only; LIKE escapes with a backslash) ──────
_DELIVERABLE = sa.text(
    """
    SELECT o.id, o.title, o.artifact_type, o.preview_url, o.deleted_at, o.created_at
      FROM v_workspace_outputs o
     WHERE o.id = :id AND o.workspace_id = :workspace_id
    """
)

_DELIVERABLE_SEARCH = sa.text(
    """
    SELECT o.id, o.title, o.artifact_type, o.preview_url, o.created_at
      FROM v_workspace_outputs o
     WHERE o.workspace_id = :workspace_id
       AND o.deleted_at IS NULL
       AND (lower(COALESCE(o.title, '')) LIKE :pattern ESCAPE '\\'
            OR lower(COALESCE(o.summary, '')) LIKE :pattern ESCAPE '\\')
     ORDER BY o.created_at DESC
     LIMIT :limit
    """
)

_REPORT = sa.text(
    """
    SELECT r.id, r.title, r.summary, r.report_type, r.deleted_at, r.created_at
      FROM agent_reports r
     WHERE r.id = :id AND r.workspace_id = :workspace_id
    """
)

_REPORT_SEARCH = sa.text(
    """
    SELECT r.id, r.title, r.summary, r.report_type, r.created_at
      FROM agent_reports r
     WHERE r.workspace_id = :workspace_id
       AND r.deleted_at IS NULL
       AND (lower(COALESCE(r.title, '')) LIKE :pattern ESCAPE '\\'
            OR lower(COALESCE(r.summary, '')) LIKE :pattern ESCAPE '\\')
     ORDER BY r.created_at DESC
     LIMIT :limit
    """
)

# The newest reports at or before as_of whose metrics mention the key (the
# LIKE narrows; the key itself is checked on the parsed JSON).
_METRIC_READINGS = sa.text(
    """
    SELECT r.id, r.title, r.metrics, r.created_at
      FROM agent_reports r
     WHERE r.workspace_id = :workspace_id
       AND r.deleted_at IS NULL
       AND r.created_at <= :as_of
       AND CAST(r.metrics AS TEXT) LIKE :key_pattern ESCAPE '\\'
     ORDER BY r.created_at DESC
     LIMIT :limit
    """
).bindparams(sa.bindparam("as_of", type_=sa.DateTime(timezone=True)))

_METRIC_SEARCH = sa.text(
    """
    SELECT r.id, r.title, r.metrics, r.created_at
      FROM agent_reports r
     WHERE r.workspace_id = :workspace_id
       AND r.deleted_at IS NULL
       AND lower(CAST(r.metrics AS TEXT)) LIKE :pattern ESCAPE '\\'
     ORDER BY r.created_at DESC
     LIMIT :limit
    """
)


# ── resolving one source ────────────────────────────────────────────────────
def _resolve_deliverable(db: Any, workspace_id: UUID, ref: str, as_of: Optional[datetime]) -> ResolvedSource:
    row = db.execute(_DELIVERABLE, {"id": _uuid(ref, "Deliverable"), "workspace_id": str(workspace_id)}).first()
    if row is None:
        raise SourceNotResolved("no Deliverable with this id in this workspace")
    if row.deleted_at is not None:
        raise SourceNotResolved("the Deliverable was deleted")
    return ResolvedSource(
        kind="deliverable", ref=ref, title=row.title or "", detail=row.artifact_type,
        as_of=_iso(row.created_at), preview_url=row.preview_url,
    )


def _resolve_report(db: Any, workspace_id: UUID, ref: str, as_of: Optional[datetime]) -> ResolvedSource:
    row = db.execute(_REPORT, {"id": _uuid(ref, "report"), "workspace_id": str(workspace_id)}).first()
    if row is None:
        raise SourceNotResolved("no report with this id in this workspace")
    if row.deleted_at is not None:
        raise SourceNotResolved("the report was deleted")
    return ResolvedSource(
        kind="report", ref=ref, title=row.title or "", detail=row.summary or row.report_type,
        as_of=_iso(row.created_at),
    )


def _document_row(db: Any, workspace_id: UUID, document_id: int) -> Any:
    return (
        db.query(Document.id, Document.filename, Document.original_filename, Document.description, Document.upload_date)
        .filter(Document.id == document_id, Document.workspace_id == workspace_id)
        .first()
    )


def _document_source(row: Any) -> ResolvedSource:
    return ResolvedSource(
        kind="document", ref=str(row.id), title=row.original_filename or row.filename,
        detail=row.description, as_of=_iso(row.upload_date),
    )


def _resolve_document(db: Any, workspace_id: UUID, ref: str, as_of: Optional[datetime]) -> ResolvedSource:
    row = _document_row(db, workspace_id, _document_id(ref))
    if row is None:
        raise SourceNotResolved("no document with this id in this workspace")
    return _document_source(row)


def _metric_source(name: str, value: Any, row: Any) -> ResolvedSource:
    return ResolvedSource(
        kind="metric", ref=name, title=name, detail=row.title, as_of=_iso(row.created_at),
        value=value, report_id=str(row.id),
    )


def _resolve_metric(db: Any, workspace_id: UUID, ref: str, as_of: Optional[datetime]) -> ResolvedSource:
    if as_of is None:
        raise SourceNotResolved("a metric needs as_of, the time it was read at")
    if as_of > _utcnow():
        raise SourceNotResolved("a metric's as_of cannot be in the future")
    key_pattern = f"%{_escape_like(json.dumps(ref, ensure_ascii=False))}%"
    rows = db.execute(
        _METRIC_READINGS,
        {"workspace_id": str(workspace_id), "as_of": as_of, "key_pattern": key_pattern, "limit": METRIC_SCAN_REPORTS},
    ).fetchall()
    for row in rows:
        value = report_metrics(row.metrics).get(ref)
        if _is_figure(value):
            return _metric_source(ref, value, row)
    raise SourceNotResolved(f"no report in this workspace carries the metric {ref!r} as of {as_of.isoformat()}")


def _resolve_url(db: Any, workspace_id: UUID, ref: str, as_of: Optional[datetime]) -> ResolvedSource:
    url = _url(ref)
    return ResolvedSource(
        kind="url", ref=url, title=urlsplit(url).hostname or url,
        as_of=as_of.isoformat() if as_of else None,
    )


_Resolver = Callable[[Any, UUID, str, Optional[datetime]], ResolvedSource]
_RESOLVERS: Dict[str, _Resolver] = {
    "deliverable": _resolve_deliverable,
    "report": _resolve_report,
    "document": _resolve_document,
    "metric": _resolve_metric,
    "url": _resolve_url,
}


def resolve(db: Any, workspace_id: Any, source: Mapping[str, Any]) -> ResolvedSource:
    """The source as it exists in ``workspace_id``, or :class:`SourceNotResolved`."""
    resolver = _RESOLVERS.get(source.get("kind"))
    if resolver is None:
        raise SourceNotResolved(f"{source.get('kind')!r} is not a source kind")
    ref = source.get("ref")
    if not isinstance(ref, str) or not ref.strip():
        raise SourceNotResolved("the source has no ref")
    ref = ref.strip()
    if len(ref) > REF_MAX_CHARS:
        raise SourceNotResolved(f"a ref is at most {REF_MAX_CHARS} characters")
    return resolver(db, _workspace(workspace_id), ref, _as_of(source.get("as_of")))


def unresolved(db: Any, workspace_id: Any, sources: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Claim name → why its source does not resolve in ``workspace_id``; empty when all do."""
    out: Dict[str, str] = {}
    for name, source in (sources or {}).items():
        if not isinstance(source, Mapping):
            out[name] = "the source is not an object"
            continue
        try:
            resolve(db, workspace_id, source)
        except SourceNotResolved as exc:
            out[name] = str(exc)
    return out


def require_resolved(
    db: Any,
    workspace_id: Any,
    value: Any,
    *,
    unchanged_from: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """``value`` as a post stores it, once every source the save adds or changes
    resolves in the workspace. A source kept exactly as ``unchanged_from`` holds
    it is left to the approval check, so a source deleted since the last save
    never blocks an unrelated edit. :class:`service.InvalidPost` for a bad
    shape, :class:`SourcesNotFound` naming each claim that does not resolve."""
    clean = service.validate_sources(value)
    before = unchanged_from or {}
    changed = {name: source for name, source in clean.items() if before.get(name) != source}
    missing = unresolved(db, workspace_id, changed)
    if missing:
        raise SourcesNotFound(missing)
    return clean


# ── the source picker ───────────────────────────────────────────────────────
def _search_deliverables(db: Any, workspace_id: UUID, text: str, limit: int) -> List[ResolvedSource]:
    rows = db.execute(
        _DELIVERABLE_SEARCH, {"workspace_id": str(workspace_id), "pattern": _contains(text), "limit": limit}
    ).fetchall()
    return [
        ResolvedSource(
            kind="deliverable", ref=str(row.id), title=row.title or "", detail=row.artifact_type,
            as_of=_iso(row.created_at), preview_url=row.preview_url,
        )
        for row in rows
    ]


def _search_reports(db: Any, workspace_id: UUID, text: str, limit: int) -> List[ResolvedSource]:
    rows = db.execute(
        _REPORT_SEARCH, {"workspace_id": str(workspace_id), "pattern": _contains(text), "limit": limit}
    ).fetchall()
    return [
        ResolvedSource(
            kind="report", ref=str(row.id), title=row.title or "", detail=row.summary or row.report_type,
            as_of=_iso(row.created_at),
        )
        for row in rows
    ]


def _search_documents(db: Any, workspace_id: UUID, text: str, limit: int) -> List[ResolvedSource]:
    query = db.query(
        Document.id, Document.filename, Document.original_filename, Document.description, Document.upload_date
    ).filter(Document.workspace_id == workspace_id)
    if text:
        pattern = _contains(text)
        query = query.filter(
            or_(
                func.lower(Document.filename).like(pattern, escape=LIKE_ESCAPE),
                func.lower(func.coalesce(Document.original_filename, "")).like(pattern, escape=LIKE_ESCAPE),
                func.lower(func.coalesce(Document.description, "")).like(pattern, escape=LIKE_ESCAPE),
            )
        )
    rows = query.order_by(Document.upload_date.desc(), Document.id.desc()).limit(limit).all()
    return [_document_source(row) for row in rows]


def _search_metrics(db: Any, workspace_id: UUID, text: str, limit: int) -> List[ResolvedSource]:
    """Metric names containing ``text``, each with its latest figure and the
    report it was read from: resolving the candidate gives the same figure."""
    rows = db.execute(
        _METRIC_SEARCH, {"workspace_id": str(workspace_id), "pattern": _contains(text), "limit": METRIC_SCAN_REPORTS}
    ).fetchall()
    needle = text.lower()
    seen: set = set()
    found: List[ResolvedSource] = []
    for row in rows:
        for name, value in report_metrics(row.metrics).items():
            if name in seen or needle not in name.lower() or not _is_figure(value):
                continue
            seen.add(name)
            found.append(_metric_source(name, value, row))
            if len(found) >= limit:
                return found
    return found


def _search_urls(db: Any, workspace_id: UUID, text: str, limit: int) -> List[ResolvedSource]:
    """A URL is typed, not searched: ``q`` itself, when it is an http(s) address."""
    if not text:
        return []
    try:
        url = _url(text)
    except SourceNotResolved:
        return []
    return [ResolvedSource(kind="url", ref=url, title=urlsplit(url).hostname or url, as_of=_utcnow().isoformat())]


_Searcher = Callable[[Any, UUID, str, int], List[ResolvedSource]]
_SEARCHERS: Dict[str, _Searcher] = {
    "deliverable": _search_deliverables,
    "report": _search_reports,
    "document": _search_documents,
    "metric": _search_metrics,
    "url": _search_urls,
}


def search(
    db: Any,
    workspace_id: Any,
    *,
    kind: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = SEARCH_DEFAULT_LIMIT,
) -> List[Dict[str, Any]]:
    """Candidates from ``workspace_id`` for ``kind`` (every kind when ``None``),
    newest first, at most ``limit`` per kind, matching ``q`` when it is given."""
    if kind is not None and kind not in _SEARCHERS:
        raise ValueError(f"kind must be one of {list(service.SOURCE_KINDS)}")
    text = (q or "").strip()
    workspace = _workspace(workspace_id)
    kinds = (kind,) if kind is not None else service.SOURCE_KINDS
    return [candidate.to_dict() for each in kinds for candidate in _SEARCHERS[each](db, workspace, text, limit)]
