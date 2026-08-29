"""
PRD-230 US-003 — Marketplace Packages CRUD + the pure signal matcher.
====================================================================

Read-side service over ``marketplace_packages``:
  - ``list_packages`` / ``list_showcased`` / ``get_by_slug`` — thin DB reads.
  - ``match_by_signals`` — the PURE matcher (no DB): rank packages against
    business-type signals (platforms, store URLs, commerce vocabulary). It powers
    the onboarding proposal (US-009) and ``platform_search_packages`` (US-006).

The matcher is deliberately separable and deterministic so it can be tested
without Postgres, and so the same ranking is reproducible across the tool and the
section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Scoring weights — platform / URL hits are the strongest signal, vocabulary the
# weakest. Kept as named constants (no magic numbers), tunable in one place.
_W_PLATFORM = 5
_W_URL = 5
_W_VERTICAL = 2
_W_VOCAB = 1

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


@dataclass
class PackageMatch:
    """A scored package match. ``reasons`` lists the signals that fired — used by
    the tool/section to say WHY a package was proposed."""

    package: Any
    score: int
    reasons: list[str] = field(default_factory=list)


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _signal_bag(signals: Any) -> dict[str, Any]:
    """Normalise the signals input into ``{platforms, urls, tokens, tags}``.

    Accepts a plain string (treated as free text) or a dict with any of
    ``platforms`` / ``urls`` / ``text`` / ``vertical_tags``.
    """
    if isinstance(signals, str):
        signals = {"text": signals}
    signals = signals or {}

    platforms = {p.lower() for p in _as_list(signals.get("platforms"))}
    urls = [u.lower() for u in _as_list(signals.get("urls"))]
    tags = {t.lower() for t in _as_list(signals.get("vertical_tags"))}

    # The token bag pools every free-text source so vocabulary/tag/platform terms
    # can be found however the caller supplied them.
    tokens: set[str] = set()
    tokens |= _tokens(" ".join(_as_list(signals.get("text"))))
    tokens |= platforms
    tokens |= tags
    for u in urls:
        tokens |= _tokens(u)
    return {"platforms": platforms, "urls": urls, "tokens": tokens, "tags": tags}


def _score_package(pkg: Any, bag: dict[str, Any]) -> PackageMatch:
    """Score ONE package against the normalised signal bag (pure)."""
    matching = getattr(pkg, "matching", None) or {}
    vertical_tags = {t.lower() for t in _as_list(getattr(pkg, "vertical_tags", None))}

    score = 0
    reasons: list[str] = []

    # Platform hits — the package names a platform the signals mention.
    for plat in {p.lower() for p in _as_list(matching.get("platforms"))}:
        if plat in bag["platforms"] or plat in bag["tokens"]:
            score += _W_PLATFORM
            reasons.append(f"platform:{plat}")

    # URL-pattern hits — a signal URL contains a package url pattern.
    for pat in {p.lower() for p in _as_list(matching.get("url_patterns"))}:
        if any(pat in u for u in bag["urls"]) or pat in bag["tokens"]:
            score += _W_URL
            reasons.append(f"url:{pat}")

    # Vertical-tag hits — shared vertical vocabulary.
    for tag in vertical_tags:
        if tag in bag["tokens"] or tag in bag["tags"]:
            score += _W_VERTICAL
            reasons.append(f"vertical:{tag}")

    # Commerce/vocabulary hits — the weakest signal.
    for term in {t.lower() for t in _as_list(matching.get("vocabulary"))}:
        if term in bag["tokens"]:
            score += _W_VOCAB
            reasons.append(f"vocab:{term}")

    return PackageMatch(package=pkg, score=score, reasons=reasons)


def match_by_signals(signals: Any, packages: Iterable[Any]) -> list[PackageMatch]:
    """Rank ``packages`` against business ``signals`` (PURE — no DB).

    Returns only positive-scoring matches, sorted by score desc with a
    deterministic tie-break (showcase first, then slug) so the same inputs always
    produce the same order across the tool and the onboarding section.
    """
    bag = _signal_bag(signals)
    matches = [_score_package(p, bag) for p in packages]
    matches = [m for m in matches if m.score > 0]
    matches.sort(
        key=lambda m: (
            -m.score,
            not bool(getattr(m.package, "showcase", False)),  # showcase first
            str(getattr(m.package, "slug", "")),
        )
    )
    return matches


# --------------------------------------------------------------------------- #
# DB reads (thin). The matcher above is the interesting, tested part.
# --------------------------------------------------------------------------- #


def list_packages(db: Any) -> list[Any]:
    from core.models.marketplace_packages import MarketplacePackage

    return db.query(MarketplacePackage).order_by(MarketplacePackage.name).all()


def list_showcased(db: Any) -> list[Any]:
    from core.models.marketplace_packages import MarketplacePackage

    return (
        db.query(MarketplacePackage)
        .filter(MarketplacePackage.showcase.is_(True))
        .order_by(MarketplacePackage.name)
        .all()
    )


def get_by_slug(db: Any, slug: str) -> Optional[Any]:
    from core.models.marketplace_packages import MarketplacePackage

    return (
        db.query(MarketplacePackage)
        .filter(MarketplacePackage.slug == slug)
        .one_or_none()
    )


def match_packages(db: Any, signals: Any) -> list[PackageMatch]:
    """DB-backed convenience: load all packages, then rank by signals."""
    return match_by_signals(signals, list_packages(db))
