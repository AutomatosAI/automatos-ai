"""PRD-270 W2 US-270-12b (K9, K10) — Automatos Cloud's approved plugins, as bundles with their scan report.

The rows: approval_status 'approved', active, security_status 'safe', and the scan the row names said
'safe' too (the approve route sets approval_status only, so both are checked), that scan being of this
plugin's slug and version, so the report is of the files a local install downloads. A plugin's lineage is
its slug (lineage.py): one plugin, one item, at the row's own version. The query names its columns, so the
S3 bucket and path, the source URL and repo, approved_by, the rejection reason, the enable count and the
materialised skill ids are never read.

The bundle carries the plugin's long description, author, licence, content counts, recommended models,
token estimate and its scan report (scan_report.py), its category the plugin category's slug and its icon
that category's icon, its publisher W1's display name of its original author (publishers.py). What the
upload's manifest leaves free is fitted before the strict check, so a plugin the scanner called safe is
never refused for it: the author's address dropped ('Name <a@b>' reads 'Name'; one still holding '@' reads
none), counts and the token estimate kept in the format's range, model ids off the format's pattern left
out, the long description cut to its cap. A bundle refused all the same is logged and left out.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from sqlalchemy.orm import Query, Session

from core.models.marketplace_plugins import MarketplacePlugin, PluginCategory, PluginSecurityScan
from services.marketplace_bundles.exported import (
    ExportedItem, envelope, exported_item, fitted_publisher, newest_per_lineage,
)
from services.marketplace_bundles.scan_report import cut, scan_report
from services.marketplace_bundles.schema import (
    AUTHOR_MAX, CONTENT_COUNT_MAX, CONTENT_KINDS, LONG_DESCRIPTION_MAX, MODEL_ID_PATTERN, RECOMMENDED_MODELS_MAX,
    TOKEN_ESTIMATE_MAX,
)
from services.marketplace_public.publishers import EMAIL_MARK, display_name

ITEM_TYPE = "plugin"
APPROVED = "approved"
SAFE = "safe"
ADDRESS_MARK = "<"
_MODEL_ID = re.compile(MODEL_ID_PATTERN)

_COLUMNS = (
    MarketplacePlugin.id, MarketplacePlugin.slug, MarketplacePlugin.name, MarketplacePlugin.version,
    MarketplacePlugin.description, MarketplacePlugin.long_description, MarketplacePlugin.tags,
    MarketplacePlugin.skills_count, MarketplacePlugin.commands_count, MarketplacePlugin.agents_count,
    MarketplacePlugin.hooks_count, MarketplacePlugin.original_author, MarketplacePlugin.license,
    MarketplacePlugin.recommended_models, MarketplacePlugin.token_estimate,
    PluginCategory.slug.label("marketplace_category"), PluginCategory.icon.label("marketplace_icon"),
    PluginSecurityScan.overall_verdict, PluginSecurityScan.static_scan_status, PluginSecurityScan.static_findings,
    PluginSecurityScan.llm_scan_status, PluginSecurityScan.llm_risk_score, PluginSecurityScan.llm_findings,
    PluginSecurityScan.llm_summary, PluginSecurityScan.scanned_at,
)


def exported_plugins(db: Session) -> List[ExportedItem]:
    """One ExportedItem per approved, active plugin whose scan said safe, by slug; a plugin whose bundle
    validate_bundle refuses is logged and left out."""
    rows = _selected(db.query(*_COLUMNS)).order_by(MarketplacePlugin.slug).all()
    items = (
        exported_item(ITEM_TYPE, lineage, envelope(
            ITEM_TYPE, lineage, fitted_publisher(display_name(lineage.newest.original_author)), _body(lineage.newest),
        ))
        for lineage in newest_per_lineage(ITEM_TYPE, rows)
    )
    return [item for item in items if item is not None]


def _selected(query: Query) -> Query:
    """``query`` joined to each plugin's scan and category, kept to K9's plugins."""
    return (
        query.join(PluginSecurityScan, PluginSecurityScan.id == MarketplacePlugin.security_scan_id)
        .outerjoin(PluginCategory, PluginCategory.id == MarketplacePlugin.category_id)
        .filter(
            MarketplacePlugin.approval_status == APPROVED, MarketplacePlugin.is_active.is_(True),
            MarketplacePlugin.security_status == SAFE, PluginSecurityScan.overall_verdict == SAFE,
            PluginSecurityScan.plugin_slug == MarketplacePlugin.slug,
            PluginSecurityScan.plugin_version == MarketplacePlugin.version,
        )
    )


def _body(row: Any) -> Dict[str, Any]:
    """The plugin object of ``row``'s bundle."""
    return {
        "slug": row.slug,
        "longDescription": cut(row.long_description, LONG_DESCRIPTION_MAX),
        "author": author_of(row.original_author),
        "license": row.license or "",
        "contents": {kind: bounded(getattr(row, f"{kind}_count"), CONTENT_COUNT_MAX) for kind in CONTENT_KINDS},
        "recommendedModels": models_of(row.recommended_models),
        "tokenEstimate": bounded(row.token_estimate, TOKEN_ESTIMATE_MAX),
        "scan": scan_report(row),
    }


def author_of(value: Any) -> str:
    """The upload manifest's author as a name: an address after '<' dropped, cut to the format's length;
    none when it is no text or still holds an '@' (K9: never an email)."""
    if not isinstance(value, str):
        return ""
    name = value.split(ADDRESS_MARK, 1)[0].strip()[:AUTHOR_MAX].rstrip()
    return "" if EMAIL_MARK in name else name


def bounded(value: Any, top: int) -> int:
    """A whole number from 0 to ``top``: a larger one reads as ``top``, anything else as 0."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return 0
    return min(value, top)


def models_of(value: Any) -> List[str]:
    """The distinct model ids the format allows, in their order, at most RECOMMENDED_MODELS_MAX."""
    if not isinstance(value, (list, tuple)):
        return []
    fitting = (model for model in value if isinstance(model, str) and _MODEL_ID.fullmatch(model) is not None)
    return list(dict.fromkeys(fitting))[:RECOMMENDED_MODELS_MAX]


__all__ = ["author_of", "bounded", "exported_plugins", "models_of"]
