"""PRD-184 US-005 — the 7 driverless legacy channel adapters are deleted + de-routed.

The Channels primitive moved to a per-driver registry (``channels/drivers/*.py``:
discord / slack / telegram / whatsapp / webhook). The live ``/api/channels`` connect
/ verify / webhook / poll flow uses ``channels.drivers`` EXCLUSIVELY, and
``GET /api/channels/platforms`` (what the UI offers as connectable) is sourced from
the driver registry — never the legacy adapter map.

Seven platforms never got a driver: teams / google_chat / signal / imessage / irc /
matrix / line. Their ``channels/*_adapter.py`` files (~1,570 LOC) were reachable ONLY
via the legacy ``ChannelManager._ADAPTER_MAP`` (a string→class registry, trimmed in
the same commit — analogous to US-003's dispatch de-route) and the PRD-142 W3-S13
contract-test ``ADAPTERS`` list (trimmed to the 4 survivors in the same commit). No
live product code imported the seven adapter classes.

``api/channels.py::_ping_platform_legacy`` (a driver-era inline pinger handling only
telegram/slack/discord) had ZERO callers and is removed too.

BOUNDARY (must survive): the 4 live adapters (telegram/slack/discord/whatsapp) keep
both their ``*_adapter.py`` file AND their ``_ADAPTER_MAP`` entry.

Pure/static — file reads only, imports no app package.
"""
from __future__ import annotations

import pathlib
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_CHANNELS = _ORCH / "channels"
_MANAGER = _CHANNELS / "manager.py"
_API_CHANNELS = _ORCH / "api" / "channels.py"

# The 7 deleted driverless adapters (module basename → adapter class token).
_DELETED_ADAPTERS = (
    "teams_adapter",
    "google_chat_adapter",
    "signal_adapter",
    "imessage_adapter",
    "irc_adapter",
    "matrix_adapter",
    "line_adapter",
)
# The 4 adapters that MUST survive (they have a matching channels/drivers/ driver).
_SURVIVING_ADAPTERS = ("telegram_adapter", "slack_adapter", "discord_adapter", "whatsapp_adapter")

# Live source trees to sweep for dangling references (NOT tests/, which holds
# this guard + its own token list).
_SCAN_DIRS = ("channels", "api", "modules", "services", "core", "consumers")


def _live_py_files():
    for d in _SCAN_DIRS:
        root = _ORCH / d
        if not root.exists():
            continue
        yield from root.rglob("*.py")
    main = _ORCH / "main.py"
    if main.exists():
        yield main


def test_no_legacy_channel_adapters():
    """The 7 adapter files are gone and no live source names their modules."""
    for mod in _DELETED_ADAPTERS:
        assert not (_CHANNELS / f"{mod}.py").exists(), (
            f"channels/{mod}.py must stay deleted (PRD-184 US-005)"
        )
    offenders = {}
    for path in _live_py_files():
        text = path.read_text(errors="ignore")
        hits = [m for m in _DELETED_ADAPTERS if m in text]
        if hits:
            offenders[str(path.relative_to(_ORCH))] = hits
    assert not offenders, (
        f"dangling references to deleted legacy channel adapters: {offenders}"
    )


def test_adapter_map_de_routed():
    """ChannelManager._ADAPTER_MAP no longer registers any of the 7 driverless
    platforms (else it would lazy-import a deleted module)."""
    manager = _MANAGER.read_text()
    for platform in ("teams", "google_chat", "signal", "imessage", "irc", "matrix", "line"):
        assert f'"{platform}":' not in manager, (
            f"_ADAPTER_MAP must not register the driverless platform {platform!r}"
        )


def test_ping_platform_legacy_removed():
    """The zero-caller _ping_platform_legacy inline pinger is gone."""
    api = _API_CHANNELS.read_text()
    assert "_ping_platform_legacy" not in api, (
        "_ping_platform_legacy (0 callers) must stay removed from api/channels.py"
    )
    # Tree-wide: nothing anywhere references it.
    offenders = [
        str(p.relative_to(_ORCH))
        for p in _live_py_files()
        if "_ping_platform_legacy" in p.read_text(errors="ignore")
    ]
    assert not offenders, f"dangling _ping_platform_legacy references: {offenders}"


def test_live_channel_adapters_survive():
    """Boundary proof: the 4 driver-backed adapters keep both their file AND
    their _ADAPTER_MAP registration."""
    manager = _MANAGER.read_text()
    for mod in _SURVIVING_ADAPTERS:
        assert (_CHANNELS / f"{mod}.py").exists(), (
            f"live adapter channels/{mod}.py MUST survive"
        )
        assert f".{mod}" in manager, (
            f"_ADAPTER_MAP MUST keep the live {mod} registration"
        )
    # The driver registry (the active path) is untouched.
    for platform in ("telegram", "slack", "discord", "whatsapp"):
        assert f'"{platform}":' in manager, (
            f"_ADAPTER_MAP MUST keep the live platform {platform!r}"
        )
