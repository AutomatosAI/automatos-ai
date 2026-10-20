"""PRD-264 FR-15 — the Article 50 notices take their public name from the one shared rule.

``core/public_principal.py`` is the only place that decides whether a name may appear in public
output. These tests hold the notice surfaces built on this branch to it: the widget's default
line (``modules.policy.ai_disclosure.default_line``) and the channel bots' notice
(``channels.ai_notice.notice_for``) name exactly what the kernel returns, and the rule's
literals live nowhere else in the orchestrator, so a second copy can't drift back in.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from core.public_principal import public_site_name

ORCH_ROOT = Path(__file__).resolve().parent.parent
KERNEL = ORCH_ROOT / "core" / "public_principal.py"
WS = str(uuid4())

NAMES = (
    None, "", "  ", 42, "Harbour Coffee", " Harbour Coffee Online ", "owner@harbour.example",
    "Ada Lovelace's Workspace", "ada@example.com's Workspace", "Jo’s workspace", "Workspace Coffee",
)

# The packages that build public output (the notices, the agent prompt, the widget API).
SOURCE_PACKAGES = ("api", "channels", "consumers", "core", "modules", "services")

# What only the FR-15 rule needs: the auto-created workspace form and the name reads.
RULE_LITERALS = (
    "\"'s workspace\"",
    "FROM business_profiles",
    "SELECT display_name FROM sites",
)


@pytest.mark.parametrize("name", NAMES)
def test_the_widget_line_names_the_site_only_when_the_kernel_says_it_is_public(name):
    from modules.policy.ai_disclosure import WIDGET_NOTICE_LINE, WIDGET_NOTICE_LINE_UNNAMED, default_line

    public = public_site_name(name)
    expected = WIDGET_NOTICE_LINE.format(site=public) if public else WIDGET_NOTICE_LINE_UNNAMED
    assert default_line(name) == expected


@pytest.mark.parametrize("principal", ("Harbour Coffee", None))
def test_the_channel_notice_names_whoever_the_kernel_returns(monkeypatch, principal):
    import channels.ai_notice as ai_notice
    from modules.policy.ai_disclosure import channel_notice

    db = object()
    asked: list = []

    def kernel(session, workspace_id):
        asked.append((session, workspace_id))
        return principal

    monkeypatch.setattr(ai_notice, "_get_redis", lambda: None)  # unreadable state: the notice is due
    monkeypatch.setattr(ai_notice, "workspace_principal_name", kernel)

    notice = asyncio.run(ai_notice.notice_for(db, workspace_id=WS, connection_id="conn-1", person_id="p-1"))

    assert notice.text == channel_notice(principal)
    assert asked == [(db, WS)]


def test_the_rule_has_one_copy_in_the_orchestrator():
    """Only ``core/public_principal.py`` carries the FR-15 rule's literals."""
    paths = sorted(path for package in SOURCE_PACKAGES for path in (ORCH_ROOT / package).rglob("*.py"))
    assert KERNEL in paths
    offenders = []
    for path in paths:
        if path == KERNEL:
            continue
        relative = path.relative_to(ORCH_ROOT)
        source = path.read_text(encoding="utf-8", errors="replace")
        offenders.extend(f"{relative}: {literal}" for literal in RULE_LITERALS if literal in source)
    assert offenders == []
