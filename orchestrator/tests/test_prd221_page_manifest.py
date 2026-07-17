"""PRD-221 S1 — page manifest contract tests (pure: no DB, no network).

Locks the manifest to reality: every action name exists in the ActionRegistry
(imported, never a copied list), keys are unique kebab-case, routes are
well-formed, unknown lookups return None instead of raising, and subpath
routes resolve to their owning page via longest-prefix.
"""
from __future__ import annotations

import re

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from core.page_manifest import (  # noqa: E402
    all_pages,
    get_page,
    list_actions,
    resolve_route,
)
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402

_KEBAB = re.compile(r"^[a-z][a-z0-9-]*$")


def test_page_manifest_actions_all_registered():
    registered = {action.name for action in get_action_registry().get_all()}
    assert registered, "ActionRegistry returned no actions — registry failed to initialize"
    for page in all_pages():
        missing = set(page.actions) - registered
        assert not missing, f"page '{page.key}' references unregistered actions: {sorted(missing)}"


def test_page_manifest_shape():
    pages = all_pages()
    assert len(pages) == 12, f"expected the 12 seeded pages, got {len(pages)}"

    keys = [page.key for page in pages]
    assert len(keys) == len(set(keys)), "page keys must be unique"

    for page in pages:
        assert _KEBAB.match(page.key), f"key '{page.key}' is not kebab-case"
        assert page.route.startswith("/"), f"route '{page.route}' must start with '/'"
        assert page.title.strip(), f"page '{page.key}' has an empty title"
        assert page.purpose.strip(), f"page '{page.key}' has an empty purpose"
        assert page.actions, f"page '{page.key}' lists no actions"
        for prompt in page.quick_prompts:
            assert prompt.text.strip(), f"page '{page.key}' has an empty quick prompt"


def test_page_manifest_unknown_returns_none():
    assert get_page("nope") is None
    assert get_page(None) is None
    assert get_page("") is None
    assert resolve_route("/nope") is None
    assert resolve_route(None) is None
    assert resolve_route("not-a-route") is None
    assert list_actions("nope") == ()


def test_route_resolution_subpaths():
    assert resolve_route("/command-center") == "command-center"
    assert resolve_route("/missions/abc-123") == "missions"
    assert resolve_route("/chat/xyz") == "chat"
    # longest matching route wins for nested surfaces
    assert resolve_route("/marketplace/widgets/5") == "marketplace"


def test_quick_prompt_admin_flag_parsed():
    team = get_page("team")
    assert team is not None
    assert any(p.admin_only for p in team.quick_prompts), "team should carry an admin-only prompt"
    assert any(not p.admin_only for p in team.quick_prompts), "team should carry member-visible prompts too"


def test_list_actions_round_trip():
    cc = get_page("command-center")
    assert cc is not None
    assert list_actions("command-center") == cc.actions
    assert "platform_get_activity_feed" in cc.actions
