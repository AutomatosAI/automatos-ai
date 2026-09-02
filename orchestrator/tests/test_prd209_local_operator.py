"""PRD-209 — the local edition has an operator identity (live-test finding).

With no users row, POST /api/chat 500'd with "No users found" (api/chat.py resolves
the caller by email, then users.id=1, then any user). The entrypoint now seeds users
id 1 with config.LOCAL_OPERATOR_EMAIL in local mode (fail-closed), and hybrid.py's
anonymous local lane carries that email so chat/attribution resolve a real user.
PRD-233 S6 adds the editable profile + greeting on top. Pure source guards.
"""
from __future__ import annotations

import pathlib
import re

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent


def test_entrypoint_seeds_the_local_operator_fail_closed():
    entry = (_REPO / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "INSERT INTO users (id, username, email, name, is_active) VALUES (1," in entry
    assert "LOCAL_OPERATOR_EMAIL" in entry
    assert "ON CONFLICT (id) DO NOTHING" in entry
    # fail-closed: a failed seed aborts the boot in local mode
    seg = entry.split("Local operator user present")[1][:400]
    assert "exit 1" in seg, "operator seed must fail closed"


def test_local_anonymous_session_carries_operator_email_only_in_local():
    hybrid = (_ORCH / "core" / "auth" / "hybrid.py").read_text(encoding="utf-8")
    assert re.search(r'if config\.AUTH_EDITION == "local":\s*\n(?:.*\n){0,4}.*UserContext\(email=config\.LOCAL_OPERATOR_EMAIL, system_role="super_admin"\)', hybrid), (
        "local anonymous lane must carry LOCAL_OPERATOR_EMAIL + super_admin"
    )
    assert "anon_user = UserContext()" in hybrid, "saas anonymous lane must stay the plain default UserContext()"


def test_config_and_defaults_declare_the_operator_email():
    cfg = (_ORCH / "config.py").read_text(encoding="utf-8")
    assert 'LOCAL_OPERATOR_EMAIL: str = os.getenv("LOCAL_OPERATOR_EMAIL", "local@automatos.local")' in cfg
    defaults = (_REPO / "envs" / "api.defaults").read_text(encoding="utf-8")
    assert "LOCAL_OPERATOR_EMAIL=local@automatos.local" in defaults
