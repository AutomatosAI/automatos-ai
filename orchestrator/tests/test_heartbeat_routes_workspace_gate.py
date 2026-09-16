"""Per-agent heartbeat routes are an ordinary agent setting (2026-09-16).

``api/heartbeat.py`` used to be locked router-wide to the super admin (PRD-143
S6). The agent editor's heartbeat tab therefore 403'd for every workspace owner
while the scheduler kept firing whatever the saved block said — WATCHTOWER ran
hourly for weeks and nobody could see or change it from the product. The router
now carries two tiers, declared per route:

- per-agent routes → ``require_workspace_permission`` on the same matrix that
  gates editing the agent (``modules/policy/roles.py``): ``agents:read`` to
  see, ``agents:update`` to change, ``agents:execute`` to fire a tick;
- observability routes (``/status`` = every scheduler job across all
  workspaces, ``/analytics``, the orchestrator heartbeat) →
  ``require_super_admin``, unchanged.

Pinned here:
1. the full route table — every route carries exactly ONE of the two gates and
   is listed in ``EXPECTED``; a new endpoint fails until it is classified, so
   nothing on this router can land open (the concern that justified the old
   router-wide lock);
2. the lanes through the real FastAPI dependencies with the blessed fake-db
   pattern (``tests/test_prd143_obs_routers_batch1.py``): owner/admin/editor
   read, change and fire; viewer reads only; non-member and API-key principals
   are refused; an agent from another workspace is refused even for an owner;
   the super admin passes everything; the obs routes still say
   "Super admin only".
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (blessed pattern, see
# test_prd143_obs_routers_batch1.py). Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.super_admin import require_super_admin  # noqa: E402
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR  # noqa: E402
from core.database.database import get_db  # noqa: E402
import core.security.web_access as wa  # noqa: E402

import api.heartbeat as heartbeat_api  # noqa: E402

_WS = uuid.uuid4()
_OTHER_WS = uuid.uuid4()
_AGENT = 578
# What the DB holds — deliberately NOT the form default (60), so a 200 proves
# the saved state reaches the caller.
_SAVED_BLOCK = {"enabled": True, "interval_minutes": 1440, "prompt": "Daily ops report"}

SU = "super_admin"

# (method, path) → the ONE gate it must carry. Adding a route to api/heartbeat.py
# means adding it here — classified, never open.
EXPECTED = {
    ("GET", "/api/heartbeat/agents/{agent_id}/config"): "agents:read",
    ("PUT", "/api/heartbeat/agents/{agent_id}/config"): "agents:update",
    ("GET", "/api/heartbeat/agents/{agent_id}/last"): "agents:read",
    ("POST", "/api/heartbeat/agents/{agent_id}/run"): "agents:execute",
    ("GET", "/api/heartbeat/agents/{agent_id}/history"): "agents:read",
    ("GET", "/api/heartbeat/workspace"): "agents:read",
    ("PATCH", "/api/heartbeat/{heartbeat_id}/toggle"): "agents:update",
    ("GET", "/api/heartbeat/{heartbeat_id}/executions"): "agents:read",
    ("POST", "/api/heartbeat/orchestrator/run"): SU,
    ("GET", "/api/heartbeat/orchestrator/history"): SU,
    ("GET", "/api/heartbeat/status"): SU,
    ("GET", "/api/heartbeat/analytics"): SU,
}


# ── 1. The route table ────────────────────────────────────────────

def _flatten_calls(dependant, acc):
    """Every dependency callable reachable from a route (the authz-sweep idiom,
    tests/authz_sweep_probe.py) — decorator ``dependencies=[...]`` included."""
    for sub in dependant.dependencies:
        acc.append(sub.call)
        _flatten_calls(sub, acc)
    return acc


def _gate_of(route) -> str:
    calls = _flatten_calls(route.dependant, [])
    perms = {
        getattr(c, PERMISSION_MARKER_ATTR)
        for c in calls
        if hasattr(c, PERMISSION_MARKER_ATTR)
    }
    gates = ([SU] if require_super_admin in calls else []) + sorted(perms)
    assert len(gates) == 1, (
        f"{sorted(route.methods)} {route.path}: expected exactly one gate, "
        f"found {gates or 'NONE'}"
    )
    return gates[0]


def test_every_heartbeat_route_carries_exactly_one_gate():
    seen = {}
    for route in heartbeat_api.router.routes:
        for method in sorted(route.methods or []):
            seen[(method, route.path)] = _gate_of(route)

    unexpected = sorted(set(seen) - set(EXPECTED))
    missing = sorted(set(EXPECTED) - set(seen))
    changed = {k: (seen[k], EXPECTED[k]) for k in seen.keys() & EXPECTED.keys() if seen[k] != EXPECTED[k]}
    assert seen == EXPECTED, (
        "heartbeat route table drifted — classify it in EXPECTED: "
        f"unexpected={unexpected} missing={missing} changed(got, expected)={changed}"
    )
    assert heartbeat_api.router.dependencies == [], "the router-wide lock must not come back"


# ── 2. The lanes ──────────────────────────────────────────────────

def _user(name: str) -> UserContext:
    return UserContext(
        id=f"u-{name}", email=f"{name}@example.test",
        system_role="user", clerk_user_id=f"clerk_{name}",
    )


SUPER_ADMIN = UserContext(id="u-gerard", system_role="super_admin", clerk_user_id="clerk_su")
# core/auth/hybrid.py mints the env API-key principal as system_role='admin' with no clerk identity.
API_KEY_ADMIN = UserContext(id="api_key", email=None, system_role="admin")


def _db(*, role=None, agent_ws=_WS, block=_SAVED_BLOCK) -> MagicMock:
    """Query-level fake answering the SQL shapes the gate and the handlers issue:
    the membership lookup (``role``), the agent ownership check (``agent_ws``)
    and the saved configuration (``block``). Everything else is empty."""

    def execute(stmt, params=None):
        sql = str(stmt)
        res = MagicMock()
        res.fetchall.return_value = []
        res.fetchone.return_value = None
        if "SELECT wm.role" in sql:
            res.fetchone.return_value = (role,) if role else None
        elif "SELECT workspace_id FROM agents" in sql:
            res.fetchone.return_value = SimpleNamespace(workspace_id=str(agent_ws))
        elif "SELECT configuration FROM agents" in sql:
            res.fetchone.return_value = SimpleNamespace(configuration={"heartbeat": dict(block)})
        return res

    db = MagicMock()
    db.execute.side_effect = execute
    q = MagicMock()
    for chain in ("filter", "filter_by", "order_by", "limit"):
        getattr(q, chain).return_value = q
    q.all.return_value = []
    q.first.return_value = None
    db.query.return_value = q
    return db


class _FakeHeartbeatService:
    """The scheduler side of the handlers — records what the route asked for."""

    def __init__(self):
        self.scheduled = []
        self.unscheduled = []
        self.ran = []

    def schedule_agent_heartbeat(self, agent_id, workspace_id, hb_config):
        self.scheduled.append((agent_id, workspace_id, hb_config))

    def unschedule_heartbeat(self, job_id):
        self.unscheduled.append(job_id)

    async def run_agent_heartbeat(self, agent_id):
        self.ran.append(agent_id)
        return {"status": "success", "agent_id": agent_id}

    async def run_orchestrator_heartbeat(self, workspace_id):
        return {"status": "success", "workspace_id": workspace_id}

    def get_status(self):
        return {"active": False, "jobs": []}


@pytest.fixture
def service(monkeypatch) -> _FakeHeartbeatService:
    import services.heartbeat_service as hbs

    fake = _FakeHeartbeatService()
    monkeypatch.setattr(hbs, "get_heartbeat_service", lambda: fake)
    return fake


def _client(user: UserContext, db: MagicMock, auth_type: str = "clerk") -> TestClient:
    app = FastAPI()
    app.include_router(heartbeat_api.router)

    def _override_ctx():
        return RequestContext(workspace_id=_WS, user=user, auth_type=auth_type)

    def _override_db():
        yield db

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, raise_server_exceptions=False)


CONFIG = f"/api/heartbeat/agents/{_AGENT}/config"
RUN = f"/api/heartbeat/agents/{_AGENT}/run"
TOGGLE = f"/api/heartbeat/{_AGENT}/toggle"
PAYLOAD = {"enabled": True, "interval_minutes": 1440, "prompt": "Daily ops report"}


@pytest.mark.parametrize("role", ["owner", "admin", "editor"])
def test_roles_that_edit_the_agent_read_change_and_fire_its_heartbeat(role, service):
    c = _client(_user(role), _db(role=role))

    got = c.get(CONFIG)
    assert got.status_code == 200, got.text
    assert got.json()["interval_minutes"] == 1440  # the SAVED block, not the form default

    put = c.put(CONFIG, json=PAYLOAD)
    assert put.status_code == 200, put.text
    assert service.scheduled and service.scheduled[-1][0] == _AGENT
    assert service.scheduled[-1][2]["interval_minutes"] == 1440

    run = c.post(RUN)
    assert run.status_code == 200, run.text
    assert service.ran == [_AGENT]


def test_viewer_reads_but_cannot_change_or_fire(service):
    c = _client(_user("viewer"), _db(role="viewer"))
    assert c.get(CONFIG).status_code == 200

    put = c.put(CONFIG, json=PAYLOAD)
    assert put.status_code == 403, put.text
    assert put.json()["detail"] == "Permission denied: agents:update"

    run = c.post(RUN)
    assert run.status_code == 403, run.text
    assert run.json()["detail"] == "Permission denied: agents:execute"

    assert c.patch(TOGGLE).status_code == 403
    assert not service.scheduled and not service.ran


def test_non_member_and_api_key_principals_are_refused(service):
    stranger = _client(_user("stranger"), _db(role=None))
    resp = stranger.get(CONFIG)
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Permission denied: agents:read"

    # PRD-195 G3: no clerk identity ⇒ no membership ⇒ refused, exactly like
    # PUT /api/agents/{id}. The env API key is not a workspace member.
    api_key = _client(API_KEY_ADMIN, _db(role=None), auth_type="api_key")
    assert api_key.put(CONFIG, json=PAYLOAD).status_code == 403
    assert not service.scheduled


def test_agent_from_another_workspace_is_refused_even_for_the_owner(service):
    # 404, the same answer as "no such agent": agent ids are not an oracle
    # across tenants, and history answers like its siblings.
    c = _client(_user("owner"), _db(role="owner", agent_ws=_OTHER_WS))
    got = c.get(CONFIG)
    assert got.status_code == 404, got.text
    assert got.json()["detail"] == f"Agent {_AGENT} not found in this workspace"
    assert c.put(CONFIG, json=PAYLOAD).status_code == 404
    assert c.get(f"/api/heartbeat/agents/{_AGENT}/history").status_code == 404
    assert not service.scheduled


# ── 3. The webhook destination is checked at save time ────────────

PUBLIC_IP = "93.184.216.34"
_DNS = {"hooks.example": PUBLIC_IP, "evil.example": "10.0.0.5"}


def _fake_getaddrinfo(host, port, proto=None):
    """No network: two names with fixed answers, literal addresses as
    themselves, everything else NXDOMAIN (the PRD-240 test idiom)."""
    import ipaddress
    import socket

    ip = _DNS.get(host)
    if ip is None:
        try:
            ip = str(ipaddress.ip_address(host))
        except ValueError:
            raise socket.gaierror(f"no fake DNS for {host}")
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


@pytest.fixture
def fake_dns(monkeypatch):
    monkeypatch.setattr(wa, "_getaddrinfo", _fake_getaddrinfo)


@pytest.mark.parametrize(
    "url, fragment",
    [
        ("https://evil.example/hook", "blocked range"),
        ("http://169.254.169.254/latest/meta-data", "blocked range"),
        ("http://127.0.0.1:5432/", "blocked range"),
        ("ftp://hooks.example/hook", "Only http and https"),
    ],
)
def test_put_refuses_a_webhook_that_points_inside(url, fragment, service, fake_dns):
    db = _db(role="owner")
    c = _client(_user("owner"), db)
    resp = c.put(CONFIG, json={**PAYLOAD, "report_to": "webhook", "webhook_url": url})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith("webhook_url refused: ")
    assert fragment in resp.json()["detail"]
    # Refused BEFORE anything was written or scheduled.
    assert not any("UPDATE agents" in str(call.args[0]) for call in db.execute.call_args_list)
    assert not service.scheduled


def test_put_accepts_a_public_webhook_even_with_web_access_off(service, fake_dns, monkeypatch):
    # The WEB_ACCESS switch governs agent web access, not an operator's
    # destination; the blocked ranges and the denylist are what apply.
    monkeypatch.setattr(wa.config, "WEB_ACCESS", False)
    c = _client(_user("owner"), _db(role="owner"))
    resp = c.put(CONFIG, json={**PAYLOAD, "report_to": "webhook", "webhook_url": "https://hooks.example/hook"})
    assert resp.status_code == 200, resp.text
    assert service.scheduled[-1][2]["webhook_url"] == "https://hooks.example/hook"

    monkeypatch.setattr(wa.config, "WEB_ACCESS_DENY", ("hooks.example",))
    resp = c.put(CONFIG, json={**PAYLOAD, "report_to": "webhook", "webhook_url": "https://hooks.example/hook"})
    assert resp.status_code == 400, resp.text
    assert "WEB_ACCESS_DENY" in resp.json()["detail"]


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/api/heartbeat/status"),
        ("GET", "/api/heartbeat/analytics"),
        ("POST", "/api/heartbeat/orchestrator/run"),
        ("GET", "/api/heartbeat/orchestrator/history"),
    ],
)
def test_observability_routes_stay_super_admin_only(method, path, service):
    for role in ("owner", "admin"):
        resp = _client(_user(role), _db(role=role)).request(method, path)
        assert resp.status_code == 403, f"{role} {method} {path}: {resp.status_code} {resp.text}"
        assert resp.json()["detail"] == "Super admin only"

    resp = _client(SUPER_ADMIN, _db()).request(method, path)
    assert resp.status_code not in (401, 403), (
        f"super admin must pass {method} {path}: {resp.status_code} {resp.text}"
    )


def test_super_admin_passes_the_per_agent_routes_without_membership(service):
    c = _client(SUPER_ADMIN, _db(role=None))
    assert c.get(CONFIG).status_code == 200
    assert c.put(CONFIG, json=PAYLOAD).status_code == 200
    assert service.scheduled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
