"""PRD-235 W3 — the backend side of "always on, always current".

Every heartbeat answer carries the host contract fingerprint and the expected
host version; the health view tells the board whether a host is online and
how many Claude Code tickets wait. Pure tests with fakes.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from services import cli_host_service as svc


def test_contract_fields_are_stable_and_versioned():
    f = svc.contract_fields()
    assert re.fullmatch(r"[0-9a-f]{16}", f["host_contract"])
    # 0.4.0: PRD-239 S7 v2 — terminal grants carry a launch; TerminalOpened/Closed events.
    # 0.5.0: session results and TerminalClosed carry the turn's token usage (analytics).
    assert f["expected_host_version"] == svc.EXPECTED_CLI_HOST_VERSION == "0.6.0"
    assert svc.contract_fields() == f


class _Host:
    def __init__(self, online: bool, seen_minutes_ago: int, status="paired"):
        self.status = status
        self.last_seen_at = datetime.now(timezone.utc) - timedelta(minutes=seen_minutes_ago)
        self._online = online
    def is_online(self, **_):
        return self._online
    def to_dict(self):
        return {"id": "h1", "name": "mac", "last_seen_at": self.last_seen_at.isoformat()}


class _Q:
    def __init__(self, rows):
        self.rows = rows
    def filter(self, *a, **k):
        return self
    def all(self):
        return self.rows
    def count(self):
        return len(self.rows)


class _DB:
    def __init__(self, hosts, agents, tasks):
        self._by = {"CliHost": hosts, "Agent": agents, "BoardTask": tasks}
    def query(self, model):
        return _Q(self._by[model.__name__])


def test_host_health_reports_offline_with_waiting_tickets():
    hosts = [_Host(online=False, seen_minutes_ago=90)]
    agents = [SimpleNamespace(id=15, configuration={"runtime": "cli"}), SimpleNamespace(id=1, configuration={})]
    tasks = [SimpleNamespace(id=92), SimpleNamespace(id=75)]
    out = svc.host_health(_DB(hosts, agents, tasks), "ws")
    assert out["online"] is False and out["paired_hosts"] == 1 and out["online_hosts"] == []
    assert out["cli_agents"] == 1 and out["waiting_tickets"] == 2
    assert out["last_seen_at"] and out["host_contract"] == svc.HOST_CONTRACT


def test_host_health_reports_online():
    out = svc.host_health(_DB([_Host(online=True, seen_minutes_ago=0)], [SimpleNamespace(id=15, configuration={"runtime": "cli"})], []), "ws")
    assert out["online"] is True and len(out["online_hosts"]) == 1 and out["waiting_tickets"] == 0


def test_health_and_the_lane_see_which_clis_the_online_hosts_run():
    """CLI adapter design §8.2/§8.3: ``providers_online`` is the union over ONLINE
    hosts of what each announced (``capabilities.providers``); an offline host's
    CLIs do not count, and the same CLI on two hosts is listed once."""
    on = _Host(online=True, seen_minutes_ago=0)
    on.capabilities = {"providers": ["claude"]}
    on2 = _Host(online=True, seen_minutes_ago=0)
    on2.capabilities = {"providers": ["codex", "claude"]}
    off = _Host(online=False, seen_minutes_ago=90)
    off.capabilities = {"providers": ["grok"]}
    db = _DB([on, on2, off], [], [])
    assert svc.serving_providers(db, "ws") == ["claude", "codex"]
    assert svc.host_health(db, "ws")["providers_online"] == ["claude", "codex"]
    # What a host said, read strictly: never said → None (no filter); said none → [].
    assert svc.served_providers_of(SimpleNamespace(capabilities=None)) is None
    assert svc.served_providers_of(SimpleNamespace(capabilities={"claude": {"version": "2"}})) is None
    assert svc.served_providers_of(SimpleNamespace(capabilities={"providers": []})) == []
    assert svc.served_providers_of(SimpleNamespace(capabilities={"providers": ["claude", 3, ""]})) == ["claude"]


def test_a_host_that_announced_no_cli_claims_nothing_without_touching_the_board():
    class _Untouchable:
        def query(self, *a, **k):
            raise AssertionError("a host with no CLI must not reach the claim statement")
    host = SimpleNamespace(id="h1", workspace_id="ws", capabilities={"providers": []})
    assert svc.claim_for_host(_Untouchable(), host, 5) == {"tasks": [], "parked": []}


def test_route_manifest_lists_the_health_endpoint():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json").read_text())
    assert {"method": "GET", "path": "/api/v1/cli-hosts/health"} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
