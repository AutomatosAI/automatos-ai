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
    assert f["expected_host_version"] == svc.EXPECTED_CLI_HOST_VERSION == "0.2.0"
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


def test_route_manifest_lists_the_health_endpoint():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json").read_text())
    assert {"method": "GET", "path": "/api/v1/cli-hosts/health"} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
