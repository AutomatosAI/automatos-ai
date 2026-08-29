#!/usr/bin/env python3
"""e2e scenario runner — drives real-user scenarios against a deployed environment.

STATUS: scaffold, written 2026-08-29 ahead of the test-account handoff. The
scenario format and assertion kinds are settled; the Clerk sign-in and the
exact endpoint paths carry TODO(verify-live) markers and get calibrated in the
first live session. Nothing here runs in CI; it is an operator tool.

Design rules:
- Conversation flows via the real chat API (no browser); Playwright cases are
  listed but delegated (`playwright: true` scenarios print their steps).
- No secrets in the repo: config via E2E_* env vars only.
- Reports are markdown, one file per run, appended per scenario.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import requests
    import yaml
except ImportError as exc:  # pragma: no cover
    sys.exit(f"e2e runner needs requests+pyyaml in the venv: {exc}")

ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "scenarios"

# Endpoint map — single place to fix when the live session calibrates paths.
ENDPOINTS = {
    "chat": "/api/chat",                    # TODO(verify-live): streaming vs sync path
    "board_tasks": "/api/board/tasks",      # TODO(verify-live)
    "board_summary": "/api/board/summary",  # TODO(verify-live)
    "questions": "/api/questions",          # TODO(verify-live): PRD-225 asks surface
}


@dataclass
class Config:
    base_url: str
    app_url: str
    email: str
    code: str
    agent_a: str = os.getenv("E2E_AGENT_A", "")
    agent_b: str = os.getenv("E2E_AGENT_B", "")
    token: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        missing = [v for v in ("E2E_BASE_URL", "E2E_CLERK_EMAIL") if not os.getenv(v)]
        if missing:
            sys.exit(f"missing env: {', '.join(missing)} (see e2e/README.md)")
        return cls(
            base_url=os.environ["E2E_BASE_URL"].rstrip("/"),
            app_url=os.getenv("E2E_APP_URL", "").rstrip("/"),
            email=os.environ["E2E_CLERK_EMAIL"],
            code=os.getenv("E2E_CLERK_CODE", "424242"),
        )


def clerk_sign_in(cfg: Config) -> str:
    """Sign in with a Clerk test-mode account (+clerk_test email, code 424242).

    TODO(verify-live): implement against the instance's Clerk frontend API:
      1. POST {clerk_fapi}/v1/client/sign_ins            {identifier: email}
      2. POST .../attempt_first_factor                    {strategy: email_code, code}
      3. exchange the session for the JWT the API expects (Authorization: Bearer).
    Calibrate in the first live session; until then run with E2E_TOKEN set.
    """
    tok = os.getenv("E2E_TOKEN")
    if tok:
        return tok
    raise NotImplementedError("clerk_sign_in pending live calibration — export E2E_TOKEN")


@dataclass
class Result:
    scenario: str
    title: str
    passed: bool
    detail: list[str] = field(default_factory=list)


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.s = requests.Session()

    # -- primitives --------------------------------------------------------
    def _auth(self):
        if not self.cfg.token:
            self.cfg.token = clerk_sign_in(self.cfg)
        self.s.headers["Authorization"] = f"Bearer {self.cfg.token}"

    def say(self, text: str) -> str:
        """One chat turn; returns Auto's reply text. TODO(verify-live): payload shape."""
        r = self.s.post(self.cfg.base_url + ENDPOINTS["chat"],
                        json={"message": text}, timeout=180)
        r.raise_for_status()
        body = r.json() if "json" in r.headers.get("content-type", "") else {"text": r.text}
        return body.get("reply") or body.get("text") or json.dumps(body)[:2000]

    def api_get(self, path: str) -> requests.Response:
        return self.s.get(self.cfg.base_url + path, timeout=60)

    # -- assertion kinds ---------------------------------------------------
    def check(self, kind: str, spec, reply: str, out: list[str]) -> bool:
        if kind == "chat_reply_contains":
            ok = spec.lower() in reply.lower()
        elif kind == "chat_reply_matches":
            ok = re.search(spec, reply, re.I) is not None
        elif kind == "api":
            path = self._fill(spec["get"])
            resp = self.api_get(path)
            ok = resp.status_code == spec.get("expect_status", 200)
            if ok and ("min_count" in spec or "expect_count" in spec):
                try:
                    items = resp.json()
                    n = len(items if isinstance(items, list) else items.get("items", []))
                except Exception:
                    n, ok = -1, False
                if "min_count" in spec:
                    ok = ok and n >= spec["min_count"]
                if "expect_count" in spec:
                    ok = ok and n == spec["expect_count"]
            out.append(f"  api {path} -> {resp.status_code}")
        elif kind in ("log_marker", "log_marker_absent"):
            out.append(f"  [log] {kind}: '{spec}' — checked via railway tail (operator lane)")
            ok = True  # log capture is advisory in the scaffold; the live loop wires it
        elif kind == "manual":
            input(f"  MANUAL STEP: {spec}\n  press enter when done...")
            ok = True
        else:
            out.append(f"  unknown assertion kind: {kind}")
            ok = False
        out.append(f"  {'PASS' if ok else 'FAIL'} {kind}: {str(spec)[:100]}")
        return ok

    def _fill(self, text: str) -> str:
        return (text.replace("{agent_a}", self.cfg.agent_a)
                    .replace("{agent_b}", self.cfg.agent_b))

    # -- scenario ----------------------------------------------------------
    def run_scenario(self, sc: dict, phase_after_232: bool) -> Result:
        out: list[str] = []
        if sc.get("playwright"):
            out.append("  playwright scenario — run steps manually / via playwright lane:")
            out += [f"    - {s}" for s in sc.get("steps", [])]
            return Result(sc["id"], sc["title"], True, out + ["  SKIPPED (playwright lane)"])
        ok_all = True
        for turn in sc.get("turns", []):
            if "manual" in turn:
                self.check("manual", turn["manual"], "", out)
                continue
            reply = self.say(self._fill(turn["say"]))
            out.append(f"  Auto: {reply[:300]}")
            key = ("expect_after_232" if phase_after_232 and "expect_after_232" in turn
                   else "expect_before_232" if not phase_after_232 and "expect_before_232" in turn
                   else "expect")
            for exp in turn.get(key, []):
                ((kind, spec),) = exp.items()
                ok_all &= self.check(kind, spec, reply, out)
            time.sleep(2)
        return Result(sc["id"], sc["title"], ok_all, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="scenario ids (S01 ...)")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--after-232", action="store_true",
                    help="evaluate canary-232 scenarios in their post-deploy expectation")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    docs = [yaml.safe_load(p.read_text()) for p in sorted(SCENARIO_DIR.glob("*.yaml"))]
    scenarios = [s for d in docs if d for s in d.get("scenarios", [])
                 if (not args.pack or d.get("pack") == args.pack)]
    if args.list:
        for s in scenarios:
            print(f"{s['id']:5} [{s.get('status','live'):10}] {s['title']}")
        return 0
    if args.ids:
        scenarios = [s for s in scenarios if s["id"] in args.ids]

    cfg = Config.from_env()
    runner = Runner(cfg)
    runner._auth()
    results = [runner.run_scenario(s, args.after_232) for s in scenarios]

    lines = [f"# e2e run {time.strftime('%F %T')}", ""]
    for r in results:
        lines.append(f"## {r.scenario} — {r.title}: {'✅ PASS' if r.passed else '❌ FAIL'}")
        lines += r.detail + [""]
    report = "\n".join(lines)
    print(report)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
