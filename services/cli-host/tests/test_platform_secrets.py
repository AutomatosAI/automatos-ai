"""F042 — a session may work in the folder that holds the platform; it may never
hold the platform's secrets.

Night 1 (2026-09-18): an OPS ticket session started in ~/Development, went into
the Automatos checkout, ran ``set -a && . ./.env && set +a`` and then psql
against the platform's own database. ALWAYS_ASK_BASH put a card in front of
that shape of Bash — and the Read tool read the same file with no card at all.
Now the platform's .env family and credential key, and the host's own state,
are a hard deny from every tool; any other secrets file asks.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from automatos_cli_host import policy
from automatos_cli_host.adapters.base import ToolClass, ToolIntent


@pytest.fixture
def layout(tmp_path):
    dev = tmp_path / "Development"
    platform = dev / "automatos-ai"
    (platform / "orchestrator").mkdir(parents=True)
    (platform / ".env").write_text("DATABASE_URL=postgres://platform\n")
    (platform / "orchestrator" / ".env").write_text("OPENROUTER_API_KEY=sk-live\n")
    (platform / ".env.example").write_text("DATABASE_URL=\n")
    (platform / ".credential_key").write_text("k")
    (platform / "README.md").write_text("# platform\n")
    shop = dev / "shop"
    shop.mkdir()
    (shop / ".env").write_text("SHOP_KEY=x\n")
    state = tmp_path / "state"
    (state / "sessions" / "42").mkdir(parents=True)
    (state / "host.json").write_text('{"token": "host-token"}')
    (state / "sessions" / "42" / "ticket.md").write_text("# ticket\n")
    (state / "sessions" / "43").mkdir()
    return dict(dev=dev, platform=platform, shop=shop, state=state)


def _ctx(layout, cwd=None, **over):
    base = dict(cwd=cwd or layout["dev"], secret_roots=(layout["platform"],), off_limits=(layout["state"],),
                extra_dirs=(layout["state"] / "sessions" / "42",))
    base.update(over)
    return policy.PolicyContext(**base)


def _read(path, tool="Read", globs=()):
    return ToolIntent(tool=tool, cls=ToolClass.FILE_READ, paths=(str(path),) if path else (), globs=tuple(globs))


def _write(path):
    return ToolIntent(tool="Edit", cls=ToolClass.FILE_WRITE, paths=(str(path),))


# ── the file tools ──────────────────────────────────────────────────────────

def test_the_platform_env_is_refused_to_the_read_tool(layout):
    ctx = _ctx(layout)
    for target in (layout["platform"] / ".env", layout["platform"] / "orchestrator" / ".env"):
        verdict = policy.decide(_read(target), ctx)
        assert verdict.behavior == "deny" and "platform's own secrets" in verdict.reason, target


def test_no_approval_path_exists_for_the_credential_key(layout):
    assert policy.decide(_write(layout["platform"] / ".credential_key"), _ctx(layout)).behavior == "deny"


def test_the_platform_code_and_its_example_env_stay_readable(layout):
    ctx = _ctx(layout)
    assert policy.decide(_read(layout["platform"] / "README.md"), ctx).behavior == "allow"
    assert policy.decide(_read(layout["platform"] / ".env.example"), ctx).behavior == "allow"


def test_another_projects_env_is_a_question_not_a_refusal(layout):
    verdict = policy.decide(_read(layout["shop"] / ".env"), _ctx(layout))
    assert verdict.behavior == "ask" and ".env" in verdict.reason


def test_a_symlink_to_the_platform_env_is_caught_by_where_it_points(layout):
    link = layout["dev"] / "notes.txt"
    os.symlink(layout["platform"] / ".env", link)
    assert policy.decide(_read(link), _ctx(layout)).behavior == "deny"


def test_a_search_for_env_files_through_the_platform_is_refused(layout):
    ctx = _ctx(layout)
    over_dev = policy.decide(_read(layout["dev"], tool="Grep", globs=[".env"]), ctx)
    assert over_dev.behavior == "deny" and "platform's own secrets" in over_dev.reason
    assert policy.decide(_read(layout["shop"], tool="Grep", globs=["*.env"]), ctx).behavior == "ask"
    assert policy.decide(_read(None, tool="Glob", globs=["**/*.py"]), ctx).behavior == "allow"


def test_the_hosts_own_state_is_refused_but_the_sessions_own_folder_is_not(layout):
    home = layout["state"].parent                       # a session whose folder happens to hold the state
    ctx = _ctx(layout, cwd=home)
    assert policy.decide(_read(layout["state"] / "host.json"), ctx).behavior == "deny"
    assert policy.decide(_read(layout["state"] / "sessions" / "43" / "ticket.md"), ctx).behavior == "deny"
    assert policy.decide(_read(layout["state"] / "sessions" / "42" / "ticket.md"), ctx).behavior == "allow"


# ── Bash ────────────────────────────────────────────────────────────────────

def test_night_ones_line_is_refused(layout):
    line = f"cd {layout['platform']} && set -a && . ./.env && set +a && psql -c 'select 1'"
    verdict = policy.decide_bash(line, _ctx(layout))
    assert verdict.behavior == "deny" and "platform's own secrets" in verdict.reason


def test_the_platform_env_is_refused_however_it_is_named(layout):
    ctx = _ctx(layout)
    for line in (f"cat {layout['platform']}/orchestrator/.env",
                 "cat automatos-ai/.env",
                 "cd automatos-ai/orchestrator && grep KEY .env",
                 f"source {layout['platform']}/.env"):
        assert policy.decide_bash(line, ctx).behavior == "deny", line


def test_the_hosts_token_is_refused_to_bash(layout):
    assert policy.decide_bash(f"cat {layout['state']}/host.json", _ctx(layout)).behavior == "deny"


def test_another_projects_env_still_asks_and_ordinary_work_is_untouched(layout):
    ctx = _ctx(layout, cwd=layout["shop"])
    assert policy.decide_bash(". ./.env", ctx).behavior == "ask"
    assert policy.decide_bash("cat README.md", _ctx(layout, cwd=layout["platform"])).behavior == "allow"


def test_without_secret_roots_nothing_changes_for_bash(layout):
    bare = policy.PolicyContext(cwd=layout["platform"])
    assert policy.decide_bash(". ./.env", bare).behavior == "ask"      # the night-1 card, as before


def test_this_host_finds_the_checkout_it_runs_from():
    [root] = policy.platform_secret_roots()
    assert (root / "orchestrator").is_dir() and (root / "services" / "cli-host").is_dir()
