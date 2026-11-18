"""Dev-box mode on the host (``dev_box.py``, ``env.github_env``, ``sandbox.GITHUB_DOMAINS``, the brief).

The token is a file the owner puts in the host's state directory, mode 0600. It never
reaches a session the host does not sandbox. A sandboxed Claude Code session gets it as
GH_TOKEN, with git's credential config, gh's own config folder and the host user's git
identity beside it, and nothing else new; a missing or loose file is a warning, and the
session runs exactly as it would without the mode. The value is in no file the host
writes for the session and in nothing it reports. GitHub's hosts are reachable only in
the mode; the credential read-deny list is the same either way. The brief appears only
in the mode and is the same for every ticket. The flag survives the login service and
reaches the fleet view.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from automatos_cli_host import dev_box, env, session, session_prompt
from automatos_cli_host.adapters import adapter_for
from automatos_cli_host.adapters.base import LaunchContext
from automatos_cli_host.config import DEV_BOX_TOKEN_FILENAME, HostConfig, parse_args
from automatos_cli_host.presets import CLAUDE
from automatos_cli_host.sandbox import GITHUB_DOMAINS, SessionSandbox, claude_settings
from automatos_cli_host.service import service_argv

from conftest import FAKE_CLAUDE
from test_session_fake_claude import _cfg, _run, _ticket

# Low entropy on purpose: nothing a secret scanner would read as a credential.
TOKEN = "devboxfixturetoken_" + "0" * 12
IDENTITY = ("Ada Fixture", "ada@example.invalid")
DEV_BOX_LOG = "automatos.cli_host.dev_box"
NO_SANDBOX = SessionSandbox(enabled=False)


def _token_file(state: Path, mode: int = 0o600, text: str = TOKEN + "\n") -> Path:
    state.mkdir(parents=True, exist_ok=True)
    path = state / DEV_BOX_TOKEN_FILENAME
    path.write_text(text)
    os.chmod(path, mode)
    return path


@pytest.fixture
def identity(monkeypatch):
    """The host user's git identity, as the start-up read would find it."""
    monkeypatch.setattr(dev_box, "git_identity", lambda: IDENTITY)
    return IDENTITY


# ── the token file ───────────────────────────────────────────────────────────

def test_a_0600_token_file_in_the_state_directory_loads(tmp_path, identity):
    cfg = HostConfig(state_dir=tmp_path / "state", dev_box=True)
    assert cfg.dev_box_token_path == tmp_path / "state" / "agent-github-token"
    _token_file(cfg.state_dir)
    assert dev_box.read_token(cfg.dev_box_token_path) == (TOKEN, None)
    credential = dev_box.SessionCredential.for_session(cfg, "claude")
    assert credential.carried and credential.token == TOKEN and credential.identity == IDENTITY


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o606])
def test_a_token_file_other_users_can_open_is_refused_with_a_warning(tmp_path, caplog, mode):
    cfg = HostConfig(state_dir=tmp_path / "state", dev_box=True)
    _token_file(cfg.state_dir, mode=mode)
    with caplog.at_level(logging.WARNING, logger=DEV_BOX_LOG):
        credential = dev_box.SessionCredential.for_session(cfg, "claude")
    assert not credential.carried
    assert any("chmod 600" in r.getMessage() and "without GitHub access" in r.getMessage() for r in caplog.records)
    assert all(TOKEN not in r.getMessage() for r in caplog.records)


def test_a_missing_token_file_is_a_warning_and_no_token(tmp_path, caplog):
    cfg = HostConfig(state_dir=tmp_path / "state", dev_box=True)
    with caplog.at_level(logging.WARNING, logger=DEV_BOX_LOG):
        assert dev_box.SessionCredential.for_session(cfg, "claude").carried is False
    assert any("no GitHub token at" in r.getMessage() for r in caplog.records)


def test_a_link_or_a_file_that_is_not_a_token_loads_nothing(tmp_path):
    real = _token_file(tmp_path / "elsewhere")
    state = tmp_path / "state"
    state.mkdir()
    (state / DEV_BOX_TOKEN_FILENAME).symlink_to(real)
    token, problem = dev_box.read_token(state / DEV_BOX_TOKEN_FILENAME)
    assert token is None and "link" in problem
    for text in ("", "two words\n", f"{TOKEN}\nsecond line\n", "short\n"):
        token, problem = dev_box.read_token(_token_file(tmp_path / "bad", text=text))
        assert token is None and "does not hold a GitHub token" in problem, text


def test_the_mode_off_or_another_cli_reads_no_token(tmp_path, caplog):
    state = tmp_path / "state"
    _token_file(state)
    with caplog.at_level(logging.WARNING, logger=DEV_BOX_LOG):
        assert dev_box.SessionCredential.for_session(HostConfig(state_dir=state), "claude").carried is False
        # GitHub Copilot would take GH_TOKEN as its own login; Codex has no host list to open.
        for cli in ("copilot", "codex"):
            assert dev_box.SessionCredential.for_session(HostConfig(state_dir=state, dev_box=True), cli).carried is False
    assert not caplog.records


# ── no sandbox, no token ─────────────────────────────────────────────────────

def test_the_token_never_reaches_a_session_the_host_does_not_sandbox(tmp_path, caplog):
    state = tmp_path / "state"
    _token_file(state)
    cfg = HostConfig(state_dir=state, dev_box=True, session_sandbox=NO_SANDBOX)
    with caplog.at_level(logging.WARNING, logger=DEV_BOX_LOG):
        assert dev_box.SessionCredential.for_session(cfg, "claude").carried is False
    assert not caplog.records                          # said once, at start (below), not per session
    assert dev_box.enabled(cfg) is False and dev_box.serves(cfg, "claude") is False


def test_the_host_says_once_at_start_why_the_mode_is_off_without_the_sandbox(tmp_path, caplog):
    cfg = HostConfig(state_dir=tmp_path / "state", dev_box=True, session_sandbox=NO_SANDBOX)
    _token_file(cfg.state_dir)
    with caplog.at_level(logging.INFO, logger=DEV_BOX_LOG):
        dev_box.log_summary(cfg)
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "--no-session-sandbox" in caplog.records[0].getMessage()


def test_an_unsandboxed_dev_box_host_runs_the_session_as_before(short_tmp, fake_home, env_clean, monkeypatch):
    _token_file(short_tmp / "state")
    monkeypatch.setenv("FAKE_CLAUDE_PRINT_ENV", "GH_TOKEN")
    monkeypatch.delenv("GH_TOKEN", raising=False)              # nothing of the host's own either
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    _, out = _run(short_tmp, _ticket(workdir), cfg=_cfg(short_tmp, dev_box=True, session_sandbox=NO_SANDBOX))
    assert out.status == "success", out
    sdir = short_tmp / "state" / "sessions" / "42"
    assert "GH_TOKEN=<unset>" in (sdir / "terminal.log").read_text(encoding="utf-8", errors="replace")
    assert session_prompt.DEV_BOX_SECTION not in (sdir / "system_prompt.md").read_text()
    push = next(d for d in out.permission_denials if "git push" in json.dumps(d))
    assert "sessions do not push or escalate" in push["reason"]


def test_the_host_announces_the_mode_only_with_the_sandbox(tmp_path):
    base = dict(state_dir=tmp_path / "state", cli_binaries={"claude": str(tmp_path / "no-such-claude")})
    assert session.host_capabilities(HostConfig(**base))["dev_box"] is False
    assert session.host_capabilities(HostConfig(**base, dev_box=True))["dev_box"] is True
    assert session.host_capabilities(HostConfig(**base, dev_box=True, session_sandbox=NO_SANDBOX))["dev_box"] is False


def test_the_host_says_at_start_whether_sessions_will_publish(tmp_path, caplog, identity):
    cfg = HostConfig(state_dir=tmp_path / "state", dev_box=True)
    with caplog.at_level(logging.INFO, logger=DEV_BOX_LOG):
        dev_box.log_summary(cfg)
        _token_file(cfg.state_dir)
        dev_box.log_summary(cfg)
    first, second = caplog.records[-2:]
    assert first.levelno == logging.WARNING and "run without GitHub access" in first.getMessage()
    assert second.levelno == logging.INFO and "push their branch" in second.getMessage()


# ── the session's environment ────────────────────────────────────────────────

def test_the_token_reaches_a_session_as_gh_token_with_git_config_and_identity_only():
    parent = {"PATH": "/x", "HOME": "/h", "GITHUB_TOKEN": "operator-fixture", "GH_TOKEN": "operator-gh-fixture",
              "GH_ENTERPRISE_TOKEN": "x", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "credential.helper",
              "GIT_CONFIG_VALUE_0": "store", "GIT_CONFIG_PARAMETERS": "'credential.helper'=''",
              "GIT_AUTHOR_NAME": "Someone Else"}
    before = env.build_session_env(CLAUDE, parent, path="/p")
    after = env.build_session_env(CLAUDE, parent, path="/p", github=env.github_env(TOKEN, "/nowhere/gh", IDENTITY))
    count = len(env.GITHUB_GIT_CONFIG)
    identity_vars = {var for pair in env.IDENTITY_VARS for var in pair}
    new = {"GH_TOKEN", "GH_CONFIG_DIR", "GIT_CONFIG_COUNT", *identity_vars,
           *(f"GIT_CONFIG_KEY_{i}" for i in range(count)), *(f"GIT_CONFIG_VALUE_{i}" for i in range(count))}
    kept = {key for key in before
            if key not in env.GITHUB_INHERITED and not key.startswith(env.GITHUB_INHERITED_PREFIXES)}
    assert set(after) == kept | new                       # what the host inherited of GitHub's or git's is gone
    assert after["GH_TOKEN"] == TOKEN and "GITHUB_TOKEN" not in after and after["GH_CONFIG_DIR"] == "/nowhere/gh"
    assert [key for key, value in after.items() if TOKEN in value] == ["GH_TOKEN"]    # the value, once
    assert (after["GIT_AUTHOR_NAME"], after["GIT_AUTHOR_EMAIL"]) == IDENTITY
    assert (after["GIT_COMMITTER_NAME"], after["GIT_COMMITTER_EMAIL"]) == IDENTITY
    pairs = [(after[f"GIT_CONFIG_KEY_{i}"], after[f"GIT_CONFIG_VALUE_{i}"]) for i in range(int(after["GIT_CONFIG_COUNT"]))]
    assert pairs[0] == ("credential.https://github.com.helper", "")                 # earlier helpers forgotten
    assert pairs[1][0] == "credential.https://github.com.helper" and "$GH_TOKEN" in pairs[1][1]
    assert ("push.default", "simple") in pairs
    assert env.forbidden_keys_present(after, CLAUDE) == []


def test_without_an_identity_no_identity_is_set():
    built = env.build_session_env(CLAUDE, {"PATH": "/x", "GIT_AUTHOR_NAME": "Inherited"}, path="/p",
                                  github=env.github_env(TOKEN, "/nowhere/gh"))
    assert not {var for pair in env.IDENTITY_VARS for var in pair} & set(built)


def test_without_a_token_the_session_environment_is_what_it_was():
    parent = {"PATH": "/x", "HOME": "/h", "GITHUB_TOKEN": "operator-fixture"}
    assert env.build_session_env(CLAUDE, parent, path="/p", github=None) == env.build_session_env(
        CLAUDE, parent, path="/p")
    assert dev_box.SessionCredential(None).environment() is None


def test_a_sessions_gh_config_folder_is_its_own_and_removed_at_the_end(tmp_path):
    credential = dev_box.SessionCredential(TOKEN, IDENTITY)
    folder = credential.gh_config_dir
    assert folder is not None and not folder.exists()
    assert credential.environment()["GH_CONFIG_DIR"] == str(folder)
    assert dev_box.SessionCredential(TOKEN).gh_config_dir != folder          # one per session
    (folder / "sub").mkdir(parents=True)
    credential.discard()
    assert not folder.exists()
    target = tmp_path / "keep"
    target.mkdir()
    linked = dev_box.SessionCredential(TOKEN)
    linked.gh_config_dir.symlink_to(target)
    linked.discard()                                     # the link goes, never what it points at
    assert target.exists() and not linked.gh_config_dir.exists()


# ── the host user's git identity ─────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_identity_is_the_host_users_own_global_git_config(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    with caplog.at_level(logging.WARNING, logger=DEV_BOX_LOG):
        assert dev_box.read_git_identity() is None                           # unset: a warning, no identity
    assert any("user.name" in r.getMessage() for r in caplog.records)
    (tmp_path / ".gitconfig").write_text(f"[user]\n\tname = {IDENTITY[0]}\n\temail = {IDENTITY[1]}\n")
    assert dev_box.read_git_identity() == IDENTITY


# ── the sandbox ──────────────────────────────────────────────────────────────

def test_the_github_hosts_are_what_git_and_gh_need_and_no_upload_or_release_host():
    assert GITHUB_DOMAINS == ("github.com", "api.github.com", "codeload.github.com", "raw.githubusercontent.com")
    assert "uploads.github.com" not in GITHUB_DOMAINS and "objects.githubusercontent.com" not in GITHUB_DOMAINS


def test_github_is_reachable_only_in_the_mode_and_the_deny_list_is_unchanged(tmp_path):
    off = claude_settings(SessionSandbox(), off_limits=(tmp_path / "state",))
    on = claude_settings(SessionSandbox(), off_limits=(tmp_path / "state",), github=True)
    assert not set(GITHUB_DOMAINS) & set(off["network"]["allowedDomains"])
    assert set(GITHUB_DOMAINS) <= set(on["network"]["allowedDomains"])
    assert "ghcr.io" not in on["network"]["allowedDomains"]
    assert on["filesystem"] == off["filesystem"]
    deny = on["filesystem"]["denyRead"]
    assert "~/.config/gh" in deny and "~/.git-credentials" in deny
    assert str(tmp_path / "state") in deny                # the token file sits in the host's state: unreadable
    assert on["network"]["strictAllowlist"] is True and on["allowUnsandboxedCommands"] is False


def test_a_dev_box_launch_writes_github_into_the_settings_and_never_the_token(short_tmp, fake_home):
    cwd = short_tmp / "ws"
    cwd.mkdir()
    session_dir = short_tmp / "state" / "sessions" / "7"
    adapter = adapter_for("claude", {"claude": str(FAKE_CLAUDE)}, SessionSandbox())
    for carried in (False, True):
        ctx = LaunchContext(cwd=cwd, session_dir=session_dir, ticket_path=session_dir / "ticket.md",
                            system_prompt_path=session_dir / "system_prompt.md", task_id="7", session_id="s",
                            state_dir=short_tmp / "state", dev_box=carried)
        prepared = adapter.prepare(ctx)
        text = Path(prepared.args[prepared.args.index("--settings") + 1]).read_text()
        assert ("github.com" in json.loads(text)["sandbox"]["network"]["allowedDomains"]) is carried
        assert TOKEN not in text


# ── the brief ────────────────────────────────────────────────────────────────

def test_the_brief_appears_only_in_the_mode_and_is_the_same_for_every_ticket():
    a = session_prompt.build_system_prompt({"agent_name": "Coder", "task_id": 1, "title": "x"}, dev_box=True)
    b = session_prompt.build_system_prompt({"agent_name": "Coder", "task_id": 2, "title": "y"}, dev_box=True)
    plain = session_prompt.build_system_prompt({"agent_name": "Coder", "task_id": 1, "title": "x"})
    assert a == b and session_prompt.DEV_BOX_SECTION in a
    assert session_prompt.DEV_BOX_SECTION not in plain and "never push, publish or open pull requests" in plain
    assert "never push, publish or open pull requests" not in a          # no rule contradicting the brief
    for words in ("git commit -s", "type(scope): description", "git push -u origin <branch>",
                  "gh pr create --base <default> --head <branch>", "not a draft", "what you did NOT verify",
                  "pull request template", "Never merge", "never force-push a shared branch",
                  "The owner reviews and merges", "`update_ticket`", "`pull_request_url`", "`branch`", "`commits`",
                  "`{sha, subject}`", "`tested`", "could not run tests", "`note`", "Never send a status",
                  "the ticket goes to the owner for review"):
        assert words in a, words
    assert "moves to review" not in a
    assert "#1" not in a and "#2" not in a


def test_the_session_rules_without_the_mode_are_unchanged():
    assert session_prompt.SESSION_RULES.count("never push, publish or open pull requests") == 1
    assert session_prompt.DEV_BOX_SESSION_RULES.endswith(session_prompt.DEV_BOX_SECTION)


# ── redaction ────────────────────────────────────────────────────────────────

def test_the_token_is_replaced_in_text_bytes_and_reports():
    credential = dev_box.SessionCredential(TOKEN)
    report = {"reason": f"saw {TOKEN}", "input": {"command": f"echo {TOKEN}"}, "list": [TOKEN, 3], "n": None}
    clean = credential.scrub(report)
    assert TOKEN not in json.dumps(clean) and clean["list"][1] == 3 and clean["n"] is None
    assert credential.scrub(f"a{TOKEN}b".encode()) == f"a{dev_box.REDACTED}b".encode()
    untouched = {"x": 1}
    assert dev_box.SessionCredential(None).scrub(untouched) is untouched      # no token: nothing is copied


def test_a_token_split_between_two_reads_is_still_replaced():
    raw = f"one {TOKEN} two".encode()
    for cut in range(1, len(raw)):
        stream = dev_box.SessionCredential(TOKEN).stream()
        shown = stream.feed(raw[:cut]) + stream.feed(raw[cut:]) + stream.flush()
        assert shown == f"one {dev_box.REDACTED} two".encode(), cut
    passthrough = dev_box.SessionCredential(None).stream()
    assert passthrough.feed(raw) == raw and passthrough.flush() == b""


def test_events_are_scrubbed_on_their_way_into_the_queue():
    events = dev_box.ScrubbedQueue(dev_box.SessionCredential(TOKEN).scrub)
    events.put({"event": "PreToolUse", "subject": f"git push https://x:{TOKEN}@github.com/o/r"})
    events.put_nowait({"event": "Notification", "message": TOKEN})
    seen = [events.get_nowait(), events.get_nowait()]
    assert TOKEN not in json.dumps(seen)


# ── one ticket, end to end, against the fake claude ──────────────────────────

def test_a_dev_box_session_carries_the_token_and_the_host_reports_none_of_it(short_tmp, fake_home, env_clean,
                                                                            monkeypatch, identity):
    _token_file(short_tmp / "state")
    monkeypatch.setenv("FAKE_CLAUDE_PRINT_ENV", "GH_TOKEN,GIT_CONFIG_COUNT,GITHUB_TOKEN,GIT_AUTHOR_NAME,GIT_COMMITTER_EMAIL")
    monkeypatch.setenv("FAKE_CLAUDE_MKDIR_ENV", "GH_CONFIG_DIR")      # as gh may make its config folder
    monkeypatch.setenv("GITHUB_TOKEN", "operator-fixture")      # the host's own: never the session's
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    s, out = _run(short_tmp, _ticket(workdir), cfg=_cfg(short_tmp, dev_box=True))
    assert out.status == "success", out
    sdir = short_tmp / "state" / "sessions" / "42"
    shown = (sdir / "terminal.log").read_text(encoding="utf-8", errors="replace")
    assert f"GH_TOKEN={dev_box.REDACTED}" in shown and TOKEN not in shown       # it reached the session, not the log
    assert f"GIT_CONFIG_COUNT={len(env.GITHUB_GIT_CONFIG)}" in shown and "GITHUB_TOKEN=<unset>" in shown
    assert f"GIT_AUTHOR_NAME={IDENTITY[0]}" in shown and f"GIT_COMMITTER_EMAIL={IDENTITY[1]}" in shown
    assert not s._github.gh_config_dir.exists()               # the session made it; the host removed it
    for name in ("settings.json", "ticket.md", "system_prompt.md"):
        assert TOKEN not in (sdir / name).read_text(), name
    settings = json.loads((sdir / "settings.json").read_text())
    assert "github.com" in settings["sandbox"]["network"]["allowedDomains"]
    assert session_prompt.DEV_BOX_SECTION in (sdir / "system_prompt.md").read_text()
    push = next(d for d in out.permission_denials if "git push" in json.dumps(d))
    assert "main is a protected branch" in push["reason"] and "dev-box mode" in push["reason"]
    events = []
    while not s.events.empty():
        events.append(s.events.get_nowait())
    assert TOKEN not in json.dumps(out.as_result_payload(1)) and TOKEN not in json.dumps(events)


def test_a_dev_box_host_without_its_token_runs_the_session_as_before(short_tmp, fake_home, env_clean, caplog):
    workdir = short_tmp / "ws" / "repo"
    workdir.mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger=DEV_BOX_LOG):
        _, out = _run(short_tmp, _ticket(workdir), cfg=_cfg(short_tmp, dev_box=True))
    assert out.status == "success", out
    sdir = short_tmp / "state" / "sessions" / "42"
    assert session_prompt.DEV_BOX_SECTION not in (sdir / "system_prompt.md").read_text()
    assert "github.com" not in json.loads((sdir / "settings.json").read_text())["sandbox"]["network"]["allowedDomains"]
    push = next(d for d in out.permission_denials if "git push" in json.dumps(d))
    assert "sessions do not push or escalate" in push["reason"]
    assert any("no GitHub token at" in r.getMessage() for r in caplog.records)


# ── the repository's own rules ───────────────────────────────────────────────

def test_outside_a_git_repository_the_named_list_alone_applies(tmp_path):
    rules = dev_box.repo_rules(tmp_path)
    assert rules.origin_owner is None and rules.protected_branch("main") and not rules.protected_branch("feat/x")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_rules_read_origins_owner_and_default_branch_from_the_checkout(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("remote", "add", "origin", "git@github.com:Octo-Org/repo.git")
    git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/integration")
    rules = dev_box.repo_rules(tmp_path)
    assert rules.origin_owner == "octo-org"
    assert rules.protected_branch("integration") and rules.protected_branch("main")


@pytest.mark.parametrize("name", ["dev", "live", "stable", "next", "hotfix/x", "staging", "uat", "qa", "trunk",
                                  "mainline", "main.2", "Release/7"])
def test_the_named_list_protects_the_usual_long_lived_branches(name):
    assert dev_box.repo_rules(Path("/nonexistent-folder")).protected_branch(name), name


# ── the flag, the service and the fleet view ─────────────────────────────────

def test_the_flag_is_off_by_default_and_the_service_keeps_it(tmp_path):
    argv = ["--dir", str(tmp_path)]
    assert parse_args(argv).dev_box is False and "--dev-box" not in service_argv(parse_args(argv))
    on = parse_args([*argv, "--dev-box"])
    assert on.dev_box is True and "--dev-box" in service_argv(on)
    assert on.session_sandbox == parse_args(argv).session_sandbox          # the sandbox flags say nothing of it
