"""F042, hardened — the platform's secrets reached WITHOUT spelling their names.

The security review of 28ec7cca9 (2026-09-22) got past the name checks in
several ways, each proved on this machine: case on APFS (``.ENV`` opens
``.env``), recursive search (BSD ``grep -r`` walks dot-files; ``rg --hidden``;
an ``rg -g`` include glob overrides .gitignore), ``find -exec``, globs and
braces (``cat .en*``, ``cat .{e,x}nv``), a ``$(…)`` the gate cannot read,
``pushd``/a variable ``cd``, a hard link under a harmless name, archivers
(``tar cf - . | tar xOf -``). Each is refused here; ordinary work in the same
folders is not.
"""
from __future__ import annotations

import os

import pytest

from automatos_cli_host import policy, secret_reach
from automatos_cli_host.adapters.base import ToolClass, ToolIntent


@pytest.fixture(autouse=True)
def fresh_inventory():
    secret_reach.clear_inventory_cache()
    yield
    secret_reach.clear_inventory_cache()


@pytest.fixture
def layout(tmp_path):
    dev = tmp_path / "Development"
    platform = dev / "automatos-ai"
    (platform / "orchestrator" / "modules").mkdir(parents=True)
    (platform / "orchestrator" / "core").mkdir()
    (platform / ".env").write_text("DATABASE_URL=postgres://platform\n")
    (platform / "orchestrator" / ".env").write_text("OPENROUTER_API_KEY=sk-live\n")
    (platform / "orchestrator" / "core" / ".credential_key").write_text("k")
    (platform / ".env.example").write_text("DATABASE_URL=\n")
    (platform / "orchestrator" / "modules" / "a.py").write_text("x = 1\n")
    (platform / "README.md").write_text("# platform\n")
    shop = dev / "shop"
    shop.mkdir()
    (shop / ".env").write_text("SHOP_KEY=x\n")
    (shop / "notes.md").write_text("notes\n")
    state = tmp_path / "state"
    (state / "sessions" / "42").mkdir(parents=True)
    (state / "host.json").write_text('{"token": "host-token"}')
    return dict(dev=dev, platform=platform, shop=shop, state=state)


def _ctx(layout, cwd, **over):
    base = dict(cwd=cwd, secret_roots=(layout["platform"],), off_limits=(layout["state"],),
                extra_dirs=(layout["state"] / "sessions" / "42",))
    base.update(over)
    return policy.PolicyContext(**base)


def _bash(layout, where, line, **over):
    return policy.decide_bash(line, _ctx(layout, layout[where] if isinstance(where, str) else where, **over))


# ── case ────────────────────────────────────────────────────────────────────

def test_a_recased_name_is_the_same_file_on_a_case_insensitive_disk(layout):
    ctx = _ctx(layout, layout["platform"])
    read = ToolIntent(tool="Read", cls=ToolClass.FILE_READ, paths=(str(layout["platform"] / ".ENV"),))
    assert policy.decide(read, ctx).behavior == "deny"
    assert _bash(layout, "platform", "cat .ENV").behavior == "deny"
    assert _bash(layout, "platform", "cat orchestrator/.Env").behavior == "deny"


def test_a_recased_path_to_the_hosts_state_is_still_the_hosts_state(layout):
    recased = str(layout["state"]).replace("/state", "/STATE")
    assert _bash(layout, layout["state"].parent, f"cat {recased}/host.json").behavior == "deny"


# ── globs, braces and spliced quotes ────────────────────────────────────────

@pytest.mark.parametrize("line", ["cat .en*", "cat .e?v", "cat .{e,x}nv", 'cat .e""nv', "cat '.e'nv",
                                  "cat orchestrator/.e*", "head -n1 orchestrator/core/.cred*"])
def test_a_word_that_expands_to_a_secret_is_refused(layout, line):
    verdict = _bash(layout, "platform", line)
    assert verdict.behavior == "deny", line


@pytest.mark.parametrize("line", ["cat automatos-ai/.e*", "cat */.e*", "cat automatos-ai/{.env,README.md}"])
def test_from_the_folder_above_the_platform_too(layout, line):
    assert _bash(layout, "dev", line).behavior == "deny", line


@pytest.mark.parametrize("line", ["cat *.md", "cat README.md", "ls -la", "wc -l README.md", "cat shop/notes.md"])
def test_ordinary_words_are_untouched(layout, line):
    cwd = "dev" if line.startswith("cat shop") else "platform"
    assert _bash(layout, cwd, line).behavior == "allow", line


def test_dotglob_on_the_line_makes_a_star_reach_dot_files(layout):
    assert _bash(layout, "platform", "cat *").behavior == "allow"
    assert _bash(layout, "platform", "shopt -s dotglob; cat *").behavior == "deny"
    assert _bash(layout, "platform", "GLOBIGNORE=x; cat *").behavior == "deny"


def test_only_setting_the_option_counts_not_the_word(layout):
    """Review 2026-09-22: the word on the line flipped dot-matching for all of it."""
    assert _bash(layout, "platform", "echo dotglob; cat *env").behavior == "allow"
    assert _bash(layout, "platform", "cat *; shopt -s dotglob").behavior != "deny"    # set after, not before (shopt itself asks)
    assert _bash(layout, "platform", "GLOBIGNORE=x cat *").behavior == "allow"        # a prefix: not this expansion


# ── recursive search ────────────────────────────────────────────────────────

@pytest.mark.parametrize("line", ["grep -r KEY .", "grep -rn KEY orchestrator", "grep -R --color KEY",
                                  "egrep -ri 'key|url' .", "grep -d recurse KEY .", "rg --hidden KEY .",
                                  "rg -uu KEY .", "rg -g '*' KEY .", "rg --glob=.e* KEY ."])
def test_a_recursive_search_that_would_read_a_secret_is_refused(layout, line):
    verdict = _bash(layout, "platform", line)
    assert verdict.behavior == "deny" and "exclude them" in verdict.reason, line


@pytest.mark.parametrize("line", ["grep -r --exclude='.env*' --exclude='.cred*' KEY .",
                                  "grep -rn --include='*.py' KEY .", "grep -r KEY orchestrator/modules",
                                  "grep KEY README.md", "rg KEY .", "rg -t py --hidden KEY .",
                                  "rg --hidden -g '!.env*' -g '!.cred*' KEY ."])
def test_a_search_that_filters_the_secrets_out_or_never_meets_them_runs(layout, line):
    assert _bash(layout, "platform", line).behavior == "allow", line


def test_the_folder_a_search_runs_in_follows_the_lines_cd(layout):
    assert _bash(layout, "dev", "grep -r KEY .").behavior == "deny"
    assert _bash(layout, "dev", "cd shop && grep -r KEY .").behavior == "allow"
    assert _bash(layout, "dev", "(cd shop && grep -r KEY .) && grep -r KEY .").behavior == "deny"
    assert _bash(layout, "dev", "for d in shop; do (cd $d && grep -rn KEY .); done").behavior == "allow"


# ── find -exec ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line, expected", [
    ("find . -type f -exec cat {} +", "deny"),
    ("find orchestrator -exec head -5 {} \\;", "deny"),
    ("find . -name '*.py' -exec cat {} +", "allow"),
    ("find . ! -name '.env*' ! -name '.cred*' -exec cat {} +", "allow"),
    ("find . -exec ls -l {} \\;", "allow"),
    ("find . -name '*.py' -exec ls {} \\; -exec cat {} \\;", "allow"),
    ("find . -exec ls {} \\; -exec cat {} \\;", "deny"),
])
def test_find_exec_is_judged_by_what_its_walk_selects(layout, line, expected):
    assert _bash(layout, "platform", line).behavior == expected, line


# ── what the gate cannot read ───────────────────────────────────────────────

@pytest.mark.parametrize("line", ["cat $(rev <<< vne.)", "cat $(printf '\\056env')", "cat \"$(ls -a | head -3)\"",
                                  "X=$(rev <<< vne.); cat $X"])
def test_a_file_named_by_a_substitution_is_refused_where_secrets_live(layout, line):
    verdict = _bash(layout, "platform", line)
    assert verdict.behavior == "deny" and "cannot read" in verdict.reason, line


def test_a_substitution_where_no_secret_lives_is_left_to_the_other_rules(layout):
    assert _bash(layout, "shop", "cat $(ls -t | head -1)").behavior != "deny"
    # Review 2026-09-22: inside the platform's checkout but with no secret beneath
    # the folder, an ordinary $(…) path is not refused.
    modules = layout["platform"] / "orchestrator" / "modules"
    line = 'cat "$(git rev-parse --show-toplevel)/README.md"'
    assert _bash(layout, modules, line).behavior != "deny"


@pytest.mark.parametrize("line", ["pushd automatos-ai && cat .env", "D=automatos-ai; cd $D && cat .env",
                                  'cd "$(pwd)/automatos-ai" && cat .env', "cd $SOMEWHERE && cat .ENV"])
def test_every_way_of_moving_into_the_platform_first(layout, line):
    assert _bash(layout, "dev", line).behavior == "deny", line


# ── links and archives ──────────────────────────────────────────────────────

def test_a_hard_link_under_a_harmless_name_is_the_secret(layout):
    link = layout["dev"] / "notes.txt"
    os.link(layout["platform"] / ".env", link)
    ctx = _ctx(layout, layout["dev"])
    read = ToolIntent(tool="Read", cls=ToolClass.FILE_READ, paths=(str(link),))
    assert policy.decide(read, ctx).behavior == "deny"
    assert _bash(layout, "dev", "cat notes.txt").behavior == "deny"


def test_making_a_link_is_always_the_operators_call(layout):
    assert _bash(layout, "shop", "ln -s notes.md n2").behavior == "ask"
    assert _bash(layout, "shop", "ln -s notes.md n2", unlisted_bash="allow").behavior == "ask"


@pytest.mark.parametrize("line, expected", [
    ("tar cf - . | tar xOf -", "deny"),
    ("zip -r out.zip orchestrator", "deny"),
    ("cp -r orchestrator backup", "deny"),
    ("rsync -a . backup/", "deny"),
    ("tar czf out.tgz orchestrator/modules", "allow"),
    ("cp README.md README.bak", "allow"),
])
def test_archivers_and_recursive_copies_read_the_whole_tree(layout, line, expected):
    assert _bash(layout, "platform", line, unlisted_bash="allow").behavior == expected, line


# ── the file tools' own searches ────────────────────────────────────────────

def _search(tool, path, glob):
    return ToolIntent(tool=tool, cls=ToolClass.FILE_READ, paths=(str(path),), globs=(glob,))


@pytest.mark.parametrize("tool, glob, expected", [
    ("Grep", "*", "deny"),            # a Grep glob overrides .gitignore and --hidden is on
    ("Grep", "**/.e*", "deny"),
    ("Grep", "*.py", "allow"),
    ("Glob", "**/.e*", "deny"),
    ("Glob", "**/.ENV", "deny"),
    ("Glob", "**/*", "allow"),        # names only, and not aimed at secrets
    ("Glob", "**/*.py", "allow"),
])
def test_file_tool_searches_over_the_platform(layout, tool, glob, expected):
    verdict = policy.decide(_search(tool, layout["platform"], glob), _ctx(layout, layout["dev"]))
    assert verdict.behavior == expected, (tool, glob, verdict.reason)


def test_a_grep_glob_over_a_folder_without_secrets_is_fine(layout):
    ctx = _ctx(layout, layout["dev"])
    assert policy.decide(_search("Grep", layout["platform"] / "orchestrator" / "modules", "*"), ctx).behavior == "allow"
    assert policy.decide(_search("Grep", layout["shop"], "*.env"), ctx).behavior == "ask"


# ── the inventory and the host's word on it ─────────────────────────────────

def test_the_inventory_is_the_secret_files_that_exist(layout):
    found = {p.relative_to(layout["platform"]).as_posix()
             for p in secret_reach.secret_files(layout["platform"], policy._is_secret_name)}
    assert found == {".env", "orchestrator/.env", "orchestrator/core/.credential_key"}


def test_the_host_says_what_it_protects():
    summary = policy.secret_protection_summary()
    assert "out of every session's reach under" in summary and summary.rstrip().endswith(
        str(policy.platform_secret_roots()[0]))
