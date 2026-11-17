"""A pattern or a program a tool reads is data, not a command or a path (shell_data).

17 Nov: one ticket, in Auto mode with the session sandbox on, was held three times
and refused twice for text its commands never ran. The migration-tool rule matched
``|alembic`` inside a quoted awk program, the ``..`` check matched jq's recursive
descent operator, and the path check read a grep pattern (``/tests/``) as a file.
The commands are the ticket's, with this layout in place of the operator's folders;
the stored text of 2 was cut after its pipe, and 6's refusal named a grep pattern, so
both shapes of 6 are here. The cases the gate must still catch sit beside them:
quoting a path hides nothing, and text a shell runs is still a command.

An awk program is NOT data: it is read as written, as before this change, because
blanking its regex and string literals could hide a ``system(`` behind a ``/"/``
(review, twice). So 1, 2 and 6, which pipe into ``awk '!/tests\\/|alembic/'``, are
still held for ``alembic``: that rule answers before the awk escape check.
"""
from __future__ import annotations

import pytest

from automatos_cli_host import policy, secret_reach
from automatos_cli_host.adapters.base import ToolClass, ToolIntent


@pytest.fixture(autouse=True)
def fresh_inventory():
    secret_reach.clear_inventory_cache()
    yield
    secret_reach.clear_inventory_cache()


@pytest.fixture
def gate(tmp_path):
    """The ticket's layout: it worked in a checkout of the platform (its .env inside,
    the platform's secrets guarded), with its ticket folder under the host's state and
    a deliverables folder of its own, in Auto mode."""
    repo = tmp_path / "work" / "repo"
    (repo / "orchestrator" / "modules" / "socials").mkdir(parents=True)
    (repo / ".env").write_text("DATABASE_URL=postgres://platform\n")
    state = tmp_path / "host-state"
    ticket = state / "sessions" / "7"
    (ticket / "brand").mkdir(parents=True)
    deliverables = tmp_path / "work" / "deliverables" / "tasks" / "7"
    deliverables.mkdir(parents=True)
    ctx = policy.PolicyContext(cwd=repo, extra_dirs=(ticket, deliverables), permission_mode="auto",
                               secret_roots=(repo,), off_limits=(state,))

    def decide(line):
        line = line.replace("<REPO>", str(repo)).replace("<TICKET>", str(ticket)).replace("<DELIVERABLES>",
                                                                                         str(deliverables))
        return policy.decide(ToolIntent(tool="Bash", cls=ToolClass.SHELL, command=line), ctx)
    return decide


# The commands of 17 Nov. 3 and 4 (``$TMPDIR/service.py``) are not here: the gate
# still holds a path built from a variable the line does not define.
HELD_OR_REFUSED_ON_17_NOV = [
    # 5, refused: "path traversal ('..')" (jq's recursive descent)
    "ls -la <TICKET>/ <TICKET>/brand 2>&1; cd <DELIVERABLES> && jq '.. | objects | select(has(\"name\"))' some.json",
    # 6, refused: "grep outside the session directory: /tests/" (a pattern read as a path)
    "git grep -n -i -E \"a|b\" origin/main -- orchestrator | grep -v \"/tests/\" | head -30",
    "git grep -n -i -E \"a|b\" origin/main -- orchestrator | grep -v -E '/tests/|alembic' | head -30",
]
# …and the ones that pipe into an awk program, which is read as written.
HELD_ON_17_NOV_FOR_AN_AWK_PROGRAM = [
    # 1, held: "this command runs a database migration tool" (``|alembic`` in the awk program)
    "git grep -n \"planned_for\" origin/main -- orchestrator | awk '!/tests\\/|alembic/' | cut -c1-210 | head -40",
    # 2, held: the same (the stored text ended after the pipe; this is its ending)
    "cd <REPO> && git grep -n -i -E \"pending_approval|awaiting_approval|submit_social_post|approve_social|social_post\" "
    "origin/main -- orchestrator | awk '!/tests\\/|alembic/'",
    # 6, as written: its awk half
    "git grep -n -E \"status ==|SCHEDULED|planned_for\" origin/main -- orchestrator/modules/socials/schedule_jobs.py "
    "| cut -c1-220 | head -30; echo ----; git grep -n -i -E \"a|b\" origin/main -- orchestrator "
    "| awk '!/tests\\/|alembic/'",
]


@pytest.mark.parametrize("line", HELD_OR_REFUSED_ON_17_NOV, ids=["5", "6-pattern", "6-alternation"])
def test_the_commands_held_or_refused_on_17_nov_now_run(gate, line):
    decision = gate(line)
    assert decision.behavior == "allow", (line, decision)


@pytest.mark.parametrize("line", HELD_ON_17_NOV_FOR_AN_AWK_PROGRAM, ids=["1", "2", "6"])
def test_the_commands_of_17_nov_that_pipe_into_awk_are_still_held(gate, line):
    """HELD, not refused: the migration-tool rule reads ``alembic`` in the awk program
    and asks before the awk escape check refuses its ``|``. As on 17 Nov."""
    decision = gate(line)
    assert decision.behavior == "ask" and "migration" in decision.reason, (line, decision)


# ── what the gate still catches ─────────────────────────────────────────────

@pytest.mark.parametrize("line", [
    "alembic upgrade head",
    "cd orchestrator && alembic upgrade head",
    "echo ok | flask db upgrade",
    "django-admin migrate",
    'bash -c "alembic upgrade head"',                 # a shell runs its quoted text
    "sh -c 'cd orchestrator; flask db upgrade'",
    'eval "django-admin migrate"',
    'echo "$(alembic upgrade head)"',                # so does a substitution
    # Held for the ``&& alembic`` inside ssh's argument, not because the gate reads
    # ssh: ``ssh db-host 'alembic upgrade head'`` alone runs in Auto mode (unchanged).
    "ssh db-host 'cd app && alembic upgrade head'",
])
def test_a_migration_run_as_a_command_is_still_held(gate, line):
    decision = gate(line)
    assert decision.behavior == "ask" and "migration" in decision.reason, (line, decision)


@pytest.mark.parametrize("line", ["time alembic upgrade head", "env PGPASSWORD=x psql -h db"])
def test_a_wrapper_does_not_hide_a_command_the_operator_is_asked_about(gate, line):
    """``time``/``env`` are off the allowlist and Auto runs them; what they run was
    read only for the never-allowed list, so a migration or ``psql`` ran unasked."""
    assert gate(line).behavior == "ask", line


@pytest.mark.parametrize("line", [
    "cat ../../outside/secret",
    'cat "../../outside/secret"',                    # quoting a path hides nothing
    'grep -e token "../../outside/secret"',          # nor does a pattern beside it
    'grep "$(cat ../../outside/secret)" orchestrator/x.py',   # a pattern's substitution still runs
    "jq '.' ../outside/x.json",
    "jq -n 'import \"../../outside/x\" as $d; $d'",  # a jq module names a file
    "git grep -n token -- ../../outside",
    'bash -c "cat ../../outside/secret"',            # the line a shell runs is a command line
    "cat $'../x'",
])
def test_a_real_dotdot_is_still_refused(gate, line):
    assert gate(line).behavior == "deny", line


@pytest.mark.parametrize("line", [
    "grep -r token /etc",
    'sed -n 1p "/etc/passwd"',
    "echo x > /etc/x",
    "grep -f /etc/passwd orchestrator/x.py",          # a pattern FILE is a path
    "grep --frobnicate '/etc/passwd' orchestrator/x.py",   # an unknown option: no word is data
    "rg --files /etc",                                # no pattern at all: every operand is a path
    "awk /etc/passwd /etc/passwd",                    # a file that spells the program is still a file
])
def test_a_path_outside_the_session_is_still_refused(gate, line):
    assert gate(line).behavior == "deny", line


# ── a program is read the way its tool reads it ─────────────────────────────

@pytest.mark.parametrize("line", [
    "sed -e'1e cat /etc/passwd' orchestrator/x.txt",  # a glued -e is a program, not a flag
    "sed -n p orchestrator/x.txt -e '1e id'",         # GNU sed reads an -e after the file too
    'awk \'BEGIN { system("alembic upgrade head") }\' orchestrator/x.txt',
    "awk '@include \"lib.awk\"; BEGIN { }' orchestrator/x.txt",   # code from a file the gate cannot read
    'awk \'{ print $1 | "sort" }\' orchestrator/x.txt',
    'awk \'$1 || $2 { print | "sh" }\' orchestrator/x.txt',
])
def test_a_program_that_can_run_a_command_is_refused(gate, line):
    assert gate(line).behavior == "deny", line


@pytest.mark.parametrize("line", [
    "rg -n -e 'x|docker' orchestrator",                       # a pattern drives no Docker
    "git log --grep='fix|docker' --oneline -5",
    "git grep -n '/etc/x' HEAD -- orchestrator",              # git grep's pattern is not a path
    "bash -c \"jq '.. | objects' orchestrator/x.json\"",       # nor is a program inside a shell's line
])
def test_a_pattern_or_a_program_is_text(gate, line):
    assert gate(line).behavior == "allow", line


# ── the review of the change: where reading text as data must fail closed ───

@pytest.mark.parametrize("line", [
    'awk \'/[/]"/ { system("id") }\'',
    'awk \'/[/]"/ { print | "sh" }\'',
    'awk \'/[/]"/ { "id" | getline x }\'',
    'awk \'/[/]"/ { print > "/etc/x" } /"/\'',
    'awk \'$0 ~ /[/]"/ { system("id") }\'',
    'awk \'{print /"/;system("id");print /"/}\' f',          # a regex after print, not a division
    'awk \'x = 1 + /"/; system("id")\' f',
    'awk \'{printf /"/;system("id");printf /"/}\' f',
    'awk \'$1 == "a" || $2 == "b"\' orchestrator/x.txt',     # a logical or: refused, as before
    'awk \'{ print "a|b > c" }\' orchestrator/x.txt',        # a string: refused, as before
])
def test_an_awk_program_is_read_as_written(gate, line):
    """Where a regex or a string literal ends in an awk program is not certain (gawk
    lets ``/`` sit in a bracket expression; ``print /"/`` opens a regex where ``1 /``
    divides), and blanking literals hid ``system(`` twice. The program is read as
    written, as before this change: a ``|``, ``system(``, ``getline`` or ``print >``
    anywhere in it is refused."""
    decision = gate(line)
    assert decision.behavior == "deny" and "runs a command of its own" in decision.reason, (line, decision)


@pytest.mark.parametrize("line", [
    "sed 'r../../etc/passwd'",
    "sed -n 'w../../x'",
    "sed 'R../../x' f",
    "sed 's/a/b/w../../x'",
    "sed 'r/etc/shadow'",
    "sed -n 'w/etc/cron.d/x'",
    "sed 's/a/b/gw /etc/x' f",                       # the w flag after another flag
])
def test_a_sed_file_command_is_refused_with_its_filename_glued(gate, line):
    """A program is not path-checked, so sed's ``r``/``R``/``w``/``W`` and the ``w``
    flag of ``s`` are refused — with the filename glued to them, too."""
    assert gate(line).behavior == "deny", line


@pytest.mark.parametrize("line", [
    "sed '1r/etc/x'",
    "sed '/x/r ../y'",
    "sed '/x/!w out'",
    "sed -e 'w out'",
    "sed 's|a|b|w /etc/x'",
    "sed '$ R ../z'",
    "sed '/x/Ir/etc/x'",                             # an address flag, then the command
    "sed -e 's/a\\' -e '/x/w /etc/x' orchestrator/f",   # an -e that ends inside a regex
])
def test_a_sed_command_that_reaches_a_file_is_refused_where_sed_reads_it(gate, line):
    """After an address, a ``!``, a ``;`` or a block, and as a flag after the third
    delimiter of ``s`` whatever the delimiter: where sed reads a command or a flag."""
    decision = gate(line)
    assert decision.behavior == "deny" and "runs a command of its own" in decision.reason, (line, decision)


@pytest.mark.parametrize("line", [
    "sed 's/x/replacement/' orchestrator/f",
    "sed -n '/warning/p' orchestrator/f",
    "sed -n '/ERROR\\|WARN/p' orchestrator/f",
    "sed -n '1,/rows/p' orchestrator/f",
    "sed 's/a/b/g;s/c/d/' orchestrator/f",
    "sed 's|/usr|/opt|g' orchestrator/f",
    "sed -n '/^r/p' orchestrator/f",
    "sed '/w/d' orchestrator/f",
    "sed 's/[^/]*$//' orchestrator/f",              # a delimiter inside brackets, read alike by every sed
])
def test_a_letter_in_a_sed_regex_or_replacement_is_text(gate, line):
    """``r``, ``w`` or ``e`` inside a regex, a replacement or a word is not a command."""
    assert gate(line).behavior == "allow", line


@pytest.mark.parametrize("script", [
    "s/a/b",                     # an unterminated s
    "/abc",                      # an unterminated address
    "/[[:alpha:]/]x/p",          # GNU and BSD end this regex at one place, BusyBox at another
    "1,p",                       # a range with no second address
    "pq",                        # text after a command
    "k",                         # not a command
])
def test_a_sed_script_the_gate_cannot_follow_is_refused(script):
    from automatos_cli_host.sed_script import sed_escape

    assert sed_escape(script), script


def test_a_sed_script_holding_a_control_character_is_refused(gate):
    """GNU ends a label at any space character (``\\v``, ``\\f``, ``\\r``); the reader
    ends one at a blank only. So ``:a<VT>r/etc/passwd`` would read as one label here
    and as a label and an ``r`` in GNU sed: a control character but tab and newline
    refuses the script outright."""
    from automatos_cli_host.sed_script import sed_escape

    vertical_tab = chr(0x0B)
    assert sed_escape(f":a{vertical_tab}r/etc/passwd")
    decision = gate(f"sed ':a{vertical_tab}r/etc/passwd' orchestrator/f")
    assert decision.behavior == "deny" and "runs a command of its own" in decision.reason, decision


@pytest.mark.parametrize("line", [
    'jq "$F" x',
    "jq $F x",
    "A='{system(\"id\")}'; awk $A",
    "A='e id'; sed \"$A\" f",
])
def test_a_pattern_or_program_held_in_a_variable_is_not_data(gate, line):
    """Its text is only known when the line runs: every word is judged, and a program
    the line itself spells in a variable is read as that program."""
    assert gate(line).behavior == "deny", line


@pytest.mark.parametrize("line", [
    "rg --pre cat x .",
    "rg --pre=cat x .",
    "rg --pre-glob '*.gz' x .",
    "rg --hostname-bin ./hostname.sh x .",
    "git grep -O x",
    "git grep -Ovim x",
    "git grep --open-files-in-pager x",
    "git grep --open-files-in-pager=vim x",
    "git grep --op=sh -e x",                         # git takes a unique prefix of a long option
    "git grep --open=sh x",
    "git grep --open-files=sh x",
])
def test_a_search_option_that_runs_a_program_is_refused(gate, line):
    decision = gate(line)
    assert decision.behavior == "deny" and "never allowed" in decision.reason, (line, decision)


@pytest.mark.parametrize("line", [
    'cat ."."/x',
    "cat .\\./x",
    "cat .\\./.\\./etc/passwd",
    "cat ..{/,}etc/passwd",
    "cat {..,x}/etc/passwd",
    "cat {.,}./x",
])
def test_a_dotdot_the_shell_assembles_is_refused(gate, line):
    """Quotes, escapes and braces put ``..`` together only when the shell reads the
    line, so the raw text never shows it: the check reads the words, braces expanded."""
    assert gate(line).behavior == "deny", line


# ── the scan of the pushed PR: more holes where text was read as data ───────

@pytest.mark.parametrize("line", [
    "grep /etc/* orchestrator/x",
    "rg /etc/* orchestrator/x",
    "git grep /etc/* orchestrator",
    "grep '/etc/*' orchestrator/x",                  # quoted: the gate cannot tell after tokenising
])
def test_an_unquoted_glob_pattern_is_judged_as_a_path(gate, line):
    """Unquoted, ``/etc/*`` expands into real files outside the session before the tool
    runs; after tokenising a quoted one looks the same. So a pattern holding a glob or a
    brace is not data — it is judged as a path, like a word holding ``$``."""
    assert gate(line).behavior == "deny", line


@pytest.mark.parametrize("line", [
    "grep -n 'foo.*bar' orchestrator/x.py",          # a relative glob stays inside the session
    "grep '^$' orchestrator/x.py",
])
def test_a_relative_glob_pattern_still_runs(gate, line):
    """Judged as a path, a relative one resolves under the session directory, so it is
    allowed — only an absolute or ``..`` path is refused."""
    assert gate(line).behavior == "allow", line


def test_a_brace_operand_is_expanded_before_the_path_check(gate):
    """``{x,/etc/shadow}`` is two words, one a real path outside the session; the path
    check expands braces, as the ``..`` check does, and judges each word."""
    assert gate("grep {x,/etc/shadow} orchestrator/x").behavior == "deny"


@pytest.mark.parametrize("line", [
    "awk -i evil.awk 'p' orchestrator/x",            # gawk's include loads awk source
    "awk -l evil 'p' orchestrator/x",                # --load loads a shared library
    "awk --load evil 'p' orchestrator/x",
    "awk -ievil.awk 'p' orchestrator/x",             # glued short option
    "awk --load=evil 'p' orchestrator/x",            # glued long option
])
def test_an_awk_option_that_loads_code_from_a_file_is_refused(gate, line):
    decision = gate(line)
    assert decision.behavior == "deny" and "from a file" in decision.reason, (line, decision)


@pytest.mark.parametrize("line", [
    "RIPGREP_CONFIG_PATH=./c rg foo",
    "env RIPGREP_CONFIG_PATH=./c rg foo",
    "export RIPGREP_CONFIG_PATH=./c; rg foo",
])
def test_setting_ripgrep_config_path_is_refused(gate, line):
    """An rg config can carry ``--pre`` (a program rg runs), so setting, exporting or
    env-assigning ``RIPGREP_CONFIG_PATH`` is never allowed in a session."""
    decision = gate(line)
    assert decision.behavior == "deny" and "never allowed" in decision.reason, (line, decision)


@pytest.mark.parametrize("line", [
    "sed --expr='r /etc/passwd' p orchestrator/x",   # --expr is --expression; r reads a file
    "sed --fi=prog.sed 's/a/b/' orchestrator/x",      # --fi is --file
    "awk --so='BEGIN{system(\"id\")}' p orchestrator/x",   # --so is --source
    "sed --expression='w /etc/x' orchestrator/x",
    "awk --source='BEGIN{system(\"id\")}' orchestrator/x",
])
def test_an_abbreviated_sed_or_awk_long_option_is_read_or_refused(gate, line):
    """getopt_long takes a unique prefix, so an abbreviation reaches the real option and
    its program is read; an abbreviation that matches none or several fails closed."""
    assert gate(line).behavior == "deny", line


@pytest.mark.parametrize("line", [
    "sed -n '1,5p' orchestrator/x",
    "sed --quiet '1,5p' orchestrator/x",             # --quiet is known in full
])
def test_a_known_sed_option_still_runs(gate, line):
    assert gate(line).behavior == "allow", line
