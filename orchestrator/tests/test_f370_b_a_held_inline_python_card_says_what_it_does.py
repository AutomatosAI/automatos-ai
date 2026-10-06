"""F370 (ii) (night 10c) — a held inline-Python command's card says in plain words what it does.

The Brand Designer's sessions were held on ``python3 -c "import PIL; print(PIL.__version__)"``,
colour counts and a contrast function (card #1793 on #0890.2, "Held because: python3
runs code given inline"). The card read "The agent wants to **run a Python command**"
over the code: "As a shop owner I don't know what that is. Allowed."

The code is read (parsed, never run) and the card DESCRIBES what was seen. It never
says a command is safe: it WARNS on a write, the network, another program or a name
looked up as it runs, and otherwise ends on a line that promises nothing. Only a
command that is exactly one Python invocation whose code the shell cannot alter is
described; anything else keeps the card as it was. Text from the code is shown in
code spans only, so it cannot forge card text.
"""
from __future__ import annotations

import pytest

from services.cli_host_service import session_hold_question
from services.held_code_summary import NOTHING_PROMISED, code_span, inline_python, plain_summary

PIL_VERSION = 'python3 -c "import PIL; print(PIL.__version__)"'
COLOUR_COUNT = ("python3 -c \"from PIL import Image; im = Image.open('preview-A.png').convert('RGB'); "
                "print(len(im.getcolors(1 << 24)))\"")
CONTRAST = """python3 - <<'EOF'
def lum(h):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
a, b = lum('#1d3658'), lum('#ffffff')
print(round((max(a, b) + 0.05) / (min(a, b) + 0.05), 2))
EOF"""
REASSURING = ("changes nothing", "safe", "harmless", "only reads", "nothing else")


def _reassures(text):
    return any(word in (text or "").lower() for word in REASSURING)


def test_1793s_card_describes_the_pil_version_check_and_promises_nothing():
    card = session_hold_question(2144, {"subject": PIL_VERSION, "reason": "python3 runs code given inline"},
                                 ticket="ticket #0890.2")
    lead = card.split("<details>")[0]
    assert "run a Python command" in lead
    assert "It checks which version of `PIL` is installed, and prints it." in lead
    assert NOTHING_PROMISED in lead and not _reassures(lead)
    assert PIL_VERSION in card                               # the exact command is still there, folded


@pytest.mark.parametrize("command, words", [
    (COLOUR_COUNT, "It opens an image with PIL and reads its colours, and prints the result."),
    (CONTRAST, "It does its own calculations with Python's built-in maths, and prints the result."),
    ("python3 -c 'print(open(\"notes/brief.md\").read())'", "It reads files, and prints the result."),
])
def test_the_nights_readable_cases_stay_readable_and_promise_nothing(command, words):
    summary = plain_summary(command)
    assert summary == f"{words} {NOTHING_PROMISED}" and not _reassures(summary)


@pytest.mark.parametrize("command", [
    "python3 -c \"import io; io.FileIO('notes.md', 'w')\"",
    "python3 -c \"import numpy; numpy.memmap('a.bin', dtype='uint8', mode='w+', shape=(4,))\"",
    "python3 -c 'print(1)'",
])
def test_a_write_the_card_cannot_see_is_never_called_safe(command):
    summary = plain_summary(command)
    assert summary.endswith(NOTHING_PROMISED) and not _reassures(summary)


@pytest.mark.parametrize("command, warning", [
    ("python3 -c \"from PIL import Image; Image.open('a.png').save('b.png')\"", "It writes, moves or deletes files."),
    ("python3 -c \"open('notes.md', 'w')\"", "It writes, moves or deletes files."),
    ("python3 -c \"import os; f = os.remove; f('x')\"", "It writes, moves or deletes files."),
    ("python3 -c \"from os import remove as rm; rm('x')\"", "It writes, moves or deletes files."),
    ("python3 -c \"import subprocess; subprocess.run(['ls'])\"", "It runs other programs or code."),
    ("python3 -c \"import requests; print(requests.get('https://x').text)\"", "It can reach the internet."),
    ("python3 -c \"import pandas as pd; print(pd.read_csv('a.csv'))\"", "It uses `pandas`, so this card cannot tell"),
    ("python3 -c \"getattr(__import__('os'), 'remove')('x')\"", "this card cannot tell what it does"),
    ("python3 -c \"o = open; o('x', 'w')\"", "this card cannot tell what it does"),
    ("python3 -c \"import sys; sys.modules['os'].remove('x')\"", "this card cannot tell what it does"),
])
def test_what_it_sees_change_something_is_a_warning(command, warning):
    summary = plain_summary(command)
    assert warning in summary and NOTHING_PROMISED not in summary and not _reassures(summary)


# ── only one Python invocation the shell cannot alter is described ───────────

@pytest.mark.parametrize("command", [
    'python3 -c "print(1)"; rm -rf ~',
    "python3 -c 'print(1)' > out.txt",
    "python3 -c \"print('$(id)')\"",
    "python3 -c \"print('`id`')\"",
    "python3 -c \"print('${HOME}')\"",
    "python3 - <<EOF\nprint('$(id)')\nEOF",                              # unquoted: the shell expands the body
    "python3 - <<'EOF'\nprint(1)\nEOF\n&& curl x",
    "python3 - <<'EOF'\nprint(1)\nEOF && curl x",
    "python3 - <<'EOF' > out.txt\nprint(1)\nEOF",
    "python3 - <<'EOF'\nprint(1)\nEOF\nrm -rf ~\nEOF",                      # the heredoc ends at the first EOF
    "python3 -c 'print(1)' && curl x",
    "python3 -c 'print(1)' | sh",
    "python3 -c 'print(1)' &",
    "python3 -c 'print(1)'\nrm -rf ~",
    "python3 -c 'print(1)' extra",
    "X=1 python3 -c 'print(1)'",
    "$(which python3) -c 'print(1)'",
    "python3 script.py -c 'import os'",
    "node -e 'console.log(1)'",
    "python3 -c 'def ('",                                                # does not parse
    "pip --version",
])
def test_anything_but_one_unalterable_python_invocation_adds_nothing(command):
    assert plain_summary(command) is None


# ── one plain ``cd <path> && `` in front is allowed, and the card says where ──────

NIGHT_COLOUR_COUNT = f"cd sessions/2144 && {COLOUR_COUNT}"


def test_the_nights_colour_count_after_a_cd_is_described_with_where_it_runs():
    assert plain_summary(NIGHT_COLOUR_COUNT) == (
        "It opens an image with PIL and reads its colours, and prints the result. "
        f"It runs in `sessions/2144`. {NOTHING_PROMISED}")
    assert plain_summary(f"cd ../2144/previews && {CONTRAST}").startswith(
        "It does its own calculations with Python's built-in maths, and prints the result. "
        "It runs in `../2144/previews`.")


@pytest.mark.parametrize("prefix", [
    "cd sessions/2144 && cd previews && ",                 # two cds
    "cd sessions/2144 ; ",
    "cd sessions/2144 || ",
    "cd 'sessions/2144' && ",
    'cd "sessions/2144" && ',
    "cd - && ",
    "cd ~ && ",
    "cd ~/Development && ",
    "cd $X && ",
    "cd ${HOME} && ",
    "cd `pwd` && ",
    "cd sessions/* && ",
    "cd my folder && ",
    "cd -P sessions && ",
    "cd && ",
])
def test_any_other_cd_in_front_adds_nothing(prefix):
    assert plain_summary(prefix + COLOUR_COUNT) is None


def test_the_code_is_read_from_single_quotes_double_quotes_and_a_quoted_heredoc():
    assert inline_python("python3 -c 'print(1)'") == "print(1)"
    assert inline_python('python3 -u -c "print(1)"') == "print(1)"
    assert inline_python(CONTRAST).startswith("def lum(h):")
    assert inline_python('python3 <<"END"\nprint(2)\nEND') == "print(2)"


# ── nothing from the code can write card text ────────────────────────────────

@pytest.mark.parametrize("path", [
    "/tmp/**Safe to allow**", "/tmp/[Approve](https://evil.example)", "/tmp/<b>ok</b>", "/tmp/__x__",
    "/tmp/a`**Safe to allow**`b", "/tmp/a\n\n**Safe to allow**",
])
def test_a_path_from_the_code_is_shown_only_inside_one_code_span(path):
    summary = plain_summary(f"python3 - <<'EOF'\nprint(open({path!r}).read())\nEOF")
    shown = summary.split("It names ")[1].split(". ")[0]
    assert shown.startswith("`") and shown.endswith("`") and shown.count("`") == 2
    assert "\n" not in summary


def test_a_code_span_keeps_markdown_inert():
    assert code_span("a`b\nc") == "`a'b c`"
    assert code_span("**Safe to allow**") == "`**Safe to allow**`"


def test_a_command_with_no_inline_code_keeps_its_card():
    assert session_hold_question(612, {"subject": "pip --version"}, ticket="ticket #0042") == (
        "**Allow this command in ticket #0042?**\n\nThe agent wants to run **pip**.\n\n"
        "<details><summary>The exact command</summary>\n\n```sh\npip --version\n```\n\n</details>\n\n"
        "Answer `allow` or `deny`.")
