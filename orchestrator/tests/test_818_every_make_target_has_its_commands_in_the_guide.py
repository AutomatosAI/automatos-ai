"""Issue #818: a Windows user without `make` can still run every target.

`make` is not installed on Windows, and the docs gave a plain command only for
`make up`. Section 1b of the self-hosting guide now lists the commands behind
every Makefile target. This test reads the Makefile's recipes and checks that
the section names each target and carries its docker commands and session-host
flags, so a Makefile change cannot leave the guide behind.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAKEFILE = REPO / "Makefile"
GUIDE = REPO / "docs" / "getting-started" / "self-hosting.md"
SECTION_START = "## 1b. "
COMPOSE_VARS = ("COMPOSE", "DEV_COMPOSE", "IMAGES_COMPOSE")
TARGET_HEADER = re.compile(r"^([a-z][a-z-]*):\s*$")
DOCKER_COMMAND = re.compile(r"\bdocker (?:image|builder|system) [a-z]+(?: -f)?")
HOST_FLAG = re.compile(r"(--[a-z][a-z-]*)")
# Formatting that `make status` adds and the guide leaves out.
OUTPUT_ONLY_FLAG = " --format"


def _section() -> str:
    text = GUIDE.read_text(encoding="utf-8")
    start = text.index(SECTION_START)
    end = text.find("\n## ", start + len(SECTION_START))
    return text[start:] if end == -1 else text[start:end]


def _compose_variables(makefile: str) -> dict:
    found = {}
    for name in COMPOSE_VARS:
        match = re.search(rf"^{name}\s*\?=\s*(.+)$", makefile, re.MULTILINE)
        assert match, f"Makefile no longer defines {name}"
        found[name] = match.group(1).strip()
    return found


def _recipes(makefile: str) -> dict:
    recipes, current = {}, None
    for line in makefile.splitlines():
        header = TARGET_HEADER.match(line)
        if header:
            current = header.group(1)
            recipes[current] = []
        elif current and line.startswith("\t"):
            recipes[current] = [*recipes[current], line.strip().lstrip("-@")]
        elif line.strip():
            current = None
    return recipes


def _expected_commands(recipe: list, variables: dict) -> list:
    expected = []
    for line in recipe:
        for name, value in variables.items():
            if line.startswith(f"$({name}) "):
                expected.append(f"{value} {line[len(name) + 3:].strip()}".split(OUTPUT_ONLY_FLAG)[0])
        expected.extend(DOCKER_COMMAND.findall(line))
        if "automatos_cli_host" in line:
            expected.extend(HOST_FLAG.findall(line.split("automatos_cli_host", 1)[1]))
    return expected


def test_the_guide_names_every_make_target():
    targets = _recipes(MAKEFILE.read_text(encoding="utf-8"))
    assert {"up", "down", "cli-host-install"} <= set(targets), "Makefile parse drift"
    section = _section()
    missing = [target for target in targets if f"make {target}" not in section]
    assert not missing, f"self-hosting.md §1b does not cover {missing}"


def test_the_guide_has_the_commands_behind_each_target():
    makefile = MAKEFILE.read_text(encoding="utf-8")
    variables = _compose_variables(makefile)
    section = _section()
    missing = {
        target: [command for command in _expected_commands(recipe, variables) if command not in section]
        for target, recipe in _recipes(makefile).items()
    }
    missing = {target: commands for target, commands in missing.items() if commands}
    assert not missing, f"self-hosting.md §1b is missing the commands behind: {missing}"
