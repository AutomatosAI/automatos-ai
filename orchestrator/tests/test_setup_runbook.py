"""SETUP.md, the install runbook people and their AI agents follow, stays true.

It is the one install path written for a non-expert on any OS (including Windows
through WSL2), and AGENTS.md sends agents there when they are asked to install
Automatos rather than change it. These checks keep it in step with compose.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "SETUP.md"


def _required_compose_variables() -> set:
    compose = (REPO / "docker-compose.yml").read_text()
    return set(re.findall(r"\$\{([A-Z_]+):\?", compose))


def test_setup_generates_every_secret_compose_requires():
    runbook = SETUP.read_text()
    generated = re.search(r"for key in ([A-Z_ ]+); do", runbook)
    assert generated, "SETUP.md must generate the required secrets in a loop"
    assert set(generated.group(1).split()) == _required_compose_variables()


def test_setup_covers_every_platform_and_the_agent_rules():
    runbook = SETUP.read_text()
    for heading in ("## For agents", "## Path W", "## Path A", "## Troubleshooting"):
        assert heading in runbook, f"SETUP.md lost its '{heading}' section"
    assert "wsl --install" in runbook
    assert "Never ask the person to paste an API key" in runbook


def test_entry_points_link_to_setup():
    for doc in ("AGENTS.md", "README.md", "QUICKSTART.md"):
        assert "(SETUP.md)" in (REPO / doc).read_text(), f"{doc} must link SETUP.md"
