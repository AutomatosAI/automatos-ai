"""Container entrypoints keep LF line endings on every checkout (issue #818).

Git for Windows defaults to core.autocrlf=true. Without .gitattributes, a clone
wrote the shell scripts with CRLF, the shebang became "#!/bin/bash\\r", and the
backend and workspace-worker crash-looped with "exec ... no such file or directory".
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    REPO / "orchestrator" / "docker-entrypoint.sh",
    REPO / "services" / "workspace-worker" / "entrypoint.sh",
)


def _pinned_patterns() -> dict:
    rules = {}
    for line in (REPO / ".gitattributes").read_text().splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            rules[parts[0]] = parts[1:]
    return rules


def test_gitattributes_pins_lf_for_what_runs_in_the_containers():
    rules = _pinned_patterns()
    for pattern in ("*.sh", "*.py", "Dockerfile*", "Makefile"):
        assert "eol=lf" in rules.get(pattern, []), f"{pattern} must be pinned to eol=lf"


def test_gitattributes_does_not_renormalise_everything():
    assert "*" not in _pinned_patterns(), "a blanket rule would rewrite files git misdetects as text"


def test_entrypoints_are_checked_out_with_lf():
    for script in ENTRYPOINTS:
        assert b"\r" not in script.read_bytes(), f"{script.name} has CRLF line endings"
