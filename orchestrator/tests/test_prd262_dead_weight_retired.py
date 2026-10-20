"""PRD-262 US-018 (D6): what nobody used is deleted, and stays deleted.

- Gotenberg: the compose service (profile ``all``), GOTENBERG_URL in compose,
  envs/api.defaults and config.py, the gotenberg-client package and its only client,
  ConversionService, which nothing called. test_prd262_unused_packages_go.py keeps
  the package out of the requirements, the code and both built images.
- The root Dockerfile.backend. Nothing built it; Railway builds orchestrator/Dockerfile.
- promtail-config.yml, which nothing loaded and which shipped logs to a hard-coded
  public IP, and envs/observability.defaults, which documented a
  ``--profile observability`` that compose never had. No compose file read it.
- frontend/tsconfig.tsbuildinfo, a TypeScript build artefact that .gitignore already
  lists.

This suite checks that:
- none of those files is tracked;
- nothing outside the history (PRDs, Ralph kits, graph snapshots) points at one;
- the runtime config and the self-host docs no longer describe Gotenberg;
- every ``--profile X`` or ``COMPOSE_PROFILES=X`` that the env defaults and the
  self-host docs tell a reader to use names a profile docker-compose.yml has.

Pure file reads and git: no Docker, no network.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"
THIS = Path(__file__).resolve().relative_to(ROOT).as_posix()

# Deleted path -> why it went.
RETIRED = {
    "Dockerfile.backend": "nothing built it; Railway builds orchestrator/Dockerfile",
    "promtail-config.yml": "nothing loaded it, and it shipped logs to a hard-coded public IP",
    "envs/observability.defaults": "it documented a compose profile that never existed, and nothing read it",
    "orchestrator/modules/documents/conversion_service.py": "Gotenberg's only client, and nothing called it",
    "frontend/tsconfig.tsbuildinfo": "a TypeScript build artefact (.gitignore: *.tsbuildinfo)",
}
# Names that would lead a reader to a retired file. tsconfig.tsbuildinfo is not one of
# them: .gitignore and .dockerignore files rightly name the artefact.
POINTERS = ("Dockerfile.backend", "promtail-config", "observability.defaults", "conversion_service")
# History may name them.
HISTORY = ("docs/PRDS", "scripts/ralph", "graphify-out")

# Where a running stack's config and the self-host docs live: none of it may describe
# Gotenberg. The workflows are left out, because they name gotenberg-client to prove
# it is gone.
GOTENBERG_FREE = (
    "docker-compose*.yml", "envs", ".env.example", "orchestrator",
    "QUICKSTART.md", "SETUP.md", "README.md", "CONTRIBUTING.md", "docs/getting-started",
    "docs/documentation-prd-corpus/runbooks-operational-docs.md", "docs/knowledge-base-rag",
    "charts", "infrastructure", ".github/dependabot.yml",
)
GOTENBERG_FREE_EXCEPT = ("orchestrator/tests",)  # guard tests name it to keep it out

# Where a reader is told which compose profile to start: the env defaults, the
# self-host docs, and the compose file's own comments.
PROFILE_DOCS = (
    "envs", ".env.example", "README.md", "QUICKSTART.md", "SETUP.md", "CONTRIBUTING.md",
    "docs/getting-started", "docs/deployment-infrastructure", "docker-compose.yml",
)
# Compose profile names are [a-zA-Z0-9][a-zA-Z0-9_.-]*. A placeholder such as
# `--profile <name>` does not match.
_PROFILE_FLAG = re.compile(r"--profile(?:=|[ \t]+)([A-Za-z0-9][\w.-]*)")
_PROFILES_VAR = re.compile(r"\bCOMPOSE_PROFILES=([\w.,-]+)")


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"git {args[0]} failed: {proc.stderr}"
    return proc.stdout


def _git_grep(*args: str) -> list[str]:
    """``git grep`` output lines; no match is an empty list, not a failure."""
    proc = subprocess.run(
        ["git", "grep", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    assert proc.returncode in (0, 1), f"git grep failed: {proc.stderr}"
    return proc.stdout.splitlines()


def _excluded(paths: tuple[str, ...]) -> list[str]:
    return [f":(exclude){path}" for path in paths]


# ── the files ───────────────────────────────────────────────────────────────


def test_the_retired_files_are_no_longer_tracked():
    tracked = set(_git("ls-files", "-z").split("\0"))
    assert "orchestrator/Dockerfile" in tracked, "git ls-files listed the wrong tree"
    still = {path: why for path, why in RETIRED.items() if path in tracked}
    assert not still, f"these were deleted by PRD-262 US-018 and are tracked again: {still}"


def test_no_build_artefact_of_tsc_is_tracked():
    assert _git("ls-files", "--", "*.tsbuildinfo").split() == []


def test_nothing_outside_the_history_points_at_a_retired_file():
    patterns = [arg for name in POINTERS for arg in ("-e", name)]
    hits = _git_grep("-n", "-F", *patterns, "--", ".", *_excluded((*HISTORY, THIS)))
    assert not hits, "these point at a file PRD-262 US-018 deleted:\n" + "\n".join(hits)


def test_the_runtime_config_and_the_self_host_docs_no_longer_describe_gotenberg():
    hits = _git_grep("-n", "-i", "gotenberg", "--", *GOTENBERG_FREE, *_excluded(GOTENBERG_FREE_EXCEPT))
    assert not hits, "Gotenberg was retired (PRD-262 US-018), but these still name it:\n" + "\n".join(hits)


# ── every documented profile exists ─────────────────────────────────────────


def profiles_named(text: str) -> list[tuple[int, str]]:
    """``(line, profile)`` for each ``--profile X`` and each name in ``COMPOSE_PROFILES=…``."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        found += [(number, match.group(1)) for match in _PROFILE_FLAG.finditer(line)]
        for match in _PROFILES_VAR.finditer(line):
            found += [(number, name) for name in match.group(1).split(",") if name]
    return found


def _compose_profiles() -> set[str]:
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    return {profile for body in services.values() for profile in ((body or {}).get("profiles") or [])}


def _documented_profiles() -> list[tuple[str, int, str]]:
    found = []
    for path in _git("ls-files", "-z", "--", *PROFILE_DOCS).split("\0"):
        if not path:
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # not a document
        found += [(path, line, profile) for line, profile in profiles_named(text)]
    return found


def test_the_profile_reader_on_fixtures():
    text = (
        "docker compose --profile observability up\n"
        "COMPOSE_PROFILES=media,search in .env, or `--profile=all`\n"
        "docker compose --profile <name> up, and --profiles is not a flag\n"
        "COMPOSE_PROFILES= (empty) and `COMPOSE_PROFILES=memory`.\n"
    )
    assert profiles_named(text) == [
        (1, "observability"), (2, "all"), (2, "media"), (2, "search"), (4, "memory"),
    ]


def test_every_profile_the_env_defaults_and_the_self_host_docs_name_exists_in_compose():
    compose = _compose_profiles()
    assert {"all", "media"} <= compose, f"docker-compose.yml was misread: {sorted(compose)}"
    documented = _documented_profiles()
    files = {path for path, _, _ in documented}
    assert {"envs/api.defaults", "docs/getting-started/self-hosting.md"} <= files, (
        f"the scan found profile mentions only in {sorted(files)}"
    )
    missing = [f"{path}:{line}: --profile {profile}" for path, line, profile in documented
               if profile not in compose]
    assert not missing, (
        f"docker-compose.yml has the profiles {sorted(compose)}; these tell a reader to start "
        "one it does not have:\n" + "\n".join(missing)
    )
