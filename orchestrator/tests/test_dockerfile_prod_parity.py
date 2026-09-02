"""Dockerfile ARG defaults are PRODUCTION values; compose opts the local edition out.

Railway builds the three Dockerfiles (railway.json: builder DOCKERFILE, target
production). On 2026-08-30 a `NEXT_PUBLIC_AUTH_EDITION=local` default shipped the
Clerk-less local edition to production (#650 -> #668), and the same slim pass
left the graph-extras and browser installs defaulted OFF. This guard makes the
rule mechanical: every build ARG with a default must be registered here with
its production value, the local edition must pass its own choices explicitly in
docker-compose.yml, and the deploy config must still say what we assume.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]

# Registered production defaults. Adding a defaulted ARG to a Dockerfile without
# a row here fails on purpose: decide the PRODUCTION value first.
PROD_DEFAULTS = {
    "frontend/Dockerfile": {
        "NEXT_PUBLIC_AUTH_EDITION": "saas",
        "NEXT_PUBLIC_CLERK_SIGN_IN_URL": "/sign-in",
        "NEXT_PUBLIC_CLERK_SIGN_UP_URL": "/sign-up",
        "NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL": "/dashboard",
        "NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL": "/dashboard",
        "FRONTEND_CACHE_BUST": None,  # any value: a cache key, not behaviour
    },
    "orchestrator/Dockerfile": {"INSTALL_GRAPH_EXTRAS": "true"},
    "services/workspace-worker/Dockerfile": {"INSTALL_BROWSER": "true"},
}

# What the LOCAL edition must pass explicitly (docker-compose.yml build args).
LOCAL_BUILD_ARGS = {
    "frontend": {"NEXT_PUBLIC_AUTH_EDITION": "local"},
    "backend": {"INSTALL_GRAPH_EXTRAS": "false"},
    "workspace-worker": {"INSTALL_BROWSER": "false"},
}

_ARG = re.compile(r"^ARG\s+([A-Z0-9_]+)(?:=(\S*))?\s*$", re.M)


def _arg_defaults(dockerfile: str) -> dict[str, str]:
    text = (_ROOT / dockerfile).read_text()
    # `ARG NAME` (no default) matches with an empty group — that is NOT a default.
    return {name: default for name, default in _ARG.findall(text) if default != ""}


def test_every_defaulted_arg_is_registered_with_its_production_value():
    for dockerfile, registered in PROD_DEFAULTS.items():
        found = _arg_defaults(dockerfile)
        unregistered = sorted(set(found) - set(registered))
        assert not unregistered, (
            f"{dockerfile}: ARG(s) {unregistered} have a default that is not registered in "
            f"PROD_DEFAULTS — a Dockerfile default is a PRODUCTION decision (Railway builds this file)."
        )
        for name, expected in registered.items():
            if expected is None or name not in found:
                continue
            assert found[name] == expected, (
                f"{dockerfile}: ARG {name} defaults to {found[name]!r}; production requires {expected!r}. "
                f"The local edition opts out in docker-compose.yml, never here."
            )


def test_the_local_edition_passes_its_choices_explicitly():
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
    for service, args in LOCAL_BUILD_ARGS.items():
        build = compose["services"][service].get("build") or {}
        got = {k: str(v) for k, v in (build.get("args") or {}).items()}
        for name, expected in args.items():
            assert name in got, f"docker-compose.yml: {service}.build.args must set {name} explicitly"
            value = got[name]
            # `${VAR:-local}` style is fine as long as the fallback is the local value.
            assert expected in value, f"docker-compose.yml: {service}.build.args.{name}={value!r}, expected the local value {expected!r}"


def test_deploy_config_builds_the_dockerfiles():
    railway = json.loads((_ROOT / "railway.json").read_text())
    assert railway["build"]["builder"] == "DOCKERFILE"
    assert railway["build"]["dockerfileTarget"] == "production"
    manifest = json.loads((_ROOT / "infrastructure" / "railway-manifest.json").read_text())
    builders = set()

    def walk(o):
        if isinstance(o, dict):
            if "builder" in o:
                builders.add(o["builder"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(manifest)
    assert builders <= {"DOCKERFILE"}, (
        f"infrastructure/railway-manifest.json declares builders {sorted(builders)} — the stale RAILPACK "
        "claim caused the 2026-08-30 outage; railway.json (DOCKERFILE) is the truth"
    )
