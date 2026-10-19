#!/usr/bin/env python3
"""The app images `make up` and `make build` run (PRD-262 US-036).

The default stack pulls the images CI builds, scans and publishes to GHCR instead of
building them. ``make up`` runs ``stack_images.py up``, which picks ONE tag for the
three app images (backend = ``automatos-api``, frontend, workspace-worker), says which
and why, gets each image (a pull, a pull and a re-tag, or a build) and writes the tag
to ``.env`` as ``AUTOMATOS_IMAGE_TAG=…``, so later plain ``docker compose`` commands in
this checkout run the same images. The Makefile then starts the stack with
``docker compose up -d --no-build``, so compose never builds under an official tag.

The rule (docs/getting-started/self-hosting.md §1b says it in words):

1. ``AUTOMATOS_IMAGE_TAG`` from the shell or the make command line is used as given.
   An official tag that is neither on this machine nor on GHCR stops here; a
   ``local-*`` tag that is not on this machine is built. ``edge`` is pulled on every
   call. The line make wrote to ``.env`` earlier is ignored: a pin goes in the shell.
2. On a clean checkout (``git status --porcelain`` prints nothing for orchestrator,
   frontend and services/workspace-worker): ``edge`` when every app image's
   ``org.opencontainers.image.revision`` label, read from the registry without
   pulling, is HEAD (a weekly OS rebuild of this very commit, US-011), and it is
   pulled, because compose never re-pulls a tag it has. Otherwise ``sha-<7>`` of HEAD
   when it exists. A ``sha-*`` or ``tree-*`` tag already on this machine counts as
   found, so a machine that has its images starts offline.
3. Otherwise ``local-<sha7>`` (``local-<sha7>-dirty`` when a context has changes).
   Each image whose context is clean and whose ``tree-<12>`` tag exists (the first
   12 characters of ``git rev-parse HEAD:<context>``, which images.yml publishes once
   per tree) is pulled and re-tagged; it keeps the revision label of the commit that
   built it. The others are built here, and make up names them and says why.

The published web app inlines its ``NEXT_PUBLIC_*`` build args (images.yml: the local
edition against localhost:8000 and localhost:9000). When ``.env`` sets one to another
value, the frontend is built here under a ``local-*`` tag (the others re-tagged from
the official tag found) and make up says why. A non-default ``API_PORT`` or
``MINIO_PORT`` whose ``NEXT_PUBLIC_*`` values still point at the default port stops
here as a misconfiguration: the web app would call a port nothing listens on.

The core stack (``make up CORE=1``, PRD-262 US-040) runs five containers without the
workspace worker: ``COMPOSE_FILE`` names ``docker-compose.core.yml``, which puts the
worker behind its ``workspace`` profile. make up then picks the tag by the same rule but
pulls, re-tags and builds only the backend's and the frontend's images.

``build`` (make build) only picks and records ``local-<sha7>`` (or a pinned
``local-*`` tag); the Makefile runs the build. ``tree-tag`` and ``published`` are what
images.yml's merge job runs to push ``tree-<12>`` once per tree.

Python 3.9 and the standard library only: macOS's make comes with the Xcode command
line tools, which bring Python 3.9.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Protocol, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "ghcr.io/automatosai/automatos-"
EDGE = "edge"
LOCAL_PREFIX, SHA_PREFIX, TREE_PREFIX = "local-", "sha-", "tree-"
DIRTY_SUFFIX = "-dirty"
NO_GIT_TAG = "local-unversioned"  # make build outside a git checkout
SHORT_SHA = 7  # docker/metadata-action's type=sha, which images.yml publishes as sha-<7>
TREE_CHARS = 12
TAG_KEY = "AUTOMATOS_IMAGE_TAG"
REVISION_LABEL = "org.opencontainers.image.revision"
LABEL_ARCH = "amd64"  # every arch of one published index is built from the same commit
NOT_FOUND = 404
GIT_TIMEOUT_SECONDS = 60
STOP_EXIT = 2
IMAGE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
ENV_FILE = ".env"
TAG_COMMENT = ("# Written by make up / make build (PRD-262 US-036): the app images' tag that plain "
               "docker compose runs here. Rewritten on every run; pin a tag in the shell instead.")


class AppImage(NamedTuple):
    """A compose service, its published image (automatos-<name>) and its build context."""

    service: str
    name: str
    context: str


APP_IMAGES = (
    AppImage("backend", "api", "orchestrator"),
    AppImage("frontend", "frontend", "frontend"),
    AppImage("workspace-worker", "workspace-worker", "services/workspace-worker"),
)
SERVICES = tuple(app.service for app in APP_IMAGES)
FRONTEND = "frontend"
WORKER = "workspace-worker"
# The core stack's override (PRD-262 US-040), the profile it puts the worker behind, and
# compose's name for every profile.
CORE_OVERRIDE = "docker-compose.core.yml"
WORKER_PROFILE = "workspace"
ALL_PROFILES = "*"

# images.yml's frontend build args: the values the published web app was built with.
# An unset or empty value in .env is compose's default, which is the same.
PUBLISHED_FRONTEND = {
    "NEXT_PUBLIC_API_URL": "http://localhost:8000",
    "NEXT_PUBLIC_WS_URL": "ws://localhost:8000/ws",
    "NEXT_PUBLIC_MEDIA_ORIGINS": "http://localhost:9000",
    "NEXT_PUBLIC_AUTH_EDITION": "local",
    "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY": "",
}
# A published port, its default, and the NEXT_PUBLIC_* values that must follow it.
PORTS = {
    "API_PORT": ("8000", ("NEXT_PUBLIC_API_URL", "NEXT_PUBLIC_WS_URL")),
    "MINIO_PORT": ("9000", ("NEXT_PUBLIC_MEDIA_ORIGINS",)),
}
PORT_VALUES = {
    "NEXT_PUBLIC_API_URL": "http://localhost:{port}",
    "NEXT_PUBLIC_WS_URL": "ws://localhost:{port}/ws",
    "NEXT_PUBLIC_MEDIA_ORIGINS": "http://localhost:{port}",
}
SETTING_KEYS = tuple(PUBLISHED_FRONTEND) + tuple(PORTS)


class Stop(Exception):
    """make cannot go on; the message says why and what to do."""


@dataclass(frozen=True)
class Checkout:
    """HEAD, the app contexts with uncommitted changes, and each context's tree id."""

    head: Optional[str]
    dirty: FrozenSet[str]
    trees: Mapping[str, str]


@dataclass(frozen=True)
class Plan:
    """The tag the stack runs, why, and how each app image gets there."""

    tag: str
    reason: str
    pull: bool = False  # `docker compose pull` the three: edge, which compose never re-pulls
    retag: Tuple[Tuple[str, str], ...] = ()  # (service, the published ref it is re-tagged from)
    build: Tuple[Tuple[str, str], ...] = ()  # (service, why it is built here)
    services: Tuple[str, ...] = SERVICES  # the app services the stack runs; the rest are left alone


class Lookup(Protocol):
    """Where an image is: this machine's Docker, and its registry (read, never pulled)."""

    def local(self, ref: str) -> bool: ...

    def published(self, ref: str) -> Optional[bool]: ...

    def revision(self, ref: str) -> Optional[str]: ...


def image_ref(app: AppImage, tag: str) -> str:
    """``ghcr.io/automatosai/automatos-<name>:<tag>``, the name docker-compose.yml gives it."""
    return f"{REPOSITORY}{app.name}:{tag}"


def app_for(service: str) -> AppImage:
    return next(app for app in APP_IMAGES if app.service == service)


def found(lookup: Lookup, ref: str) -> bool:
    """On this machine, or on its registry."""
    return lookup.local(ref) or lookup.published(ref) is True


def _missing(lookup: Lookup, tag: str) -> List[str]:
    """The app images ``tag`` lacks, each with what the registry said."""
    missing = []
    for app in APP_IMAGES:
        ref = image_ref(app, tag)
        if lookup.local(ref):
            continue
        state = lookup.published(ref)
        if state is not True:
            where = "ghcr.io does not have it" if state is False else "ghcr.io could not be reached"
            missing.append(f"{ref} ({where})")
    return missing


# --------------------------------------------------------------------------- the rule


def local_tag(checkout: Checkout) -> str:
    """``local-<sha7>``, ``local-<sha7>-dirty`` with uncommitted changes in a context."""
    if checkout.head is None:
        return NO_GIT_TAG
    tag = LOCAL_PREFIX + checkout.head[:SHORT_SHA]
    return tag + DIRTY_SUFFIX if checkout.dirty else tag


def plan_pinned(tag: str, frontend: Sequence[str], lookup: Lookup, reason: str = "") -> Plan:
    """Rule 1: a tag from the shell or the make command line, used as given."""
    reason = reason or f"{TAG_KEY}={tag} is set in the shell or on the make command line"
    if tag.startswith(LOCAL_PREFIX):
        build = tuple((app.service, f"{image_ref(app, tag)} is not on this machine")
                      for app in APP_IMAGES if not lookup.local(image_ref(app, tag)))
        return Plan(tag, reason, build=build)
    if frontend:
        raise Stop(f"the web app has to be built here, because .env sets {'; '.join(frontend)}, and make never "
                   f"builds under an official tag such as {tag}. Unset {TAG_KEY} to let make up choose, or "
                   f"pin a local one: {TAG_KEY}=local-mine make up")
    missing = _missing(lookup, tag)
    if missing:
        raise Stop(f"{TAG_KEY}={tag}, but {'; '.join(missing)}. Pick a published tag (edge, or sha-<7> of a "
                   f"main commit CI published), or unset {TAG_KEY} to let make up choose")
    return Plan(tag, reason, pull=tag == EDGE)


def plan_official(head: str, lookup: Lookup) -> Optional[Plan]:
    """Rule 2, on a clean checkout: edge when it is a build of HEAD, else HEAD's sha-<7>."""
    if all(lookup.revision(image_ref(app, EDGE)) == head for app in APP_IMAGES):
        return Plan(EDGE, "edge is a build of this very commit (its revision label is HEAD), such as the "
                          "weekly rebuild with fresh OS packages; pulled, since compose never re-pulls a tag "
                          "it already has", pull=True)
    sha = SHA_PREFIX + head[:SHORT_SHA]
    if all(found(lookup, image_ref(app, sha)) for app in APP_IMAGES):
        return Plan(sha, f"this checkout is clean and CI published {sha} for its commit")
    return None


def _local_source(app: AppImage, checkout: Checkout, frontend: Sequence[str], official: Optional[Plan],
                  lookup: Lookup) -> Tuple[Optional[str], str]:
    """The published image ``app`` is re-tagged from, or None and why it is built here."""
    if app.service == FRONTEND and frontend:
        return None, "its NEXT_PUBLIC_* build args are not the published web app's: " + "; ".join(frontend)
    if app.context in checkout.dirty:
        return None, f"{app.context} has uncommitted changes"
    if official is not None:
        return image_ref(app, official.tag), ""
    tree = checkout.trees.get(app.context, "")
    if not OBJECT_ID.match(tree):
        return None, f"the tree of {app.context} could not be read from git"
    ref = image_ref(app, TREE_PREFIX + tree[:TREE_CHARS])
    if found(lookup, ref):
        return ref, ""
    return None, f"CI has published no {TREE_PREFIX}{tree[:TREE_CHARS]} for {app.context}'s tree"


def _local_reason(checkout: Checkout, official: Optional[Plan]) -> str:
    if checkout.dirty:
        changed = ", ".join(sorted(checkout.dirty))
        return f"{changed} {'has' if len(checkout.dirty) == 1 else 'have'} uncommitted changes"
    if official is not None:
        return f"the web app is built here, so the stack runs a local tag; the rest is {official.tag}, re-tagged"
    sha = SHA_PREFIX + (checkout.head or "")[:SHORT_SHA]
    return (f"CI has not published {sha} (a [skip ci] merge, or the minutes before images.yml publishes): "
            f"each image whose tree is published is re-tagged, the rest built here")


def plan_local(checkout: Checkout, frontend: Sequence[str], official: Optional[Plan], lookup: Lookup) -> Plan:
    """Rule 3: local-<sha7>[-dirty]; published images re-tagged, the rest built here."""
    retag, build = [], []
    for app in APP_IMAGES:
        source, why = _local_source(app, checkout, frontend, official, lookup)
        if source:
            retag.append((app.service, source))
        else:
            build.append((app.service, why))
    return Plan(local_tag(checkout), _local_reason(checkout, official), retag=tuple(retag), build=tuple(build))


def plan_up(pinned: str, checkout: Checkout, frontend: Sequence[str], lookup: Lookup) -> Plan:
    """The tag make up runs, why, and how it gets each image (the module docstring's rule)."""
    if pinned:
        return plan_pinned(pinned, frontend, lookup)
    if checkout.head is None:
        return plan_pinned(EDGE, frontend, lookup, reason=(
            "this is not a git checkout, so no build can be matched to its code: edge, the newest main "
            "build that booted in CI"))
    official = None if checkout.dirty else plan_official(checkout.head, lookup)
    if official is not None and not frontend:
        return official
    return plan_local(checkout, frontend, official, lookup)


def plan_build(pinned: str, checkout: Checkout) -> str:
    """make build: a pinned local-* tag, or local-<sha7>[-dirty]; never an official tag."""
    if pinned and not pinned.startswith(LOCAL_PREFIX):
        raise Stop(f"make build never builds under an official tag ({TAG_KEY}={pinned}): pin a local one "
                   f"({TAG_KEY}=local-mine make build) or unset it")
    return pinned or local_tag(checkout)


# --------------------------------------------------------------------------- settings


def _value(raw: str) -> str:
    """A .env value: quotes removed; an unquoted value ends at ' #'."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    return raw.split(" #", 1)[0].rstrip()


def read_dotenv(text: str) -> Dict[str, str]:
    """``KEY=value`` pairs of a .env file, as compose reads them; the last one wins."""
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        key = key.strip()
        key = key[len("export "):].strip() if key.startswith("export ") else key
        values[key] = _value(raw)
    return values


def compose_setting(key: str, dotenv: Mapping[str, str], environ: Mapping[str, str]) -> str:
    """A variable as compose reads it: the shell's value wins over .env's."""
    return environ[key] if key in environ else dotenv.get(key, "")


def settings(dotenv: Mapping[str, str], environ: Mapping[str, str]) -> Dict[str, str]:
    """The frontend build args and ports compose will use: the shell's value wins over .env's."""
    return {key: compose_setting(key, dotenv, environ) for key in SETTING_KEYS}


def stack_services(dotenv: Mapping[str, str], environ: Mapping[str, str]) -> Tuple[str, ...]:
    """The app services compose runs here (PRD-262 US-040).

    All three, unless COMPOSE_FILE names docker-compose.core.yml (make up CORE=1 sets it):
    that override puts the worker behind the ``workspace`` profile, which COMPOSE_PROFILES
    can still turn on. COMPOSE_FILE is split as compose splits it.
    """
    separator = compose_setting("COMPOSE_PATH_SEPARATOR", dotenv, environ) or os.pathsep
    files = {Path(part.strip()).name for part in compose_setting("COMPOSE_FILE", dotenv, environ).split(separator)}
    profiles = {part.strip() for part in compose_setting("COMPOSE_PROFILES", dotenv, environ).split(",")}
    if CORE_OVERRIDE not in files or profiles & {WORKER_PROFILE, ALL_PROFILES}:
        return SERVICES
    return tuple(service for service in SERVICES if service != WORKER)


def only(plan: Plan, services: Sequence[str]) -> Plan:
    """``plan`` for the app services the stack runs: the others' images are neither pulled, re-tagged nor built."""
    keep = tuple(service for service in SERVICES if service in services)
    return replace(plan, services=keep, retag=tuple(step for step in plan.retag if step[0] in keep),
                   build=tuple(step for step in plan.build if step[0] in keep))


def frontend_reasons(values: Mapping[str, str]) -> List[str]:
    """The NEXT_PUBLIC_* values that differ from what the published web app inlines."""
    return [f"{key}={values[key]} (the published web app has {published or 'none'})"
            for key, published in PUBLISHED_FRONTEND.items()
            if values.get(key) and values[key] != published]


def misconfigurations(values: Mapping[str, str]) -> List[str]:
    """A moved API or MinIO port whose NEXT_PUBLIC_* values still point at the default one."""
    problems = []
    for port_key, (default, keys) in PORTS.items():
        port = values.get(port_key) or default
        if port == default:
            continue
        stale = [key for key in keys if (values.get(key) or PUBLISHED_FRONTEND[key]) == PUBLISHED_FRONTEND[key]]
        if stale:
            wanted = " and ".join(f"{key}={PORT_VALUES[key].format(port=port)}" for key in stale)
            problems.append(f"{port_key}={port}, but the web app would still call port {default}: set {wanted} "
                            f"in .env")
    return problems


def pinned_tag(environ: Mapping[str, str]) -> str:
    """AUTOMATOS_IMAGE_TAG from the shell or the make command line; empty counts as unset."""
    tag = environ.get(TAG_KEY, "").strip()
    if tag and not IMAGE_TAG.match(tag):
        raise Stop(f"{TAG_KEY}={tag!r} is not an image tag")
    return tag


def with_tag(text: str, tag: str) -> str:
    """``text`` (a .env file) with make's tag line, and only that, replaced at its end."""
    kept = [line for line in text.splitlines()
            if line.strip() != TAG_COMMENT and not re.match(rf"^\s*(export\s+)?{TAG_KEY}=", line)]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept + ([""] if kept else []) + [TAG_COMMENT, f"{TAG_KEY}={tag}"]) + "\n"


def write_tag(path: Path, tag: str) -> None:
    """Replace make's AUTOMATOS_IMAGE_TAG line in ``path``, keeping every other line and the mode."""
    target = path.resolve() if path.exists() else path
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(with_tag(text, tag))
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except OSError:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


# --------------------------------------------------------------------------- git, docker and the registry


def git(root: Path, *args: str) -> Optional[str]:
    """``git -C root <args>``'s output, or None when git is missing or the command fails."""
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                timeout=GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_checkout(root: Path, run_git=git) -> Checkout:
    """HEAD, the dirty contexts (an unreadable status counts as dirty) and the context trees."""
    head = run_git(root, "rev-parse", "--verify", "HEAD")
    if not head or not OBJECT_ID.match(head):
        return Checkout(None, frozenset(), {})
    dirty = frozenset(app.context for app in APP_IMAGES
                      if run_git(root, "status", "--porcelain", "--", app.context) != "")
    trees = {app.context: run_git(root, "rev-parse", f"HEAD:{app.context}") or "" for app in APP_IMAGES}
    return Checkout(head, dirty, trees)


NO_DOCKER = "the docker command is not installed or not on PATH: install Docker (self-hosting.md §1)"


class Docker:
    """This machine's Docker CLI."""

    def local(self, ref: str) -> bool:
        command = ["docker", "image", "inspect", "--format", "{{.Id}}", ref]
        try:
            return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except OSError as error:
            raise Stop(NO_DOCKER) from error

    def run(self, *args: str, env: Optional[Mapping[str, str]] = None) -> int:
        try:
            return subprocess.call(["docker", *args], env=dict(env) if env is not None else None)
        except OSError as error:
            raise Stop(NO_DOCKER) from error

    def check(self, *args: str, env: Optional[Mapping[str, str]] = None) -> None:
        code = self.run(*args, env=env)
        if code != 0:
            raise Stop(f"`docker {' '.join(args)}` failed (exit {code})")


def _load_registry_client():
    """scripts/ci/image_registry.py, the standard-library GHCR reader CI uses, loaded by path."""
    name = "stack_images_registry"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / "image_registry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the CI scripts do
    spec.loader.exec_module(module)
    return module


def _unreadable(ref: str, error: Exception) -> None:
    print(f"warning: could not read {ref} from its registry: {error}", file=sys.stderr)


class Registry:
    """Tags and labels on the registry, read without pulling (anonymous for public images)."""

    def __init__(self, client=None):
        self.client = client or _load_registry_client()

    def published(self, ref: str) -> Optional[bool]:
        """True when the tag exists, False when the registry says it does not, None unread."""
        try:
            self.client.resolve(ref)
        except urllib.error.HTTPError as error:
            if error.code == NOT_FOUND:
                return False
            return _unreadable(ref, error)
        except self.client.READ_ERRORS as error:
            return _unreadable(ref, error)
        return True

    def revision(self, ref: str) -> Optional[str]:
        """The image's org.opencontainers.image.revision label, or None."""
        try:
            _, config, _ = self.client.image(ref, LABEL_ARCH)
        except urllib.error.HTTPError as error:
            if error.code == NOT_FOUND:
                return None
            return _unreadable(ref, error)
        except self.client.READ_ERRORS as error:
            return _unreadable(ref, error)
        return ((config.get("config") or {}).get("Labels") or {}).get(REVISION_LABEL)


class Images:
    """A Lookup over this machine's Docker and the registry."""

    def __init__(self, docker: Docker, registry: Registry):
        self.docker, self.registry = docker, registry

    def local(self, ref: str) -> bool:
        return self.docker.local(ref)

    def published(self, ref: str) -> Optional[bool]:
        return self.registry.published(ref)

    def revision(self, ref: str) -> Optional[str]:
        return self.registry.revision(ref)


# --------------------------------------------------------------------------- doing it


def describe(plan: Plan) -> List[str]:
    """What make up prints: the tag and why, then what it pulls, re-tags and builds."""
    lines = [f"make up: {TAG_KEY}={plan.tag} ({plan.reason})"]
    if plan.pull:
        lines.append(f"  pulling {', '.join(plan.services)} at {plan.tag}")
    lines += [f"  {service}: {source}, re-tagged {plan.tag}" for service, source in plan.retag]
    lines += [f"  {service}: built here, because {why}" for service, why in plan.build]
    lines += [f"  {service}: not run ({CORE_OVERRIDE}), so its image is left alone"
              for service in SERVICES if service not in plan.services]
    if not plan.build:
        lines.append("  nothing is built")
    return lines


def _pull_all(plan: Plan, docker: Docker, env: Mapping[str, str]) -> None:
    """``docker compose pull`` the stack's app images; a failed pull is fine only when all are here."""
    if docker.run("compose", "pull", *plan.services, env=env) == 0:
        return
    if all(docker.local(image_ref(app_for(service), plan.tag)) for service in plan.services):
        print(f"warning: could not pull {plan.tag}; starting the {plan.tag} images this machine already has",
              file=sys.stderr)
        return
    raise Stop(f"docker compose pull failed, and this machine does not have the {plan.tag} images")


def execute(plan: Plan, docker: Docker, env: Mapping[str, str]) -> None:
    """Pull, re-tag and build what ``plan`` says; compose gets the tag through ``env``."""
    if plan.pull:
        _pull_all(plan, docker, env)
    for service, source in plan.retag:
        if source.endswith(f":{EDGE}") or not docker.local(source):
            docker.check("pull", source)
        docker.check("tag", source, image_ref(app_for(service), plan.tag))
    if plan.build:
        docker.check("compose", "build", *[service for service, _ in plan.build], env=env)


def _dotenv(root: Path) -> Dict[str, str]:
    """The checkout's .env, as compose reads it; none is empty."""
    env_file = root / ENV_FILE
    return read_dotenv(env_file.read_text(encoding="utf-8") if env_file.exists() else "")


def _settings(dotenv: Mapping[str, str]) -> Dict[str, str]:
    """The checkout's frontend settings; a misconfigured port stops here."""
    values = settings(dotenv, os.environ)
    problems = misconfigurations(values)
    if problems:
        raise Stop("; ".join(problems))
    return values


def cmd_up(args: argparse.Namespace) -> int:
    """make up: pick the tag, get the images, write the tag to .env."""
    root = args.root
    dotenv = _dotenv(root)
    values = _settings(dotenv)
    pinned = pinned_tag(os.environ)
    print("make up: picking the app images' tag (this checkout, this machine and ghcr.io)...")
    docker = Docker()
    plan = plan_up(pinned, read_checkout(root), frontend_reasons(values), Images(docker, Registry()))
    plan = only(plan, stack_services(dotenv, os.environ))
    for line in describe(plan):
        print(line)
    execute(plan, docker, {**os.environ, TAG_KEY: plan.tag})
    write_tag(root / ENV_FILE, plan.tag)
    print(f"make up: {TAG_KEY}={plan.tag} written to .env, so plain docker compose runs these images too")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """make build: pick the local-* tag and write it to .env; the Makefile builds."""
    _settings(_dotenv(args.root))
    tag = plan_build(pinned_tag(os.environ), read_checkout(args.root))
    write_tag(args.root / ENV_FILE, tag)
    print(f"make build: building backend, frontend and workspace-worker from this checkout as {TAG_KEY}={tag} "
          f"(written to .env)")
    return 0


def tree_tag(root: Path, context: str, run_git=git) -> str:
    """``tree-<12>`` of ``context`` at HEAD: the tag images.yml publishes once per tree."""
    tree = run_git(root, "rev-parse", f"HEAD:{context}") or ""
    if not OBJECT_ID.match(tree):
        raise Stop(f"git has no tree for {context} at HEAD")
    return TREE_PREFIX + tree[:TREE_CHARS]


def cmd_tree_tag(args: argparse.Namespace) -> int:
    print(tree_tag(args.root, args.context))
    return 0


def cmd_published(args: argparse.Namespace) -> int:
    """Exit 0 when ``ref`` is on its registry, 1 when it is not, 2 when the registry was not read."""
    state = Registry().published(args.ref)
    print(f"{args.ref}: {({True: 'published', False: 'not published'}).get(state, 'unknown')}")
    return {True: 0, False: 1}.get(state, STOP_EXIT)


COMMANDS = {"up": cmd_up, "build": cmd_build, "tree-tag": cmd_tree_tag, "published": cmd_published}
MAKE_TARGETS = ("up", "build")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root.add_argument("--root", type=Path, default=ROOT, help="the checkout (default: this script's)")
    commands = root.add_subparsers(dest="name", required=True)
    commands.add_parser("up", help="make up: pick the tag, get the images, write the tag to .env")
    commands.add_parser("build", help="make build: pick the local-* tag and write it to .env")
    tree = commands.add_parser("tree-tag", help="print tree-<12> of a build context at HEAD")
    tree.add_argument("--context", required=True, help="orchestrator, frontend or services/workspace-worker")
    published = commands.add_parser("published", help="exit 0 if the image ref is on its registry, 1 if not")
    published.add_argument("ref")
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.name](args)
    except Stop as stop:
        who = f"make {args.name}" if args.name in MAKE_TARGETS else f"stack_images.py {args.name}"
        print(f"{who}: {stop}", file=sys.stderr)
        return STOP_EXIT


if __name__ == "__main__":
    sys.exit(main())
