"""PRD-262: the worker image's code cuts from the 19 Oct size breakdown, pinned.

Each cut keeps behaviour: agents keep gcc, g++, make, node, npm, pnpm, uv, git and
Perl. The Dockerfile edits are checked here against the in-image proofs that images.yml
runs on the image it built, on both arches:

- row 16: gcc's lto-dump is deleted in the apt layer that installs build-essential;
- row 19: the API's git trim (US-019) is applied to the worker: the same server-side
  and mail binaries go, and git-core's git becomes a link to /usr/bin/git;
- row 21: Node's binary comes from a `node-stripped` stage that media-render carries
  byte for byte, copied with --link as the production stage's first instruction, so
  the two images' layers are one blob on the same base;
- row 47, gosu: gone from the apt list; entrypoint.sh drops to the worker with setpriv
  (test_prd262_worker_boot.py runs the entrypoint, and test.yml's parity job checks the
  worker's uid and HOME in the Railway build).

scripts/ci/image_worker_toolchain.py proves inside the image that ``gcc -flto`` still
links, that git still commits, branches, clones, pulls and clones over HTTPS, that the
stripped node runs, keeps .dynsym and loads a native N-API addon, and that setpriv drops
root to the worker with the worker's HOME.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
IMAGES = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text())
BUILD_STEPS = IMAGES["jobs"]["build"]["steps"]
TOOLCHAIN_SCRIPT = "scripts/ci/image_worker_toolchain.py"


def _load(relative: str):
    """A CI script, loaded by path as the other image tests load theirs."""
    name = Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


toolchain = _load(TOOLCHAIN_SCRIPT)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


INSTRUCTIONS = _instructions(DOCKERFILE)


def _only(predicate) -> str:
    found = [line for line in INSTRUCTIONS if predicate(line)]
    assert len(found) == 1, f"expected one matching instruction, found {found}"
    return found[0]


def _commands(run: str) -> list[list[str]]:
    """The commands of a RUN, split at && and ;, each as its words."""
    body = run[len("RUN "):] if run.startswith("RUN ") else run
    return [shlex.split(part) for part in re.split(r"&&|;", body) if part.strip()]


APT_RUN = _only(lambda line: line.startswith("RUN") and "build-essential" in line)


def _rm_paths(run: str) -> list[str]:
    return [word for words in _commands(run) if words[:2] == ["rm", "-f"] for word in words[2:]]


def _worker_step(script: str) -> dict:
    steps = [step for step in BUILD_STEPS if script in step.get("run", "")]
    assert len(steps) == 1, f"images.yml's build job runs {script} in {len(steps)} steps"
    return steps[0]


# --------------------------------------------------------------------------- row 16


def test_lto_dump_goes_in_the_layer_that_installs_gcc():
    assert "/usr/bin/*lto-dump*" in _rm_paths(APT_RUN)
    assert toolchain.LTO_DUMP_GLOB == "/usr/bin/*lto-dump*", "the in-image check looks where the Dockerfile deletes"
    # The toolchain stays: gcc, g++ and make all come with build-essential.
    assert "build-essential" in APT_RUN


def test_the_in_image_lto_check_links_two_units_with_flto():
    """Two units, so the link really runs lto-wrapper and lto1 across them."""
    assert set(toolchain.LTO_UNITS) == {"main.c", "answer.c"}
    source = (ROOT / TOOLCHAIN_SCRIPT).read_text()
    assert '["gcc", "-flto", "-O2", "-c", name' in source
    assert '["gcc", "-flto", "-O2", *objects' in source


# --------------------------------------------------------------------------- row 19


GIT_TRIM = {
    "/usr/bin/git-shell", "/usr/bin/scalar",
    "/usr/lib/git-core/git-daemon", "/usr/lib/git-core/git-http-backend",
    "/usr/lib/git-core/git-http-push", "/usr/lib/git-core/git-imap-send",
    "/usr/lib/git-core/git-shell", "/usr/lib/git-core/scalar",
}


def test_the_git_trim_deletes_the_server_side_binaries_and_links_git_core_git():
    removed = set(_rm_paths(APT_RUN)) - {"/usr/bin/*lto-dump*"}
    assert removed == GIT_TRIM
    assert ["ln", "-sf", "/usr/bin/git", "/usr/lib/git-core/git"] in _commands(APT_RUN)
    # git-remote-http(s) clones, pulls and pushes over HTTPS: it stays.
    assert not [path for path in removed if "remote" in path]


def test_the_in_image_check_looks_for_exactly_what_the_trim_deletes():
    assert set(toolchain.GIT_REMOVED) == GIT_TRIM
    assert toolchain.GIT_CORE_GIT == "/usr/lib/git-core/git" and toolchain.GIT_BINARY == "/usr/bin/git"
    assert toolchain.HTTPS_REPO.startswith("https://")


# --------------------------------------------------------------------------- row 21

# Shared with services/media-render/Dockerfile, byte for byte: the same stage on the same
# node digest strips to the same file, and COPY --link makes it the same layer blob.
NODE_STRIPPED_STAGE = (
    "# Node, stripped. The same stage, byte for byte, in services/workspace-worker/Dockerfile\n"
    "# and services/media-render/Dockerfile, so the two images share one node layer on a host.\n"
    "FROM node AS node-stripped\n"
    "RUN apt-get update \\\n"
    " && apt-get install -y --no-install-recommends binutils \\\n"
    " && strip --strip-unneeded --preserve-dates /usr/local/bin/node \\\n"
    " && rm -rf /var/lib/apt/lists/*\n"
)
NODE_COPY = "COPY --link --from=node-stripped /usr/local/bin/node /usr/local/bin/node"
NODE_DOCKERFILES = ("services/workspace-worker/Dockerfile", "services/media-render/Dockerfile")


def _stage_lines(stage: str) -> list[str]:
    """The instructions of the worker's stage named ``stage``, its FROM line first."""
    start = next(i for i, line in enumerate(INSTRUCTIONS) if re.fullmatch(rf"FROM \S+ AS {stage}", line))
    end = next((i for i in range(start + 1, len(INSTRUCTIONS)) if INSTRUCTIONS[i].startswith("FROM ")),
               len(INSTRUCTIONS))
    return INSTRUCTIONS[start:end]


def test_the_node_stripped_stage_follows_the_node_stage_word_for_word():
    node_at = DOCKERFILE.index(" AS node\n")
    assert DOCKERFILE[node_at:].startswith(" AS node\n\n" + NODE_STRIPPED_STAGE + "\n")


def test_every_dockerfile_with_the_stage_carries_it_byte_for_byte():
    """media-render carries the same stage (its own lane adds it), never a variant of it."""
    for path in NODE_DOCKERFILES:
        text = (ROOT / path).read_text()
        if "AS node-stripped" in text:
            assert text.count(NODE_STRIPPED_STAGE) == 1, f"{path}'s node-stripped stage is not the shared one"
            assert text.count(NODE_COPY) == 1, f"{path} copies the stripped node some other way"


def test_the_worker_copies_node_first_with_link_and_only_from_the_stripped_stage():
    production = _stage_lines("production")
    assert production[0].startswith("FROM python:3.12-slim@sha256:") and production[1] == NODE_COPY
    assert [line for line in INSTRUCTIONS if "/usr/local/bin/node " in line and line.startswith("COPY")] == [NODE_COPY]
    assert INSTRUCTIONS[-1] == "CMD []" and production[-1] == "CMD []", "production stays the last stage"


def test_the_in_image_node_checks_look_for_the_strip():
    assert toolchain.NODE == "/usr/local/bin/node"
    assert set(toolchain.NODE_STRIPPED_SECTIONS) == {".symtab", ".strtab"}
    assert "napi_register_module_v1" in toolchain.NODE_ADDON_SOURCE
    assert set(toolchain.NODE_TOOLS) == {"npm", "npx", "corepack", "pnpm"}


# --------------------------------------------------------------------------- images.yml


def test_images_runs_the_toolchain_check_in_the_worker_it_built():
    step = _worker_step(TOOLCHAIN_SCRIPT)
    condition = " ".join(step["if"].split())
    assert "!cancelled()" in condition and "steps.build.outcome == 'success'" in condition
    assert "matrix.name == 'workspace-worker'" in condition
    assert step["env"] == {"IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    assert 'docker pull --quiet "${IMAGE}@${DIGEST}"' in step["run"]
    assert f'--entrypoint python "${{IMAGE}}@${{DIGEST}}" - < {TOOLCHAIN_SCRIPT}' in step["run"]
    scan_at = next(i for i, other in enumerate(BUILD_STEPS) if other.get("name") == "Vulnerability scan")
    assert BUILD_STEPS.index(step) > scan_at, "after the reports and the scan, so they run first"


# --------------------------------------------------------------------------- row 47


def _apt_packages(run: str) -> list[str]:
    words = next(words for words in _commands(run) if words[:2] == ["apt-get", "install"])
    return [word for word in words[2:] if not word.startswith("-")]


def test_gosu_is_gone_and_the_in_image_check_drops_as_the_entrypoint_does():
    assert "gosu" not in _apt_packages(APT_RUN)
    assert not [line for line in INSTRUCTIONS if "gosu" in line]
    exec_line = (WORKER / "entrypoint.sh").read_text().rstrip().splitlines()[-1]
    assert exec_line == f'exec {" ".join(toolchain.SETPRIV_DROP)} python -m main "$@"'
    assert toolchain.WORKER_IDENTITY == "1000 1000 /home/worker"


def test_every_toolchain_check_names_its_breakdown_row():
    rows = [title.split(":")[0] for title, _ in toolchain.CHECKS]
    assert rows == ["row 16", "row 19", "row 19", "row 19", "row 21", "row 21", "row 21", "row 47"]
