#!/usr/bin/env python3
"""PRD-262 US-019: the throwaway test image, for pytest runs inside an API image.

US-014 took pytest out of the production image, so a test that must run inside an image
runs in one built from it: ``FROM <the image under test>``, then ``pip install --user
--require-hashes`` of orchestrator/requirements-dev.lock as the image's own user
(US-022: the lock is compiled against the image's own lock, so the versions the two
share are the ones the image already has, and pip leaves them alone). The tests are
mounted read-only at /app/tests. The image is built in the CI job and never pushed, and
orchestrator/Dockerfile gains no stage for it, so ``production`` stays its last stage
(US-010). Its Dockerfile is written here and sent on stdin, with a build context that
holds only the dev lock.

    image_test_image.py build IMAGE TAG
        Builds TAG from IMAGE, a local or registry image.

    image_test_image.py pytest TAG [--network NAME] [PYTEST_ARG...]
        Runs ``python -m pytest PYTEST_ARG...`` in TAG from /app, with the checkout's
        orchestrator/tests mounted read-only over /app/tests and no bytecode written.
        ``--network none`` keeps the run off the network.

The run gets the throwaway database settings image_site_packages.py imports the app
with (``APP_IMPORT_ENV``): the image sets none, and core/database/database.py raises
when imported without them, so a test whose module reaches it (the skill loader's
does) would fail to collect. Nothing listens there, and nothing connects on import.

test.yml's prod-image-parity runs US-019's clone tests in it, on the network:
importing the skill loader loads modules/rag/service.py, which fetches tiktoken's
encoding when imported until US-033. US-028 runs its per-language parser tests here
with ``--network none``. Stdlib only.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


site_packages = _sibling("image_site_packages")

ROOT = Path(__file__).resolve().parents[2]
DEV_LOCK = ROOT / "orchestrator" / "requirements-dev.lock"
TESTS = ROOT / "orchestrator" / "tests"
IMAGE_TESTS = "/app/tests"
USAGE = "usage: image_test_image.py build IMAGE TAG | pytest TAG [--network NAME] [PYTEST_ARG...]"


def dockerfile(image: str) -> str:
    """The test image's Dockerfile: the image under test plus the test tools, installed as its user. Pure."""
    return (
        f"FROM {image}\n"
        f"COPY {DEV_LOCK.name} /tmp/{DEV_LOCK.name}\n"
        "RUN python -m pip install --user --no-cache-dir --no-warn-script-location --require-hashes "
        f"-r /tmp/{DEV_LOCK.name}\n"
    )


def build_command(tag: str, context: str) -> List[str]:
    """``docker build`` of TAG from a Dockerfile on stdin. Pure."""
    return ["docker", "build", "--tag", tag, "--file", "-", context]


def pytest_command(tag: str, pytest_args: List[str], network: Optional[str] = None) -> List[str]:
    """``docker run`` of pytest in TAG with the tests mounted read-only and the app's
    throwaway database settings. Pure."""
    command = ["docker", "run", "--rm", "--workdir", "/app", "--entrypoint", "python",
               "--volume", f"{TESTS}:{IMAGE_TESTS}:ro", "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for key, value in site_packages.APP_IMPORT_ENV.items():
        command += ["--env", f"{key}={value}"]
    if network:
        command += ["--network", network]
    return command + [tag, "-m", "pytest", "-p", "no:cacheprovider", *pytest_args]


def build(image: str, tag: str) -> int:
    context = tempfile.mkdtemp(prefix="prd262-test-image-")
    try:
        shutil.copy(DEV_LOCK, Path(context) / DEV_LOCK.name)
        print(dockerfile(image), end="")
        return subprocess.run(build_command(tag, context), input=dockerfile(image), text=True, check=False).returncode
    finally:
        shutil.rmtree(context, ignore_errors=True)


def pytest_split(rest: List[str]) -> Tuple[Optional[str], List[str]]:
    """(the --network value, pytest's arguments) from what follows the tag. Pure."""
    if rest[:1] == ["--network"] and len(rest) > 1:
        return rest[1], rest[2:]
    return None, rest


def main(argv: List[str]) -> int:
    if len(argv) == 3 and argv[0] == "build":
        return build(argv[1], argv[2])
    if len(argv) >= 2 and argv[0] == "pytest":
        network, pytest_args = pytest_split(argv[2:])
        return subprocess.run(pytest_command(argv[1], pytest_args, network), check=False).returncode
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
