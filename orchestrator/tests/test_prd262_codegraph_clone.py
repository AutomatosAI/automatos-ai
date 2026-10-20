"""PRD-262 US-019: the code graph and the skill loader clone with the image's git.

US-019 trims git in the API image: its server-side and mail binaries go, and
/usr/lib/git-core/git, a second copy of /usr/bin/git, becomes a link to it. Every
builtin in git-core (git-clone, git-fetch, git-upload-pack, …) is a relative link to
that file, and a clone or a pull runs git-upload-pack from there. These tests clone for
real, through the code's own paths, from a repository made on the spot, so the clones
need no network:

- the code graph: ``CodeGraphService._clone_github_repo`` (GitPython, then the git CLI),
  one commit deep on a branch, and a failed clone that leaves no directory behind;
- the skill loader: ``SkillLoader._git_clone`` and ``_git_pull`` (the git CLI).

CI runs them in the backend shards and, inside the API image, in the throwaway test
image (scripts/ci/image_test_image.py, test.yml's prod-image-parity), with the app's
throwaway database settings. That run stays on the network: importing the skill loader
loads modules/rag/service.py, which fetches tiktoken's encoding when imported (until
PRD-262 US-033). The code graph's HTTPS-and-allowlist URL check has tests of its own
(tests/security, test_prd154_s12); here it lets a file:// URL through.
prod-image-parity also clones a public repository over HTTPS in the image.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, List

import pytest

import modules.codegraph.codegraph_service as codegraph_service
from modules.agents.services.skill_loader import SkillLoader
from modules.codegraph.codegraph_service import CodeGraphService

BRANCH = "main"
CLONE_PREFIX = "codegraph_"  # _clone_github_repo's mkdtemp prefix
GIT_TIMEOUT_S = 60
IDENTITY = ("-c", "user.name=PRD-262 fixture", "-c", "user.email=prd262@example.invalid")
FIRST = ("README.md", "first commit\n")
SECOND = ("app.py", "def answer():\n    return 42\n")
THIRD = ("CHANGELOG.md", "pulled\n")


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a fixed identity; its stdout, stripped."""
    done = subprocess.run(["git", *IDENTITY, *args], cwd=cwd, capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_S, check=False)
    assert done.returncode == 0, f"git {' '.join(args)} in {cwd}: {done.stderr}"
    return done.stdout.strip()


def commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A repository with two commits on ``main``."""
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q", "-b", BRANCH)
    commit(repo, *FIRST)
    commit(repo, *SECOND)
    return repo


@pytest.fixture
def any_url(monkeypatch):
    """The code graph accepts the fixture's file:// URL (its URL check is tested elsewhere)."""
    monkeypatch.setattr(codegraph_service, "validate_git_url", lambda url: (True, None))


@pytest.fixture
def made_dirs(monkeypatch) -> Iterator[List[str]]:
    """The directories the code graph's clone makes (``codegraph_*``), recorded as it makes them
    and removed after the test."""
    made: List[str] = []
    real = codegraph_service.tempfile.mkdtemp

    def recording(*args, **kwargs):
        path = real(*args, **kwargs)
        if Path(path).name.startswith(CLONE_PREFIX):
            made.append(path)
        return path

    monkeypatch.setattr(codegraph_service.tempfile, "mkdtemp", recording)
    yield made
    for path in made:
        shutil.rmtree(path, ignore_errors=True)


def clone(url: str, branch: str) -> Path:
    service = CodeGraphService.__new__(CodeGraphService)
    return Path(asyncio.run(service._clone_github_repo(url, branch, None)))


def test_the_code_graph_clones_a_branch_one_commit_deep(source, any_url, made_dirs):
    checkout = clone(source.as_uri(), BRANCH)

    assert [str(checkout)] == made_dirs
    assert (checkout / SECOND[0]).read_text() == SECOND[1]
    assert git(checkout, "rev-list", "--count", "HEAD") == "1"
    assert (checkout / ".git" / "shallow").is_file()
    assert git(checkout, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH
    assert git(checkout, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")


def test_a_failed_clone_raises_and_leaves_no_directory(tmp_path, any_url, made_dirs):
    with pytest.raises(ValueError, match="Failed to clone repository"):
        clone((tmp_path / "missing").as_uri(), BRANCH)

    assert len(made_dirs) == 1 and not Path(made_dirs[0]).exists(), made_dirs


def test_the_skill_loader_clones_and_pulls(source, tmp_path):
    loader = SkillLoader.__new__(SkillLoader)
    local = tmp_path / "skills"

    cloned = loader._git_clone(source.as_uri(), str(local), BRANCH)
    assert cloned["success"], cloned
    assert cloned["commit_sha"] == git(source, "rev-parse", "HEAD")
    assert (local / FIRST[0]).read_text() == FIRST[1]

    commit(source, *THIRD)
    pulled = loader._git_pull(str(local))
    assert pulled["success"], pulled
    assert pulled["commit_sha"] == git(source, "rev-parse", "HEAD")
    assert (local / THIRD[0]).read_text() == THIRD[1]
