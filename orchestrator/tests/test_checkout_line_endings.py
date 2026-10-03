"""Linux runtime files must survive a Git checkout with Windows defaults."""

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    [
        "orchestrator/docker-entrypoint.sh",
        "services/workspace-worker/entrypoint.sh",
        "Dockerfile.backend",
        "services/workspace-worker/Dockerfile",
        "Makefile",
        "orchestrator/config.py",
        "docker-compose.yml",
        "charts/automatos/Chart.yaml",
        ".env.example",
        "envs/api.defaults",
    ],
)
def test_runtime_files_checkout_with_lf(tmp_path: Path, relative_path: str) -> None:
    """Exercise Git's conversion, including the bind-mounted backend entrypoint."""
    attributes = REPO_ROOT / ".gitattributes"
    if attributes.exists():
        (tmp_path / ".gitattributes").write_bytes(attributes.read_bytes())

    # Isolate the checkout from a developer's global attribute overrides.
    global_attributes = tmp_path / "empty-attributes"
    global_attributes.write_bytes(b"")
    git = [
        "git", "-c", "core.autocrlf=true", "-c", "core.safecrlf=false",
        "-c", f"core.attributesFile={global_attributes}",
    ]
    subprocess.run([*git, "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True)

    expected = (REPO_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
    checked_out = tmp_path / relative_path
    checked_out.parent.mkdir(parents=True, exist_ok=True)
    checked_out.write_bytes(expected)
    subprocess.run([*git, "add", "--", relative_path], cwd=tmp_path, check=True, capture_output=True)
    checked_out.unlink()
    subprocess.run(
        [*git, "checkout-index", "--", relative_path],
        cwd=tmp_path, check=True, capture_output=True,
    )

    assert checked_out.read_bytes() == expected, f"{relative_path} acquired CRLF on checkout"
