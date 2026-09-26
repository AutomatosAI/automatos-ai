"""F178 (26 Sep, the F173 security review) — the worker never serves a deploy key.

The workspace worker's listing and content routes refused sensitive names
(.ssh, .gitconfig, .aws, .gcp, .workspace_meta.json, .canvas, .task_env_*), but
its download route checked none. The backend's GET /api/workspaces/{id}/files/raw
passes a member's path straight to it, and the agent tool workspace_get_public_url
uploads what it downloads to the public image store. And the deploy key a clone
task injects into <workspace>/.ssh/ was never removed: clear_credentials() had
no caller. Now all three routes check one list (workspace_manager.SENSITIVE_NAMES),
and the last task running in a workspace takes the key and git identity with it.

The worker runs in-process over a temporary volume. No Redis, no database.
"""
import asyncio
import sys
import uuid

import pytest

from tests.helpers_workspace_worker import WORKER_DIR, answer, task_runner, worker_server

KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nnot-a-real-key\n-----END OPENSSH PRIVATE KEY-----\n"
SENSITIVE = [
    ".ssh/id_ed25519", ".gitconfig", ".aws/credentials", ".gcp/key.json",
    ".workspace_meta.json", ".canvas/transcript.jsonl", ".task_env_task-1",
]


@pytest.mark.parametrize("secret", SENSITIVE)
@pytest.mark.asyncio
async def test_no_file_route_serves_a_sensitive_file(monkeypatch, tmp_path, secret):
    ws = str(uuid.uuid4())
    path = tmp_path / ws / secret
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(KEY)
    async with worker_server(monkeypatch, tmp_path) as (base, http):
        downloaded = await answer(await http.get(f"{base}/workspaces/{ws}/files/download", params={"path": secret}))
        read = await answer(await http.get(f"{base}/workspaces/{ws}/files/content", params={"path": secret}))
        listed = await answer(await http.get(f"{base}/workspaces/{ws}/files", params={"path": "."}))

    assert downloaded == (403, {"error": "Access denied"}), downloaded
    assert read == (403, {"error": "Access denied"})
    assert secret.split("/")[0] not in [entry["name"] for entry in listed[1]["entries"]]


@pytest.mark.asyncio
async def test_the_last_task_out_takes_the_deploy_key_with_it(monkeypatch, tmp_path):
    worker = task_runner(monkeypatch, tmp_path)
    import executor as worker_executor  # the worker's, on the path the loader set

    released = asyncio.Event()

    async def _step(self, step):
        if step.get("action") == "hold":
            await released.wait()
        return {}

    monkeypatch.setattr(worker_executor.WorkspaceToolExecutor, "execute_step", _step)
    ws = str(uuid.uuid4())
    credentials = {"ssh_private_key": KEY, "git_name": "Auto", "git_email": "auto@example.com"}
    pushing = asyncio.create_task(worker._execute_task(
        {"task_id": "task-push", "workspace_id": ws, "credentials": credentials, "steps": [{"action": "hold"}]}))
    await asyncio.sleep(0.05)

    await worker._execute_task({"task_id": "task-read", "workspace_id": ws, "credentials": credentials, "steps": []})
    assert (tmp_path / ws / ".ssh" / "id_ed25519").read_text() == KEY  # the push still needs it

    released.set()
    await pushing
    assert not (tmp_path / ws / ".ssh").exists()
    assert not (tmp_path / ws / ".gitconfig").exists()


def test_cleanup_clears_credentials_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(WORKER_DIR), *sys.path])
    from workspace_manager import WorkspaceManager

    ws = WorkspaceManager(str(uuid.uuid4()), str(tmp_path))
    ws.ensure_workspace_exists()
    ws.inject_credentials("task-1", {"ssh_private_key": KEY, "git_name": "Auto"})

    ws.cleanup_task("task-1")
    assert (ws.root / ".ssh" / "id_ed25519").exists() and (ws.root / ".gitconfig").exists()
    ws.cleanup_task("task-1", clear_credentials=True)
    assert not (ws.root / ".ssh").exists() and not (ws.root / ".gitconfig").exists()


# ── the security review of 52dae577b ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_html_to_png_never_renders_a_protected_file(monkeypatch, tmp_path):
    """A screenshot of the key would be an ordinary, downloadable PNG."""
    monkeypatch.setattr(sys, "path", [str(WORKER_DIR), *sys.path])
    from executor import WorkspaceToolExecutor
    from workspace_manager import WorkspaceManager

    ws = WorkspaceManager(str(uuid.uuid4()), str(tmp_path))
    ws.ensure_workspace_exists()
    ws.inject_credentials("task-1", {"ssh_private_key": KEY})

    rendered = await WorkspaceToolExecutor(ws).html_to_png(
        url=f"file://{ws.root / '.ssh' / 'id_ed25519'}", viewport_w=100, viewport_h=100, output_path="key.png")

    assert rendered == {"success": False, "error": "file:// URL must not point at a protected file"}
    assert not (ws.root / "key.png").exists()


@pytest.mark.asyncio
async def test_a_task_whose_key_injection_fails_strands_no_key(monkeypatch, tmp_path):
    worker = task_runner(monkeypatch, tmp_path)
    import workspace_manager as worker_workspace  # the worker's, on the path the loader set

    inject = worker_workspace.WorkspaceManager.inject_credentials

    def _inject_then_fill_the_disk(self, task_id, credentials):
        inject(self, task_id, credentials)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(worker_workspace.WorkspaceManager, "inject_credentials", _inject_then_fill_the_disk)
    ws = str(uuid.uuid4())

    await worker._execute_task({"task_id": "task-full-disk", "workspace_id": ws,
                                "credentials": {"ssh_private_key": KEY, "git_name": "Auto"}, "steps": []})

    assert not (tmp_path / ws / ".ssh").exists() and not (tmp_path / ws / ".gitconfig").exists()
    assert worker._redis.statuses[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_a_protected_name_is_refused_in_any_case(monkeypatch, tmp_path):
    """.SSH is .ssh on a case-insensitive filesystem."""
    ws = str(uuid.uuid4())
    key = tmp_path / ws / ".SSH" / "id_ed25519"
    key.parent.mkdir(parents=True)
    key.write_text(KEY)
    async with worker_server(monkeypatch, tmp_path) as (base, http):
        downloaded = await answer(await http.get(
            f"{base}/workspaces/{ws}/files/download", params={"path": ".SSH/id_ed25519"}))

    assert downloaded == (403, {"error": "Access denied"})
