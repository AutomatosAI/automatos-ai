"""F178 (b, 26 Sep): a page html_to_png renders loads no protected file and
nothing from outside its workspace.

html_to_png launches Chromium with --allow-file-access-from-files and
--disable-web-security, and checked only the top page's file:// URL. A page
inside the workspace could iframe .ssh/<key> or another workspace's files, and
the screenshot, an ordinary PNG that download serves and F179 A will still let
be published, would show them. A probe in the worker image with Chromium
(Playwright 1.52) rendered both iframes' text ("SECRET-KEY-LINE",
"OTHER-TENANT"). With a context route the same page got Chromium's error page
in both frames. Every request of a rendered page now passes one gate.

The headline test drives the real html_to_png through a fake Playwright whose
page requests what the HTML embeds, as Chromium does. No browser, except in
the last test, which runs only where Chromium is installed.
"""
import contextlib
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers_workspace_worker import WORKER_DIR

KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nnot-a-real-key\n-----END OPENSSH PRIVATE KEY-----\n"


class _Route:
    def __init__(self, url, outcomes):
        self.request = SimpleNamespace(url=url)
        self._outcomes = outcomes

    async def abort(self):
        self._outcomes[self.request.url] = "refused"

    async def continue_(self):
        self._outcomes[self.request.url] = "loaded"


class _Context:
    """A browser context whose page loads its URL and then what that page embeds."""

    def __init__(self, embeds):
        self.embeds, self.gate, self.outcomes = embeds, None, {}

    async def route(self, pattern, handler):
        assert pattern == "**/*"
        self.gate = handler

    async def new_page(self):
        context = self

        class _Page:
            async def goto(self, url, **kwargs):
                for requested in [url, *context.embeds]:
                    if context.gate is None:
                        context.outcomes[requested] = "loaded"  # Chromium loads what nothing refuses
                    else:
                        await context.gate(_Route(requested, context.outcomes))

            async def wait_for_selector(self, *args, **kwargs):
                pass

            async def screenshot(self, path, **kwargs):
                Path(path).write_bytes(b"\x89PNG")

        return _Page()

    async def close(self):
        pass


def _fake_playwright(monkeypatch, context):
    class _Browser:
        async def new_context(self, **kwargs):
            return context

        async def close(self):
            pass

    async def _launch(**kwargs):
        return _Browser()

    @contextlib.asynccontextmanager
    async def async_playwright():
        yield SimpleNamespace(chromium=SimpleNamespace(launch=_launch))

    api = types.ModuleType("playwright.async_api")
    api.async_playwright, api.TimeoutError = async_playwright, TimeoutError
    package = types.ModuleType("playwright")
    package.async_api = api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.async_api", api)


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(WORKER_DIR), *sys.path])
    from executor import WorkspaceToolExecutor
    from workspace_manager import WorkspaceManager

    ws = WorkspaceManager(str(uuid.uuid4()), str(tmp_path))
    ws.ensure_workspace_exists()
    ws.inject_credentials("task-1", {"ssh_private_key": KEY})
    other = tmp_path / str(uuid.uuid4()) / "reports" / "q3.md"
    other.parent.mkdir(parents=True)
    other.write_text("another workspace's report")
    page = ws.root / "content" / "card.html"
    page.write_text("<iframe src='../.ssh/id_ed25519'></iframe>")
    return SimpleNamespace(ws=ws, executor=WorkspaceToolExecutor(ws), other=other, page=page)


@pytest.mark.asyncio
async def test_a_workspace_page_cannot_frame_a_key_or_another_workspace(monkeypatch, workspace):
    root = workspace.ws.root
    key = f"file://{root / '.ssh' / 'id_ed25519'}"
    theirs = f"file://{workspace.other}"
    font = f"file://{root / 'content' / 'brand.woff2'}"
    cdn = "https://cdn.example.com/brand.css"
    context = _Context(embeds=[key, theirs, font, cdn])
    _fake_playwright(monkeypatch, context)

    rendered = await workspace.executor.html_to_png(
        url=f"file://{workspace.page}", viewport_w=100, viewport_h=100,
        output_path="content/card.png", wait_for=None)

    assert rendered["success"] is True, rendered
    assert context.outcomes == {
        f"file://{workspace.page}": "loaded", key: "refused", theirs: "refused", font: "loaded", cdn: "loaded",
    }


def test_the_gate_refuses_what_a_page_may_not_load(workspace, tmp_path):
    root, blocked = workspace.ws.root, workspace.executor._file_url_blocked
    (root / "content" / "shortcut").symlink_to(root / ".ssh" / "id_ed25519")

    assert blocked(f"file://{root / 'content' / 'card.html'}") is False
    assert blocked(f"file://{root / '.canvas' / 'transcript.jsonl'}") is True
    assert blocked(f"file://{root / 'content' / 'shortcut'}") is True  # a symlink into .ssh/
    assert blocked(f"file://{workspace.other}") is True
    assert blocked(f"file://{root}/content/%2e%2e/%2e%2e/{workspace.other.parent.parent.name}/reports/q3.md") is True
    assert blocked("file:///etc/passwd") is True
    assert blocked("https://cdn.example.com/brand.css") is False
    assert blocked("data:text/html,<p>hi</p>") is False


@pytest.mark.asyncio
async def test_in_chromium_a_framed_key_is_refused(monkeypatch, workspace):
    """Where Chromium is installed (the production worker image), the real browser
    routes the iframe's file:// request through the gate."""
    pytest.importorskip("playwright.async_api")
    refused = []
    gate = workspace.executor._gate_render_request

    async def _recording_gate(route):
        if workspace.executor._file_url_blocked(route.request.url):
            refused.append(route.request.url)
        await gate(route)

    monkeypatch.setattr(workspace.executor, "_gate_render_request", _recording_gate)
    rendered = await workspace.executor.html_to_png(
        url=f"file://{workspace.page}", viewport_w=100, viewport_h=100,
        output_path="content/card.png", wait_for=None)
    if not rendered.get("success") and "Executable doesn't exist" in rendered.get("error", ""):
        pytest.skip("no Chromium in this image")

    assert rendered["success"] is True, rendered
    assert refused == [f"file://{workspace.ws.root / '.ssh' / 'id_ed25519'}"]
