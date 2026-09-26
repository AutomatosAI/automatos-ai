"""PRD-251 Wave 1, US-117 — two tools agents already use, extended.

Pins:

* **generate_document renders the social formats.** An agent calls it with
  ``social_video`` and a seeded template (Data story, the real starter): the
  real service builds the bundle, media-render (``httpx.MockTransport``, the
  real client) renders it, and the tool answers with a video Deliverable (the
  first test to run the tool with a template_id: that path raised NameError on
  ``UUID`` since PRD-167). A
  social format with no template is refused before anything renders. Both
  lanes (the chat lane's inline schema and the ToolRegistry spec) offer every
  format ``generate()`` dispatches, in one wording that names social images
  and videos, and the unified executor hands a social format to the same
  handler. The tool's summary calls an MP4 a video and a PNG an image.
* **A playbook renders a social template as a FIXED step.** Through the real
  step loop: step 1 (an agent) saves the variables with ``scratchpad_write``,
  step 2 (``generate_document``, no agent, no prompt) reads them with
  ``"data": "{{ step_1.variables }}"`` and renders; the file is a playbook
  Deliverable. A reference no earlier step answers fails the step, naming it,
  and nothing renders (it used to be replaced by the whole run's context).
  The references themselves (``core/services/playbook_step_refs.py``) are
  pinned as a pure contract.
* **platform_generate_cover_image without a post_id** makes a still in the
  aspect ratio asked for, saves it to the same image store and registers an
  image Deliverable; no blog post is read or written. With a post_id it is the
  blog cover, attached as before, and no Deliverable is added. Every image
  takes the cover model's one cost path.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import httpx  # noqa: E402

import core.llm  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import core.services.blog_service as blog_service  # noqa: E402
import core.services.image_store as image_store  # noqa: E402
import modules.documents.generation_service as generation_service  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
import services.knowledge_flywheel as knowledge_flywheel  # noqa: E402
from api import recipe_executor as rex  # noqa: E402
from config import config  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models import Agent  # noqa: E402
from core.models.core import (  # noqa: E402
    DOCUMENT_TEMPLATE_FORMATS,
    PLAYBOOK_DOCUMENT_STEP,
    BoardTask,
    DocumentTemplate,
    RecipeExecution,
    WorkflowTemplate,
)
from core.models.workspaces import Workspace  # noqa: E402
from core.music_credit import bundle_track  # noqa: E402
from core.services.playbook_scratchpad import PlaybookScratchpad  # noqa: E402
from core.services.playbook_step_refs import (  # noqa: E402
    UnresolvedStepReference,
    resolve_step_references,
    step_values,
    structured,
)
from modules.documents.generation_service import DocumentGenerationService  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.tools.builtin.scratchpad_tool import handle_scratchpad_write  # noqa: E402
from modules.tools.discovery import handlers_blog  # noqa: E402
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from modules.tools.formatting.result_formatter import ToolResultFormatter  # noqa: E402
from tests.helpers_playbook_run import _Session, done, patch_edges  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000117a1")
TEMPLATE_ID = uuid.UUID("00000000-0000-0000-0000-0000000117b1")
POST_ID = uuid.UUID("00000000-0000-0000-0000-0000000117c1")
AGENT = SimpleNamespace(id=7, workspace_id=WS, user_id=None, name="Social Media Director", configuration={})
DATA_STORY = next(s for s in social_starters("social_video") if s["slug"] == "data-story")

RENDER_URL = "http://media-render:8090"
JOB_ID = "e" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"frames " * 200
VIDEO = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "duration": 40.0}


def _template_row(starter=DATA_STORY):
    return SimpleNamespace(
        id=TEMPLATE_ID, workspace_id=WS, name=starter["name"], format=starter["format"],
        blocks=starter["blocks"], is_active=True, version=1,
    )


def _workspace_row():
    return SimpleNamespace(
        id=WS, name="Acme", plan="basic", plan_limits={}, settings={},
        deleted_at=None, paused_at=None, paused_reason=None,
    )


class _Renderer:
    """media-render on httpx.MockTransport: takes a bundle, renders one MP4, and
    reports the library track the bundle asked for (a CC BY credit line)."""

    def __init__(self):
        self.bundles = []

    def _report(self):
        track = bundle_track(self.bundles[-1]) if self.bundles else None
        if not track:
            return {"check": {"ok": True}}
        return {"check": {"ok": True}, "music": {
            "track": track, "title": "Reference track", "artist": "Someone", "licence": "CC BY 4.0",
            "attribution": f'Music: "{track}", licensed CC BY 4.0.', "credit_required": True,
            "start": 0.0, "end": 40.0,
        }}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/render":
            self.bundles.append(json.loads(request.content))
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            return httpx.Response(
                200, json={"id": JOB_ID, "status": "done", "outputs": [VIDEO], "report": self._report()},
            )
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/{VIDEO['name']}":
            return httpx.Response(200, content=MP4)
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {path}"})


@pytest.fixture
def render_env(monkeypatch, tmp_path):
    """The real generation service, rendering through the real client over a mock
    transport; quota, booking, the Deliverable write and the flywheel are recorded."""
    for cfg in {id(m.config): m.config for m in (generation_service, media_render_client)}.values():
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_URL", RENDER_URL, raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_TOKEN", "", raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_POLL_SECONDS", 0, raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_MAX_WAIT_SECONDS", 60, raising=False)
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    monkeypatch.setattr(generation_service, "brand_kit_for_media_render", lambda kit: dict(kit))
    state = SimpleNamespace(quota=[], booked=[], registered=[], ingested=[], renderer=_Renderer())
    monkeypatch.setattr(generation_service, "enforce_render_quota", lambda db, workspace: state.quota.append(workspace.id))
    monkeypatch.setattr(generation_service, "book_render_seconds", lambda **kwargs: state.booked.append(kwargs))

    class Deliverables:
        def __init__(self, db, workspace_id):
            self.workspace_id = workspace_id

        def register(self, **kwargs):
            state.registered.append({**kwargs, "workspace_id": self.workspace_id})
            return {"success": True, "deliverable_id": f"d-{len(state.registered)}"}

    async def ingest(db, workspace_id, **kwargs):
        state.ingested.append(kwargs)

    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)
    monkeypatch.setattr(knowledge_flywheel, "ingest_agent_output", ingest)
    http = httpx.AsyncClient(transport=httpx.MockTransport(state.renderer.handler))
    monkeypatch.setattr(DocumentGenerationService, "_render_client", lambda self: MediaRenderClient(http))
    yield state
    asyncio.run(http.aclose())


def _platform_tools(session):
    from modules.agents.services.agent_platform_tools import AgentPlatformTools

    with patch("modules.agents.services.agent_platform_tools.RAGService"), patch(
        "modules.agents.services.agent_platform_tools.CodeGraphService"
    ):
        return AgentPlatformTools(db_session=session)


def _variables_rendered(bundle, supplied):
    return {name: bundle["variables"].get(name) for name in supplied}


# ---------------------------------------------------------------------------
# (a) generate_document: the social formats
# ---------------------------------------------------------------------------


def test_generate_document_renders_a_seeded_social_video_and_returns_its_deliverable(render_env):
    tools = _platform_tools(_Session({Agent: [AGENT], DocumentTemplate: [_template_row()], Workspace: [_workspace_row()]}))
    sample = DATA_STORY["sample_data"]

    result = asyncio.run(tools.execute_tool(
        "generate_document",
        {"title": "Market data story", "format": "social_video", "template_id": str(TEMPLATE_ID), "data": dict(sample)},
        agent_id=AGENT.id,
    ))

    assert result["success"] is True, result
    (answer,) = result["results"]
    assert answer["format"] == "mp4" and answer["deliverable_id"] == "d-1"
    assert (answer["template_id"], answer["template_name"]) == (str(TEMPLATE_ID), "Data story")
    # The seeded template, filled with what the agent sent, went to media-render once.
    (bundle,) = render_env.renderer.bundles
    assert bundle["reference"] == f"document_template:{TEMPLATE_ID}"
    assert _variables_rendered(bundle, sample) == sample
    assert render_env.quota == [WS] and render_env.booked[0]["seconds"] == VIDEO["duration"]
    # One video Deliverable, attributed to the agent and the template, with its music.
    (registered,) = render_env.registered
    assert (registered["artifact_type"], registered["file_type"]) == ("video", "mp4")
    assert registered["source_type"] == "agent_output" and registered["workspace_id"] == WS
    assert (registered["agent_id"], registered["agent_name"]) == (AGENT.id, AGENT.name)
    assert registered["extra"]["template_id"] == str(TEMPLATE_ID)
    assert registered["extra"]["music"]["track"] == bundle_track(bundle)


@pytest.mark.parametrize("fmt", ["social_video", "social_image"])
def test_a_social_format_without_a_template_is_refused_before_anything_renders(render_env, fmt):
    tools = _platform_tools(_Session({Agent: [AGENT], Workspace: [_workspace_row()]}))

    result = asyncio.run(tools.execute_tool(
        "generate_document", {"title": "Launch", "format": fmt, "data": {"headline": "Launch day"}}, agent_id=AGENT.id,
    ))

    assert not result["success"] and result["status"] == "error"
    assert f"rendered from a {fmt} template" in result["error"] and "template_id" in result["error"]
    assert render_env.renderer.bundles == [] and render_env.quota == [] and render_env.registered == []


def test_both_lanes_offer_every_format_generate_renders_in_one_wording():
    from modules.agents.services.agent_platform_tools import AgentPlatformTools
    from modules.tools.registry.tool_registry import (
        GENERATE_DOCUMENT_DESCRIPTION,
        GENERATE_DOCUMENT_FORMAT_DESCRIPTION,
        ToolRegistry,
    )

    inline = next(t for t in AgentPlatformTools.get_available_tools(object()) if t["name"] == "generate_document")
    spec = ToolRegistry().get_tool("generate_document")
    registry_format = next(p for p in spec.parameters if p.name == "format")
    inline_format = inline["parameters"]["properties"]["format"]

    assert inline_format["enum"] == registry_format.enum == list(DOCUMENT_TEMPLATE_FORMATS)
    assert {"social_image", "social_video"} <= set(registry_format.enum)
    assert spec.to_openai_format()["parameters"]["properties"]["format"]["enum"] == list(DOCUMENT_TEMPLATE_FORMATS)
    assert inline_format["description"] == registry_format.description == GENERATE_DOCUMENT_FORMAT_DESCRIPTION
    assert inline["description"] == GENERATE_DOCUMENT_DESCRIPTION
    assert spec.description.startswith(GENERATE_DOCUMENT_DESCRIPTION)
    # Semantic tool routing ranks by description: a social image or video request must land here.
    for text in (inline["description"], spec.description):
        assert "social image" in text and "social video" in text and "platform_create_social_post" in text


def test_the_unified_executor_hands_a_social_format_to_the_same_handler():
    from modules.tools.execution.exec_document import execute_generate_document

    calls = []

    class Tools:
        async def execute_tool(self, tool_name, parameters, agent_id):
            calls.append((tool_name, parameters, agent_id))
            return {"success": True}

    params = {"title": "Card", "format": "social_image", "template_id": str(TEMPLATE_ID), "data": {"headline": "Hi"}}
    asyncio.run(execute_generate_document(SimpleNamespace(platform_tools=Tools()), "generate_document", params, 7))
    assert calls == [("generate_document", params, 7)]


@pytest.mark.parametrize("fmt, kind", [("mp4", "video"), ("png", "image"), ("pdf", "document")])
def test_the_tool_summary_says_what_was_made(fmt, kind):
    result = {"success": True, "results": [{
        "filename": f"launch.{fmt}", "format": fmt, "size_kb": 3,
        "download_url": f"/api/documents/generated/launch.{fmt}", "deliverable_id": "d-1",
    }]}
    assert f"Generated {fmt.upper()} {kind}: launch.{fmt} (3 KB)" in ToolResultFormatter.format_for_llm(result, "generate_document")


# ---------------------------------------------------------------------------
# (a) the playbook step: a FIXED generate_document step, fed by an earlier step
# ---------------------------------------------------------------------------

EXECUTION_ID = "exec-117"
PLAYBOOK_ID = 117


class _Hash:
    """A Redis hash in memory: the scratchpad's own code runs on it."""

    def __init__(self):
        self.fields = {}

    def hset(self, key, field, value):
        self.fields[field] = value

    def hget(self, key, field):
        return self.fields.get(field)

    def hgetall(self, key):
        return dict(self.fields)

    def expire(self, key, ttl):
        pass


class _Pad(PlaybookScratchpad):
    def __init__(self, execution_id, redis_client=None):
        super().__init__(execution_id, redis_client=_Hash())


def _run_launch_video(monkeypatch, *, document_step, saved):
    """'Launch video' through the real step loop: the Director (step 1) saves
    ``saved`` with scratchpad_write, then the fixed generate_document step runs."""

    async def director(**kwargs):
        for key, value in saved.items():
            handle_scratchpad_write(key=key, value=value, scratchpad=kwargs["scratchpad"], step_order=kwargs["step_order"])
        return done("I wrote the data story's variables and saved them as 'variables'.", tokens=1200)

    steps = [
        {"step_id": "s1", "order": 1, "agent_id": AGENT.id, "error_handling": "stop", "max_retries": 0,
         "prompt_template": "Write the Data story template's variables and save them as 'variables'."},
        {"step_id": "s2", "order": 2, "type": PLAYBOOK_DOCUMENT_STEP, "error_handling": "stop", **document_step},
    ]
    execution = SimpleNamespace(
        execution_id=EXECUTION_ID, recipe_id=PLAYBOOK_ID, workspace_id=WS, status="pending", current_step=0,
        step_results=None, error_message=None, completed_at=None, started_at=None, output_data=None,
        execution_metadata={},
    )
    card = SimpleNamespace(id=1170, status="in_progress", result=None, error_message=None,
                           review_feedback=None, completed_at=None)
    session = _Session({
        WorkflowTemplate: [SimpleNamespace(id=PLAYBOOK_ID, name="Launch video", steps=steps, execution_config={})],
        RecipeExecution: [execution],
        Workspace: [_workspace_row()],
        Agent: [AGENT],
        BoardTask: [card],
        DocumentTemplate: [_template_row()],
    })
    patch_edges(monkeypatch, session=session, step=director, pad=_Pad)
    asyncio.run(rex._execute_recipe_inner(EXECUTION_ID, PLAYBOOK_ID, WS, {}, None))
    return execution


def test_a_playbook_renders_a_social_template_with_the_variables_an_earlier_step_saved(monkeypatch, render_env):
    sample = DATA_STORY["sample_data"]

    execution = _run_launch_video(
        monkeypatch,
        document_step={
            "title": "Launch video: {{ step_1.variables.product_name }}",
            "format": "social_video",
            "template_id": str(TEMPLATE_ID),
            "data": "{{ step_1.variables }}",
        },
        saved={"variables": json.dumps(sample)},
    )

    assert execution.status == "completed", execution.error_message
    (bundle,) = render_env.renderer.bundles
    assert bundle["reference"] == f"document_template:{TEMPLATE_ID}"
    assert _variables_rendered(bundle, sample) == sample
    (registered,) = render_env.registered
    assert (registered["artifact_type"], registered["source_type"]) == ("video", "playbook")
    assert registered["source_id"] == EXECUTION_ID
    assert registered["title"] == f"Launch video: {sample['product_name']}"
    assert registered["extra"]["template_id"] == str(TEMPLATE_ID)
    assert json.loads(execution.output_data["final_output"])["deliverable_id"] == "d-1"


def test_a_reference_no_earlier_step_answers_fails_the_step_naming_it_and_renders_nothing(monkeypatch, render_env):
    execution = _run_launch_video(
        monkeypatch,
        document_step={
            "title": "Launch video",
            "format": "social_video",
            "template_id": str(TEMPLATE_ID),
            "data": {"hook_line_1": "{{ step_1.hook }}", "product_name": "{{ step_3.name }}"},
        },
        saved={"variables": "{}"},
    )

    assert execution.status == "failed"
    assert "{{ step_1.hook }}" in execution.error_message and "{{ step_3.name }}" in execution.error_message
    assert render_env.renderer.bundles == [] and render_env.quota == [] and render_env.registered == []


def test_a_fixed_step_needs_no_agent_or_prompt_and_an_agent_step_still_does():
    fixed = {"step_id": "s2", "order": 2, "type": PLAYBOOK_DOCUMENT_STEP, "format": "social_video",
             "template_id": str(TEMPLATE_ID), "data": "{{ step_1.variables }}"}
    assert WorkflowTemplate(steps=[fixed]).validate_steps() == (True, None)
    assert WorkflowTemplate(steps=[{k: v for k, v in fixed.items() if k != "order"}]).validate_steps() == (
        False, "Step 0 missing required field: order",
    )
    agent_step = {"step_id": "s1", "order": 1, "prompt_template": "Write it."}
    assert WorkflowTemplate(steps=[agent_step]).validate_steps() == (False, "Step 0 missing required field: agent_id")


def test_the_step_config_keeps_one_reference_as_its_data_and_it_must_resolve_to_an_object():
    config_ = rex._document_step_config(
        {"type": PLAYBOOK_DOCUMENT_STEP, "format": "social_image", "data": "{{ step_1.variables }}"}
    )
    assert config_["data"] == "{{ step_1.variables }}" and config_["format"] == "social_image"
    assert rex._document_step_config({"data": 7})["data"] == {}
    with pytest.raises(ValueError, match="data must be an object"):
        rex._resolved_document_step({"title": "T", "data": "{{ step_1.output }}"}, {1: step_values("plain words")})


def test_the_scratchpad_gives_a_step_only_the_keys_it_saved():
    pad = _Pad("exec-pad")
    handle_scratchpad_write(key="headline", value="One", scratchpad=pad, step_order=1)
    # The loop then files every export so far under step 1 ...
    pad.write_step_results(step_order=1, tool_calls=[], agent_output="done", agent_exports=pad.get_exports())
    handle_scratchpad_write(key="variables", value="{}", scratchpad=pad, step_order=2)
    # ... so a step's own keys are read before that happens to it.
    assert pad.step_exports(2) == {"variables": "{}"}
    assert pad.step_exports(1) == {"headline": "One"}
    assert pad.step_exports(3) == {}


# ---------------------------------------------------------------------------
# The references (core/services/playbook_step_refs.py): a pure contract
# ---------------------------------------------------------------------------

VARIABLES = {"headline": "Launch day", "stat": 21, "tags": ["a", "b"]}
FINISHED = {
    1: step_values("Here are the variables.", {"variables": json.dumps(VARIABLES), "version": "3.10"}),
    2: step_values('```json\n{"deliverable_id": "d-9", "card": {"title": "T"}}\n```'),
}


def test_a_whole_reference_takes_the_value_and_json_text_is_its_object():
    assert resolve_step_references("{{ step_1.variables }}", FINISHED) == VARIABLES
    assert resolve_step_references("{{step_1.variables.stat}}", FINISHED) == 21
    assert resolve_step_references({"tags": " {{ step_1.variables.tags }} "}, FINISHED) == {"tags": ["a", "b"]}
    # Text stays text: "3.10" is never the number 3.1.
    assert resolve_step_references({"v": "{{ step_1.version }}"}, FINISHED) == {"v": "3.10"}


def test_a_reference_inside_text_is_replaced_by_its_text():
    text = "Day {{ step_1.variables.stat }}: {{ step_1.variables.headline }} {{ step_1.variables.tags }}"
    assert resolve_step_references(text, FINISHED) == 'Day 21: Launch day ["a", "b"]'


def test_a_step_offers_its_output_its_saved_keys_and_its_json_answer():
    assert resolve_step_references("{{ step_1.output }}", FINISHED) == "Here are the variables."
    assert resolve_step_references("{{ step_2.deliverable_id }}", FINISHED) == "d-9"
    answer = '{"headline": "from the answer", "output": "x"}'
    values = step_values(answer, {"headline": "saved"})
    # A saved key wins over the answer's; output is always the whole answer.
    assert values["headline"] == "saved" and values["output"] == answer
    assert step_values(None) == {"output": ""}


def test_every_unanswered_reference_is_named_once():
    data = {
        "a": "{{ step_1.nope }}",
        "b": ["{{ step_4.output }}", "and {{ step_1.nope }}"],
        "c": "{{ step_1.variables.nope }}",
        "d": "{{ step_1.version.major }}",
    }
    with pytest.raises(UnresolvedStepReference) as unanswered:
        resolve_step_references(data, FINISHED)
    assert unanswered.value.references == [
        "{{ step_1.nope }}", "{{ step_4.output }}", "{{ step_1.variables.nope }}", "{{ step_1.version.major }}",
    ]
    assert "{{ step_N.output }}" in str(unanswered.value) and "scratchpad_write" in str(unanswered.value)


def test_resolving_hands_out_copies_and_leaves_its_input_alone():
    data = {"card": "{{ step_2.card }}"}
    first = resolve_step_references(data, FINISHED)
    first["card"]["title"] = "changed"
    assert resolve_step_references(data, FINISHED) == {"card": {"title": "T"}}
    assert data == {"card": "{{ step_2.card }}"}


def test_structured_reads_only_a_json_object_or_array():
    assert structured('[1, {"a": 2}]') == [1, {"a": 2}]
    assert structured('```\n{"a": 1}\n```') == {"a": 1}
    for text in ("21", "true", "{not json", "plain words", ""):
        assert structured(text) == text


# ---------------------------------------------------------------------------
# (b) platform_generate_cover_image: one image tool, with or without a post
# ---------------------------------------------------------------------------

IMAGE_ID = "9f1c2d3e-4b5a-4c6d-8e7f-0a1b2c3d4e5f"
IMAGE_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"pixels" * 20).decode("ascii")
IMAGE_MODEL = "image-model-under-test"
BACKEND = "https://api.example"
COST_PATH = {
    "service_name": "blog_cover_gen", "provider": "openrouter", "model": IMAGE_MODEL,
    "workspace_id": str(WS), "request_type": "cover_image",
}


class _ImageModel:
    """create_llm_manager's stand-in: records each manager and each prompt."""

    def __init__(self):
        self.content = f"Here it is: ![image](data:image/png;base64,{IMAGE_B64})"
        self.managers = []
        self.prompts = []

    def create(self, **kwargs):
        self.managers.append(kwargs)
        model = self

        class Manager:
            async def generate_response(self, messages):
                model.prompts.append(messages[0]["content"])
                return SimpleNamespace(content=model.content)

        return Manager()


@pytest.fixture
def image_env(monkeypatch):
    state = SimpleNamespace(model=_ImageModel(), saved=[], registered=[], blog=[])
    monkeypatch.setattr(core.llm, "create_llm_manager", state.model.create)
    monkeypatch.setattr(type(config), "BLOG_COVER_MODEL", IMAGE_MODEL)
    monkeypatch.setattr(config, "BACKEND_URL", BACKEND)

    class Store:
        async def save_image(self, b64, mime_type="image/png", workspace_id=None):
            state.saved.append((b64, mime_type, workspace_id))
            return IMAGE_ID

    class Deliverables:
        def __init__(self, db, workspace_id):
            self.workspace_id = workspace_id

        def register(self, **kwargs):
            state.registered.append({**kwargs, "workspace_id": self.workspace_id})
            return {"success": True, "deliverable_id": "d-image"}

    class Blog:
        def __init__(self, db, workspace_id):
            state.blog.append(("open", workspace_id))

        def get_post(self, post_id):
            state.blog.append(("get", post_id))
            return SimpleNamespace(id=post_id, title="RAG in production", slug="rag-in-production")

        async def update_post(self, post_id, **fields):
            state.blog.append(("update", post_id, fields))
            return SimpleNamespace(id=post_id, title="RAG in production", slug="rag-in-production")

    monkeypatch.setattr(image_store, "get_image_store", lambda: Store())
    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)
    monkeypatch.setattr(blog_service, "BlogService", Blog)
    return state


def _image(db, params):
    return asyncio.run(handlers_blog.generate_cover_image(db, WS, params))


def test_without_a_post_the_image_is_an_image_deliverable_and_no_blog_post_is_touched(image_env):
    db = MagicMock()
    result = _image(db, {
        "prompt": "a lighthouse at dusk", "aspect_ratio": "4:5", "title": "Launch still",
        "_agent_id": 7, "_agent_name": "Brand Designer",
    })

    assert result == {
        "success": True, "deliverable_id": "d-image", "image_id": IMAGE_ID,
        "image_url": f"{BACKEND}/api/generated-images/{IMAGE_ID}", "title": "Launch still", "aspect_ratio": "4:5",
        "message": "Image 'Launch still' (4:5) saved to Deliverables.",
    }
    # No blog post was opened, read or written; nothing else touched the session.
    assert image_env.blog == [] and db.method_calls == []
    assert image_env.saved == [(IMAGE_B64, "image/png", str(WS))]
    (registered,) = image_env.registered
    assert registered == {
        "file_path": f"generated-images/{WS}/{IMAGE_ID}.png",
        "title": "Launch still",
        "source_type": "agent_output",
        "agent_id": 7,
        "agent_name": "Brand Designer",
        "artifact_type": "image",
        "storage_type": "s3",
        "file_type": "png",
        "file_size_bytes": len(base64.b64decode(IMAGE_B64)),
        "preview_url": f"/api/generated-images/{IMAGE_ID}",
        "preview_type": "image",
        "extra": {"image_id": IMAGE_ID, "prompt": "a lighthouse at dusk", "aspect_ratio": "4:5"},
        "workspace_id": WS,
    }
    (prompt,) = image_env.model.prompts
    assert prompt.startswith("Generate a 4:5 image. Image direction: a lighthouse at dusk.")
    assert "no embedded text" in prompt and "blog" not in prompt


def test_an_image_without_a_title_is_named_by_its_prompt(image_env):
    prompt = "a very long direction " * 10
    result = _image(MagicMock(), {"prompt": prompt})
    assert result["aspect_ratio"] == "16:9"
    assert result["title"] == prompt.strip()[: handlers_blog.IMAGE_TITLE_CHARS]


def test_with_a_post_the_cover_is_attached_as_before_and_no_deliverable_is_added(image_env):
    result = _image(MagicMock(), {"post_id": str(POST_ID), "prompt": "abstract data streams"})

    cover_url = f"{BACKEND}/api/generated-images/{IMAGE_ID}"
    assert result == {
        "success": True, "post_id": str(POST_ID), "title": "RAG in production", "slug": "rag-in-production",
        "cover_image_url": cover_url, "image_id": IMAGE_ID,
        "message": "Cover image generated and attached to post 'RAG in production'.",
    }
    assert image_env.blog == [("open", WS), ("get", POST_ID), ("update", POST_ID, {"cover_image_url": cover_url})]
    assert image_env.registered == []
    (prompt,) = image_env.model.prompts
    assert prompt == (
        "Generate a 16:9 cover image for a blog post titled: 'RAG in production'. "
        "Image direction: abstract data streams. "
        "Style: abstract/conceptual, modern and clean, no embedded text "
        "(title overlay handled by CSS). Output the image only."
    )


def test_every_image_takes_the_cover_models_one_cost_path(image_env):
    _image(MagicMock(), {"prompt": "a still", "aspect_ratio": "9:16"})
    _image(MagicMock(), {"post_id": str(POST_ID), "prompt": "a cover"})
    assert image_env.model.managers == [COST_PATH, COST_PATH]


@pytest.mark.parametrize(
    "params, error",
    [
        ({"prompt": "x", "post_id": str(POST_ID), "aspect_ratio": "9:16"}, "A blog cover is 16:9"),
        ({"prompt": "x", "aspect_ratio": "3:2"}, "aspect_ratio must be one of 16:9, 1:1, 4:5, 9:16"),
        ({"aspect_ratio": "1:1"}, "prompt is required"),
        ({"post_id": str(POST_ID), "prompt": "  "}, "prompt is required"),
    ],
)
def test_a_request_the_tool_cannot_make_is_refused_before_the_model_runs(image_env, params, error):
    result = _image(MagicMock(), params)
    assert result["success"] is False and error in result["error"]
    assert image_env.model.prompts == [] and image_env.saved == [] and image_env.blog == []


def test_a_model_answer_without_an_image_saves_and_registers_nothing(image_env):
    image_env.model.content = "Sorry, I can only describe it."
    result = _image(MagicMock(), {"prompt": "x"})
    assert result == {"success": False, "error": "Image model did not return base64 image data — try a clearer prompt"}
    assert image_env.saved == [] and image_env.registered == []


def test_an_image_that_cannot_join_deliverables_says_where_it_was_saved(image_env, monkeypatch):
    class Refusing:
        def __init__(self, db, workspace_id):
            pass

        def register(self, **kwargs):
            return {"success": False, "error": "register failed: boom"}

    monkeypatch.setattr(deliverable_service, "DeliverableService", Refusing)
    result = _image(MagicMock(), {"prompt": "x"})
    assert result["success"] is False and "boom" in result["error"]
    assert result["image_url"] == f"{BACKEND}/api/generated-images/{IMAGE_ID}" and result["image_id"] == IMAGE_ID


def test_the_action_needs_only_a_prompt_and_offers_the_handlers_ratios():
    action = get_action_registry().get("platform_generate_cover_image")
    assert action.parameters["required"] == ["prompt"]
    assert tuple(action.parameters["properties"]["aspect_ratio"]["enum"]) == handlers_blog.IMAGE_ASPECT_RATIOS
    assert "post_id" in action.parameters["properties"] and "title" in action.parameters["properties"]
    assert "social post" in action.description and "Deliverables" in action.description


def test_the_image_key_is_where_the_store_saves_it():
    assert image_store.image_key("i-1", "image/jpeg", "ws-1") == "generated-images/ws-1/i-1.jpg"
    assert image_store.image_key("i-1", "image/unknown") == "generated-images/default/i-1.png"
    assert image_store.generated_image_path("i-1") == "/api/generated-images/i-1"
