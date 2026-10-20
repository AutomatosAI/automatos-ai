"""PRD-256 F430 (A): a CLI agent's sessions are offered the Socials post tools.

Socials day run (14 Oct): the Social Media Director (#348, runtime cli) refused three times,
"platform_create_social_post isn't available in this session". No session spec ran a Socials
action and the group catalogue had no Socials group, so the claim payload, the host's gate,
``tools/list`` and ``tools/call`` all left the posts out, and the prompt told the session every
``platform_*`` tool was unavailable. Now:

* the catalogue's last group is ``socials``, on by default, so #348 gets it with no migration;
* the claim, ``tools/list`` (byte-identical across sessions) and the host's names carry the
  five tools, and an agent whose owner saved a list without Socials is refused in words;
* the Director's skill maps every post tool it names to the session's spelling;
* its persona, in a ticket session, gets the same line (an installed copy keeps its own
  persona), stable per agent;
* neither new module imports anything under ``modules`` at load time.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from services import session_tool_groups as groups
from services import session_tools as st
from services import session_tools_rpc as rpc
from services.cli_session_prompt import session_system_prompt, session_tool_gaps

ORCH = Path(__file__).resolve().parents[1]
POST_TOOLS = ("create_social_post", "update_social_post", "submit_social_post", "get_social_post",
              "list_social_posts")
API_POST_ACTIONS = ("platform_create_social_post", "platform_update_social_post", "platform_submit_social_post",
                    "platform_get_social_post")


def _agent(**config):
    return SimpleNamespace(id=348, name="Social Media Director", configuration={"runtime": "cli", **config})


def _session(agent, task_id):
    return st.SessionContext(task_id=task_id, agent_id=agent.id, agent_name=agent.name, workspace_id="ws-shop",
                             offered=groups.agent_tool_names(agent))


async def _unused(tool, arguments, context):
    raise AssertionError("nothing runs")


def _rpc(ctx, method, params=None):
    message = {"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params else {})}
    return asyncio.run(rpc.handle_message(message, ctx, server_version="1.0", call=_unused))


def test_socials_is_the_last_group_and_on_by_default():
    assert groups.GROUP_IDS[-1] == "socials"
    assert groups.DEFAULT_SESSION_TOOL_GROUPS == groups.GROUP_IDS
    assert groups.group_of("create_social_post").label == "Socials"


def test_a_cli_agent_on_the_default_groups_is_offered_the_post_tools():
    names = groups.agent_tool_names(_agent())
    assert set(POST_TOOLS) <= set(names)
    assert names[-len(POST_TOOLS):] == POST_TOOLS                          # appended last, in the group's order
    # AC1, literally: the API spelling maps to a tool the session has
    assert st.equivalent_of("platform_create_social_post") == "create_social_post"
    assert "create_social_post" in names


def test_tools_list_carries_them_byte_identical_across_sessions():
    agent = _agent()
    first = _rpc(_session(agent, 2001), "tools/list")["result"]["tools"]
    second = _rpc(_session(agent, 2002), "tools/list")["result"]["tools"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [t["name"] for t in first][-len(POST_TOOLS):] == list(POST_TOOLS)


def test_the_claim_carries_the_post_tools(monkeypatch):
    from services import cli_host_service as svc

    agent = _agent()
    task = SimpleNamespace(id=2001, workspace_id="ws-shop", assigned_agent_id=348, title="t", review_mode="auto",
                           attachment_ids=[])
    ref = {"provider": "claude", "model": "sonnet", "cwd": "/tmp/x", "session_id": "s", "attempt": 1}
    monkeypatch.setattr(svc, "_session_system_prompt", lambda a: "")
    payload = svc._claim_payload(task, agent, agent.configuration, ref, "do it", "tok", ("auto", False))
    assert set(POST_TOOLS) <= set(payload["session_tools"])


def test_an_agent_whose_owner_saved_a_list_without_socials_is_refused_in_words():
    agent = _agent(session_tool_groups=["documents"])
    assert not set(POST_TOOLS) & set(groups.agent_tool_names(agent))
    reply = _rpc(_session(agent, 2001), "tools/call", {"name": "create_social_post", "arguments": {"title": "x"}})
    assert reply["result"]["isError"] is True
    assert "Socials tool group" in reply["result"]["content"][0]["text"]


def _director_skill():
    body = (ORCH / "core/seeds/skills/social-video-director.md").read_text(encoding="utf-8")
    return SimpleNamespace(id=1, name="social-video-director", description="", is_active=True,
                           prompt_template=body, content_hash=None, tools_schema=None)


def test_the_directors_skill_maps_every_post_tool_to_the_sessions_spelling():
    agent = SimpleNamespace(**vars(_agent()), skills=[_director_skill()])
    (gap,) = session_tool_gaps(agent, groups.tools_for_groups(groups.DEFAULT_SESSION_TOOL_GROUPS))
    assert not [name for name in gap["tools"] if name.endswith("_social_post")]
    assert {name: gap["instead"][name] for name in API_POST_ACTIONS} == {
        name: name.removeprefix("platform_") for name in API_POST_ACTIONS}


def _director_with_its_persona():
    from core.seeds.seed_socials_package import _DIRECTOR_PERSONA

    return SimpleNamespace(**vars(_agent()), description=None, use_custom_persona=True,
                           custom_persona_prompt=_DIRECTOR_PERSONA, persona=None, skills=[])


def test_the_persona_in_a_ticket_session_says_what_its_tools_are_called_here():
    agent = _director_with_its_persona()
    text = session_system_prompt(agent)
    header = text.index("## Persona & Communication Style")
    assert text.index("`create_social_post` instead of `platform_create_social_post`") > header
    # the ones no session has are named as missing, never swapped
    assert "Not available in a session: `platform_list_deliverables`, `platform_browse_reports`." in text
    assert session_system_prompt(agent) == text                              # stable per agent (prompt cache)


def test_the_operators_own_terminal_gets_the_persona_as_written():
    text = session_system_prompt(_director_with_its_persona(), ticket_session=False)
    assert "instead of `platform_create_social_post`" not in text


def _top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_the_new_modules_import_nothing_under_modules_at_load_time():
    for relative in ("services/session_socials_tools.py", "core/social_post_schema.py", "services/session_tool_scope.py"):
        imported = list(_top_level_imports(ORCH / relative))
        assert not [name for name in imported if name == "modules" or name.startswith("modules.")], relative
