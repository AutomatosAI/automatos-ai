"""Unit tests for IdentitySection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.context.sections.base import SectionContext
from modules.context.sections.identity import IdentitySection


@pytest.fixture
def identity():
    return IdentitySection()


def _ctx(agent=None, workspace_id="ws_1", workspace_name="TestWS", **kwargs):
    """Build a minimal SectionContext."""
    if agent is None:
        agent = SimpleNamespace(
            id=1,
            name="Test Agent",
            agent_type="assistant",
            description=None,
            use_custom_persona=False,
            custom_persona_prompt=None,
            persona=None,
        )
    return SectionContext(
        agent=agent,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        kwargs=kwargs,
    )


class TestIdentitySectionBasic:
    """Tests for non-personality (task_execution, heartbeat, etc.) identity."""

    @pytest.mark.asyncio
    async def test_renders_agent_name_and_role(self, identity):
        ctx = _ctx()
        result = await identity.render(ctx)

        assert "Test Agent" in result
        assert "assistant" in result
        assert "TestWS" in result

    @pytest.mark.asyncio
    async def test_renders_workspace_id_when_no_name(self, identity):
        ctx = _ctx(workspace_name=None, workspace_id="ws_fallback")
        result = await identity.render(ctx)

        assert "ws_fallback" in result

    @pytest.mark.asyncio
    async def test_includes_description_when_present(self, identity):
        agent = SimpleNamespace(
            id=2,
            name="Desc Agent",
            agent_type="researcher",
            description="I research things deeply",
            use_custom_persona=False,
            custom_persona_prompt=None,
            persona=None,
        )
        ctx = _ctx(agent=agent)
        result = await identity.render(ctx)

        assert "I research things deeply" in result

    @pytest.mark.asyncio
    async def test_no_description_when_none(self, identity):
        ctx = _ctx()
        result = await identity.render(ctx)

        # Should not contain stray "None" text
        assert "None" not in result

    @pytest.mark.asyncio
    async def test_includes_custom_persona(self, identity):
        agent = SimpleNamespace(
            id=3,
            name="Persona Agent",
            agent_type="writer",
            description=None,
            use_custom_persona=True,
            custom_persona_prompt="I am a creative writing specialist.",
            persona=None,
        )
        ctx = _ctx(agent=agent)
        result = await identity.render(ctx)

        assert "creative writing specialist" in result

    @pytest.mark.asyncio
    async def test_includes_db_persona(self, identity):
        persona_obj = SimpleNamespace(system_prompt="Be extremely concise.")
        agent = SimpleNamespace(
            id=4,
            name="DB Persona Agent",
            agent_type="ops",
            description=None,
            use_custom_persona=False,
            custom_persona_prompt=None,
            persona=persona_obj,
        )
        ctx = _ctx(agent=agent)
        result = await identity.render(ctx)

        assert "extremely concise" in result


class TestIdentitySectionGracefulFallback:
    """Tests for graceful fallback on missing/broken agent data."""

    @pytest.mark.asyncio
    async def test_none_agent(self, identity):
        ctx = _ctx(agent=None)
        result = await identity.render(ctx)

        # Should still produce something, not crash
        assert "Agent" in result

    @pytest.mark.asyncio
    async def test_agent_without_name_attr(self, identity):
        """Agent missing 'name' attribute uses fallback."""
        agent = SimpleNamespace(id=99)
        ctx = _ctx(agent=agent)
        result = await identity.render(ctx)

        assert "Agent" in result  # default fallback

    @pytest.mark.asyncio
    async def test_persona_raises_exception(self, identity):
        """If persona loading blows up, identity still renders."""
        agent = MagicMock()
        agent.name = "Broken Persona Agent"
        agent.agent_type = "tester"
        agent.description = None
        agent.use_custom_persona = True
        # Accessing custom_persona_prompt raises
        type(agent).custom_persona_prompt = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        ctx = _ctx(agent=agent)
        result = await identity.render(ctx)

        # Should still have basic identity
        assert "Broken Persona Agent" in result


class TestIdentitySectionAttributes:
    """Tests for section metadata."""

    def test_name(self, identity):
        assert identity.name == "identity"

    def test_priority(self, identity):
        assert identity.priority == 1

    def test_max_tokens(self, identity):
        assert identity.max_tokens == 500
