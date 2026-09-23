"""F105-B (night 3) — an API agent run does not hold a pool connection while the model thinks.

Night 3's pool dumps showed agent runs "idle in transaction" for 55–57 s, their
last statement the agent row: the run read, then called the model again and
again on the same open transaction. Every model call of an agent run now goes
through ``AgentFactory._call_model``, which first ends a transaction that has
only read (the same rule and dial as the chat turn).
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace as NS

import pytest
import sqlalchemy
from sqlalchemy import Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from core.llm.clients.base import LLMResponse
from modules.agents.factory.agent_factory import AgentFactory


class _Base(DeclarativeBase):
    pass


class _Agent(_Base):
    __tablename__ = "f105_run_agents"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)


@pytest.fixture
def engine(test_db_url):
    """A pool of two with no overflow: the third holder waits 0.5 s, then fails."""
    eng = create_engine(test_db_url, pool_size=2, max_overflow=0, pool_timeout=0.5)
    _Base.metadata.drop_all(eng)
    _Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO f105_run_agents (id, name) VALUES (294, 'Auto')"))
    yield eng
    _Base.metadata.drop_all(eng)
    eng.dispose()


class _SlowModel:
    """A model that takes 0.3 s per call."""

    def __init__(self):
        self.calls = 0

    async def generate_response(self, messages, tools=None):
        self.calls += 1
        await asyncio.sleep(0.3)
        return LLMResponse(content="done", tool_calls=None)


def _factory(session, release):
    factory = AgentFactory(db_session=session)
    factory._release_db_dial = release
    return factory


def test_six_concurrent_agent_runs_with_a_slow_model_do_not_run_the_pool_dry(engine):
    model = _SlowModel()
    runtime = NS(llm_manager=model)

    async def run(release: bool):
        session = Session(engine)
        factory = _factory(session, release)
        try:
            for _ in range(2):
                session.execute(select(_Agent)).all()                  # the run reads its agent, tools…
                await factory._call_model(runtime, [{"role": "user", "content": "go"}])
            session.commit()
        finally:
            session.close()

    async def six(release: bool):
        await asyncio.gather(*(run(release) for _ in range(6)))

    asyncio.run(six(release=True))
    assert model.calls == 12
    with pytest.raises(sqlalchemy.exc.TimeoutError):
        asyncio.run(six(release=False))                                 # as before: the third run gives up


def test_a_run_that_wrote_keeps_its_transaction(engine):
    runtime = NS(llm_manager=_SlowModel())
    with Session(engine) as session:
        session.add(_Agent(id=295, name="Analyst"))
        session.flush()
        asyncio.run(_factory(session, True)._call_model(runtime, [{"role": "user", "content": "go"}]))
        assert session.in_transaction() and engine.pool.checkedout() == 1
        session.rollback()


def test_every_model_call_of_an_agent_run_goes_through_the_seam():
    source = inspect.getsource(AgentFactory)
    body_calls = source.count("llm_manager.generate_response(")
    # _call_model itself, plus the connection check that runs outside any run
    assert body_calls == 2, body_calls
    assert source.count("self._call_model(agent_runtime") == 3
