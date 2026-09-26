"""F105-B (night 3) — a chat turn does not hold a pool connection while the model thinks.

Night 3's pool ran dry on chat turns that opened a transaction with their first
read and kept it, with its connection, through every model call of the tool
loop: up to 9 connections "idle in transaction" for 16–38 s. Before each model
call a turn that has only read now ends its transaction; a turn that wrote (or
took a lock) keeps it exactly as before.
"""
from __future__ import annotations

import asyncio
import inspect
import re

import pytest
import sqlalchemy
from sqlalchemy import Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from core.database.read_release import is_plain_read, release_if_read_only


class _Base(DeclarativeBase):
    pass


class _Item(_Base):
    __tablename__ = "f105_turn_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text)


@pytest.fixture
def engine(test_db_url):
    """A pool of two with no overflow: the third holder has to wait (0.5 s), then fails."""
    eng = create_engine(test_db_url, pool_size=2, max_overflow=0, pool_timeout=0.5)
    _Base.metadata.drop_all(eng)
    _Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("INSERT INTO f105_turn_items (id, name) VALUES (1, 'old')"))
    yield eng
    _Base.metadata.drop_all(eng)
    eng.dispose()


def test_a_read_only_turn_gives_its_connection_back_and_rereads_what_it_loaded(engine):
    with Session(engine) as turn:
        item = turn.get(_Item, 1)
        assert item.name == "old" and engine.pool.checkedout() == 1
        assert release_if_read_only(turn) is True
        assert engine.pool.checkedout() == 0                        # nothing held while the model thinks
        with engine.begin() as other:                               # meanwhile the row changes
            other.execute(text("UPDATE f105_turn_items SET name = 'new' WHERE id = 1"))
        assert item.name == "new"                                   # expired, read again on next use
        assert release_if_read_only(turn) is True                   # and given back before the next call


def test_a_turn_that_wrote_keeps_its_transaction_exactly_as_today(engine):
    with Session(engine) as turn:
        turn.add(_Item(id=2, name="draft"))
        turn.flush()
        assert release_if_read_only(turn) is False and turn.in_transaction()
        with engine.connect() as other:
            assert other.execute(text("SELECT count(*) FROM f105_turn_items WHERE id = 2")).scalar() == 0
        turn.commit()
    with engine.connect() as other:
        assert other.execute(text("SELECT name FROM f105_turn_items WHERE id = 2")).scalar() == "draft"


@pytest.mark.parametrize("statement", [
    "INSERT INTO f105_turn_items (id, name) VALUES (3, 'raw')",
    "SELECT id FROM f105_turn_items WHERE id = 1 FOR UPDATE",
    "SELECT pg_advisory_xact_lock(105)",
])
def test_a_raw_write_or_a_lock_counts_as_writing(engine, statement):
    with Session(engine) as turn:
        turn.execute(text(statement))
        assert release_if_read_only(turn) is False and turn.in_transaction()
        turn.rollback()


def test_only_a_plain_select_is_a_read():
    assert is_plain_read("SELECT users.clerk_user_id FROM users WHERE users.id = %(id)s")
    assert is_plain_read("  /* ping */ SELECT 1")
    assert not is_plain_read("SELECT * FROM board_tasks FOR UPDATE SKIP LOCKED")
    assert not is_plain_read("SELECT set_config('app.ws', 'c1', true)")
    assert not is_plain_read("WITH moved AS (DELETE FROM t RETURNING *) SELECT * FROM moved")
    assert not is_plain_read("UPDATE agents SET status = 'active'")


def test_concurrent_read_only_turns_with_slow_model_calls_do_not_run_the_pool_dry(engine):
    """Six turns on a pool of two, each reading then waiting 0.3 s on the model."""
    async def turn(release: bool):
        session = Session(engine)
        try:
            for _ in range(2):
                session.execute(select(_Item)).all()
                if release:
                    release_if_read_only(session)
                await asyncio.sleep(0.3)                            # the model thinking
            session.commit()
        finally:
            session.close()

    async def six(release: bool):
        await asyncio.gather(*(turn(release) for _ in range(6)))

    asyncio.run(six(release=True))
    with pytest.raises(sqlalchemy.exc.TimeoutError):
        asyncio.run(six(release=False))                             # as before: the third turn gives up


# ── the chat turn ───────────────────────────────────────────────────────────

def test_the_chat_turn_releases_before_its_model_calls_unless_the_dial_is_off(monkeypatch):
    from consumers.chatbot import service as chat

    released = []
    monkeypatch.setattr(chat, "release_if_read_only", lambda db: released.append(db) or True)
    for dial, per_call in ((True, 1), (False, 0)):
        monkeypatch.setattr(type(chat.config), "CHATBOT_RELEASE_DB_BETWEEN_MODEL_CALLS",
                            property(lambda self, on=dial: on))
        svc = chat.StreamingChatService.__new__(chat.StreamingChatService)
        svc.db = object()
        released.clear()
        svc._before_model_call()
        svc._before_model_call()
        assert released == [svc.db] * (2 * per_call)


def test_every_model_call_in_the_chat_turn_releases_first():
    from consumers.chatbot import service as chat

    source = inspect.getsource(chat.StreamingChatService)
    calls = [m.start() for m in re.finditer(r"generate_response\(", source)]
    assert len(calls) >= 6       # the scan found them (the dead stream_response held two more)
    for pos in calls:
        window = "\n".join(source[:pos].splitlines()[-3:])
        assert "self._before_model_call()" in window, window
