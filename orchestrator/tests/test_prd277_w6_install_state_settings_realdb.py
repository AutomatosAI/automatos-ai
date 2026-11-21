"""PRD-277 US-601 against Postgres: what the install last started as is its own, not a setting anyone edits.

The marker (``services.mobile.edition_change``, category ``MARKER_CATEGORY``) is left out of every list the
System Settings API answers (the list, its categories, by category, the stats), and every route that acts on one
setting answers for it as for a setting that does not exist: read, write and delete are 404 ``Setting not
found``, a bulk update skips it, a reset of its category leaves it, and a new setting in its category is
400. Auto's settings tools neither list it nor change it. Another category's setting is answered as before, for
contrast.

Each test runs in a transaction rolled back at the end; ``commit`` releases a savepoint.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import system_settings as settings_api  # noqa: E402
from core.models.system_settings import SystemSetting, SystemSettingCreate, SystemSettingUpdate  # noqa: E402
from modules.tools.discovery import handlers_workspace as tools  # noqa: E402
from services.mobile import edition_change  # noqa: E402

ADMIN = SimpleNamespace(auth_type="clerk", user=SimpleNamespace(system_role="admin"))
MARKED = "enterprise:pairing_on"
ELSEWHERE = "prd277_w6_contrast"


@pytest.fixture
def db_session(test_engine):
    """A session whose commit releases a savepoint and whose rollback returns to one."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def rows(db_session):
    """The marker (with a default, so a reset would move it) and another category's setting."""
    db_session.execute(text("DELETE FROM system_settings WHERE category = :c"), {"c": edition_change.MARKER_CATEGORY})
    edition_change._store_edition(db_session, MARKED)
    db_session.flush()
    db_session.execute(text("UPDATE system_settings SET default_value = 'local' WHERE category = :c"),
                       {"c": edition_change.MARKER_CATEGORY})
    other = SystemSetting(category=ELSEWHERE, key=f"key-{uuid.uuid4().hex[:8]}", value="before",
                          value_type="string", is_sensitive=False, is_required=False, created_by="system")
    db_session.add(other)
    db_session.commit()
    marker = db_session.query(SystemSetting).filter(SystemSetting.category == edition_change.MARKER_CATEGORY).one()
    return SimpleNamespace(marker=marker.id, other=other.id)


def _marker_value(db) -> str:
    db.expire_all()
    return edition_change.stored_edition(db)


def _not_found(call) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (404, settings_api.SETTING_NOT_FOUND)
    return refused.value


def test_the_lists_leave_the_marker_out(db_session, rows):
    listed = {setting.id for setting in settings_api.list_system_settings(db=db_session, ctx=ADMIN)}
    assert rows.other in listed and rows.marker not in listed
    assert settings_api.list_system_settings(category=edition_change.MARKER_CATEGORY, db=db_session, ctx=ADMIN) == []

    categories = settings_api.list_setting_categories(db=db_session, ctx=ADMIN)
    assert ELSEWHERE in categories and edition_change.MARKER_CATEGORY not in categories

    grouped = {group.category for group in settings_api.get_settings_by_category(db=db_session, ctx=ADMIN)}
    assert ELSEWHERE in grouped and edition_change.MARKER_CATEGORY not in grouped

    stats = settings_api.get_settings_stats(db=db_session, ctx=ADMIN)
    assert ELSEWHERE in stats.categories and edition_change.MARKER_CATEGORY not in stats.categories


def test_the_marker_cannot_be_read_written_or_deleted(db_session, rows):
    _not_found(lambda: settings_api.get_system_setting(setting_id=rows.marker, db=db_session, ctx=ADMIN))
    _not_found(lambda: settings_api.update_system_setting(
        setting_id=rows.marker, update_data=SystemSettingUpdate(value="local"), db=db_session, ctx=ADMIN))
    _not_found(lambda: settings_api.delete_system_setting(setting_id=rows.marker, db=db_session, ctx=ADMIN))

    assert _marker_value(db_session) == MARKED
    assert settings_api.get_system_setting(setting_id=rows.other, db=db_session, ctx=ADMIN).id == rows.other


def test_a_bulk_update_skips_the_marker_as_a_setting_that_does_not_exist(db_session, rows):
    answer = settings_api.bulk_update_settings(
        request=None, db=db_session, ctx=ADMIN,
        updates=[settings_api.BulkUpdateItem(id=rows.marker, value="local"),
                 settings_api.BulkUpdateItem(id=rows.other, value="after")])

    assert answer == {"message": "Updated 1 settings successfully"}
    assert _marker_value(db_session) == MARKED
    assert db_session.get(SystemSetting, rows.other).value == "after"


def test_a_reset_leaves_the_marker(db_session, rows):
    """The marker has a default here, so a reset of its category would move it if the API reached it."""
    answer = settings_api.reset_settings_to_defaults(category=edition_change.MARKER_CATEGORY, db=db_session, ctx=ADMIN)
    assert answer == {"message": "Reset 0 settings to default values"} and _marker_value(db_session) == MARKED


@pytest.mark.parametrize("key", [edition_change.MARKER_KEY, "anything"])
def test_no_setting_is_made_in_the_markers_category(db_session, rows, key):
    with pytest.raises(HTTPException) as refused:
        settings_api.create_system_setting(
            setting_data=SystemSettingCreate(category=edition_change.MARKER_CATEGORY, key=key, value="local"),
            db=db_session, ctx=ADMIN)

    assert (refused.value.status_code, refused.value.detail) == (400, settings_api.RESERVED_CATEGORY_MESSAGE)
    db_session.expire_all()
    assert db_session.query(SystemSetting).filter(
        SystemSetting.category == edition_change.MARKER_CATEGORY).count() == 1


def test_autos_settings_tools_neither_list_nor_change_the_marker(db_session, rows):
    listed = asyncio.run(tools.list_system_settings(db_session, None, {}))
    assert {setting["category"] for setting in listed["settings"]} >= {ELSEWHERE}
    assert edition_change.MARKER_CATEGORY not in {setting["category"] for setting in listed["settings"]}

    changed = asyncio.run(tools.update_system_setting(
        db_session, None, {"category": edition_change.MARKER_CATEGORY, "key": edition_change.MARKER_KEY,
                           "value": "local"}))

    assert changed["success"] is False and _marker_value(db_session) == MARKED
