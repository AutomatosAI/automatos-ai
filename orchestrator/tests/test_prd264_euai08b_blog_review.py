"""PRD-264 EUAI-08b — agent-written blog posts are labelled AI-generated until a person reviews them.

Article 50(4) ¶2 of the AI Act: AI-written text published to inform the public is
disclosed unless a person reviewed it and holds editorial responsibility.

* **The flag.** ``ai_generated`` is true when an agent wrote the post
  (``author_agent_id`` set) and no review covers its current text. Both blog APIs,
  the dashboard's and the public widget's, return it with ``reviewed_at``; who
  reviewed it, and the reviewed text's hash, stay out of the public answer.
* **Marking.** ``POST /api/blog/posts/{post_id}/review`` (``documents:update``)
  records who, when and the SHA-256 of the text, for a post in the caller's own
  workspace only.
* **Editing.** An update that changes the post clears the review; publishing, or
  saving without a change, keeps it.
* **The file.** The text lives in a workspace file an agent's file tools can rewrite
  without ``update_post``. A review counts only while the current text still hashes
  to the reviewed value, so a rewritten file labels the post AI-generated again.
* **The migration.** One revision, ``prd264_blog_review``, adds the three nullable
  columns with ``IF NOT EXISTS``; both head pins name it.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import operator
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
MIGRATION = ORCH / "alembic" / "versions" / "prd264_blog_review.py"
ROUTE = "/api/blog/posts/{post_id}/review"
WS = UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e08")
OTHER_WS = UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e09")
AGENT_ID = 12
REVIEWED = datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc)
TEXT = "Roasted on Thursday. Rested for a week."
TEXT_SHA256 = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
REWRITTEN = TEXT + " Now 20% off, this week only."  # what an agent's file tool wrote over it
POST_FILE = "content/blog/harbour-blend.md"


# --------------------------------------------------------------------------- fakes


def _post(workspace_id=WS, *, agent=AGENT_ID, reviewed_at=None, **fields):
    """A transient BlogPost: real columns and ``to_dict``, never saved. Reviewed means reviewed over TEXT."""
    from core.models import BlogPost

    values = dict(
        id=uuid4(), workspace_id=workspace_id, author_agent_id=agent, author_name="Harbour Roasters",
        title="Harbour blend", slug="harbour-blend", excerpt="Roasted on Thursday.", content=TEXT,
        file_path=None, tags=["coffee"], category="Roasting", status="published",
        reviewed_by_user_id="user_ana" if reviewed_at else None, reviewed_at=reviewed_at,
        reviewed_content_sha256=TEXT_SHA256 if reviewed_at else None,
    )
    return BlogPost(**{**values, **fields})


def _holds(row, criterion) -> bool:
    """One ``Column == value`` criterion, evaluated on a row as Postgres would."""
    assert criterion.operator is operator.eq, criterion
    return getattr(row, criterion.left.name) == criterion.right.value


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *criteria):
        return _Query([row for row in self.rows if all(_holds(row, c) for c in criteria)])

    def first(self):
        return self.rows[0] if self.rows else None


class _Db:
    """The session calls BlogService makes for one post: query/filter/first, commit, refresh."""

    def __init__(self, *rows):
        self.rows = list(rows)
        self.commits = 0

    def query(self, model):
        return _Query(self.rows)

    def commit(self):
        self.commits += 1

    def refresh(self, row):
        return None


def _service(post, workspace_id=WS, stored_text=TEXT):
    """A real BlogService over ``_Db``; the post's text lives in a fake workspace file."""
    from core.services.blog_service import BlogService

    service = BlogService(_Db(post), workspace_id)
    service.written = []

    async def write(path, content):
        service.written.append((path, content))
        return True

    async def read(_post):
        return stored_text

    service._write_content = write
    service.get_content = read
    return service


class _WorkspaceFiles:
    """The workspace file store behind ``WorkspaceClient``: BlogService reads and writes it,
    and an agent's file-write tool rewrites it directly, without ``update_post``."""

    files: dict = {}

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def read_file(self, path):
        if path not in self.files:
            return {"success": False, "error": "not found"}
        return {"success": True, "content": self.files[path]}

    async def write_file(self, path, content):
        self.files[path] = content
        return {"success": True}


def _ctx(workspace_id=WS, user_id="user_ana", email=None):
    return NS(workspace_id=workspace_id, user=NS(id=user_id, email=email), auth_type="clerk")


def _mark(post_id, db, ctx):
    import api.blog as blog

    return asyncio.run(blog.mark_post_reviewed(post_id, db=db, ctx=ctx))


# --------------------------------------------------------------------------- the flag


def test_the_flag_in_each_state():
    from core.services.blog_post_view import is_ai_generated, post_view

    unreviewed, reviewed, persons = _post(), _post(reviewed_at=REVIEWED), _post(agent=None)

    assert is_ai_generated(unreviewed, TEXT) is True
    assert is_ai_generated(reviewed, TEXT) is False
    assert is_ai_generated(reviewed, REWRITTEN) is True  # the review covers TEXT, not this
    assert is_ai_generated(reviewed) is True  # text not read: no review can be shown to cover it
    assert is_ai_generated(persons, TEXT) is False
    assert is_ai_generated(_post(agent=None, reviewed_at=REVIEWED), REWRITTEN) is False

    assert (post_view(unreviewed)["ai_generated"], post_view(unreviewed)["reviewed_at"]) == (True, None)
    view = post_view(reviewed, current_text=TEXT)
    assert (view["ai_generated"], view["reviewed_at"]) == (False, REVIEWED.isoformat())
    stale = post_view(reviewed, current_text=REWRITTEN)
    assert (stale["ai_generated"], stale["reviewed_at"]) == (True, None)
    assert post_view(persons)["ai_generated"] is False


def test_a_review_covers_the_text_that_was_reviewed_and_nothing_else():
    from core.services.blog_review import content_sha256, review_covers

    assert content_sha256(TEXT) == TEXT_SHA256
    assert content_sha256("  " + TEXT + "\n") == TEXT_SHA256  # the editor trims what it sends
    assert content_sha256(None) == content_sha256("")

    reviewed = _post(reviewed_at=REVIEWED)
    assert review_covers(reviewed, TEXT) is True
    assert review_covers(reviewed, REWRITTEN) is False
    assert review_covers(reviewed, None) is False
    assert review_covers(_post(reviewed_at=REVIEWED, reviewed_content_sha256=None), TEXT) is False
    assert review_covers(_post(), TEXT) is False


def test_who_reviewed_a_post_and_the_texts_hash_stay_out_of_the_view():
    """The widget's answer is public: it says when a post was reviewed, never by whom."""
    from core.services.blog_post_view import post_view

    view = post_view(_post(reviewed_at=REVIEWED), include_content=True, current_text=TEXT)
    assert "reviewed_by_user_id" not in view and "reviewed_content_sha256" not in view
    dumped = json.dumps(view, default=str)
    assert "user_ana" not in dumped and TEXT_SHA256 not in dumped


class _WidgetPosts:
    rows: list = []
    text: str = TEXT

    def __init__(self, db, workspace_id):
        pass

    def list_posts(self, **filters):
        return {"posts": self.rows, "total": len(self.rows), "page": 1, "per_page": 10, "total_pages": 1}

    def get_post_by_slug(self, slug):
        return next(row for row in self.rows if row.slug == slug)

    def increment_views(self, post_id):
        return None

    async def get_content(self, post):
        return self.text


def test_the_public_widget_labels_only_the_unreviewed_agent_post(monkeypatch):
    import api.widgets.blog as widget_blog

    rows = [_post(slug="unreviewed"), _post(slug="reviewed", reviewed_at=REVIEWED), _post(slug="persons", agent=None)]
    monkeypatch.setattr(_WidgetPosts, "rows", rows)
    monkeypatch.setattr(widget_blog, "BlogService", _WidgetPosts)

    listed = asyncio.run(widget_blog.list_published_posts(page=1, per_page=10, category=None, tag=None,
                                                          workspace_id=WS, db=object()))
    posts = json.loads(listed.body)["posts"]
    assert [(p["slug"], p["ai_generated"]) for p in posts] == [("unreviewed", True), ("reviewed", False),
                                                                ("persons", False)]
    assert [p["reviewed_at"] for p in posts] == [None, REVIEWED.isoformat(), None]
    assert all("reviewed_by_user_id" not in p and "reviewed_content_sha256" not in p for p in posts)

    one = json.loads(asyncio.run(widget_blog.get_published_post("reviewed", workspace_id=WS, db=object())).body)
    assert (one["ai_generated"], one["reviewed_at"]) == (False, REVIEWED.isoformat())


# --------------------------------------------------------------------------- marking


def test_a_member_marks_a_post_reviewed_and_the_label_comes_off():
    post = _post()
    db = _Db(post)
    before = datetime.now(timezone.utc)

    data = _mark(post.id, db, _ctx())

    assert post.reviewed_by_user_id == "user_ana" and before <= post.reviewed_at <= datetime.now(timezone.utc)
    assert post.reviewed_content_sha256 == TEXT_SHA256  # the text the reviewer read (the post's own column here)
    assert db.commits == 1
    assert data["ai_generated"] is False and data["reviewed_at"] == post.reviewed_at.isoformat()


def test_the_local_operator_is_recorded_by_email_and_no_one_is_a_401():
    post = _post()
    _mark(post.id, _Db(post), _ctx(user_id=None, email="owner@harbour.example"))
    assert post.reviewed_by_user_id == "owner@harbour.example"

    untouched = _post()
    with pytest.raises(HTTPException) as refused:
        _mark(untouched.id, _Db(untouched), _ctx(user_id=None, email=None))
    assert refused.value.status_code == 401
    assert (untouched.reviewed_at, untouched.reviewed_content_sha256) == (None, None)


def test_another_workspaces_post_cannot_be_marked():
    theirs = _post(OTHER_WS)
    db = _Db(theirs)

    with pytest.raises(HTTPException) as missing:
        _mark(theirs.id, db, _ctx(WS))

    assert missing.value.status_code == 404
    assert (theirs.reviewed_by_user_id, theirs.reviewed_at, theirs.reviewed_content_sha256, db.commits) == (
        None, None, None, 0)


def test_the_route_is_gated_for_editors_and_listed_in_the_manifest():
    import api.blog as blog

    source = (ORCH / "api" / "blog.py").read_text()
    assert re.search(r'@router\.post\("/posts/\{post_id\}/review", dependencies=\[Depends\('
                     r'require_workspace_permission\("documents:update"\)\)\]\)\nasync def mark_post_reviewed\(',
                     source)
    params = blog.mark_post_reviewed.__code__.co_varnames[:blog.mark_post_reviewed.__code__.co_argcount]
    assert params == ("post_id", "db", "ctx")  # ctx: get_request_context_hybrid, the workspace it scopes to

    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text())
    assert {"method": "POST", "path": ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


# --------------------------------------------------------------------------- the file under the post


def test_an_agent_rewriting_the_post_file_outside_update_post_labels_it_again(monkeypatch):
    """The text lives in content/blog/<slug>.md; an agent's file tool can rewrite it without
    update_post, which leaves reviewed_at set. The review no longer covers the text, so the
    dashboard says AI-generated again and shows no review date."""
    import api.blog as blog
    import core.services.blog_service as blog_service

    monkeypatch.setattr(_WorkspaceFiles, "files", {POST_FILE: TEXT})
    monkeypatch.setattr(blog_service, "WorkspaceClient", _WorkspaceFiles)
    post = _post(file_path=POST_FILE, content=TEXT[:10])  # the row keeps only a truncated fallback
    db = _Db(post)

    marked = _mark(post.id, db, _ctx())
    assert marked["ai_generated"] is False and post.reviewed_content_sha256 == TEXT_SHA256

    _WorkspaceFiles.files[POST_FILE] = REWRITTEN  # the agent's file-write tool; update_post never runs

    data = asyncio.run(blog.get_post(post.id, db=db, ctx=_ctx()))
    assert (data["ai_generated"], data["reviewed_at"], data["content"]) == (True, None, REWRITTEN)
    assert post.reviewed_at is not None  # the row still says reviewed: only the hash disagrees

    _WorkspaceFiles.files[POST_FILE] = TEXT + "\n"  # put back as reviewed (whitespace aside)
    again = asyncio.run(blog.get_post(post.id, db=db, ctx=_ctx()))
    assert (again["ai_generated"], again["reviewed_at"]) == (False, post.reviewed_at.isoformat())


def test_both_blog_apis_label_a_reviewed_post_whose_file_changed(monkeypatch):
    import api.blog as blog
    import api.widgets.blog as widget_blog

    rows = [_post(slug="reviewed", reviewed_at=REVIEWED), _post(slug="persons", agent=None, reviewed_at=REVIEWED)]
    monkeypatch.setattr(_WidgetPosts, "rows", rows)
    monkeypatch.setattr(_WidgetPosts, "text", REWRITTEN)
    monkeypatch.setattr(widget_blog, "BlogService", _WidgetPosts)
    monkeypatch.setattr(blog, "BlogService", _WidgetPosts)

    listed = asyncio.run(widget_blog.list_published_posts(page=1, per_page=10, category=None, tag=None,
                                                          workspace_id=WS, db=object()))
    widget_posts = json.loads(listed.body)["posts"]
    assert [(p["slug"], p["ai_generated"], p["reviewed_at"]) for p in widget_posts] == [
        ("reviewed", True, None), ("persons", False, None)]

    one = json.loads(asyncio.run(widget_blog.get_published_post("reviewed", workspace_id=WS, db=object())).body)
    assert (one["ai_generated"], one["reviewed_at"]) == (True, None)

    dashboard = asyncio.run(blog.list_posts(status=None, category=None, tag=None, page=1, per_page=20,
                                            db=object(), ctx=_ctx()))
    assert [(p["slug"], p["ai_generated"]) for p in dashboard["posts"]] == [("reviewed", True), ("persons", False)]


def test_a_list_reads_the_text_only_of_reviewed_posts():
    """An unreviewed post's flag needs no file read, so a list of them reads none."""
    from core.services.blog_post_view import read_post_view

    reads = []

    class Service:
        async def get_content(self, post):
            reads.append(post.slug)
            return TEXT

    posts = [_post(slug="unreviewed"), _post(slug="reviewed", reviewed_at=REVIEWED), _post(slug="persons", agent=None)]
    views = [asyncio.run(read_post_view(Service(), p)) for p in posts]
    assert reads == ["reviewed"]
    assert [v["ai_generated"] for v in views] == [True, False, False]


# --------------------------------------------------------------------------- editing


def _update(post, **changes):
    service = _service(post)
    updated = asyncio.run(service.update_post(post.id, **changes))
    return updated, service


def test_editing_the_text_clears_the_review():
    from core.services.blog_post_view import post_view

    post = _post(reviewed_at=REVIEWED)
    updated, service = _update(post, content=TEXT + " Now with tasting notes.")

    assert updated is post
    assert (post.reviewed_by_user_id, post.reviewed_at, post.reviewed_content_sha256) == (None, None, None)
    assert post_view(post, current_text=service.written[0][1])["ai_generated"] is True
    assert service.written and service.written[0][1].endswith("tasting notes.")


@pytest.mark.parametrize("changes", [
    {"title": "Harbour blend, second roast"},
    {"excerpt": "A new summary."},
    {"tags": ["coffee", "espresso"]},
    {"cover_image_url": "https://cdn.harbour.example/cover.png"},
])
def test_editing_any_other_field_clears_it_too(changes):
    post = _post(reviewed_at=REVIEWED)
    _update(post, **changes)
    assert (post.reviewed_by_user_id, post.reviewed_at, post.reviewed_content_sha256) == (None, None, None)


def test_saving_without_a_change_or_publishing_keeps_the_review():
    """The editor sends every field on Save, trimmed; then Publish. Neither undoes the review,
    and the review still covers the text the save wrote."""
    from core.services.blog_review import review_covers

    post = _post(reviewed_at=REVIEWED)
    _, service = _update(post, title="Harbour blend", content=TEXT + "\n", excerpt="Roasted on Thursday.",
                         tags=["coffee"], category="Roasting")
    assert (post.reviewed_by_user_id, post.reviewed_at, post.reviewed_content_sha256) == (
        "user_ana", REVIEWED, TEXT_SHA256)
    assert review_covers(post, service.written[0][1])

    _update(post, status="draft")
    assert (post.reviewed_by_user_id, post.reviewed_at, post.reviewed_content_sha256) == (
        "user_ana", REVIEWED, TEXT_SHA256)


def test_an_unreviewed_post_edited_stays_unreviewed():
    post = _post()
    _update(post, content="Fresh text.")
    assert (post.reviewed_by_user_id, post.reviewed_at, post.reviewed_content_sha256) == (None, None, None)


def test_the_edit_rule_compares_text_without_its_whitespace():
    from core.services.blog_review import NOT_AN_EDIT, REVIEW_CLEARED, edits_change_the_post

    post = _post(excerpt=None, category=None, tags=[])
    assert edits_change_the_post(post, {"excerpt": "", "category": None, "tags": None}, None) is False
    assert edits_change_the_post(post, {"content": "  same  "}, "same") is False
    assert edits_change_the_post(post, {"content": "other"}, "same") is True
    assert edits_change_the_post(post, {"status": "archived"}, None) is False
    assert NOT_AN_EDIT == frozenset({"status"})
    assert dict(REVIEW_CLEARED) == {"reviewed_by_user_id": None, "reviewed_at": None, "reviewed_content_sha256": None}
    with pytest.raises(TypeError):
        REVIEW_CLEARED["reviewed_at"] = REVIEWED  # read-only: no caller can change what clearing writes


# --------------------------------------------------------------------------- the migration


def test_the_model_and_the_migration_add_three_nullable_columns():
    from core.models import BlogPost

    columns = BlogPost.__table__.columns
    assert columns["reviewed_by_user_id"].nullable and columns["reviewed_at"].nullable
    assert columns["reviewed_content_sha256"].nullable and columns["reviewed_content_sha256"].type.length == 64
    assert columns["reviewed_at"].type.timezone is True

    src = MIGRATION.read_text()
    assert 'revision = "prd264_blog_review"' in src
    assert 'down_revision = "prd264_content_provenance"' in src
    assert "ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS reviewed_by_user_id VARCHAR(255);" in src
    assert "ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;" in src
    assert "ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS reviewed_content_sha256 VARCHAR(64);" in src
    assert "ALTER TABLE blog_posts DROP COLUMN IF EXISTS reviewed_content_sha256;" in src
    assert "ALTER TABLE blog_posts DROP COLUMN IF EXISTS reviewed_at;" in src
    assert "ALTER TABLE blog_posts DROP COLUMN IF EXISTS reviewed_by_user_id;" in src


def test_both_head_pins_name_the_revision():
    guard = (ORCH / "tests" / "test_prd209_alembic_single_head.py").read_text()
    chain = (ORCH / "tests" / "test_prd236_w1_routes.py").read_text()
    assert 'EXPECTED_HEAD = "prd264_blog_review"' in guard
    assert 'assert \'EXPECTED_HEAD = "prd264_blog_review"\' in guard' in chain
    assert '(versions / "prd264_blog_review.py")' in chain


def _upgrade_statements():
    spec = importlib.util.spec_from_file_location("prd264_blog_review_under_test", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements = []
    migration.op = NS(execute=statements.append)
    migration.upgrade()
    return statements


def _review_columns(db):
    return db.execute(text(
        "SELECT column_name, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'blog_posts' "
        "AND column_name IN ('reviewed_by_user_id', 'reviewed_at', 'reviewed_content_sha256') "
        "ORDER BY column_name")).all()


def test_the_upgrade_runs_twice_over_a_create_all_table_and_restores_missing_columns(db_session):
    """create_all built blog_posts with the columns; the upgrade is a no-op over them and puts back
    a dropped one. Everything runs inside db_session's transaction and is rolled back."""
    expected = [("reviewed_at", "YES"), ("reviewed_by_user_id", "YES"), ("reviewed_content_sha256", "YES")]
    assert [tuple(row) for row in _review_columns(db_session)] == expected
    for statement in _upgrade_statements():
        db_session.execute(text(statement))

    db_session.execute(text("ALTER TABLE blog_posts DROP COLUMN reviewed_at"))
    db_session.execute(text("ALTER TABLE blog_posts DROP COLUMN reviewed_content_sha256"))
    for statement in _upgrade_statements():
        db_session.execute(text(statement))
    assert [tuple(row) for row in _review_columns(db_session)] == expected


def test_on_the_real_table_only_the_owning_workspace_can_mark_its_post(db_session, seed_workspace):
    from core.models import BlogPost

    ours, theirs = UUID(seed_workspace()), UUID(seed_workspace())
    post = BlogPost(workspace_id=theirs, author_name="Harbour Roasters", title="Harbour blend",
                    slug=f"harbour-blend-{uuid4().hex[:8]}", content=TEXT, status="published", tags=[])
    db_session.add(post)
    db_session.flush()

    with pytest.raises(HTTPException) as missing:
        _mark(post.id, db_session, _ctx(ours, user_id="user_ben"))
    assert missing.value.status_code == 404
    db_session.refresh(post)
    assert (post.reviewed_by_user_id, post.reviewed_at, post.reviewed_content_sha256) == (None, None, None)

    data = _mark(post.id, db_session, _ctx(theirs, user_id="user_ana"))
    db_session.refresh(post)
    assert post.reviewed_by_user_id == "user_ana" and post.reviewed_at is not None
    assert post.reviewed_content_sha256 == TEXT_SHA256
    assert data["reviewed_at"] is not None
