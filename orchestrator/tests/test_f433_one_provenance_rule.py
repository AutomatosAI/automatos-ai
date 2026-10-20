"""P256-FIX-SOC-2 (F433, socials day 14 Oct): one provenance rule for every post.

Post b142ac9a came in through POST /api/socials/posts with 987/500/300/187 in a Data story's
figures, no ``claim`` keys and no sources, and was approved: the gate counted only the
variables the stored post marked ``claim: true``, and the REST routes stored whatever flags the
caller sent. Now the template decides. On the S0.3b API harness (``tests/test_prd251_api.py``,
its ``document_templates`` table), pinned:

* (a) b142ac9a's body is stored with its four figures marked claims (the names are not), and its
  approval is 422 naming them until the approver overrides, which the history records;
* (b) an explicit ``claim: false`` on a template claim is stored as a claim;
* (c) a PATCH that un-claims a figure is marked again;
* (d) a PATCH that only gives a post the template marks its stored figure;
* (e) a post stored without flags (before this rule) is gated at approval all the same;
* (f) a blank figure claims nothing, and a claim with a url source needs no override;
* (g) a post with no template has no template claims, and the lookup never queries.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import sqlalchemy as sa

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import service, template_claims  # noqa: E402
from tests.test_prd251_api import WS_A, _create, _post  # noqa: E402

api = api_harness.api

# A Data-story-like template: its figures are claims (D7), the items' names are not.
DATA_STORY_SCHEMA = {
    "stat_value": {"type": "text", "label": "Big number", "claim": True},
    "item_1_name": {"type": "text", "label": "Top item 1: name"},
    "item_1_value": {"type": "number", "label": "Top item 1: value", "claim": True},
    "item_2_name": {"type": "text", "label": "Top item 2: name"},
    "item_2_value": {"type": "number", "label": "Top item 2: value", "claim": True},
    "item_3_name": {"type": "text", "label": "Top item 3: name"},
    "item_3_value": {"type": "number", "label": "Top item 3: value", "claim": True},
}
FIGURES = ["item_1_value", "item_2_value", "item_3_value", "stat_value"]
# b142ac9a's variables: plain values, no claim keys.
B142 = {
    "stat_value": {"value": "987"},
    "item_1_name": {"value": "Sourdough"}, "item_1_value": {"value": 500},
    "item_2_name": {"value": "Rye"}, "item_2_value": {"value": 300},
    "item_3_name": {"value": "Spelt"}, "item_3_value": {"value": 187},
}


def claims_template(env, schema=None, ws=WS_A):
    """A social video template of ``ws`` whose ``variables_schema`` is ``schema`` (the Data story's)."""
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks, is_active, version) "
            "VALUES (:id, :ws, 'Data story', 'social_video', '{}', :blocks, 1, 1)"
        ),
        {"id": template_id.hex, "ws": ws.hex, "blocks": json.dumps({"variables_schema": schema or DATA_STORY_SCHEMA})},
    )
    env.session.commit()
    return template_id


def _stored(env, post_id):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(post_id)).variables


def _submitted(env, post_id):
    resp = _post(env, post_id, "submit")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_the_b142ac9a_body_is_marked_and_its_approval_needs_sources_or_the_override(api):
    template_id = claims_template(api)
    post = _create(api, format="video", template_id=str(template_id), variables=B142)

    stored = _stored(api, post["id"])
    assert [name for name, spec in sorted(stored.items()) if spec.get("claim") is True] == FIGURES
    assert stored["item_1_name"] == {"value": "Sourdough"}  # a name is no claim, and is kept as sent
    submitted = _submitted(api, post["id"])
    refused = _post(api, post["id"], "approve", {"content_hash": submitted["content_hash"], "override_unsourced": False})
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["claims"] == FIGURES
    assert "override_unsourced" in refused.json()["detail"]["message"]

    approved = _post(api, post["id"], "approve", {"content_hash": submitted["content_hash"], "override_unsourced": True})
    assert approved.status_code == 200, approved.text
    assert approved.json()["review_log"][-1]["overridden_claims"] == FIGURES


def test_b_an_explicit_false_on_a_template_claim_is_stored_as_a_claim(api):
    template_id = claims_template(api)
    post = _create(api, format="video", template_id=str(template_id),
                   variables={"stat_value": {"value": "987", "claim": False}})
    assert _stored(api, post["id"])["stat_value"] == {"value": "987", "claim": True}


def test_c_a_patch_that_unclaims_a_figure_is_marked_again(api):
    template_id = claims_template(api)
    post = _create(api, format="video", template_id=str(template_id), variables={"stat_value": {"value": "12"}})
    resp = api.client.patch(f"/api/socials/posts/{post['id']}", json={"variables": {"stat_value": {"value": "987", "claim": False}}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["variables"]["stat_value"] == {"value": "987", "claim": True}


def test_d_a_patch_that_only_gives_the_template_marks_the_stored_figure(api):
    template_id = claims_template(api)
    post = _create(api, format="video", variables={"stat_value": {"value": "987"}})
    assert _stored(api, post["id"])["stat_value"] == {"value": "987"}  # no template: nothing to mark
    resp = api.client.patch(f"/api/socials/posts/{post['id']}", json={"template_id": str(template_id)})
    assert resp.status_code == 200, resp.text
    assert _stored(api, post["id"])["stat_value"] == {"value": "987", "claim": True}


def test_e_a_post_stored_without_flags_is_gated_at_approval(api):
    template_id = claims_template(api)
    legacy = service.create_draft(api.session, workspace_id=WS_A, created_by="member-1", title="Legacy",
                                  format="video", template_id=template_id, variables=B142)
    service.submit(legacy, "member-1")
    api.session.commit()
    assert legacy.variables["stat_value"] == {"value": "987"}  # stored as the caller sent it

    resp = _post(api, str(legacy.id), "approve", {"content_hash": legacy.content_hash})

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["claims"] == FIGURES
    api.session.expire_all()
    assert api.session.get(SocialPost, legacy.id).status == "needs_approval"


def test_f_a_blank_figure_claims_nothing_and_a_url_source_needs_no_override(api):
    template_id = claims_template(api)
    post = _create(
        api, format="video", template_id=str(template_id),
        variables={"stat_value": {"value": ""}, "item_1_value": {"value": 500}},
        sources={"item_1_value": {"kind": "url", "ref": "https://example.com/roast-log"}},
    )
    stored = _stored(api, post["id"])
    assert stored["stat_value"] == {"value": ""} and stored["item_1_value"] == {"value": 500, "claim": True}
    submitted = _submitted(api, post["id"])
    resp = _post(api, post["id"], "approve", {"content_hash": submitted["content_hash"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["override_unsourced"] is False and "overridden_claims" not in resp.json()["review_log"][-1]


class _NoQueries:
    def query(self, *args, **kwargs):
        raise AssertionError("a post with no template has no template to read")


def test_g_no_template_means_no_template_claims_and_no_query():
    assert template_claims.names(_NoQueries(), WS_A, None) == frozenset()
