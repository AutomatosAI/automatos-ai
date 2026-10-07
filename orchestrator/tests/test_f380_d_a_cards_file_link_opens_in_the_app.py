"""F380 (night 11, 7 Oct): a ticket's answer links a generated file by its Deliverable.

#2160's answer linked its card by the file's object-storage address
(``localhost:9000/…/generated-documents/…png``), which answered "Access Denied". Before a
ticket closes, each storage address of a generated file in its answer, signed or not,
becomes the owner's link: the Deliverables page opened on that file's Deliverable, or the
Deliverables page when the file has none. The app's own file route is left alone.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from modules.tools.execution.generate_document_tool import deliverable_open_url
from services import result_document_links as links

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
CARD = "20261007_011012_Top_3_Cafes.png"
DELIVERABLE = "515bb59b-0000-4000-8000-000000000001"
SIGNED = (f"http://localhost:9000/automatos-ai/workspaces/{WS}/generated-documents/{CARD}"
          "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=604800&X-Amz-Signature=78e47a4d")
BARE = f"localhost:9000/automatos-ai/workspaces/{WS}/generated-documents/{CARD}"


def _db(found=True):
    def execute(_statement, params):
        assert params == {"ws": WS, "name": CARD}                         # scoped to the workspace
        return NS(fetchone=lambda: (DELIVERABLE,) if found else None)
    return NS(execute=execute)


def test_storage_addresses_are_found_signed_or_not():
    answer = f"The card is waiting: {BARE}. Signed copy: {SIGNED}"

    assert links.storage_links(answer) == {BARE: CARD, SIGNED: CARD}


def test_the_apps_own_links_are_not_storage_addresses():
    answer = (f"Open it here: /api/documents/generated/{CARD} or "
              f"{deliverable_open_url(DELIVERABLE)}")

    assert links.storage_links(answer) == {}


def test_2160s_link_becomes_the_owners_link_to_its_deliverable():
    answer = f"This draft is ready ({BARE}). Or: {SIGNED}"

    linked = links.with_owners_links(_db(), WS, answer)

    assert "generated-documents" not in linked
    assert linked == f"This draft is ready ({deliverable_open_url(DELIVERABLE)}). Or: {deliverable_open_url(DELIVERABLE)}"


def test_a_file_with_no_deliverable_gets_the_deliverables_page():
    linked = links.with_owners_links(_db(found=False), WS, f"See {BARE}")

    assert linked == f"See {deliverable_open_url(None)}"


def test_the_writers_keywords_carry_the_owners_link_and_nothing_else_changes():
    kwargs = {"task_id": 2160, "workspace_id": WS, "agent_id": 348,
              "exec_result": {"status": "success", "result": f"See {BARE}"}}

    linked = links._linked_kwargs(_db(), kwargs)

    assert linked["exec_result"] == {"status": "success", "result": f"See {deliverable_open_url(DELIVERABLE)}"}
    assert kwargs["exec_result"]["result"] == f"See {BARE}"              # the caller's dict is not changed
    plain = {**kwargs, "exec_result": {"status": "success", "result": "The card is in Deliverables."}}
    assert links._linked_kwargs(_db(), plain) is plain


def test_a_lookup_that_fails_leaves_the_answer_as_it_was():
    def broken(*_a, **_k):
        raise RuntimeError("database gone")

    kwargs = {"task_id": 2160, "workspace_id": WS, "exec_result": {"status": "success", "result": f"See {BARE}"}}

    assert links._linked_kwargs(NS(execute=broken), kwargs) is kwargs
