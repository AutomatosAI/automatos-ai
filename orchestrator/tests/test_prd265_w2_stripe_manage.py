"""PRD-265 W2 v2 — managing a subscription from inside Studio (``services/billing/stripe_manage.py``).

There is no portal: the Billing tab calls these directly through
``StripeProvider`` (which mixes in :class:`StripeManagement`). Every id that
goes into a URL path is checked against its shape (``checked``) before any
HTTP call, so a malformed id never reaches the network.
"""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qsl

import httpx
import pytest

from services.billing.provider import CardOnFile, InvoiceLine, PlanChangePreview, PlanChangeResult
from services.billing.stripe_api import StripeApi
from services.billing.stripe_manage import checked
from services.billing.stripe_provider import StripeProvider

CUSTOMER = "cus_1"
SUBSCRIPTION = "sub_1"


def _provider(handler) -> StripeProvider:
    api = StripeApi("sk_test_1", api_version="2026-09-30.endive", timeout=5.0,
                    transport=httpx.MockTransport(handler), sleep=lambda s: None)
    return StripeProvider(api, webhook_secret="whsec_t1", tolerance_seconds=300)


def _form(request: httpx.Request) -> dict:
    return dict(parse_qsl(request.content.decode()))


CARD_METHOD = {"id": "pm_1", "type": "card", "card": {"brand": "visa", "last4": "4242",
                                                       "exp_month": 12, "exp_year": 2028}}


# --------------------------------------------------------------------------- #
# card_on_file
# --------------------------------------------------------------------------- #


def test_card_on_file_prefers_the_subscriptions_default():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1/subscriptions/{SUBSCRIPTION}"
        return httpx.Response(200, json={"id": SUBSCRIPTION, "default_payment_method": CARD_METHOD})

    card = _provider(handler).card_on_file(customer_id=CUSTOMER, subscription_id=SUBSCRIPTION)
    assert card == CardOnFile(brand="visa", last4="4242", exp_month=12, exp_year=2028)


def test_card_on_file_falls_back_to_the_customers_default():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.startswith("/v1/subscriptions/"):
            return httpx.Response(200, json={"id": SUBSCRIPTION, "default_payment_method": None})
        return httpx.Response(200, json={"id": CUSTOMER,
                                         "invoice_settings": {"default_payment_method": CARD_METHOD}})

    card = _provider(handler).card_on_file(customer_id=CUSTOMER, subscription_id=SUBSCRIPTION)
    assert card == CardOnFile(brand="visa", last4="4242", exp_month=12, exp_year=2028)
    assert seen == [f"/v1/subscriptions/{SUBSCRIPTION}", f"/v1/customers/{CUSTOMER}"]


def test_card_on_file_is_none_with_neither():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/subscriptions/"):
            return httpx.Response(200, json={"id": SUBSCRIPTION, "default_payment_method": None})
        return httpx.Response(200, json={"id": CUSTOMER, "invoice_settings": {"default_payment_method": None}})

    assert _provider(handler).card_on_file(customer_id=CUSTOMER, subscription_id=SUBSCRIPTION) is None


def test_card_on_file_with_no_subscription_reads_the_customer_directly():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1/customers/{CUSTOMER}"
        return httpx.Response(200, json={"id": CUSTOMER,
                                         "invoice_settings": {"default_payment_method": CARD_METHOD}})

    card = _provider(handler).card_on_file(customer_id=CUSTOMER, subscription_id=None)
    assert card == CardOnFile(brand="visa", last4="4242", exp_month=12, exp_year=2028)


# --------------------------------------------------------------------------- #
# start_card_change
# --------------------------------------------------------------------------- #


def test_start_card_change_posts_an_off_session_setup_intent():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "seti_1", "client_secret": "seti_1_secret_abc"})

    secret = _provider(handler).start_card_change(customer_id=CUSTOMER, workspace_id="ws-1")
    assert secret == "seti_1_secret_abc"
    (request,) = seen
    form = _form(request)
    assert request.url.path == "/v1/setup_intents"
    assert form["customer"] == CUSTOMER
    assert form["usage"] == "off_session"
    assert form["automatic_payment_methods[enabled]"] == "true"
    assert form["metadata[workspace_id]"] == "ws-1"


# --------------------------------------------------------------------------- #
# finish_card_change
# --------------------------------------------------------------------------- #


def _setup_intent(**overrides) -> dict:
    base = {"id": "seti_1", "customer": CUSTOMER, "status": "succeeded", "payment_method": "pm_1"}
    base.update(overrides)
    return base


def test_finish_card_change_sets_the_customer_and_subscription_default():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v1/setup_intents/seti_1":
            return httpx.Response(200, json=_setup_intent())
        if request.url.path == f"/v1/customers/{CUSTOMER}":
            return httpx.Response(200, json={"id": CUSTOMER})
        if request.url.path == f"/v1/subscriptions/{SUBSCRIPTION}":
            return httpx.Response(200, json={"id": SUBSCRIPTION})
        if request.url.path == "/v1/payment_methods/pm_1":
            return httpx.Response(200, json=CARD_METHOD)
        raise AssertionError(request.url.path)

    card = _provider(handler).finish_card_change(customer_id=CUSTOMER, subscription_id=SUBSCRIPTION,
                                                 setup_id="seti_1")
    assert card == CardOnFile(brand="visa", last4="4242", exp_month=12, exp_year=2028)
    customer_post = next(c for c in calls if c.method == "POST" and c.url.path == f"/v1/customers/{CUSTOMER}")
    assert _form(customer_post)["invoice_settings[default_payment_method]"] == "pm_1"
    sub_post = next(c for c in calls if c.method == "POST" and c.url.path == f"/v1/subscriptions/{SUBSCRIPTION}")
    assert _form(sub_post)["default_payment_method"] == "pm_1"


def test_finish_card_change_refuses_another_customers_setup():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_setup_intent(customer="cus_other"))

    with pytest.raises(ValueError):
        _provider(handler).finish_card_change(customer_id=CUSTOMER, subscription_id=None, setup_id="seti_1")


def test_finish_card_change_refuses_an_unfinished_setup():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_setup_intent(status="requires_payment_method"))

    with pytest.raises(ValueError):
        _provider(handler).finish_card_change(customer_id=CUSTOMER, subscription_id=None, setup_id="seti_1")


# --------------------------------------------------------------------------- #
# invoices
# --------------------------------------------------------------------------- #


def test_invoices_maps_the_list_newest_first():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/invoices"
        return httpx.Response(200, json={"data": [
            {"id": "in_1", "number": "ABCD-0001", "status": "paid", "total": 2900, "currency": "usd",
             "created": 1_700_000_000},
        ]})

    items = _provider(handler).invoices(customer_id=CUSTOMER)
    assert items == [InvoiceLine(id="in_1", number="ABCD-0001", status="paid", total_cents=2900,
                                 currency="usd", created=datetime.fromtimestamp(1_700_000_000, tz=timezone.utc))]


# --------------------------------------------------------------------------- #
# invoice_pdf
# --------------------------------------------------------------------------- #


def test_invoice_pdf_refuses_another_customers_invoice():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "in_1", "customer": "cus_other", "invoice_pdf": "https://x"})

    with pytest.raises(ValueError):
        _provider(handler).invoice_pdf(customer_id=CUSTOMER, invoice_id="in_1")


def test_invoice_pdf_refuses_an_invoice_with_no_pdf_link():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "in_1", "customer": CUSTOMER, "invoice_pdf": None})

    with pytest.raises(ValueError):
        _provider(handler).invoice_pdf(customer_id=CUSTOMER, invoice_id="in_1")


def test_invoice_pdf_fetches_the_file_and_sanitises_the_filename():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/invoices/in_1":
            return httpx.Response(200, json={"id": "in_1", "customer": CUSTOMER, "number": "ABCD/0001 *",
                                             "invoice_pdf": "https://files.stripe.com/v1/files/f_1/contents"})
        return httpx.Response(200, content=b"%PDF-1.4 bytes")

    content, name = _provider(handler).invoice_pdf(customer_id=CUSTOMER, invoice_id="in_1")
    assert content == b"%PDF-1.4 bytes"
    assert name == "ABCD0001.pdf"  # non-alphanumerics stripped


def test_invoice_pdf_falls_back_to_the_invoice_id_with_no_number():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/invoices/in_1":
            return httpx.Response(200, json={"id": "in_1", "customer": CUSTOMER, "number": None,
                                             "invoice_pdf": "https://files.stripe.com/v1/files/f_1/contents"})
        return httpx.Response(200, content=b"%PDF-1.4 bytes")

    _content, name = _provider(handler).invoice_pdf(customer_id=CUSTOMER, invoice_id="in_1")
    assert name == "in1.pdf"  # the invoice id's underscore is stripped too (only [A-Za-z0-9-] survives)


# --------------------------------------------------------------------------- #
# set_cancel_at_period_end
# --------------------------------------------------------------------------- #


def test_set_cancel_at_period_end_posts_the_flag_and_returns_the_period_end():
    end = int(datetime(2026, 12, 1, tzinfo=timezone.utc).timestamp())

    def handler(request: httpx.Request) -> httpx.Response:
        assert _form(request)["cancel_at_period_end"] == "true"
        return httpx.Response(200, json={"id": SUBSCRIPTION, "current_period_end": end, "items": {"data": []}})

    ends = _provider(handler).set_cancel_at_period_end(subscription_id=SUBSCRIPTION, cancel=True)
    assert ends == datetime(2026, 12, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# preview_plan_change
# --------------------------------------------------------------------------- #


def test_preview_plan_change_sends_the_subscription_item_and_sums_tax():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/subscriptions/{SUBSCRIPTION}":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "items": {"data": [{"id": "si_1"}]}})
        assert request.url.path == "/v1/invoices/create_preview"
        form = _form(request)
        assert form["subscription_details[items][0][id]"] == "si_1"
        assert form["subscription_details[items][0][price]"] == "price_pro"
        assert form["subscription_details[proration_behavior]"] == "always_invoice"
        assert form["subscription"] == SUBSCRIPTION and form["customer"] == CUSTOMER
        return httpx.Response(200, json={"amount_due": 1350, "currency": "usd",
                                         "total_taxes": [{"amount": 100}, {"amount": 50}]})

    preview = _provider(handler).preview_plan_change(customer_id=CUSTOMER, subscription_id=SUBSCRIPTION,
                                                      price_id="price_pro")
    assert preview == PlanChangePreview(amount_due_cents=1350, tax_cents=150, currency="usd")


# --------------------------------------------------------------------------- #
# upgrade_now
# --------------------------------------------------------------------------- #


def test_upgrade_now_with_no_pending_update_is_done():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "items": {"data": [{"id": "si_1"}]}})
        form = _form(request)
        assert form["payment_behavior"] == "pending_if_incomplete"
        assert form["items[0][id]"] == "si_1" and form["items[0][price]"] == "price_pro"
        return httpx.Response(200, json={"id": SUBSCRIPTION})  # no pending_update

    result = _provider(handler).upgrade_now(subscription_id=SUBSCRIPTION, price_id="price_pro")
    assert result == PlanChangeResult(status="done")


def test_upgrade_now_with_a_pending_update_requires_action():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "items": {"data": [{"id": "si_1"}]}})
        return httpx.Response(200, json={
            "id": SUBSCRIPTION, "pending_update": {"expires_at": 1},
            "latest_invoice": {"confirmation_secret": {"client_secret": "pi_1_secret_xyz"}},
        })

    result = _provider(handler).upgrade_now(subscription_id=SUBSCRIPTION, price_id="price_pro")
    assert result == PlanChangeResult(status="requires_action", client_secret="pi_1_secret_xyz")


def test_upgrade_now_with_a_pending_update_but_no_confirmable_secret_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "items": {"data": [{"id": "si_1"}]}})
        return httpx.Response(200, json={"id": SUBSCRIPTION, "pending_update": {"expires_at": 1},
                                         "latest_invoice": None})

    with pytest.raises(ValueError):
        _provider(handler).upgrade_now(subscription_id=SUBSCRIPTION, price_id="price_pro")


def test_upgrade_now_releases_an_attached_schedule_first():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "schedule": "sub_sched_1",
                                             "items": {"data": [{"id": "si_1"}]}})
        if request.url.path == "/v1/subscription_schedules/sub_sched_1/release":
            return httpx.Response(200, json={"id": "sub_sched_1", "status": "released"})
        return httpx.Response(200, json={"id": SUBSCRIPTION})

    _provider(handler).upgrade_now(subscription_id=SUBSCRIPTION, price_id="price_pro")
    assert ("POST", "/v1/subscription_schedules/sub_sched_1/release") in calls
    release_index = calls.index(("POST", "/v1/subscription_schedules/sub_sched_1/release"))
    upgrade_index = calls.index(("POST", f"/v1/subscriptions/{SUBSCRIPTION}"))
    assert release_index < upgrade_index  # released before the upgrade is applied


# --------------------------------------------------------------------------- #
# downgrade_at_period_end
# --------------------------------------------------------------------------- #


def test_downgrade_at_period_end_schedules_two_phases():
    now_phase = {"start_date": 1_700_000_000, "end_date": 1_702_000_000,
                "items": [{"price": "price_basic", "quantity": 1}]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "items": {"data": [{"id": "si_1"}]}})
        if request.url.path == "/v1/subscription_schedules":
            return httpx.Response(200, json={"id": "sub_sched_1", "phases": [now_phase]})
        assert request.url.path == "/v1/subscription_schedules/sub_sched_1"
        form = _form(request)
        assert form["end_behavior"] == "release"
        assert form["phases[0][items][0][price]"] == "price_basic"
        assert form["phases[1][items][0][price]"] == "price_basic_new"
        assert form["phases[1][duration][interval]"] == "month"
        return httpx.Response(200, json={"id": "sub_sched_1"})

    result = _provider(handler).downgrade_at_period_end(subscription_id=SUBSCRIPTION, price_id="price_basic_new")
    assert result.status == "scheduled" and result.schedule_id == "sub_sched_1"
    assert result.effective_at == datetime.fromtimestamp(1_702_000_000, tz=timezone.utc)


def test_downgrade_at_period_end_releases_an_attached_schedule_first():
    calls = []
    now_phase = {"start_date": 1, "end_date": 2, "items": []}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"id": SUBSCRIPTION, "schedule": "sub_sched_old",
                                             "items": {"data": [{"id": "si_1"}]}})
        if request.url.path == "/v1/subscription_schedules/sub_sched_old/release":
            return httpx.Response(200, json={})
        if request.url.path == "/v1/subscription_schedules":
            return httpx.Response(200, json={"id": "sub_sched_new", "phases": [now_phase]})
        return httpx.Response(200, json={"id": "sub_sched_new"})

    _provider(handler).downgrade_at_period_end(subscription_id=SUBSCRIPTION, price_id="price_basic")
    assert ("POST", "/v1/subscription_schedules/sub_sched_old/release") in calls


# --------------------------------------------------------------------------- #
# cancel_scheduled_change
# --------------------------------------------------------------------------- #


def test_cancel_scheduled_change_posts_release():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    _provider(handler).cancel_scheduled_change(schedule_id="sub_sched_1")
    (request,) = seen
    assert request.method == "POST" and request.url.path == "/v1/subscription_schedules/sub_sched_1/release"


# --------------------------------------------------------------------------- #
# checked — refuses a malformed id before any HTTP call
# --------------------------------------------------------------------------- #


def test_checked_accepts_a_well_formed_id():
    assert checked("cus", CUSTOMER) == CUSTOMER


@pytest.mark.parametrize("kind, value", [("cus", "not-a-stripe-id"), ("sub", ""), ("seti", None)])
def test_checked_refuses_a_malformed_id(kind, value):
    with pytest.raises(ValueError):
        checked(kind, value)


def test_a_malformed_id_raises_before_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        raise AssertionError("no HTTP call should be made for a malformed id")

    with pytest.raises(ValueError):
        _provider(handler).card_on_file(customer_id="not-a-stripe-id", subscription_id=None)
