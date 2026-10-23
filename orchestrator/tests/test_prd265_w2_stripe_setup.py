"""PRD-265 W2: ``scripts/stripe_setup.py`` against an in-memory Stripe.

No network. An ``httpx.MockTransport`` plays Stripe's prices, portal
configurations, webhook endpoints and account: it records every request and answers
the way Stripe does (list pages, create payloads, a lookup key moved only with
``transfer_lookup_key``, the webhook signing secret on create). The tests pin what
the script sends (the form fields of every create), that a second run creates
nothing, the live/test key guard, ``--dry-run``, the stop on a Price that doesn't
match config, and that neither the secret key nor the webhook signing secret is ever
printed (the Dashboard reveals it).
"""
from __future__ import annotations

import io
import itertools
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import parse_qsl

import httpx
import pytest

from scripts import stripe_setup
from services.billing.stripe_events import WEBHOOK_EVENTS

# Fake values, short and low-entropy on purpose (the secrets scan reads tests too).
TEST_KEY = "sk_test_fake"
LIVE_KEY = "sk_live_fake"
RESTRICTED_LIVE_KEY = "rk_live_fake"
WEBHOOK_SECRET = "whsec_fake"
ACCOUNT_ID = "acct_fake"
API_VERSION = "2025-03-31.basil"
FRONTEND_URL = "https://app.example.test"
WEBHOOK_URL = "https://api.example.test/api/v1/billing/webhooks/stripe"
RUN_ID = "run1"

PRICES = "/v1/prices"
PORTALS = "/v1/billing_portal/configurations"
WEBHOOKS = "/v1/webhook_endpoints"
CREATE_PATHS = (PRICES, PORTALS, WEBHOOKS)

LOOKUP_KEYS = {
    "automatos_own_keys_monthly": "STRIPE_PRICE_BASIC",
    "automatos_pro_monthly": "STRIPE_PRICE_PRO",
    "automatos_business_monthly": "STRIPE_PRICE_BUSINESS",
    "automatos_topup_1000": "STRIPE_PRICE_TOPUP",
}


@dataclass(frozen=True)
class Call:
    """One request the script sent to the fake Stripe."""

    method: str
    path: str
    form: tuple
    query: tuple
    idempotency_key: Optional[str]
    authorization: Optional[str]

    def field(self, name: str) -> Optional[str]:
        return dict(self.form).get(name)

    def values(self, prefix: str) -> list[str]:
        return [value for key, value in self.form if key.startswith(prefix)]


def _page(items: list[dict]) -> dict:
    return {"object": "list", "data": [dict(item) for item in items], "has_more": False}


class FakeStripe:
    """Enough of Stripe for the setup script, in memory."""

    def __init__(self) -> None:
        self.prices: list[dict] = []
        self.products: dict[str, dict] = {}
        self.portals: list[dict] = []
        self.webhooks: list[dict] = []
        self.calls: list[Call] = []
        self.failures: dict[tuple[str, str], tuple[int, dict]] = {}
        self._ids = itertools.count(1)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = request.content.decode() if request.method == "POST" else ""
        call = Call(request.method, request.url.path, tuple(parse_qsl(body, keep_blank_values=True)),
                    tuple(request.url.params.multi_items()), request.headers.get("Idempotency-Key"),
                    request.headers.get("Authorization"))
        self.calls.append(call)
        failure = self.failures.get((call.method, call.path))
        if failure is not None:
            return httpx.Response(failure[0], json=failure[1])
        return httpx.Response(200, json=self._route(call)(call))

    def _route(self, call: Call) -> Callable[[Call], dict]:
        routes = {
            ("GET", "/v1/account"): lambda _call: {"id": ACCOUNT_ID, "object": "account"},
            ("GET", PRICES): self._list_prices,
            ("POST", PRICES): self._create_price,
            ("GET", PORTALS): lambda _call: _page(self.portals),
            ("POST", PORTALS): self._create_portal,
            ("GET", WEBHOOKS): lambda _call: _page(self.webhooks),
            ("POST", WEBHOOKS): self._create_webhook,
        }
        handler = routes.get((call.method, call.path))
        if handler is not None:
            return handler
        if call.method == "POST" and call.path.startswith(PORTALS + "/"):
            return self._update_portal
        if call.method == "POST" and call.path.startswith(WEBHOOKS + "/"):
            return self._update_webhook
        raise AssertionError(f"unexpected Stripe call {call.method} {call.path}")

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids)}"

    # -- seeding -------------------------------------------------------------
    def add_price(self, lookup_key: str, unit_amount: int, *, key: str = "pro", interval: Optional[str] = "month",
                  active: bool = True, tax_behavior: str = "exclusive") -> dict:
        product = {"id": self._next_id("prod"), "object": "product", "name": f"Automatos {key}",
                   "metadata": {"automatos_key": key}}
        self.products[product["id"]] = product
        price = {"id": self._next_id("price"), "object": "price", "active": active, "lookup_key": lookup_key,
                 "unit_amount": unit_amount, "currency": "usd", "tax_behavior": tax_behavior,
                 "type": "recurring" if interval else "one_time",
                 "recurring": {"interval": interval} if interval else None, "product": product["id"]}
        self.prices.append(price)
        return price

    def add_webhook(self, events: list[str], *, api_version: str = API_VERSION, status: str = "enabled") -> dict:
        endpoint = {"id": self._next_id("we"), "object": "webhook_endpoint", "url": WEBHOOK_URL,
                    "api_version": api_version, "enabled_events": list(events), "status": status}
        self.webhooks.append(endpoint)
        return endpoint

    def price(self, lookup_key: str) -> dict:
        return next(price for price in self.prices if price["lookup_key"] == lookup_key)

    # -- endpoints -----------------------------------------------------------
    def _expand(self, price: dict) -> dict:
        return {**price, "product": dict(self.products[price["product"]])}

    def _list_prices(self, call: Call) -> dict:
        keys = {value for key, value in call.query if key.startswith("lookup_keys[")}
        active_only = ("active", "true") in call.query
        expand = ("expand[0]", "data.product") in call.query
        found = [price for price in self.prices
                 if price["lookup_key"] in keys and (price["active"] or not active_only)]
        return _page([self._expand(price) if expand else price for price in found])

    def _create_price(self, call: Call) -> dict:
        lookup_key = call.field("lookup_key")
        holders = [price for price in self.prices if price["lookup_key"] == lookup_key]
        if holders and call.field("transfer_lookup_key") != "true":
            raise AssertionError(f"lookup key {lookup_key} is taken and transfer_lookup_key wasn't set")
        for holder in holders:
            holder["lookup_key"] = None
        price = self.add_price(lookup_key, int(call.field("unit_amount")),
                               key=call.field("product_data[metadata][automatos_key]"),
                               interval=call.field("recurring[interval]"),
                               tax_behavior=call.field("tax_behavior"))
        return self._expand(price) if ("expand[0]", "product") in call.form else price

    def _create_portal(self, call: Call) -> dict:
        portal = {"id": self._next_id("bpc"), "object": "billing_portal.configuration", "active": True,
                  "metadata": {"automatos_key": call.field("metadata[automatos_key]")}}
        self.portals.append(portal)
        return portal

    def _update_portal(self, call: Call) -> dict:
        return self._by_id(self.portals, call.path)

    def _create_webhook(self, call: Call) -> dict:
        endpoint = {"id": self._next_id("we"), "object": "webhook_endpoint", "url": call.field("url"),
                    "api_version": call.field("api_version"), "enabled_events": call.values("enabled_events["),
                    "status": "enabled"}
        self.webhooks.append(endpoint)
        return {**endpoint, "secret": WEBHOOK_SECRET}

    def _update_webhook(self, call: Call) -> dict:
        endpoint = self._by_id(self.webhooks, call.path)
        endpoint["enabled_events"] = call.values("enabled_events[")
        if call.field("disabled") == "false":
            endpoint["status"] = "enabled"
        return endpoint

    @staticmethod
    def _by_id(items: list[dict], path: str) -> dict:
        object_id = path.rsplit("/", 1)[1]
        return next(item for item in items if item["id"] == object_id)


def _settings(secret_key: str = TEST_KEY) -> stripe_setup.Settings:
    plans = (stripe_setup.PlanPrice("basic", "Own keys", 9, True), stripe_setup.PlanPrice("pro", "Pro", 29, True),
             stripe_setup.PlanPrice("business", "Business", 99, False))
    return stripe_setup.Settings(secret_key=secret_key, api_version=API_VERSION, timeout=5.0,
                          frontend_url=FRONTEND_URL, plans=plans, topup_credits=1000, topup_price_usd=10)


def _options(**overrides) -> stripe_setup.Options:
    values = {"webhook_url": WEBHOOK_URL, "tax_behavior": "exclusive", "live": False, "dry_run": False}
    return stripe_setup.Options(**{**values, **overrides})


def _run(fake: FakeStripe, options: Optional[stripe_setup.Options] = None,
         settings: Optional[stripe_setup.Settings] = None) -> tuple[int, str]:
    out = io.StringIO()
    code = stripe_setup.execute(options or _options(), settings or _settings(), out=out,
                         transport=fake.transport(), run_id=RUN_ID)
    return code, out.getvalue()


def _posts(fake: FakeStripe, path: Optional[str] = None, since: int = 0) -> list[Call]:
    return [call for call in fake.calls[since:] if call.method == "POST" and (path is None or call.path == path)]


def _env(output: str, prefix: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(prefix)]


# ---------------------------------------------------------------------------
# An empty account gets everything
# ---------------------------------------------------------------------------
def test_empty_account_gets_four_prices_each_with_its_product():
    fake = FakeStripe()
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output

    creates = {call.field("lookup_key"): call for call in _posts(fake, PRICES)}
    assert set(creates) == set(LOOKUP_KEYS)
    expected = {
        "automatos_own_keys_monthly": ("900", "basic", "month", "Automatos Own keys"),
        "automatos_pro_monthly": ("2900", "pro", "month", "Automatos Pro"),
        "automatos_business_monthly": ("9900", "business", "month", "Automatos Business"),
        "automatos_topup_1000": ("1000", "topup", None, "Automatos top-up: 1,000 credits"),
    }
    for lookup_key, (amount, key, interval, name) in expected.items():
        call = creates[lookup_key]
        assert call.field("unit_amount") == amount
        assert call.field("currency") == "usd"
        assert call.field("tax_behavior") == "exclusive"
        assert call.field("recurring[interval]") == interval
        assert call.field("product_data[name]") == name
        assert call.field("product_data[tax_code]") == "txcd_10103001"
        assert call.field("product_data[metadata][automatos_key]") == key
        assert call.field("transfer_lookup_key") == "true"
        assert call.idempotency_key and call.idempotency_key.startswith(f"automatos-stripe-setup-{RUN_ID}-")
    assert len({call.idempotency_key for call in creates.values()}) == 4


def test_empty_account_gets_the_portal_configuration():
    fake = FakeStripe()
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output

    [call] = _posts(fake, PORTALS)
    basic, pro = fake.price("automatos_own_keys_monthly"), fake.price("automatos_pro_monthly")
    business = fake.price("automatos_business_monthly")
    assert call.field("default_return_url") == "https://app.example.test/settings?tab=billing"
    assert call.field("metadata[automatos_key]") == "portal"
    assert call.field("features[invoice_history][enabled]") == "true"
    assert call.field("features[payment_method_update][enabled]") == "true"
    assert call.field("features[customer_update][enabled]") == "true"
    assert call.values("features[customer_update][allowed_updates][") == ["email", "address", "tax_id"]
    assert call.field("features[subscription_cancel][enabled]") == "true"
    assert call.field("features[subscription_cancel][mode]") == "at_period_end"
    update = "features[subscription_update]"
    assert call.field(f"{update}[enabled]") == "true"
    assert call.values(f"{update}[default_allowed_updates][") == ["price"]
    assert call.field(f"{update}[proration_behavior]") == "always_invoice"
    assert call.field(f"{update}[schedule_at_period_end][conditions][0][type]") == "decreasing_item_amount"
    assert call.field(f"{update}[products][0][product]") == basic["product"]
    assert call.values(f"{update}[products][0][prices][") == [basic["id"]]
    assert call.field(f"{update}[products][1][product]") == pro["product"]
    assert call.values(f"{update}[products][1][prices][") == [pro["id"]]
    # Business is never self-serve: neither its product nor its price is in the switcher.
    assert call.field(f"{update}[products][2][product]") is None
    sent = [value for _key, value in call.form]
    assert business["id"] not in sent and business["product"] not in sent
    assert call.idempotency_key and call.idempotency_key.startswith(f"automatos-stripe-setup-{RUN_ID}-")


def test_empty_account_gets_the_webhook_at_the_pinned_version_with_every_event():
    fake = FakeStripe()
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output

    [call] = _posts(fake, WEBHOOKS)
    assert call.field("url") == WEBHOOK_URL
    assert call.field("api_version") == API_VERSION
    assert call.values("enabled_events[") == list(WEBHOOK_EVENTS)
    assert call.idempotency_key and call.idempotency_key.startswith(f"automatos-stripe-setup-{RUN_ID}-")


def test_empty_account_prints_every_env_line_and_where_to_reveal_the_signing_secret():
    fake = FakeStripe()
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output

    lines = output.splitlines()
    for lookup_key, env_name in LOOKUP_KEYS.items():
        assert f"{env_name}={fake.price(lookup_key)['id']}" in lines
    assert f"STRIPE_PORTAL_CONFIGURATION_ID={fake.portals[0]['id']}" in lines
    assert f"STRIPE_ACCOUNT_ID={ACCOUNT_ID}" in lines
    # Stripe returned the signing secret, but it never reaches the output.
    assert WEBHOOK_SECRET not in output and "STRIPE_WEBHOOK_SECRET=" not in output
    assert f"# STRIPE_WEBHOOK_SECRET: {stripe_setup.SIGNING_SECRET_WHERE}" in lines
    assert "set STRIPE_WEBHOOK_SECRET too" in output


# ---------------------------------------------------------------------------
# Idempotent
# ---------------------------------------------------------------------------
def test_a_second_run_creates_nothing_and_prints_the_same_ids():
    fake = FakeStripe()
    _, first = _run(fake)
    seen = len(fake.calls)
    code, second = _run(fake)
    assert code == stripe_setup.EXIT_OK, second

    assert [call for call in _posts(fake, since=seen) if call.path in CREATE_PATHS] == []
    # The only write is the portal brought to our settings; the webhook already matches.
    assert [call.path for call in _posts(fake, since=seen)] == [f"{PORTALS}/{fake.portals[0]['id']}"]
    assert _posts(fake, since=seen)[0].field("active") == "true"
    assert (len(fake.prices), len(fake.portals), len(fake.webhooks)) == (4, 1, 1)
    assert _env(second, "STRIPE_PRICE_") == _env(first, "STRIPE_PRICE_")
    assert _env(second, "STRIPE_PORTAL_CONFIGURATION_ID=") == _env(first, "STRIPE_PORTAL_CONFIGURATION_ID=")
    assert "STRIPE_WEBHOOK_SECRET=" not in second
    assert WEBHOOK_SECRET not in second


def test_an_archived_price_is_replaced_and_hands_over_its_lookup_key():
    fake = FakeStripe()
    archived = fake.add_price("automatos_pro_monthly", 1900, active=False)
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output

    [create] = [call for call in _posts(fake, PRICES) if call.field("lookup_key") == "automatos_pro_monthly"]
    assert create.field("unit_amount") == "2900"
    assert archived["lookup_key"] is None
    assert f"STRIPE_PRICE_PRO={fake.price('automatos_pro_monthly')['id']}" in output.splitlines()


def test_inclusive_tax_behavior_goes_on_every_price():
    fake = FakeStripe()
    code, output = _run(fake, options=_options(tax_behavior="inclusive"))
    assert code == stripe_setup.EXIT_OK, output
    assert {call.field("tax_behavior") for call in _posts(fake, PRICES)} == {"inclusive"}


# ---------------------------------------------------------------------------
# Live / test key guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", [LIVE_KEY, RESTRICTED_LIVE_KEY])
def test_a_live_key_without_live_is_refused_before_any_call(key):
    fake = FakeStripe()
    code, output = _run(fake, settings=_settings(secret_key=key))
    assert code == stripe_setup.EXIT_REFUSED
    assert fake.calls == []
    assert "--live" in output
    assert key not in output


def test_live_with_a_test_key_is_refused_before_any_call():
    fake = FakeStripe()
    code, output = _run(fake, options=_options(live=True))
    assert code == stripe_setup.EXIT_REFUSED
    assert fake.calls == []
    assert TEST_KEY not in output


@pytest.mark.parametrize("key", ["", "pk_test_fake"])
def test_a_missing_or_publishable_key_is_refused(key):
    fake = FakeStripe()
    code, _output = _run(fake, settings=_settings(secret_key=key))
    assert code == stripe_setup.EXIT_REFUSED
    assert fake.calls == []


def test_a_live_key_with_live_runs():
    fake = FakeStripe()
    code, output = _run(fake, options=_options(live=True), settings=_settings(secret_key=LIVE_KEY))
    assert code == stripe_setup.EXIT_OK, output
    assert "LIVE mode" in output
    assert LIVE_KEY not in output


@pytest.mark.parametrize("url", [
    "http://api.example.test/api/v1/billing/webhooks/stripe",
    "https://api.example.test/api/v1/billing/webhooks",
    "https://api.example.test/api/v1/billing/webhooks/stripe/",
    "https://api.example.test/api/v1/billing/webhooks/stripe?x=1",
    "api.example.test/api/v1/billing/webhooks/stripe",
])
def test_a_webhook_url_that_isnt_the_https_billing_route_is_refused(url):
    fake = FakeStripe()
    code, _output = _run(fake, options=_options(webhook_url=url))
    assert code == stripe_setup.EXIT_REFUSED
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def test_dry_run_on_an_empty_account_writes_nothing():
    fake = FakeStripe()
    code, output = _run(fake, options=_options(dry_run=True))
    assert code == stripe_setup.EXIT_OK, output
    assert _posts(fake) == []
    assert "would create" in output
    assert "# STRIPE_PRICE_PRO:" in output
    assert f"STRIPE_ACCOUNT_ID={ACCOUNT_ID}" in output
    assert "STRIPE_WEBHOOK_SECRET=" not in output


def test_dry_run_on_a_set_up_account_with_stale_events_writes_nothing():
    fake = FakeStripe()
    _run(fake)
    fake.webhooks[0]["enabled_events"] = ["invoice.paid"]
    seen = len(fake.calls)
    code, output = _run(fake, options=_options(dry_run=True))
    assert code == stripe_setup.EXIT_OK, output
    assert _posts(fake, since=seen) == []
    assert "would update" in output
    assert f"STRIPE_PRICE_PRO={fake.price('automatos_pro_monthly')['id']}" in output.splitlines()


# ---------------------------------------------------------------------------
# A Price that doesn't match config stops the run
# ---------------------------------------------------------------------------
def test_a_price_whose_amount_differs_from_config_stops_before_any_write():
    fake = FakeStripe()
    fake.add_price("automatos_pro_monthly", 1900)
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_FAILED
    assert _posts(fake) == []
    assert "PRICE MISMATCH: automatos_pro_monthly" in output
    assert "1900" in output and "2900" in output
    assert "Archive" in output
    assert _env(output, "STRIPE_PRICE_") == []


def test_a_price_with_another_interval_also_stops():
    fake = FakeStripe()
    fake.add_price("automatos_topup_1000", 1000, key="topup", interval="month")
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_FAILED
    assert _posts(fake) == []
    assert "PRICE MISMATCH: automatos_topup_1000" in output


# ---------------------------------------------------------------------------
# Existing webhook endpoint
# ---------------------------------------------------------------------------
def test_an_existing_webhook_gets_its_events_updated_and_no_secret_is_printed():
    fake = FakeStripe()
    endpoint = fake.add_webhook(["invoice.paid"])
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output

    assert _posts(fake, WEBHOOKS) == []
    [update] = _posts(fake, f"{WEBHOOKS}/{endpoint['id']}")
    assert update.values("enabled_events[") == list(WEBHOOK_EVENTS)
    assert update.field("disabled") is None
    assert "STRIPE_WEBHOOK_SECRET=" not in output
    assert WEBHOOK_SECRET not in output


def test_a_disabled_webhook_is_re_enabled():
    fake = FakeStripe()
    endpoint = fake.add_webhook(list(WEBHOOK_EVENTS), status="disabled")
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output
    [update] = _posts(fake, f"{WEBHOOKS}/{endpoint['id']}")
    assert update.field("disabled") == "false"


def test_a_webhook_at_another_api_version_warns_and_writes_no_new_endpoint():
    fake = FakeStripe()
    fake.add_webhook(list(WEBHOOK_EVENTS), api_version="2024-06-20")
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_OK, output
    assert _posts(fake, WEBHOOKS) == []
    assert "WARNING" in output and "2024-06-20" in output
    assert "delete this endpoint" in output


# ---------------------------------------------------------------------------
# Secrets and errors
# ---------------------------------------------------------------------------
def test_the_secret_key_reaches_stripe_but_never_the_output():
    fake = FakeStripe()
    outputs = [_run(fake)[1], _run(fake)[1], _run(fake, options=_options(dry_run=True))[1]]
    mismatched = FakeStripe()
    mismatched.add_price("automatos_pro_monthly", 1900)
    outputs.append(_run(mismatched)[1])
    outputs.append(_run(FakeStripe(), settings=_settings(secret_key=LIVE_KEY))[1])

    assert fake.calls[0].authorization == f"Bearer {TEST_KEY}"
    for output in outputs:
        assert TEST_KEY not in output
        assert LIVE_KEY not in output


def test_a_stripe_error_stops_with_stripes_message():
    fake = FakeStripe()
    fake.failures[("POST", PRICES)] = (400, {"error": {"type": "invalid_request_error",
                                                       "message": "Invalid tax code."}})
    code, output = _run(fake)
    assert code == stripe_setup.EXIT_FAILED
    assert "Invalid tax code." in output
    assert _posts(fake, WEBHOOKS) == []
    assert "STRIPE_PRICE_" not in output


# ---------------------------------------------------------------------------
# Config and flags
# ---------------------------------------------------------------------------
def test_settings_come_from_config_and_plan_tiers():
    from config import PLAN_TIERS, config

    settings = stripe_setup.settings_from_config()
    plans = {plan.key: plan for plan in settings.plans}
    assert set(plans) == {"basic", "pro", "business"}
    for key, plan in plans.items():
        assert plan.price_usd == float(PLAN_TIERS[key]["display_price_usd"])
        assert plan.self_serve == bool(PLAN_TIERS[key].get("self_serve"))
        assert plan.name == PLAN_TIERS[key]["display_name"]
    assert settings.api_version == config.STRIPE_API_VERSION
    assert settings.frontend_url == config.FRONTEND_URL
    assert settings.topup_credits == config.TOPUP_CREDITS
    assert settings.topup_price_usd == float(config.TOPUP_PRICE_USD)


def test_flags_default_to_a_test_mode_run_that_writes_and_excludes_tax():
    options = stripe_setup.parse_args(["--webhook-url", WEBHOOK_URL])
    assert options == stripe_setup.Options(webhook_url=WEBHOOK_URL, tax_behavior="exclusive", live=False, dry_run=False)


def test_the_webhook_url_flag_is_required():
    with pytest.raises(SystemExit):
        stripe_setup.parse_args([])
