"""Make the Stripe account match Automatos's billing config (PRD-265 W2, US-009/US-010).

One idempotent command, run by a person: in test mode first, in live mode later.
From the ``orchestrator`` directory (``/app`` inside the API container)::

    python -m scripts.stripe_setup \\
        --webhook-url https://<api host>/api/v1/billing/webhooks/stripe \\
        [--tax-behavior exclusive|inclusive] [--live] [--dry-run]

It finds each object below and creates it only when it is missing:

* **Prices**, by ``lookup_key``: Own keys, Pro and Business monthly, and the credit
  top-up (one-off), each with its own Product, tax-coded as SaaS so Stripe Tax can
  work out VAT. Stripe Prices can't be edited, so an existing Price whose amount,
  currency, interval or tax behaviour differs from config stops the run before
  anything is written: archive it in Stripe and re-run, and the new Price takes
  over the lookup key.
* **The customer-portal configuration**, by metadata ``automatos_key=portal``,
  brought to our settings on every run: invoice history, payment methods, email /
  address / tax ID, cancel at period end, and a switch between the self-serve
  plans only (Business is sold by invite), where a downgrade waits for the period end.
* **The webhook endpoint**, by exact URL, at ``config.STRIPE_API_VERSION`` with the
  events the webhook handler reads (``services.billing.stripe_events.WEBHOOK_EVENTS``).

Then it prints the ``NAME=value`` lines to set on the Railway API service. A live
key needs ``--live`` and ``--live`` needs a live key. ``--dry-run`` only reads.
The secret key is never printed. The webhook signing secret is printed once, when
the endpoint is created: Stripe's API never returns it again.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence, TextIO
from urllib.parse import urlsplit

import httpx

from config import PLAN_TIERS, config
from services.billing.checkout import BILLING_TAB_PATH
from services.billing.stripe_api import StripeApi, StripeError
from services.billing.stripe_events import WEBHOOK_EVENTS

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

LIVE_KEY_PREFIXES = ("sk_live_", "rk_live_")
TEST_KEY_PREFIXES = ("sk_test_", "rk_test_")
# The backend's Stripe webhook route (api/billing_webhooks.py: prefix + "/stripe").
WEBHOOK_PATH = "/api/v1/billing/webhooks/stripe"
# Stripe Tax product code: "Software as a service (SaaS)".
SAAS_TAX_CODE = "txcd_10103001"
CURRENCY = "usd"
MONTHLY = "month"
TAX_BEHAVIORS = ("exclusive", "inclusive")
DEFAULT_TAX_BEHAVIOR = "exclusive"
CUSTOMER_UPDATE_FIELDS = ("email", "address", "tax_id")
METADATA_KEY = "automatos_key"
PORTAL_METADATA_VALUE = "portal"
WEBHOOK_METADATA_VALUE = "webhook"
PRODUCT_NAME_PREFIX = "Automatos"
WEBHOOK_DESCRIPTION = "Automatos billing webhooks (PRD-265)"
IDEMPOTENCY_PREFIX = "automatos-stripe-setup"
LIST_PAGE_SIZE = 100
MAX_LIST_PAGES = 50

ACCOUNT_PATH = "/v1/account"
PRICES_PATH = "/v1/prices"
PORTAL_PATH = "/v1/billing_portal/configurations"
WEBHOOKS_PATH = "/v1/webhook_endpoints"

# Plan key in config.PLAN_TIERS → (its Price's lookup key, the env var the backend reads).
PLAN_PRICE_NAMES: dict[str, tuple[str, str]] = {
    "basic": ("automatos_own_keys_monthly", "STRIPE_PRICE_BASIC"),
    "pro": ("automatos_pro_monthly", "STRIPE_PRICE_PRO"),
    "business": ("automatos_business_monthly", "STRIPE_PRICE_BUSINESS"),
}
TOPUP_KEY = "topup"
TOPUP_LOOKUP_KEY = "automatos_topup_{credits}"
TOPUP_ENV_NAME = "STRIPE_PRICE_TOPUP"
PORTAL_ENV_NAME = "STRIPE_PORTAL_CONFIGURATION_ID"
ACCOUNT_ENV_NAME = "STRIPE_ACCOUNT_ID"
WEBHOOK_SIGNING_ENV_NAME = "STRIPE_WEBHOOK_SECRET"


class SetupError(Exception):
    """The account is in a state this script must not paper over; the run stops
    with this message and a non-zero exit."""


@dataclass(frozen=True)
class PlanPrice:
    """A plan's display price from config: what its Stripe Price must charge."""

    key: str
    name: str
    price_usd: float
    self_serve: bool


@dataclass(frozen=True)
class Settings:
    """What this script reads from ``config.py`` (tests build their own)."""

    secret_key: str
    api_version: str
    timeout: float
    frontend_url: str
    plans: tuple[PlanPrice, ...]
    topup_credits: int
    topup_price_usd: float


@dataclass(frozen=True)
class Options:
    """The command line."""

    webhook_url: str
    tax_behavior: str
    live: bool
    dry_run: bool


@dataclass(frozen=True)
class PriceSpec:
    """One Price the account must have, found by its lookup key."""

    automatos_key: str
    env_name: str
    lookup_key: str
    product_name: str
    unit_amount: int
    monthly: bool
    tax_behavior: str
    in_portal: bool


@dataclass(frozen=True)
class WebhookResult:
    """The endpoint's id (None on a dry run that would create it), and its signing
    secret, which Stripe returns only when the endpoint is created."""

    endpoint_id: Optional[str]
    secret: Optional[str]


@dataclass(frozen=True)
class SetupRun:
    """One run: the Stripe client, where the report goes, and whether to write."""

    api: StripeApi
    out: TextIO
    dry_run: bool
    run_id: str

    def say(self, line: str = "") -> None:
        """Write one line of the report."""
        print(line, file=self.out)

    def idempotency_key(self, name: str) -> str:
        """A key per run and per object: a create the client retries inside this run
        can't make a second copy, and a later run (after an archive) can create anew."""
        return f"{IDEMPOTENCY_PREFIX}-{self.run_id}-{name}"


def settings_from_config() -> Settings:
    """The settings from ``config.py``: the Stripe key and API version, the app's
    URL, each plan's display price (``PLAN_TIERS``) and the top-up pack."""
    plans = tuple(
        PlanPrice(key=key, name=str(PLAN_TIERS[key]["display_name"]),
                  price_usd=float(PLAN_TIERS[key]["display_price_usd"]),
                  self_serve=bool(PLAN_TIERS[key].get("self_serve")))
        for key in PLAN_PRICE_NAMES
    )
    return Settings(secret_key=config.STRIPE_SECRET_KEY, api_version=config.STRIPE_API_VERSION,
                    timeout=float(config.STRIPE_TIMEOUT_SECONDS), frontend_url=config.FRONTEND_URL,
                    plans=plans, topup_credits=int(config.TOPUP_CREDITS),
                    topup_price_usd=float(config.TOPUP_PRICE_USD))


def key_mode_problem(secret_key: str, live: bool) -> Optional[str]:
    """Why this key may not be used in this mode, or None. A live key needs
    ``--live`` and ``--live`` needs a live key, so a test run can't touch real
    money and a live run can't quietly land in test mode. Never quotes the key."""
    if not secret_key:
        return "STRIPE_SECRET_KEY is not set in this environment."
    if secret_key.startswith(LIVE_KEY_PREFIXES):
        return None if live else ("STRIPE_SECRET_KEY is a LIVE key. Re-run with --live "
                                  "if you mean to set up the live account.")
    if secret_key.startswith(TEST_KEY_PREFIXES):
        return "--live was given, but STRIPE_SECRET_KEY is a test key." if live else None
    return "STRIPE_SECRET_KEY doesn't look like a Stripe secret or restricted key (sk_… or rk_…)."


def webhook_url_problem(url: str) -> Optional[str]:
    """Why ``url`` can't be the webhook endpoint, or None: it must be https and
    end in the backend's Stripe webhook route."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        return f"--webhook-url must be an https:// URL (got {url!r})."
    if not parts.path.endswith(WEBHOOK_PATH) or parts.query or parts.fragment:
        return f"--webhook-url must end in {WEBHOOK_PATH}, the API's Stripe webhook route (got {url!r})."
    return None


def to_cents(usd: float, what: str) -> int:
    """``usd`` in cents; a missing or non-positive price stops the run."""
    cents = round(float(usd) * 100)
    if cents <= 0:
        raise SetupError(f"{what} has no positive price in config ({usd!r}).")
    return int(cents)


def price_specs(settings: Settings, tax_behavior: str) -> tuple[PriceSpec, ...]:
    """The Prices the account must have: one monthly Price per plan, and the
    top-up. Only self-serve plans go in the portal's plan switcher."""
    plans = tuple(
        PriceSpec(automatos_key=plan.key, env_name=PLAN_PRICE_NAMES[plan.key][1],
                  lookup_key=PLAN_PRICE_NAMES[plan.key][0],
                  product_name=f"{PRODUCT_NAME_PREFIX} {plan.name}",
                  unit_amount=to_cents(plan.price_usd, plan.name), monthly=True,
                  tax_behavior=tax_behavior, in_portal=plan.self_serve)
        for plan in settings.plans
    )
    topup = PriceSpec(automatos_key=TOPUP_KEY, env_name=TOPUP_ENV_NAME,
                      lookup_key=TOPUP_LOOKUP_KEY.format(credits=settings.topup_credits),
                      product_name=f"{PRODUCT_NAME_PREFIX} top-up: {settings.topup_credits:,} credits",
                      unit_amount=to_cents(settings.topup_price_usd, "The top-up"), monthly=False,
                      tax_behavior=tax_behavior, in_portal=False)
    return plans + (topup,)


def _id_of(obj: object, what: str) -> str:
    """The ``id`` of a Stripe object; an answer without one stops the run."""
    object_id = obj.get("id") if isinstance(obj, dict) else None
    if not isinstance(object_id, str) or not object_id:
        raise SetupError(f"Stripe's answer for {what} has no id.")
    return object_id


def list_all(api: StripeApi, path: str) -> list[dict]:
    """Every object of a Stripe list endpoint, page by page."""
    items: list[dict] = []
    params: dict = {"limit": LIST_PAGE_SIZE}
    for _ in range(MAX_LIST_PAGES):
        page = api.get(path, params)
        data = [item for item in (page.get("data") or []) if isinstance(item, dict)]
        items = items + data
        if not page.get("has_more") or not data:
            return items
        params = {"limit": LIST_PAGE_SIZE, "starting_after": _id_of(data[-1], path)}
    raise SetupError(f"{path} lists more than {MAX_LIST_PAGES * LIST_PAGE_SIZE} objects; tidy it up in Stripe.")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def find_prices(api: StripeApi, specs: Sequence[PriceSpec]) -> dict[str, dict]:
    """The account's ACTIVE Prices for these lookup keys, by lookup key, with each
    Product expanded (the portal needs its id). An archived Price isn't found, so
    the run after an archive creates its successor."""
    page = api.get(PRICES_PATH, {"lookup_keys": [spec.lookup_key for spec in specs], "active": True,
                                 "expand": ["data.product"], "limit": LIST_PAGE_SIZE})
    return {price["lookup_key"]: price for price in (page.get("data") or [])
            if isinstance(price, dict) and price.get("lookup_key")}


def price_drift(spec: PriceSpec, price: dict) -> list[str]:
    """How an existing Price differs from config (amount, currency, interval, tax
    behaviour), in words; empty when it matches."""
    recurring = price.get("recurring") or {}
    wanted = {"amount (cents)": spec.unit_amount, "currency": CURRENCY,
              "interval": MONTHLY if spec.monthly else None, "tax behavior": spec.tax_behavior}
    actual = {"amount (cents)": price.get("unit_amount"), "currency": price.get("currency"),
              "interval": recurring.get("interval"), "tax behavior": price.get("tax_behavior")}
    return [f"{field}: {actual[field]!r} in Stripe, {wanted[field]!r} in config"
            for field in wanted if actual[field] != wanted[field]]


def stop_on_drift(run: SetupRun, specs: Sequence[PriceSpec], found: dict[str, dict]) -> None:
    """Stop before any write when an existing Price doesn't match config. Stripe
    Prices are immutable, so changing one means archiving it and making a new one."""
    drifted = [(spec, found[spec.lookup_key]) for spec in specs
               if spec.lookup_key in found and price_drift(spec, found[spec.lookup_key])]
    if not drifted:
        return
    for spec, price in drifted:
        run.say(f"PRICE MISMATCH: {spec.lookup_key} ({price.get('id')}) doesn't match config:")
        for problem in price_drift(spec, price):
            run.say(f"  - {problem}")
    raise SetupError(
        "Stripe Prices can't be edited, so nothing was changed. Archive each Price above in the "
        "Stripe Dashboard (Product catalogue → the product → the price → Archive), then re-run: "
        "the new Price takes over the lookup key, and its new ID goes on Railway."
    )


def create_price_params(spec: PriceSpec) -> dict:
    """The form for ``POST /v1/prices``: the Price with its Product inline (named,
    tax-coded as SaaS for Stripe Tax, tagged with its Automatos key). The lookup key
    moves over from an archived Price that still holds it."""
    recurring = {"recurring": {"interval": MONTHLY}} if spec.monthly else {}
    return {
        "currency": CURRENCY,
        "unit_amount": spec.unit_amount,
        "tax_behavior": spec.tax_behavior,
        "lookup_key": spec.lookup_key,
        "transfer_lookup_key": True,
        "metadata": {METADATA_KEY: spec.automatos_key},
        "product_data": {"name": spec.product_name, "tax_code": SAAS_TAX_CODE,
                         "metadata": {METADATA_KEY: spec.automatos_key}},
        "expand": ["product"],
        **recurring,
    }


def ensure_price(run: SetupRun, spec: PriceSpec, existing: Optional[dict]) -> Optional[dict]:
    """The Price for ``spec``: the existing one, else a new one (None on a dry run)."""
    if existing is not None:
        run.say(f"Price {spec.lookup_key}: exists ({existing.get('id')}), matches config")
        return existing
    billing = "a month" if spec.monthly else "one-off"
    if run.dry_run:
        run.say(f"Price {spec.lookup_key}: missing, would create ({spec.unit_amount} cents {CURRENCY}, {billing})")
        return None
    created = run.api.post(PRICES_PATH, create_price_params(spec),
                           idempotency_key=run.idempotency_key(f"price-{spec.lookup_key}"))
    run.say(f"Price {spec.lookup_key}: created ({_id_of(created, spec.lookup_key)}), "
            f"{spec.unit_amount} cents {CURRENCY}, {billing}")
    return created


def ensure_prices(run: SetupRun, specs: Sequence[PriceSpec]) -> dict[str, Optional[dict]]:
    """Every Price, by lookup key: found, or created when missing. A mismatch on
    any of them stops the run before the first create."""
    found = find_prices(run.api, specs)
    stop_on_drift(run, specs, found)
    return {spec.lookup_key: ensure_price(run, spec, found.get(spec.lookup_key)) for spec in specs}


def product_id_of(price: dict) -> str:
    """A Price's Product id, whether Stripe expanded the Product or not."""
    product = price.get("product")
    if isinstance(product, str) and product:
        return product
    return _id_of(product, f"the product of {price.get('id')}")


# ---------------------------------------------------------------------------
# Customer portal
# ---------------------------------------------------------------------------
def portal_params(products: list[dict], return_url: str) -> dict:
    """The portal's settings. Customers see invoices, change their card, email,
    address and tax ID, and cancel at the period end. They switch only between the
    self-serve plans; an upgrade is invoiced at once (prorated), and a downgrade
    (a cheaper price) waits for the period end."""
    return {
        "default_return_url": return_url,
        "metadata": {METADATA_KEY: PORTAL_METADATA_VALUE},
        "features": {
            "invoice_history": {"enabled": True},
            "payment_method_update": {"enabled": True},
            "customer_update": {"enabled": True, "allowed_updates": list(CUSTOMER_UPDATE_FIELDS)},
            "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
            "subscription_update": {
                "enabled": True,
                "default_allowed_updates": ["price"],
                "proration_behavior": "always_invoice",
                "products": products,
                "schedule_at_period_end": {"conditions": [{"type": "decreasing_item_amount"}]},
            },
        },
    }


def portal_products(specs: Sequence[PriceSpec], prices: dict[str, Optional[dict]]) -> list[dict]:
    """The plans a customer may switch between in the portal, each with its
    monthly Price: the self-serve plans only (Business is sold by invite)."""
    return [{"product": product_id_of(prices[spec.lookup_key] or {}),
             "prices": [_id_of(prices[spec.lookup_key], spec.lookup_key)]}
            for spec in specs if spec.in_portal]


def find_portal(api: StripeApi) -> Optional[dict]:
    """Our portal configuration (metadata ``automatos_key=portal``), if any."""
    return next((item for item in list_all(api, PORTAL_PATH)
                 if (item.get("metadata") or {}).get(METADATA_KEY) == PORTAL_METADATA_VALUE), None)


def ensure_portal(run: SetupRun, specs: Sequence[PriceSpec], prices: dict[str, Optional[dict]],
                  return_url: str) -> Optional[str]:
    """The portal configuration's id: ours, brought to the settings above, or a new
    one. None on a dry run that would create it."""
    existing = find_portal(run.api)
    if run.dry_run:
        action = f"update {existing.get('id')}" if existing else "create one"
        run.say(f"Portal configuration: would {action}")
        return existing.get("id") if existing else None
    params = portal_params(portal_products(specs, prices), return_url)
    if existing is not None:
        path = f"{PORTAL_PATH}/{_id_of(existing, 'the portal configuration')}"
        portal_id = _id_of(run.api.post(path, {**params, "active": True}), "the portal configuration")
        run.say(f"Portal configuration: updated ({portal_id})")
        return portal_id
    created = run.api.post(PORTAL_PATH, params, idempotency_key=run.idempotency_key("portal"))
    portal_id = _id_of(created, "the new portal configuration")
    run.say(f"Portal configuration: created ({portal_id})")
    return portal_id


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------
def find_webhook(api: StripeApi, url: str) -> Optional[dict]:
    """The webhook endpoint at exactly ``url``, if any."""
    return next((item for item in list_all(api, WEBHOOKS_PATH) if item.get("url") == url), None)


def create_webhook(run: SetupRun, url: str, api_version: str, events: Sequence[str]) -> WebhookResult:
    """A new endpoint at our pinned API version. Its signing secret comes back
    only in this answer."""
    if run.dry_run:
        run.say(f"Webhook endpoint: missing, would create at {url} ({len(events)} events, API {api_version})")
        return WebhookResult(endpoint_id=None, secret=None)
    params = {"url": url, "api_version": api_version, "enabled_events": list(events),
              "description": WEBHOOK_DESCRIPTION, "metadata": {METADATA_KEY: WEBHOOK_METADATA_VALUE}}
    created = run.api.post(WEBHOOKS_PATH, params, idempotency_key=run.idempotency_key("webhook"))
    endpoint_id = _id_of(created, "the new webhook endpoint")
    secret = created.get("secret")
    if not isinstance(secret, str) or not secret:
        raise SetupError(f"Stripe created webhook endpoint {endpoint_id} but returned no signing secret. "
                         "Reveal it in the Dashboard (Developers → Webhooks → the endpoint).")
    run.say(f"Webhook endpoint: created ({endpoint_id}), {len(events)} events, API {api_version}")
    return WebhookResult(endpoint_id=endpoint_id, secret=secret)


def warn_on_api_version(run: SetupRun, endpoint: dict, api_version: str) -> None:
    """Warn when the endpoint sends events at another API version than config
    pins: the handlers read the pinned shapes, and Stripe can't change it."""
    current = endpoint.get("api_version")
    if current == api_version:
        return
    run.say(f"WARNING: webhook endpoint {endpoint.get('id')} sends events at API version "
            f"{current or 'the account default'}, but config pins {api_version}. Stripe can't change "
            "an endpoint's version: delete this endpoint in the Dashboard (Developers → Webhooks) "
            "and re-run. The new endpoint gets a new signing secret.")


def update_webhook(run: SetupRun, endpoint: dict, events: Sequence[str]) -> None:
    """Bring an existing endpoint's events to ours (and re-enable it if Stripe or
    someone disabled it). Writes nothing when it already matches."""
    endpoint_id = _id_of(endpoint, "the webhook endpoint")
    current = set(endpoint.get("enabled_events") or [])
    disabled = endpoint.get("status") == "disabled"
    if current == set(events) and not disabled:
        run.say(f"Webhook endpoint: exists ({endpoint_id}), events up to date")
        return
    change = (f"+{sorted(set(events) - current)} -{sorted(current - set(events))}"
              + (", re-enable" if disabled else ""))
    if run.dry_run:
        run.say(f"Webhook endpoint: exists ({endpoint_id}), would update: {change}")
        return
    params = {"enabled_events": list(events), **({"disabled": False} if disabled else {})}
    run.api.post(f"{WEBHOOKS_PATH}/{endpoint_id}", params)
    run.say(f"Webhook endpoint: exists ({endpoint_id}), updated: {change}")


def ensure_webhook(run: SetupRun, url: str, api_version: str, events: Sequence[str]) -> WebhookResult:
    """The webhook endpoint at ``url`` with our events: created when missing (the
    only time its secret is known), else updated in place."""
    existing = find_webhook(run.api, url)
    if existing is None:
        return create_webhook(run, url, api_version, events)
    warn_on_api_version(run, existing, api_version)
    update_webhook(run, existing, events)
    return WebhookResult(endpoint_id=_id_of(existing, "the webhook endpoint"), secret=None)


# ---------------------------------------------------------------------------
# Report and command line
# ---------------------------------------------------------------------------
def env_lines(specs: Sequence[PriceSpec], prices: dict[str, Optional[dict]], portal_id: Optional[str],
              account_id: str, webhook: WebhookResult) -> list[str]:
    """The ``NAME=value`` lines for Railway. The webhook secret is there only when
    the endpoint was just created; a value a dry run couldn't know is a comment."""
    values = [(spec.env_name, (prices.get(spec.lookup_key) or {}).get("id")) for spec in specs]
    values = values + [(PORTAL_ENV_NAME, portal_id), (ACCOUNT_ENV_NAME, account_id)]
    if webhook.secret:
        values = values + [(WEBHOOK_SIGNING_ENV_NAME, webhook.secret)]
    return [f"{name}={value}" if value else f"# {name}: known after a real run (this was a dry run)"
            for name, value in values]


def print_env_block(run: SetupRun, lines: Sequence[str], webhook: WebhookResult, live: bool) -> None:
    """The closing block: the lines to set on the Railway API service."""
    run.say()
    if webhook.secret:
        run.say(f"IMPORTANT: set {WEBHOOK_SIGNING_ENV_NAME} now. Stripe's API returns the signing secret "
                "only when the endpoint is created, so this script can't show it again.")
    mode = "LIVE" if live else "test"
    run.say(f"Set these on the Railway API service ({mode} mode), next to STRIPE_SECRET_KEY (not printed):")
    for line in lines:
        run.say(line)


def say_header(run: SetupRun, options: Options, settings: Settings) -> None:
    """The first lines: which mode, whether anything is written, and a warning when
    the portal's return link wouldn't reach the public app."""
    mode = "LIVE" if options.live else "test"
    dry = " (dry run: nothing is written)" if options.dry_run else ""
    run.say(f"Stripe setup, {mode} mode{dry}; API version {settings.api_version}")
    if not settings.frontend_url.startswith("https://"):
        run.say(f"WARNING: FRONTEND_URL is {settings.frontend_url!r}, so the portal sends customers back "
                "there. Run this where FRONTEND_URL is the app's public https URL (the API service).")


def _setup(run: SetupRun, options: Options, settings: Settings) -> int:
    """The steps, in order: account, Prices, portal, webhook, then the report. The
    webhook comes last, so nothing can fail between its secret and the report."""
    say_header(run, options, settings)
    specs = price_specs(settings, options.tax_behavior)
    account_id = _id_of(run.api.get(ACCOUNT_PATH), "the account")
    run.say(f"Account: {account_id}")
    prices = ensure_prices(run, specs)
    return_url = settings.frontend_url.rstrip("/") + BILLING_TAB_PATH
    portal_id = ensure_portal(run, specs, prices, return_url)
    webhook = ensure_webhook(run, options.webhook_url, settings.api_version, tuple(WEBHOOK_EVENTS))
    print_env_block(run, env_lines(specs, prices, portal_id, account_id, webhook), webhook, options.live)
    return EXIT_OK


def execute(options: Options, settings: Settings, *, out: TextIO,
            transport: Optional[httpx.BaseTransport] = None, run_id: Optional[str] = None) -> int:
    """Run the setup and return the exit code: 0 done, 1 stopped (a Price mismatch
    or a Stripe error, reason printed), 2 refused before any call to Stripe."""
    problem = key_mode_problem(settings.secret_key, options.live) or webhook_url_problem(options.webhook_url)
    if problem:
        print(f"REFUSED: {problem}", file=out)
        return EXIT_REFUSED
    api = StripeApi(settings.secret_key, api_version=settings.api_version, timeout=settings.timeout,
                    transport=transport)
    run = SetupRun(api=api, out=out, dry_run=options.dry_run, run_id=run_id or uuid.uuid4().hex)
    try:
        return _setup(run, options, settings)
    except SetupError as exc:
        run.say(f"STOPPED: {exc}")
    except StripeError as exc:
        run.say(f"STOPPED: Stripe refused a request (HTTP {exc.status}, {exc.code or 'no code'}): {exc.message}")
    except httpx.TransportError as exc:
        run.say(f"STOPPED: couldn't reach Stripe ({type(exc).__name__}). Re-run: what was "
                "already created is found again, not duplicated.")
    return EXIT_FAILED


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    """The flags (see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.stripe_setup",
        description="Make the Stripe account match Automatos's billing config, then print the "
                    "env vars to set on the Railway API service.")
    parser.add_argument("--webhook-url", required=True,
                        help=f"the API's public webhook URL, https://<api host>{WEBHOOK_PATH}")
    parser.add_argument("--tax-behavior", choices=TAX_BEHAVIORS, default=DEFAULT_TAX_BEHAVIOR,
                        help="whether plan prices exclude or include VAT (decision D-7); "
                             f"default {DEFAULT_TAX_BEHAVIOR}")
    parser.add_argument("--live", action="store_true", help="required with a live key, refused with a test key")
    parser.add_argument("--dry-run", action="store_true", help="read Stripe and report; write nothing")
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    return Options(webhook_url=namespace.webhook_url.strip(), tax_behavior=namespace.tax_behavior,
                   live=namespace.live, dry_run=namespace.dry_run)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The command: parse the flags, read config, set Stripe up, print the env lines."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return execute(parse_args(argv), settings_from_config(), out=sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
