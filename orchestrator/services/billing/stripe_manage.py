"""Managing a Stripe subscription from inside Studio (PRD-265 W2, no portal).

The Billing tab does what Stripe's hosted portal would, so the customer never
leaves Studio (Gerard, 23 Oct 2026):

- **Card**: the card the subscription is charged on (else the customer's default);
  a new one is entered in our tab through a SetupIntent and then made the default
  for both the customer and the subscription.
- **Invoices**: listed from Stripe, and each PDF fetched by our server and handed to
  the browser, so it never visits a Stripe page.
- **Cancel / resume**: ``cancel_at_period_end``; the plan runs to the period end.
- **Plan change**: an upgrade moves now and charges the prorated difference
  (``pending_if_incomplete``: if the bank wants 3-D Secure the change waits, and the
  client finishes it with the payment's client secret); a downgrade waits for the
  period end through a subscription schedule, so nothing is charged or refunded now.

Every Stripe id that goes into a URL path is checked against its shape first.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from services.billing.provider import CardOnFile, InvoiceLine, PlanChangePreview, PlanChangeResult

INVOICES_SHOWN = 24
INVOICE_PDF_MAX_BYTES = 10 * 1024 * 1024
_IDS = {
    "cus": re.compile(r"^cus_[A-Za-z0-9]+$"),
    "sub": re.compile(r"^sub_[A-Za-z0-9]+$"),
    "seti": re.compile(r"^seti_[A-Za-z0-9]+$"),
    "in": re.compile(r"^in_[A-Za-z0-9]+$"),
    "pm": re.compile(r"^pm_[A-Za-z0-9]+$"),
    "sub_sched": re.compile(r"^sub_sched_[A-Za-z0-9]+$"),
    "si": re.compile(r"^si_[A-Za-z0-9]+$"),
}


def checked(kind: str, value: Optional[str]) -> str:
    """``value`` when it has the shape of a Stripe ``kind`` id, else ValueError."""
    if not value or not _IDS[kind].match(value):
        raise ValueError(f"not a Stripe {kind} id")
    return value


def _id(value: Any) -> Optional[str]:
    return value.get("id") if isinstance(value, dict) else (value or None)


def _at(timestamp: Any) -> Optional[datetime]:
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc) if timestamp else None


def card_of(method: Any) -> Optional[CardOnFile]:
    """A payment method as the Billing tab shows it. Pure."""
    if not isinstance(method, dict):
        return None
    kind = str(method.get("type") or "card")
    details = method.get(kind) if isinstance(method.get(kind), dict) else {}
    return CardOnFile(brand=str(details.get("brand") or kind), last4=str(details.get("last4") or ""),
                      exp_month=details.get("exp_month"), exp_year=details.get("exp_year"))


class StripeManagement:
    """The management half of :class:`services.billing.stripe_provider.StripeProvider`
    (it supplies ``self._api``)."""

    def _subscription(self, subscription_id: str, expand: tuple = ()) -> dict:
        params = {"expand": list(expand)} if expand else None
        return self._api.get(f"/v1/subscriptions/{checked('sub', subscription_id)}", params)

    def card_on_file(self, *, customer_id: str, subscription_id: Optional[str]) -> Optional[CardOnFile]:
        if subscription_id:
            card = card_of(self._subscription(subscription_id, ("default_payment_method",)).get("default_payment_method"))
            if card:
                return card
        customer = self._api.get(f"/v1/customers/{checked('cus', customer_id)}",
                                 {"expand": ["invoice_settings.default_payment_method"]})
        return card_of((customer.get("invoice_settings") or {}).get("default_payment_method"))

    def start_card_change(self, *, customer_id: str, workspace_id: str) -> str:
        setup = self._api.post("/v1/setup_intents", {
            "customer": checked("cus", customer_id), "usage": "off_session",
            "automatic_payment_methods": {"enabled": True}, "metadata": {"workspace_id": workspace_id},
        })
        return setup["client_secret"]

    def finish_card_change(self, *, customer_id: str, subscription_id: Optional[str], setup_id: str) -> CardOnFile:
        setup = self._api.get(f"/v1/setup_intents/{checked('seti', setup_id)}")
        if _id(setup.get("customer")) != customer_id or setup.get("status") != "succeeded":
            raise ValueError("that card setup isn't this workspace's, or isn't finished")
        method_id = checked("pm", _id(setup.get("payment_method")))
        self._api.post(f"/v1/customers/{checked('cus', customer_id)}",
                       {"invoice_settings": {"default_payment_method": method_id}})
        if subscription_id:
            self._api.post(f"/v1/subscriptions/{checked('sub', subscription_id)}",
                           {"default_payment_method": method_id})
        return card_of(self._api.get(f"/v1/payment_methods/{method_id}"))

    def invoices(self, *, customer_id: str) -> list[InvoiceLine]:
        page = self._api.get("/v1/invoices", {"customer": checked("cus", customer_id), "limit": INVOICES_SHOWN})
        return [InvoiceLine(id=item["id"], number=item.get("number"), status=item.get("status"),
                            total_cents=int(item.get("total") or 0), currency=str(item.get("currency") or "usd"),
                            created=_at(item.get("created"))) for item in page.get("data") or []]

    def invoice_pdf(self, *, customer_id: str, invoice_id: str) -> tuple[bytes, str]:
        invoice = self._api.get(f"/v1/invoices/{checked('in', invoice_id)}")
        if _id(invoice.get("customer")) != customer_id or not invoice.get("invoice_pdf"):
            raise ValueError("not this workspace's invoice")
        name = re.sub(r"[^A-Za-z0-9-]", "", str(invoice.get("number") or invoice_id)) or "invoice"
        return self._api.fetch_file(invoice["invoice_pdf"], max_bytes=INVOICE_PDF_MAX_BYTES), f"{name}.pdf"

    def set_cancel_at_period_end(self, *, subscription_id: str, cancel: bool) -> Optional[datetime]:
        sub = self._api.post(f"/v1/subscriptions/{checked('sub', subscription_id)}", {"cancel_at_period_end": cancel})
        return period_end_of(sub)

    def preview_plan_change(self, *, customer_id: str, subscription_id: str, price_id: str) -> PlanChangePreview:
        item = first_item(self._subscription(subscription_id))
        preview = self._api.post("/v1/invoices/create_preview", {
            "customer": checked("cus", customer_id), "subscription": subscription_id,
            "subscription_details": {"items": [{"id": item, "price": price_id}],
                                     "proration_behavior": "always_invoice", "proration_date": int(time.time())},
        })
        tax = sum(int(t.get("amount") or 0) for t in preview.get("total_taxes") or [])
        return PlanChangePreview(amount_due_cents=int(preview.get("amount_due") or 0), tax_cents=tax,
                                 currency=str(preview.get("currency") or "usd"))

    def upgrade_now(self, *, subscription_id: str, price_id: str) -> PlanChangeResult:
        sub = self._subscription(subscription_id)
        if sub.get("schedule"):
            self.cancel_scheduled_change(schedule_id=_id(sub["schedule"]))
        updated = self._api.post(f"/v1/subscriptions/{checked('sub', subscription_id)}", {
            "items": [{"id": first_item(sub), "price": price_id}], "proration_behavior": "always_invoice",
            "payment_behavior": "pending_if_incomplete", "expand": ["latest_invoice.confirmation_secret"],
        })
        if not updated.get("pending_update"):
            return PlanChangeResult(status="done")
        secret = ((updated.get("latest_invoice") or {}).get("confirmation_secret") or {}).get("client_secret")
        if not secret:
            raise ValueError("the upgrade is waiting for a payment that can't be confirmed here")
        return PlanChangeResult(status="requires_action", client_secret=secret)

    def downgrade_at_period_end(self, *, subscription_id: str, price_id: str) -> PlanChangeResult:
        sub = self._subscription(subscription_id)
        if sub.get("schedule"):
            self.cancel_scheduled_change(schedule_id=_id(sub["schedule"]))
        schedule = self._api.post("/v1/subscription_schedules", {"from_subscription": subscription_id})
        now = schedule["phases"][0]
        current = [{"price": _id(item.get("price")), "quantity": int(item.get("quantity") or 1)}
                   for item in now.get("items") or []]
        self._api.post(f"/v1/subscription_schedules/{checked('sub_sched', schedule['id'])}", {
            "end_behavior": "release", "proration_behavior": "none",
            "phases": [{"items": current, "start_date": now["start_date"], "end_date": now["end_date"]},
                       {"items": [{"price": price_id, "quantity": 1}],
                        "duration": {"interval": "month", "interval_count": 1}}],
        })
        return PlanChangeResult(status="scheduled", effective_at=_at(now["end_date"]), schedule_id=schedule["id"])

    def cancel_scheduled_change(self, *, schedule_id: str) -> None:
        self._api.post(f"/v1/subscription_schedules/{checked('sub_sched', schedule_id)}/release", {})


def first_item(subscription: dict) -> str:
    """The subscription's one item (we sell one price per subscription)."""
    return checked("si", ((subscription.get("items") or {}).get("data") or [{}])[0].get("id"))


def period_end_of(subscription: dict) -> Optional[datetime]:
    items = (subscription.get("items") or {}).get("data") or []
    ends = [item.get("current_period_end") for item in items if item.get("current_period_end")]
    return _at(subscription.get("current_period_end") or (max(ends) if ends else None))
