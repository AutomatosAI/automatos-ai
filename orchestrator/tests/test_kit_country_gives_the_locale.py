"""The kit's country gives its currency and date style unless the kit sets them (Gerard, 7 Oct).

``country`` (ISO 3166-1 alpha-2) is on the Brand kit page's Locale card and in the
update tool. An empty ``currency`` is the country's and an empty ``date_style`` the
country's (``modules/documents/country_locale.py``): GB is GBP and day first, IE and
the other euro countries EUR, US USD and month first. A currency or date style the kit
sets wins. Every renderer reads them through ``locale_text``, so a kit with a country
prints its amounts in its currency and Auto never asks which (F367). Pins:

* the country is stored upper case; one the table does not hold is refused, saying
  so; empty is no country;
* the currency and the date style follow the country, and a set one wins;
* with a country, no amount is reported as printed without a currency, and the agents'
  rules say the country's currency and date style;
* the PUT saves it and the agent tool's schema takes it.
"""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from modules.documents.brand_kit import get_brand_kit, validate_brand_kit
from modules.documents.brand_system import CURRENCY_CODE, DATE_STYLE_DAY_FIRST, DATE_STYLE_MONTH_FIRST, DATE_STYLES
from modules.documents.country_locale import COUNTRY_LOCALES, EURO_AREA
from modules.documents.currency_notice import amounts_without_currency
from modules.documents.locale_text import currency_of, date_style_of, long_date
from modules.tools.discovery.action_registry import get_action_registry
from services.brand_design_rules import currency_line, date_line
from tests import test_prd255w1_palette_roles as roles_tests

api = roles_tests.api  # the documents router over one workspace holding a v1 kit
KIT_ROUTE = roles_tests.KIT_ROUTE
INVOICE = {"client_name": "Lantern Kitchen", "total": 269}
DAY = date(2026, 10, 5)


def _refusal(patch):
    with pytest.raises(ValidationError) as caught:
        validate_brand_kit(patch)
    return caught.value.errors()


def test_the_country_is_stored_upper_case_and_one_the_table_lacks_is_refused():
    assert get_brand_kit(None)["country"] == ""
    assert validate_brand_kit({"country": " gb "})["country"] == "GB"
    assert validate_brand_kit({"country": ""}, {"country": "IE"})["country"] == ""
    for bad in ("XX", "GBR", "United Kingdom", "G1"):
        (error,) = _refusal({"country": bad})
        assert error["loc"] == ("country",) and "ISO 3166-1" in error["msg"], bad


def test_the_currency_and_the_date_style_follow_the_country():
    assert (currency_of({"country": "GB"}), date_style_of({"country": "GB"})) == ("GBP", DATE_STYLE_DAY_FIRST)
    assert (currency_of({"country": "US"}), date_style_of({"country": "US"})) == ("USD", DATE_STYLE_MONTH_FIRST)
    assert currency_of({"country": "IE"}) == currency_of({"country": "DE"}) == "EUR"
    assert long_date(DAY, date_style_of(get_brand_kit({"brand_kit": {"country": "US"}}))) == "October 5, 2026"
    # No country: no currency, and dates print day first, as before.
    assert currency_of(get_brand_kit(None)) == "" and long_date(DAY, date_style_of(get_brand_kit(None))) == "5 October 2026"


def test_a_currency_or_date_style_the_kit_sets_wins():
    kit = get_brand_kit({"brand_kit": {"country": "US", "currency": "EUR", "date_style": DATE_STYLE_DAY_FIRST}})
    assert (currency_of(kit), date_style_of(kit)) == ("EUR", DATE_STYLE_DAY_FIRST)


def test_every_country_has_a_currency_code_and_a_date_style_and_the_euro_area_is_eur():
    for country, (currency, style) in COUNTRY_LOCALES.items():
        assert len(country) == 2 and CURRENCY_CODE.match(currency) and style in DATE_STYLES, country
    assert all(COUNTRY_LOCALES[country][0] == "EUR" for country in EURO_AREA)


def test_with_a_country_no_amount_is_said_to_print_without_a_currency():
    assert amounts_without_currency(get_brand_kit(None), INVOICE, "pdf") == ["total"]  # F367: ask once
    assert amounts_without_currency(get_brand_kit({"brand_kit": {"country": "GB"}}), INVOICE, "pdf") == []


def test_the_agents_rules_say_the_countrys_currency_and_dates():
    kit = get_brand_kit({"brand_kit": {"country": "US"}})
    assert currency_line(kit) == "- Currency: USD ($). Print every amount in it, never another currency."
    assert date_line(kit) == "- Dates: written as October 5, 2026."


def test_the_put_saves_the_country_and_the_tool_takes_it(api):
    saved = api.client.put(KIT_ROUTE, json={"country": "us"})
    assert saved.status_code == 200, saved.text
    assert api.client.get(KIT_ROUTE).json()["country"] == "US"
    stored = api.workspace.settings["brand_kit"]
    assert (stored["country"], stored["currency"], stored["date_style"]) == ("US", "", "")
    refused = api.client.put(KIT_ROUTE, json={"country": "XX"})
    assert refused.status_code == 422
    properties = get_action_registry().get("platform_update_brand_kit").parameters["properties"]
    assert "GB" in properties["country"]["description"] and properties["date_style"]["enum"][-1] == ""
