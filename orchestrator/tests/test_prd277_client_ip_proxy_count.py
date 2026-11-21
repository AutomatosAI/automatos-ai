"""PRD-277: ``CLIENT_IP_PROXY_COUNT`` at start-up, and the rules ``core.client_ip_settings`` holds it to.

``config.py`` reads the count as text, so importing it never raises. Proven by reloading
``config.py`` under each value:

* unset, it is empty, it is in ``reports/config-surface.json``, and start-up hands it in;
* in ``enterprise``: with ``CLIENT_IP_HEADER`` set to a header proxies append to (``X-Forwarded-For``,
  ``X-Original-Forwarded-For``, in any case, with spaces) the count must be stated; when stated it
  must be a plain whole number from 1 to ``PROXY_COUNT_MAX``; a count with no header is refused,
  named before the missing header; and ``Forwarded``, whose values are parameters, is refused as the
  header. Each refusal names the setting and never a value;
* a header the proxy sets starts with or without a count, and a header proxies append to starts
  with any count in range;
* ``local`` and ``saas`` start whatever the count and the header are;
* the chart refuses the same range, with the same bound, the same headers, and
  ``config.FORWARDED_ALLOW_IPS``.
"""
from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core import client_ip_settings  # noqa: E402
from core.client_ip_settings import PROXY_COUNT_MAX, MalformedProxyCount, proxy_count  # noqa: E402
from core.edition_start_up import OIDC_SIGN_IN_SETTINGS, EditionNotReady  # noqa: E402

ORCH = Path(__file__).resolve().parents[1]
CHART_TEMPLATE = ORCH.parent / "charts" / "automatos" / "templates" / "_enterprise_api.tpl"
ENV_KEYS = (
    "AUTH_EDITION", "REQUIRE_AUTH", "OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "FRONTEND_URL",
    "CLI_RUNTIME_ENABLED", "DEFAULT_WORKSPACE_ID", "CLERK_JWKS_URL", "CLERK_SECRET_KEY",
    "LOCAL_REMOTE_ACCESS", "API_PUBLISH_ADDRESS", "WEB_PUBLISH_ADDRESS", "CLIENT_IP_HEADER",
    "CLIENT_IP_PROXY_COUNT", "OIDC_REQUIRED_CLAIMS", "ENTERPRISE_BOOTSTRAP_ADMINS",
)
OIDC_ENV = {
    "OIDC_ISSUER": "https://idp.example.test/realms/prd277-proxy-count",
    "OIDC_CLIENT_ID": "prd277-proxy-count-client",
    "OIDC_CLIENT_SECRET": "prd277" + "-" + "proxy-count" + "-" + "marker",
    "FRONTEND_URL": "https://automatos.example.test",
}
EDITION_ENV = {
    "local": {"AUTH_EDITION": "local", "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277f1"},
    "saas": {"AUTH_EDITION": "saas", "CLERK_JWKS_URL": "https://clerk.example.test/.well-known/jwks.json",
             "CLERK_SECRET_KEY": "prd277" + "-" + "proxy-count-stand-in"},
    "enterprise": {"AUTH_EDITION": "enterprise", "CLIENT_IP_HEADER": "X-Real-IP",
                   "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277f3", **OIDC_ENV},
}
FORWARDED_FOR = "X-Forwarded-For"
ORIGINAL_FORWARDED_FOR = "X-Original-Forwarded-For"
MALFORMED = ["0", "9", "-1", "1.5", "two", "+1", "01", "١", "1 2", "0x1", "1000", "9" * 5000]


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Put ``config``'s attributes back after a reload, so the fresh singleton never leaks."""
    import config as config_mod

    saved = dict(config_mod.__dict__)
    try:
        yield
    finally:
        config_mod.__dict__.clear()
        config_mod.__dict__.update(saved)


def _fresh_config(**env):
    """Reload config.py with only ``env`` set among ``ENV_KEYS`` and return its singleton."""
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    try:
        import config as config_mod

        importlib.reload(config_mod)
        return config_mod.config
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _refusal(**env) -> str:
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**{**EDITION_ENV["enterprise"], **env}).validate_auth_edition()
    message = str(refused.value)
    assert OIDC_ENV["OIDC_CLIENT_SECRET"] not in message
    return message


def test_unset_it_is_empty_named_in_the_surface_and_handed_to_start_up():
    cfg = _fresh_config(**EDITION_ENV["saas"])
    assert cfg.CLIENT_IP_PROXY_COUNT == ""
    assert _fresh_config(**EDITION_ENV["saas"], CLIENT_IP_PROXY_COUNT=" 2 ").CLIENT_IP_PROXY_COUNT == "2"
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert "CLIENT_IP_PROXY_COUNT" in surface and surface == sorted(surface)
    assert "CLIENT_IP_PROXY_COUNT" in OIDC_SIGN_IN_SETTINGS


@pytest.mark.parametrize("header", [FORWARDED_FOR, "x-forwarded-for", "  X-FORWARDED-FOR ",
                                    ORIGINAL_FORWARDED_FOR, " x-original-forwarded-for "])
def test_enterprise_behind_a_header_proxies_append_to_must_state_the_count(header):
    message = _refusal(CLIENT_IP_HEADER=header)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: CLIENT_IP_PROXY_COUNT must be set: ")
    assert "how many of your own proxies append to it" in message
    assert header.strip() not in message


@pytest.mark.parametrize("count", MALFORMED)
def test_enterprise_refuses_a_count_out_of_range_naming_it_never_its_value(count):
    message = _refusal(CLIENT_IP_HEADER=FORWARDED_FOR, CLIENT_IP_PROXY_COUNT=count)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: CLIENT_IP_PROXY_COUNT must be a plain "
                              f"whole number from 1 to {PROXY_COUNT_MAX}: ")
    assert count not in message.split("CLIENT_IP_PROXY_COUNT", 1)[1].replace(f"1 to {PROXY_COUNT_MAX}", "")


def test_enterprise_refuses_a_count_without_a_header_naming_the_count_first():
    env = {name: value for name, value in EDITION_ENV["enterprise"].items() if name != "CLIENT_IP_HEADER"}
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**env, CLIENT_IP_PROXY_COUNT="1").validate_auth_edition()
    assert str(refused.value).startswith(
        "AUTH_EDITION=enterprise cannot start: CLIENT_IP_PROXY_COUNT needs CLIENT_IP_HEADER: ")


@pytest.mark.parametrize(("header", "count"), [("Forwarded", ""), (" forwarded ", ""), ("FORWARDED", "1")])
def test_enterprise_refuses_the_forwarded_header_whose_values_are_not_addresses(header, count):
    message = _refusal(CLIENT_IP_HEADER=header, CLIENT_IP_PROXY_COUNT=count)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: CLIENT_IP_HEADER must name a header that "
                              "holds one IP address, or IP addresses separated by commas")
    # The operator's value is never quoted: the only "Forwarded" is the example X-Forwarded-For.
    assert not re.search(rf"(?<!X-){re.escape(header.strip())}(?!-For)", message)


@pytest.mark.parametrize(("header", "count"), [
    (FORWARDED_FOR, "1"), (FORWARDED_FOR, "2"), (FORWARDED_FOR, str(PROXY_COUNT_MAX)), ("x-forwarded-for", " 1 "),
    (ORIGINAL_FORWARDED_FOR, "1"), ("X-Real-IP", ""), ("X-Real-IP", "1"), ("X-Real-IP", "   "),
])
def test_enterprise_starts_with_a_usable_pair(header, count):
    _fresh_config(**{**EDITION_ENV["enterprise"], "CLIENT_IP_HEADER": header,
                     "CLIENT_IP_PROXY_COUNT": count}).validate_auth_edition()  # must not raise


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize(("header", "count"), [(FORWARDED_FOR, ""), (ORIGINAL_FORWARDED_FOR, ""), ("Forwarded", ""),
                                              ("", "1"), ("", "two")]
                         + [(FORWARDED_FOR, count) for count in MALFORMED])
def test_local_and_saas_start_whatever_the_pair(edition, header, count):
    _fresh_config(**EDITION_ENV[edition], CLIENT_IP_HEADER=header,
                  CLIENT_IP_PROXY_COUNT=count).validate_auth_edition()  # must not raise


@pytest.mark.parametrize(("value", "count"), [(None, None), ("", None), ("  ", None), ("1", 1), (" 8 ", 8), (3, 3)])
def test_a_count_is_read_or_left_unset(value, count):
    assert proxy_count(value) == count


@pytest.mark.parametrize("value", MALFORMED + [0, 9, True, False, 1.0])
def test_anything_else_is_malformed_never_another_count(value):
    with pytest.raises(MalformedProxyCount):
        proxy_count(value)


def test_the_header_rules_read_the_header_as_start_up_does():
    required = [("CLIENT_IP_PROXY_COUNT", client_ip_settings.COUNT_REQUIRED_PROBLEM)]
    assert list(client_ip_settings.start_up_problems(" X-Forwarded-For ", None)) == required
    assert list(client_ip_settings.start_up_problems("X-ORIGINAL-FORWARDED-FOR", "")) == required
    assert list(client_ip_settings.start_up_problems("Forwarded", "2")) == [
        ("CLIENT_IP_HEADER", client_ip_settings.PARAMETER_HEADER_PROBLEM)]
    assert list(client_ip_settings.start_up_problems("X-Real-IP", None)) == []
    assert client_ip_settings.count_problem(FORWARDED_FOR, None) is None  # outside enterprise: as before


def test_the_chart_refuses_the_same_range_bound_and_headers():
    template = CHART_TEMPLATE.read_text(encoding="utf-8")
    assert f"must be a plain whole number from 1 to {PROXY_COUNT_MAX} (CLIENT_IP_PROXY_COUNT)" in template
    assert f"(gt (atoi $count) {PROXY_COUNT_MAX})" in template
    assert "`^[1-9][0-9]{0,2}$`" in template and "CLIENT_IP_HEADER CLIENT_IP_PROXY_COUNT " in template
    appended = " ".join(f'"{name}"' for name in client_ip_settings.APPENDED_HEADERS)
    assert f"(list {appended})" in template
    assert all(f'eq $header "{name}"' in template for name in client_ip_settings.PARAMETER_HEADERS)
    assert 'hasKey $config "FORWARDED_ALLOW_IPS"' in template
