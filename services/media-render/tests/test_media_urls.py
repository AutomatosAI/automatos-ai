"""The storage allowlist (US-102): media come only from our own storage."""

from __future__ import annotations

import pytest

from media_render.config import ConfigError, load_settings
from media_render.media_urls import parse_prefix, parse_prefixes, redact, url_allowed

S3 = parse_prefixes("https://automatos-media.s3.eu-west-2.amazonaws.com, http://minio:9000/automatos/")


@pytest.mark.parametrize(
    "url",
    [
        "https://automatos-media.s3.eu-west-2.amazonaws.com/ws/post/hook.mp4?X-Amz-Signature=abc",
        "https://AUTOMATOS-MEDIA.s3.eu-west-2.amazonaws.com/ws/hook.mp4",
        "https://automatos-media.s3.eu-west-2.amazonaws.com:443/ws/hook.mp4",
        "http://minio:9000/automatos/ws/hook.mp4?sig=1",
    ],
)
def test_our_storage_is_allowed(url):
    assert url_allowed(url, S3)


@pytest.mark.parametrize(
    "url",
    [
        "https://automatos-media.s3.eu-west-2.amazonaws.com.evil.example/ws/hook.mp4",
        "https://evil.example/automatos-media.s3.eu-west-2.amazonaws.com/hook.mp4",
        "http://automatos-media.s3.eu-west-2.amazonaws.com/ws/hook.mp4",
        "https://automatos-media.s3.eu-west-2.amazonaws.com:444/ws/hook.mp4",
        "https://me:pw@automatos-media.s3.eu-west-2.amazonaws.com/ws/hook.mp4",
        "https://evil.example\\@automatos-media.s3.eu-west-2.amazonaws.com/hook.mp4",
        "https://automatos-media.s3.eu-west-2.amazonaws.com/ws/hook.mp4#frag",
        "http://minio:9000/other-bucket/hook.mp4",
        "http://minio:9000/automatos/../other-bucket/hook.mp4",
        "http://minio:9000/automatos/%2E%2E/other-bucket/hook.mp4",
        "http://minio:9000/automatos%2fx/hook.mp4",
        "http://minio:9000/automatos",
        "http://minio:/automatos/hook.mp4",
        "http://minio:99999/automatos/hook.mp4",
        "ftp://minio:9000/automatos/hook.mp4",
        "https://automatos-media.s3.eu-west-2.amazonaws.com/ws/hook .mp4",
        "",
        "not a url",
    ],
)
def test_anything_else_is_refused(url):
    assert not url_allowed(url, S3)


def test_nothing_is_allowed_by_an_empty_allowlist():
    assert not url_allowed("https://automatos-media.s3.eu-west-2.amazonaws.com/x.mp4", ())


def test_a_prefix_path_always_ends_in_a_slash():
    assert parse_prefix("http://minio:9000/automatos").path == "/automatos/"
    assert parse_prefix("https://bucket.s3.amazonaws.com").path == "/"


@pytest.mark.parametrize("raw", ["minio:9000/automatos", "https://x.example/?a=1", "https://u@x.example/", "ftp://x.example/"])
def test_a_bad_prefix_stops_the_container(raw):
    with pytest.raises(ConfigError, match="MEDIA_RENDER_MEDIA_URL_PREFIXES"):
        load_settings({"MEDIA_RENDER_MEDIA_URL_PREFIXES": raw})


def test_the_allowlist_is_read_from_config():
    settings = load_settings({"MEDIA_RENDER_MEDIA_URL_PREFIXES": "https://a.example/x/ https://b.example"})
    assert [str(prefix) for prefix in settings.media_url_prefixes] == ["https://a.example:443/x/", "https://b.example:443/"]
    assert load_settings({}).media_url_prefixes == ()


def test_redact_drops_the_signature_and_credentials():
    assert redact("https://me:pw@a.example/k/v.mp4?X-Amz-Signature=abc") == "https://a.example/k/v.mp4"
