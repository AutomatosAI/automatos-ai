"""Firecrawl wire formats used by the business intake wizard."""

import json

import httpx
import pytest

from modules.intake import firecrawl_client


@pytest.fixture
def transport(monkeypatch):
    """Capture outbound requests without spending Firecrawl credits."""
    requests = []
    responses = []
    client_type = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=responses.pop(0))

    monkeypatch.setattr(
        firecrawl_client.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handle)),
    )
    return requests, responses


@pytest.mark.asyncio
async def test_schema_scrape_keeps_the_intake_extract_contract(transport):
    requests, responses = transport
    responses.append({"success": True, "data": {"markdown": "page", "json": {}}})
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    client = firecrawl_client.FirecrawlClient(api_key="test")

    result = await client.scrape("https://example.com", schema=schema)

    assert str(requests[0].url) == "https://api.firecrawl.dev/v2/scrape"
    payload = json.loads(requests[0].content)
    assert payload["formats"] == ["markdown", {"type": "json", "schema": schema}]
    assert "extract" not in payload
    assert result["extract"] == {}
    assert result["markdown"] == "page"


@pytest.mark.asyncio
async def test_plain_scrape_uses_markdown_without_extraction(transport):
    requests, responses = transport
    responses.append({"success": True, "data": {"markdown": "page"}})
    client = firecrawl_client.FirecrawlClient(api_key="test")

    result = await client.scrape("https://example.com")

    assert str(requests[0].url) == "https://api.firecrawl.dev/v2/scrape"
    assert json.loads(requests[0].content)["formats"] == ["markdown"]
    assert result["extract"] is None


@pytest.mark.asyncio
async def test_map_reads_url_objects_and_preserves_the_domain_filter(transport):
    requests, responses = transport
    responses.append(
        {
            "success": True,
            "links": [
                {"url": "https://example.com/about", "title": "About"},
                {"url": "https://other.example/about", "title": "Other"},
            ],
        }
    )
    client = firecrawl_client.FirecrawlClient(api_key="test")

    assert await client.map("example.com") == ["https://example.com/about"]
    assert str(requests[0].url) == "https://api.firecrawl.dev/v2/map"
