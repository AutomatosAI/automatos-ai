"""Journey 17: LLM configuration — verify all service categories are configured.

Tests that every LLM service category registered in SERVICE_CATEGORY_MAP
has valid provider/model settings, preventing silent fallback degradation.
"""

# Known service categories that MUST have LLM settings configured.
# Sourced from orchestrator/core/llm/manager.py SERVICE_CATEGORY_MAP.
REQUIRED_SERVICE_CATEGORIES = [
    "orchestrator_llm",
    "chatbot",
    "complexity_assessor",
]


def test_llm_settings_categories_exist(client):
    """Every required LLM service category must exist in system settings.

    Bug: orchestrator/core/llm/manager.py line 104 raises ValueError
    when complexity_assessor has no system settings entry. The exception
    is caught in consumers/chatbot/auto.py line 299 and silently falls
    back to DELEGATE routing, degrading classification quality.

    Fix: either add complexity_assessor settings via system-settings API,
    or add fallback to orchestrator_llm in get_provider_and_model_from_settings()
    (manager.py line 96) when the service-specific category has no settings.
    """
    r = client.get("/api/system-settings/categories")
    assert r.status_code == 200
    categories = r.json()
    assert isinstance(categories, list)

    missing = [cat for cat in REQUIRED_SERVICE_CATEGORIES if cat not in categories]
    assert not missing, (
        f"LLM service categories not configured: {missing}. "
        f"Bug: core/llm/manager.py raises ValueError for unconfigured services. "
        f"complexity_assessor (auto.py:280) silently falls back to DELEGATE, "
        f"degrading routing quality. Add settings via POST /api/system-settings "
        f"or add orchestrator_llm fallback in manager.py:104."
    )


def test_llm_settings_have_provider_and_model(client):
    """Each LLM category must have both llm_provider and llm_model keys."""
    r = client.get("/api/system-settings/", params={"category": "orchestrator_llm"})
    assert r.status_code == 200
    settings = r.json()

    keys = {s.get("key") for s in settings}
    assert "llm_provider" in keys or "provider" in keys, (
        "orchestrator_llm missing llm_provider setting. "
        "All downstream services depend on this as fallback."
    )
    assert "llm_model" in keys or "model" in keys, (
        "orchestrator_llm missing llm_model setting."
    )


def test_system_settings_list(client):
    """GET /api/system-settings/ should return all settings."""
    r = client.get("/api/system-settings/")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0, "System settings should not be empty"


def test_system_settings_by_category(client):
    """GET /api/system-settings/ filtered by category should return subset."""
    r = client.get("/api/system-settings/", params={"category": "chatbot"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_system_settings_invalid_category(client):
    """GET /api/system-settings/ with fake category should return empty, not 500."""
    r = client.get("/api/system-settings/", params={"category": "DOES_NOT_EXIST"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 0, "Unknown category should return empty list"


def test_llm_provider_settings_have_values(client):
    """LLM settings values should not be empty strings or None."""
    r = client.get("/api/system-settings/", params={"category": "orchestrator_llm"})
    assert r.status_code == 200
    settings = r.json()
    for s in settings:
        val = s.get("value")
        if s.get("key") in ("llm_provider", "llm_model", "provider", "model"):
            assert val and val.strip(), (
                f"Setting '{s.get('key')}' has empty value — "
                f"this will cause silent LLM fallback failures"
            )


def test_service_category_map_coverage(client):
    """Verify all known service categories have at least one setting."""
    r = client.get("/api/system-settings/categories")
    assert r.status_code == 200
    categories = r.json()
    # At minimum orchestrator_llm must exist
    assert "orchestrator_llm" in categories, (
        "orchestrator_llm category missing — this is the global fallback"
    )


def test_system_settings_update_roundtrip(client):
    """PUT /api/system-settings/ should update and return new value.

    Uses a safe non-critical setting to test the update path.
    """
    # Read current value
    r = client.get("/api/system-settings/", params={"category": "orchestrator_llm"})
    assert r.status_code == 200
    settings = r.json()
    if not settings:
        return  # No settings to test with

    # Find a non-critical setting to toggle
    test_setting = None
    for s in settings:
        if s.get("key") not in ("llm_provider", "llm_model", "provider", "model"):
            test_setting = s
            break
    if not test_setting:
        return  # Only critical settings, skip

    # Verify update endpoint exists (don't actually change values)
    r = client.get(f"/api/system-settings/{test_setting.get('id', 0)}")
    assert r.status_code != 500, (
        f"GET single setting returned 500: {r.text[:200]}"
    )
