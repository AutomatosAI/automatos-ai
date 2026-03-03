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
