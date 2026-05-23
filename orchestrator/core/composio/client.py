"""
Composio SDK Client (PRD-36)
============================

Wrapper around the Composio SDK for unified tool integration.

Features:
- Tool Router session management for intent-based action selection
- Hosted OAuth authentication flow
- App discovery and action listing
- Webhook trigger subscription
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from config import config

logger = logging.getLogger(__name__)

# Lazy imports for Composio SDK (may not be installed in all environments)
_composio = None
_composio_toolset = None


def _get_composio():
    """Lazy import of Composio SDK."""
    global _composio
    if _composio is None:
        try:
            from composio import Composio
            _composio = Composio
        except ImportError:
            logger.warning("composio-core not installed. Install with: pip install composio-core composio-openai")
            raise ImportError("Composio SDK not available. Please install composio-core and composio-openai.")
    return _composio


def _get_composio_openai_provider():
    """Lazy import of OpenAI Provider."""
    global _composio_toolset
    if _composio_toolset is None:
        try:
            from composio_openai import OpenAIProvider
            _composio_toolset = OpenAIProvider
        except ImportError:
            logger.warning("composio-openai not installed. Install with: pip install composio-openai")
            raise ImportError("Composio OpenAI SDK not available. Please install composio-openai.")
    return _composio_toolset


class ComposioClient:
    """
    Composio SDK wrapper using Tool Router and Hosted Auth.
    
    This client provides:
    - Entity management (map workspace_id to Composio entities)
    - OAuth connection initiation via Hosted Auth
    - Tool Router session creation
    - App and action discovery
    - Trigger subscription
    
    Usage:
        client = ComposioClient()
        
        # Initiate hosted-auth connection (returns dict with redirect_url +
        # the auth_config_id/scheme actually used — persist these against the
        # workspace's connection row so reconnects reuse the same scheme).
        result = client.initiate_connection(
            entity_id="workspace_123",
            app="GITHUB",
        )
        redirect_url = result["redirect_url"]
        
        # Create Tool Router session
        session = client.create_tool_router_session(
            entity_id="workspace_123",
            apps=["GITHUB", "SLACK"]
        )
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Composio client.
        
        Args:
            api_key: Composio API key. If not provided, uses COMPOSIO_API_KEY env var.
        """
        self.api_key = api_key or config.COMPOSIO_API_KEY
        if not self.api_key:
            logger.warning("COMPOSIO_API_KEY/COMPOSIO_KEY not set. Composio integration will not work.")
        
        self._composio = None
        self._toolset = None
        self._action_count_cache: Dict[str, Any] = {}
        self._action_count_ttl_seconds = 600
        # PERFORMANCE: Cache auth_config_id resolution to avoid repeated API calls
        self._auth_config_cache: Dict[str, Optional[str]] = {}
        self._auth_config_cache_ttl = 3600  # 1 hour TTL
        # PERFORMANCE: Cache all action schemas per app for exact-name lookups.
        # Bypasses SDK semantic search (which returns alphabetical, not semantic).
        self._schema_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {APP: {ACTION_NAME: schema}}
        self._schema_cache_ts: Dict[str, float] = {}
        self._schema_cache_ttl = 3600
    
    @property
    def composio(self):
        """Lazy-load Composio client."""
        if self._composio is None and self.api_key:
            Composio = _get_composio()
            # Pin toolkit versions to "latest" so manual tools.execute() calls
            # (e.g. bug-report Jira widget) don't error on the SDK's version check.
            # Note: Composio SDK sends telemetry to telemetry.composio.dev for usage tracking,
            # billing, and service monitoring. This is part of their service model.
            self._composio = Composio(
                api_key=self.api_key,
                # "latest" is rejected for manual tools.execute() calls — see
                # docs/SHOPIFY/COMPOSIO-SHOPIFY-SETUP.md gotcha #1. Pin shopify
                # to the version we've tested PRD-009 sync against; other
                # toolkits stay on "latest" via the default key.
                toolkit_versions={"default": "latest", "shopify": "20260414_00"},
            )
        return self._composio

    @property
    def toolset(self):
        """Lazy-load Composio with OpenAI Provider."""
        if self._toolset is None and self.api_key:
            # Use new API: Composio with OpenAIProvider
            Composio = _get_composio()
            OpenAIProvider = _get_composio_openai_provider()
            # Note: Composio SDK sends telemetry to telemetry.composio.dev for usage tracking,
            # billing, and service monitoring. This is part of their service model.
            # Initialize Composio client with OpenAI provider
            self._toolset = Composio(
                api_key=self.api_key,
                provider=OpenAIProvider(),
                # Mirror the pin on `composio` above so the OpenAI-provider
                # variant doesn't drift back to "latest" for shopify.
                toolkit_versions={"default": "latest", "shopify": "20260414_00"},
            )
        return self._toolset
    
    def get_entity(self, entity_id: str):
        """
        Get or create a Composio entity (user).
        
        In the new API, entities are just user IDs. This method validates
        that the entity/user exists.
        
        Args:
            entity_id: Unique identifier (typically workspace_id as string)
            
        Returns:
            Dict with entity_id
        """
        if not self.composio:
            raise ValueError("Composio client not initialized. Set COMPOSIO_API_KEY.")
        
        # In new API, entity is just a user_id string
        # Return a simple dict for compatibility
        return {"user_id": entity_id, "composio_entity_id": entity_id}
    
    def _resolve_auth_config_id(
        self,
        app_slug: str,
        preferred_auth_config_id: Optional[str] = None,
        preferred_scheme: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find an ENABLED Auth Config ID for an app slug.

        Selection order:
          1. preferred_auth_config_id, if passed (caller knows exactly which one)
          2. ENABLED config matching app_slug AND preferred_scheme (if passed)
          3. ENABLED config matching app_slug (first hit — legacy behaviour)

        PERFORMANCE: Cached for 1 hour. Scheme is part of the cache key so
        OAuth lookups don't poison API_KEY lookups for the same toolkit.
        """
        if preferred_auth_config_id:
            return preferred_auth_config_id

        scheme_norm = (preferred_scheme or "_any").upper()
        cache_key = f"{app_slug.lower()}::{scheme_norm}"
        if cache_key in self._auth_config_cache:
            return self._auth_config_cache[cache_key]

        try:
            logger.debug(f"Fetching auth_configs from Composio API for {app_slug} scheme={scheme_norm}")
            response = self.composio.auth_configs.list()
            items = response.items if hasattr(response, 'items') else response.data if hasattr(response, 'data') else []

            for c in items:
                c_slug = getattr(c.toolkit, 'slug', '') if hasattr(c, 'toolkit') else ''
                c_status = getattr(c, 'status', 'ENABLED')
                c_scheme = (getattr(c, 'auth_scheme', '') or getattr(c, 'authScheme', '') or '').upper()

                if c_slug.lower() != app_slug.lower():
                    continue
                if c_status != 'ENABLED':
                    continue
                if preferred_scheme and c_scheme != scheme_norm:
                    continue

                self._auth_config_cache[cache_key] = c.id
                return c.id

            self._auth_config_cache[cache_key] = None
            return None
        except Exception as e:
            logger.error(f"Error resolving auth config for {app_slug}: {e}")
            return None

    def _get_auth_schemes(self, app_slug: str) -> List[str]:
        """Get auth schemes for an app from Composio's toolkit metadata."""
        try:
            toolkits = self.composio.toolkits.get()
            app_info = next(
                (t for t in toolkits if getattr(t, 'slug', '').lower() == app_slug.lower()),
                None,
            )
            if app_info:
                return getattr(app_info, 'auth_schemes', []) or []
        except Exception as e:
            logger.warning(f"Failed to inspect auth schemes for {app_slug}: {e}")
        return []

    def is_no_auth_app(self, app_slug: str) -> bool:
        """Check if an app requires no authentication (e.g. composio_search)."""
        schemes = self._get_auth_schemes(app_slug)
        return schemes == ["NO_AUTH"] or (len(schemes) == 1 and "NO_AUTH" in schemes)

    def _ensure_auth_config_id(
        self,
        app_slug: str,
        preferred_auth_config_id: Optional[str] = None,
        preferred_scheme: Optional[str] = None,
    ) -> str:
        """
        Get existing Auth Config ID or create a new one with correct scheme.

        If preferred_auth_config_id or preferred_scheme is passed, those are
        threaded through resolution. Creation only happens when nothing exists.

        Raises ValueError for NO_AUTH apps — they don't need auth configs.
        """
        existing_id = self._resolve_auth_config_id(
            app_slug,
            preferred_auth_config_id=preferred_auth_config_id,
            preferred_scheme=preferred_scheme,
        )
        if existing_id:
            return existing_id

        # Check auth schemes
        schemes = self._get_auth_schemes(app_slug)
        logger.info(f"Detected auth schemes for {app_slug}: {schemes}")

        # NO_AUTH apps don't need auth configs — can't create one
        if "NO_AUTH" in schemes:
            raise ValueError(
                f"{app_slug} does not require authentication. "
                f"It can be used directly without a connected account."
            )

        try:
            if "API_KEY" in schemes:
                options = {"type": "use_custom_auth", "authScheme": "API_KEY"}
            elif "BASIC" in schemes:
                options = {"type": "use_custom_auth", "authScheme": "BASIC"}
            else:
                options = {"type": "use_composio_managed_auth"}

            logger.info(f"Creating new Auth Config for {app_slug} with options={options}")
            config = self.composio.auth_configs.create(
                toolkit=app_slug,
                options=options
            )
            return config.id
        except Exception as e:
            if "DefaultAuthConfigNotFound" in str(e) and options.get("type") == "use_composio_managed_auth":
                logger.warning(f"Managed auth unavailable for {app_slug}, falling back to custom OAUTH2")
                try:
                    fallback = {"type": "use_custom_auth", "authScheme": "OAUTH2"}
                    config = self.composio.auth_configs.create(
                        toolkit=app_slug,
                        options=fallback
                    )
                    return config.id
                except Exception as e2:
                    logger.error(f"Custom auth fallback also failed for {app_slug}: {e2}")
                    raise ValueError(f"Could not setup authentication for {app_slug}: {str(e2)}")
            logger.error(f"Failed to create auth config for {app_slug}: {e}")
            raise ValueError(f"Could not setup authentication for {app_slug}: {str(e)}")

    def initiate_connection(
        self,
        entity_id: str,
        app: str,
        callback_url: Optional[str] = None,
        auth_config_id: Optional[str] = None,
        auth_scheme: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Initiate a Composio hosted-auth connection.

        Composio's hosted-auth link handles OAuth, API Key, and Bearer Token
        flows from the same UI — the form Composio renders depends on the
        auth_config's scheme. Pass auth_scheme or auth_config_id to pin which
        scheme this workspace uses (e.g. API_KEY for Shopify merchants where
        managed-install breaks OAuth).

        Args:
            entity_id: Workspace's Composio user_id.
            app: App slug (e.g. "SHOPIFY").
            callback_url: Optional post-auth redirect.
            auth_config_id: Optional explicit auth_config to use (wins over scheme).
            auth_scheme: Optional preferred scheme ("API_KEY" | "OAUTH2" | "BASIC").

        Returns:
            dict with redirect_url + auth_config_id + auth_scheme actually used
            (so callers can persist the choice against the workspace's connection).
        """
        if not self.composio:
            raise ValueError("Composio client not initialized. Set COMPOSIO_API_KEY.")

        try:
            chosen_id = self._ensure_auth_config_id(
                app,
                preferred_auth_config_id=auth_config_id,
                preferred_scheme=auth_scheme,
            )

            connection_request = self.composio.connected_accounts.link(
                user_id=entity_id,
                auth_config_id=chosen_id,
                callback_url=callback_url,
            )

            # Best-effort: look up the chosen config's scheme so callers can
            # persist it. Falls back to whatever the caller asked for.
            resolved_scheme = (auth_scheme or "").upper() or None
            try:
                cfg = self.composio.auth_configs.get(chosen_id)
                resolved_scheme = (
                    getattr(cfg, 'auth_scheme', None)
                    or getattr(cfg, 'authScheme', None)
                    or resolved_scheme
                )
                if resolved_scheme:
                    resolved_scheme = str(resolved_scheme).upper()
            except Exception:
                pass

            return {
                "redirect_url": connection_request.redirect_url,
                "auth_config_id": chosen_id,
                "auth_scheme": resolved_scheme or "",
            }
        except Exception as e:
            logger.error(f"Failed to initiate Composio connection for {app}: {e}")
            raise
    
    def get_connection_status(self, entity_id: str, app: str) -> Optional[Dict[str, Any]]:
        """
        Check if an entity has an active connection to an app.
        
        Args:
            entity_id: Entity identifier
            app: App name
            
        Returns:
            Connection details or None if not connected
        """
        if not self.composio:
            return None
        
        try:
            # Resolve Auth Config ID to lookup connections
            auth_config_id = self._resolve_auth_config_id(app)
            if not auth_config_id:
                return None
                
            # Use list with filters to find connection
            response = self.composio.connected_accounts.list(
                user_ids=[entity_id],
                auth_config_ids=[auth_config_id]
            )
            
            # Access items/data from response
            connections = response.items if hasattr(response, 'items') else response.data if hasattr(response, 'data') else []
            
            for conn in connections:
                # ACTIVE = fully connected, INITIATED = OAuth completed but
                # Composio is still provisioning. Treat both as connected.
                if conn.status in ('ACTIVE', 'INITIATED'):
                    return {
                        "id": conn.id,
                        "status": conn.status,
                        "created_at": getattr(conn, 'created_at', None),
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get connection status: {e}")
            return None

    @staticmethod
    def _extract_token_from_connection(conn) -> Optional[str]:
        """Extract OAuth token from a connected account object (Pydantic model or dict)."""
        for attr in ('access_token', 'token'):
            val = getattr(conn, attr, None)
            if val and isinstance(val, str):
                return val

        cp = getattr(conn, 'connectionParams', None)
        if cp is None:
            return None
        # Pydantic model (SDK returns AuthConnectionParamsModel)
        token = getattr(cp, 'access_token', None) or getattr(cp, 'token', None)
        if token and isinstance(token, str):
            return token
        # Plain dict fallback
        if isinstance(cp, dict):
            token = cp.get('access_token') or cp.get('token')
            if token:
                return token
        return None

    def get_app_access_token(self, entity_id: str, app: str) -> Optional[str]:
        """
        Retrieve the OAuth access token for a connected app.

        Inspects the connected account object for token attributes.
        Returns None if unavailable (token not exposed or not connected).
        """
        if not self.composio:
            return None

        try:
            auth_config_id = self._resolve_auth_config_id(app)
            if not auth_config_id:
                return None

            response = self.composio.connected_accounts.list(
                user_ids=[entity_id],
                auth_config_ids=[auth_config_id]
            )
            connections = response.items if hasattr(response, 'items') else response.data if hasattr(response, 'data') else []

            for conn in connections:
                if conn.status not in ('ACTIVE', 'INITIATED'):
                    continue

                token = self._extract_token_from_connection(conn)
                if token:
                    return token

                # Fetch full details if list response was sparse
                try:
                    detail = self.composio.connected_accounts.get(conn.id)
                    token = self._extract_token_from_connection(detail)
                    if token:
                        return token
                    # Log what the detail object looks like for debugging
                    cp = getattr(detail, 'connectionParams', None)
                    cp_debug = {}
                    if cp:
                        for f in ('access_token', 'token', 'scope', 'token_type', 'base_url', 'refresh_token'):
                            v = getattr(cp, f, None) if not isinstance(cp, dict) else cp.get(f)
                            cp_debug[f] = f"<{len(v)} chars>" if isinstance(v, str) and v else repr(v)
                    logger.warning("get_app_access_token: conn %s status=%s connectionParams fields: %s",
                                   conn.id, conn.status, cp_debug)
                except Exception as detail_err:
                    logger.warning("get_app_access_token: .get(%s) failed: %s", conn.id, detail_err)

            return None
        except Exception as e:
            logger.error(f"Failed to get access token for {app}: {e}")
            return None

    def disconnect_app(self, entity_id: str, app: str) -> bool:
        """
        Disconnect an app from an entity.
        
        Args:
            entity_id: Entity identifier
            app: App name to disconnect
            
        Returns:
            True if disconnected successfully
        """
        if not self.composio:
            raise ValueError("Composio client not initialized.")
        
        try:
            # Resolve Auth Config ID
            auth_config_id = self._resolve_auth_config_id(app)
            if not auth_config_id:
                logger.warning(f"No auth config found for {app}, cannot disconnect")
                return False

            # First find the connection ID
            response = self.composio.connected_accounts.list(
                user_ids=[entity_id],
                auth_config_ids=[auth_config_id]
            )
            connections = response.items if hasattr(response, 'items') else response.data if hasattr(response, 'data') else []
            
            if not connections:
                logger.warning(f"No connection found for {app} to disconnect")
                return False
                
            # Disconnect the first found connection
            conn_id = connections[0].id
            self.composio.connected_accounts.delete(nanoid=conn_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect {app}: {e}")
            return False
    
    def create_tool_router_session(
        self,
        entity_id: str,
        apps: List[str]
    ) -> Dict[str, Any]:
        """
        Create a Tool Router session scoped to specific apps.
        
        Tool Router provides "meta-tools" (Search, Plan, Execute) instead of
        exposing all individual actions to the LLM. This dramatically reduces
        the number of tools the LLM needs to choose from.
        
        Args:
            entity_id: Entity identifier
            apps: List of app names to include in the session
            
        Returns:
            Tool Router session with meta-tools
        """
        if not self.composio:
            raise ValueError("Composio client not initialized.")
        
        try:
            session = self.composio.experimental.tool_router.create_session(
                user_id=entity_id,
                toolkits=apps
            )
            return {
                "session_id": session.id,
                "entity_id": entity_id,
                "apps": apps,
                "tools": session.get_tools()  # Meta-tools for LLM
            }
        except Exception as e:
            logger.error(f"Failed to create Tool Router session: {e}")
            raise
    
    def get_available_apps(self) -> List[Dict[str, Any]]:
        """
        Get all available Composio apps with full pagination.

        The Composio API defaults to 100 items per page. We paginate using
        cursor-based pagination to fetch ALL toolkits (typically 500+).

        Returns:
            List of available apps with metadata
        """
        if not self.composio:
            return []

        try:
            # Paginate through all toolkit pages (max 1000 per page)
            all_apps = []
            cursor = None
            page = 0
            while True:
                page += 1
                kwargs = {"limit": 1000}
                if cursor:
                    kwargs["cursor"] = cursor
                response = self.composio.toolkits.list(**kwargs)
                items = response.items or []
                all_apps.extend(items)
                total = getattr(response, "total_items", len(all_apps))
                logger.info(
                    f"Fetched page {page}: {len(items)} toolkits "
                    f"({len(all_apps)}/{int(total)} total)"
                )
                next_cursor = getattr(response, "next_cursor", None)
                if not next_cursor or not items or len(all_apps) >= total:
                    break
                cursor = next_cursor

            logger.info(f"Fetched {len(all_apps)} total toolkits from Composio API")

            trigger_map = self._build_trigger_map()
            results = []
            for app in all_apps:
                # Try multiple sources for triggers
                raw_triggers = None
                source = "none"
                if hasattr(app, "meta") and hasattr(app.meta, "triggers"):
                    raw_triggers = getattr(app.meta, "triggers", None)
                    source = "app.meta.triggers"
                if not raw_triggers and hasattr(app, "triggers"):
                    raw_triggers = getattr(app, "triggers", None)
                    source = "app.triggers"
                if not raw_triggers:
                    raw_triggers = trigger_map.get(app.slug.lower(), [])
                    source = f"trigger_map[{app.slug.lower()}]"

                normalized_triggers = self._normalize_triggers(raw_triggers)

                # Log if we found triggers (for debugging)
                if normalized_triggers and app.slug.lower() in ["slack", "gmail", "github"]:
                    logger.debug(f"Found {len(normalized_triggers)} triggers for {app.slug} (source: {source})")
                # Extract description from multiple possible sources
                description = ""
                if hasattr(app, "meta"):
                    description = (
                        getattr(app.meta, "description", None)
                        or getattr(app.meta, "summary", None)
                        or getattr(app, "description", None)
                        or ""
                    )
                else:
                    description = getattr(app, "description", None) or ""

                # SDK uses tools_count (not actions_count)
                action_count = 0
                if hasattr(app, "meta"):
                    action_count = int(
                        getattr(app.meta, "tools_count", 0)
                        or getattr(app.meta, "actions_count", 0)
                        or 0
                    )

                # Logo: nested under meta (Pydantic model or dict)
                logo_url = None
                if hasattr(app, "meta") and app.meta is not None:
                    if isinstance(app.meta, dict):
                        logo_url = app.meta.get("logo")
                    else:
                        logo_url = getattr(app.meta, "logo", None)

                # Categories: top-level List[str] on AppModel, fallback to meta
                categories = getattr(app, "categories", None) or []
                if not categories and hasattr(app, "meta"):
                    meta_cats = (
                        app.meta.get("categories", [])
                        if isinstance(app.meta, dict)
                        else getattr(app.meta, "categories", [])
                    )
                    categories = [
                        (c.name if hasattr(c, "name") else str(c))
                        for c in (meta_cats or [])
                    ]
                else:
                    categories = [str(c) for c in categories]

                results.append(
                    {
                        "name": app.slug,  # Use slug as the identifier (e.g. 'github')
                        "display_name": app.name,
                        "description": description,
                        "logo_url": logo_url,
                        "categories": categories,
                        "auth_schemes": app.auth_schemes or [],
                        "action_count": action_count,
                        "triggers": normalized_triggers,
                        "trigger_count": len(normalized_triggers),
                    }
                )
            logos_found = sum(1 for r in results if r.get("logo_url"))
            logger.info(f"Composio apps: {len(results)} total, {logos_found} with logos")
            return results
        except Exception as e:
            logger.error(f"Failed to get available apps: {e}")
            return []

    def _build_trigger_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a map of toolkit slug -> triggers."""
        trigger_map: Dict[str, List[Dict[str, Any]]] = {}
        triggers = self.get_trigger_types()
        logger.info(f"Fetched {len(triggers)} trigger types from Composio API")
        if not triggers:
            logger.warning("⚠️  No triggers fetched from Composio API - trigger_map will be empty")
        for trigger in triggers:
            toolkit = trigger.get("toolkit") or {}
            slug = (toolkit.get("slug") or "").lower()
            if not slug:
                # Log if we're skipping triggers due to missing toolkit slug
                logger.debug(f"Skipping trigger {trigger.get('name')} - no toolkit slug")
                continue
            trigger_map.setdefault(slug, []).append({
                "name": trigger.get("slug") or trigger.get("name"),
                "display_name": trigger.get("name"),
                "description": trigger.get("description"),
            })
        logger.info(f"Built trigger map with {len(trigger_map)} toolkits")
        return trigger_map

    def _normalize_triggers(self, triggers: Any) -> List[Dict[str, Any]]:
        """Normalize triggers into a list of {name, display_name, description} dicts."""
        if not triggers:
            return []
        if isinstance(triggers, list):
            normalized = []
            for trigger in triggers:
                if isinstance(trigger, str):
                    normalized.append({"name": trigger, "display_name": trigger, "description": ""})
                elif isinstance(trigger, dict):
                    normalized.append({
                        "name": trigger.get("name") or trigger.get("slug") or trigger.get("trigger_name"),
                        "display_name": trigger.get("display_name") or trigger.get("name") or trigger.get("slug"),
                        "description": trigger.get("description", ""),
                    })
            return normalized
        return []

    def get_trigger_types(self, toolkit_slugs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch available trigger types from Composio.

        Args:
            toolkit_slugs: Optional list of toolkit slugs to filter by.
        """
        if not self.api_key:
            return []

        try:
            import requests
        except Exception as exc:
            logger.warning(f"requests not available: {exc}")
            return []

        url = f"{config.COMPOSIO_API_BASE_URL}/triggers_types"
        # v3.1 defaults toolkit_versions to "latest" — no explicit param needed.
        params: Dict[str, Any] = {"limit": 1000}
        if toolkit_slugs:
            params["toolkit_slugs"] = toolkit_slugs

        headers = {"x-api-key": self.api_key}
        items: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        while True:
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                logger.error(
                    f"❌ Failed to fetch trigger types from Composio API: "
                    f"status={resp.status_code}, response={resp.text[:200]}"
                )
                # Don't break on first error - might be pagination issue
                # But log it clearly
                if resp.status_code == 401:
                    logger.error("⚠️  Authentication failed - check COMPOSIO_API_KEY")
                elif resp.status_code == 403:
                    logger.error("⚠️  API key doesn't have permission to fetch triggers")
                break
            data = resp.json() or {}
            items.extend(data.get("items", []) or [])
            cursor = data.get("next_cursor") or data.get("nextCursor")
            if not cursor:
                break

        return items
    
    def get_app_actions(self, app_name: str) -> List[Dict[str, Any]]:
        """
        Get all actions available for an app.
        
        Args:
            app_name: App name (e.g., "github")
            
        Returns:
            List of actions with metadata
        """
        if not self.composio:
            return []
        
        try:
            # For listing available actions, use a placeholder user_id
            # The new API requires user_id, but for discovery we can use a temporary one
            # Actions are the same regardless of user - user matters only for execution
            placeholder_user_id = "discovery_placeholder"
            
            # New API: composio.tools.get(user_id, toolkits=[app_name])
            # Set a high limit to get all actions (default is only 20!)
            tools = self.composio.tools.get(
                user_id=placeholder_user_id,
                toolkits=[app_name],
                limit=5000  # Get all actions (many apps exceed 500)
            )
            results = []
            for tool in tools:
                if tool.get("type") != "function":
                    continue
                fn = tool.get("function", {}) or {}
                action_name = fn.get("name") or tool.get("name") or ""
                if not action_name:
                    continue
                results.append(
                    {
                        "name": action_name,
                        "display_name": fn.get("name") or tool.get("display_name") or action_name,
                        "description": fn.get("description") or tool.get("description") or "",
                        "app_name": app_name,
                        "parameters": fn.get("parameters") or tool.get("parameters") or {},
                        # Extract additional metadata if available
                        "response_schema": fn.get("response_schema") or tool.get("response_schema") or {},
                        "tags": tool.get("tags") or fn.get("tags") or [],
                        "category": tool.get("category") or fn.get("category") or None,
                        "is_premium": tool.get("is_premium") or fn.get("is_premium") or False,
                        "metadata": self._sanitize_metadata({
                            k: v
                            for k, v in tool.items()
                            if k
                            not in {
                                "function",
                                "type",
                                "name",
                                "display_name",
                                "description",
                                "parameters",
                                "response_schema",
                                "tags",
                                "category",
                                "is_premium",
                            }
                        }),
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Failed to get actions for {app_name}: {e}")
            return []

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to avoid circular references."""
        import json
        sanitized = {}
        for k, v in metadata.items():
            try:
                # Test if value is JSON-serializable (breaks circular refs)
                json.dumps(v, default=str)
                sanitized[k] = v
            except (TypeError, ValueError):
                # Skip non-serializable values (likely circular refs)
                pass
        return sanitized

    def get_all_actions_bulk(self, limit: int = 500, max_pages: int = 1000) -> List[Dict[str, Any]]:
        """Fetch all actions/tools in bulk (paged), without per-toolkit calls.

        This is used by the metadata sync to avoid making one Composio API call per app.
        It hits the Composio REST endpoint directly and paginates using `next_cursor`.
        """
        if not self.api_key:
            return []

        try:
            import requests
        except Exception as exc:
            logger.warning(f"requests not available: {exc}")
            return []

        url = f"{config.COMPOSIO_API_BASE_URL}/tools"
        headers = {"x-api-key": self.api_key}
        cursor: Optional[str] = None
        out: List[Dict[str, Any]] = []
        pages = 0

        logger.info(f"Starting bulk actions fetch (limit={limit} per page)...")
        while True:
            # v3.1 defaults toolkit_versions to "latest" — no explicit param needed.
            params: Dict[str, Any] = {"limit": int(limit)}
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Failed bulk tools fetch: {resp.status_code} {resp.text}")
                break

            data = resp.json() or {}
            items = data.get("items") or data.get("data") or []
            if not isinstance(items, list):
                items = []
            
            pages += 1
            logger.info(f"  Fetched page {pages}: {len(items)} actions (total so far: {len(out) + len(items)})")

            for item in items:
                if not isinstance(item, dict):
                    continue

                fn = item.get("function") if isinstance(item.get("function"), dict) else None
                if item.get("type") and item.get("type") != "function" and fn is None:
                    # Ignore non-function tools unless function payload exists.
                    continue

                # Extract display name (human-readable) from function.name or item.name
                display_name_raw = (
                    (fn or {}).get("name")
                    or item.get("name")
                    or ""
                )
                display_name_raw = str(display_name_raw).strip()

                # Extract slug and derive API enum identifier (e.g., "GMAIL_FETCH_EMAILS")
                item_slug = str(item.get("slug") or "").strip()
                item_enum = str(item.get("enum") or "").strip()

                # Prefer enum > slug-derived > display-name-derived
                if item_enum:
                    action_name = item_enum.upper()
                elif item_slug:
                    action_name = item_slug.upper().replace("-", "_")
                elif display_name_raw:
                    action_name = display_name_raw
                else:
                    continue

                if not action_name:
                    continue

                toolkit = item.get("toolkit")
                toolkit_slug = None
                if isinstance(toolkit, dict):
                    toolkit_slug = toolkit.get("slug") or toolkit.get("name")
                elif isinstance(toolkit, str):
                    toolkit_slug = toolkit
                toolkit_slug = toolkit_slug or item.get("toolkit_slug") or item.get("toolkitSlug")

                app_name = str(toolkit_slug or "").strip()
                if not app_name and "_" in action_name:
                    # Fallback: many actions are prefixed with TOOLKIT_*
                    app_name = action_name.split("_", 1)[0]

                app_name = app_name.upper().strip()
                if not app_name:
                    continue

                # Extract all available fields from the API response
                # (the REST API may include more metadata than the SDK wrapper exposes)
                import json
                # Safely extract metadata without circular references
                extra_metadata = {}
                excluded_keys = {
                    "function", "name", "slug", "toolkit", "toolkit_slug", "toolkitSlug",
                    "type", "description", "summary", "parameters", "response_schema",
                    "tags", "category", "is_premium",
                }
                for k, v in item.items():
                    if k not in excluded_keys:
                        try:
                            # Test if value is JSON-serializable (breaks circular refs)
                            json.dumps(v, default=str)
                            extra_metadata[k] = v
                        except (TypeError, ValueError):
                            # Skip non-serializable values (likely circular refs)
                            pass
                
                out.append(
                    {
                        "app_name": app_name,
                        "name": action_name,
                        "display_name": display_name_raw or item.get("display_name") or action_name,
                        "description": (
                            (fn or {}).get("description")
                            or item.get("description")
                            or item.get("summary")
                            or ""
                        ),
                        "parameters": (fn or {}).get("parameters") or item.get("parameters") or {},
                        # Additional metadata fields (if present in API response)
                        "response_schema": (fn or {}).get("response_schema") or item.get("response_schema") or {},
                        "tags": item.get("tags") or (fn or {}).get("tags") or [],
                        "category": item.get("category") or (fn or {}).get("category") or None,
                        "is_premium": item.get("is_premium") or (fn or {}).get("is_premium") or False,
                        # Store any extra metadata (sanitized to avoid circular refs)
                        "metadata": self._sanitize_metadata(extra_metadata),
                    }
                )

            cursor = data.get("next_cursor") or data.get("nextCursor")
            if not cursor:
                logger.info(f"✅ Bulk fetch complete: {len(out)} total actions across {pages} pages")
                break
            if pages >= max_pages:
                logger.warning(f"Bulk tools fetch stopped early: max_pages ({max_pages}) reached. Total fetched: {len(out)}")
                break

        return out

    def get_app_action_count(self, app_name: str) -> int:
        """Get cached action count for an app."""
        if not app_name:
            return 0

        key = app_name.lower()
        now = datetime.utcnow().timestamp()
        cached = self._action_count_cache.get(key)
        if cached:
            timestamp, count = cached
            if now - timestamp < self._action_count_ttl_seconds:
                return count

        actions = self.get_app_actions(app_name)
        count = len(actions)
        self._action_count_cache[key] = (now, count)
        return count

    # ------------------------------------------------------------------
    # Exact action schema lookup (bypasses broken SDK semantic search)
    # ------------------------------------------------------------------

    def get_action_schemas_by_name(
        self,
        action_names: List[str],
        entity_id: str,
        app_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get OpenAI function-calling schemas for specific Composio actions by name.

        Uses a per-app cache. On first call per app, fetches all action schemas
        from the SDK (one-time cost), then subsequent lookups are instant.

        This bypasses the SDK's semantic search which returns results in
        alphabetical order regardless of query.

        Returns:
            List of dicts with ``action_name`` and ``schema`` keys.
        """
        if not self.toolset:
            logger.warning("[ComposioClient] Toolset not initialized — cannot fetch schemas")
            return []

        import time as _time
        now = _time.monotonic()

        # Ensure each relevant app is cached
        for app in app_names:
            app_upper = app.upper()
            cache_age = now - self._schema_cache_ts.get(app_upper, 0)
            if app_upper not in self._schema_cache or cache_age > self._schema_cache_ttl:
                self._populate_schema_cache(app, entity_id)

        # Look up requested actions — search all cached apps since action names
        # may have multi-word app prefixes (e.g. COMPOSIO_SEARCH_WEB → app COMPOSIO_SEARCH)
        results = []
        seen = set()
        for name in action_names:
            if name in seen:
                continue
            for app_key, app_cache in self._schema_cache.items():
                if name in app_cache:
                    results.append({
                        "action_name": name,
                        "schema": app_cache[name],
                        "app_name": app_key,
                    })
                    seen.add(name)
                    break

        # If some explicitly-requested actions weren't in the initial cache,
        # re-fetch their app with no limit so all actions are available.
        missing = set(action_names) - seen
        if missing:
            refetched_apps: set = set()
            for name in missing:
                app_key = name.split("_", 1)[0]
                if app_key in refetched_apps:
                    continue
                refetched_apps.add(app_key)
                self._populate_schema_cache(
                    app_key, entity_id,
                    limit=500,
                )
            for name in missing:
                for app_key, app_cache in self._schema_cache.items():
                    if name in app_cache:
                        results.append({
                            "action_name": name,
                            "schema": app_cache[name],
                            "app_name": app_key,
                        })
                        seen.add(name)
                        break

        logger.info(
            "[ComposioClient] get_action_schemas_by_name: requested=%d resolved=%d "
            "actions=%s",
            len(action_names), len(results),
            [r["action_name"] for r in results],
        )
        return results

    def _populate_schema_cache(self, app_name: str, entity_id: str, limit: int = 30):
        """Fetch action schemas for an app and cache them.

        Args:
            app_name: Composio app name (e.g. "GMAIL", "COMPOSIO_SEARCH")
            entity_id: Composio entity/user ID
            limit: Max actions to fetch per app (SDK returns by importance).
                   Initial fetch uses 30 for performance; fallback re-fetch
                   uses 500 to get all actions when explicit names are missing.
        """
        import time as _time
        app_upper = app_name.upper()
        cache: Dict[str, Dict[str, Any]] = {}
        t0 = _time.monotonic()

        try:
            raw_tools = self.toolset.tools.get(
                user_id=entity_id,
                toolkits=[app_name.lower()],
                limit=limit,
            )
            for tool in raw_tools:
                if isinstance(tool, dict):
                    tool_dict = tool
                elif hasattr(tool, "model_dump"):
                    tool_dict = tool.model_dump()
                elif hasattr(tool, "dict"):
                    tool_dict = tool.dict()
                elif hasattr(tool, "__dict__"):
                    tool_dict = dict(tool.__dict__)
                else:
                    continue

                if tool_dict.get("type") != "function":
                    continue
                fn = tool_dict.get("function") or {}
                name = fn.get("name") or ""
                if name:
                    cache[name] = tool_dict

            elapsed = int((_time.monotonic() - t0) * 1000)
            logger.info(
                "[ComposioClient] Cached %d schemas for app=%s in %dms",
                len(cache), app_upper, elapsed,
            )
            # Only update cache on success — preserve existing data on failure
            self._schema_cache[app_upper] = cache
            self._schema_cache_ts[app_upper] = _time.monotonic()
        except Exception as e:
            logger.warning(
                "[ComposioClient] Schema cache failed for app=%s: %s",
                app_upper, e, exc_info=True,
            )

    def get_all_schemas_for_apps(
        self,
        app_names: List[str],
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get ALL OpenAI function-calling schemas for the given apps.

        Populates the per-app schema cache (via SDK ``toolset.tools.get``),
        then returns every cached action schema.  The LLM decides which
        action to call — no semantic search, no query rewriting.

        Returns:
            List of dicts with ``action_name`` and ``schema`` keys.
        """
        if not self.toolset:
            logger.warning("[ComposioClient] Toolset not initialized")
            return []

        import time as _time
        now = _time.monotonic()

        results: List[Dict[str, Any]] = []
        seen: set = set()

        for app in app_names:
            app_upper = app.upper()
            cache_age = now - self._schema_cache_ts.get(app_upper, 0)
            if app_upper not in self._schema_cache or cache_age > self._schema_cache_ttl:
                self._populate_schema_cache(app, entity_id)

            for action_name, schema in self._schema_cache.get(app_upper, {}).items():
                if action_name not in seen:
                    results.append({"action_name": action_name, "schema": schema, "app_name": app_upper})
                    seen.add(action_name)

        logger.info(
            "[ComposioClient] get_all_schemas_for_apps: apps=%s → %d actions",
            [a.upper() for a in app_names], len(results),
        )
        return results

    def search_actions_for_step(
        self,
        search_query: str,
        app_names: List[str],
        entity_id: str,
        limit: int = 5,
        explicit_actions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use Composio SDK semantic search to find the best actions for a recipe step.

        Returns OpenAI function-calling tool schemas so each Composio action becomes
        its own top-level tool (the LLM calls ``SLACK_SEND_MESSAGE(channel, text)``
        directly instead of ``composio_execute(action="SLACK_SEND_MESSAGE", ...)``).

        Args:
            search_query: Natural language description of the step's intent
                          (e.g. "send a slack message").
            app_names: Composio toolkit slugs the agent has connected
                       (e.g. ["slack", "gmail"]).
            entity_id: Composio entity / user id (typically workspace_id as str).
            limit: Max actions to return per search (default 5).
            explicit_actions: If provided, fetch these exact action schemas instead
                              of searching.  Power-user override for ``tool_hints.explicit_actions``.

        Returns:
            List of dicts, each with:
                - action_name: e.g. "SLACK_SEND_MESSAGE"
                - schema: OpenAI function-calling tool dict (``{"type":"function","function":{...}}``)
        """
        if not self.toolset:
            logger.warning("Composio toolset not initialized — cannot search actions")
            return []

        results: List[Dict[str, Any]] = []
        seen: set = set()

        try:
            if explicit_actions:
                # Fetch exact action schemas by name
                raw_tools = self.toolset.tools.get(
                    user_id=entity_id,
                    actions=explicit_actions,
                )
                if raw_tools:
                    logger.info(
                        f"[ComposioClient] explicit_actions={explicit_actions} "
                        f"returned {len(raw_tools)} items, types={[type(t).__name__ for t in raw_tools[:3]]}"
                    )
            else:
                # Semantic search scoped to the agent's connected apps
                raw_tools = self.toolset.tools.get(
                    user_id=entity_id,
                    search=search_query,
                    toolkits=[n.lower() for n in app_names],
                    limit=limit,
                )

            for tool in raw_tools:
                # Composio SDK may return dicts or Pydantic/SDK objects.
                # Normalize to dict for consistent downstream handling.
                if isinstance(tool, dict):
                    tool_dict = tool
                elif hasattr(tool, "model_dump"):
                    tool_dict = tool.model_dump()
                elif hasattr(tool, "dict"):
                    tool_dict = tool.dict()
                elif hasattr(tool, "__dict__"):
                    tool_dict = dict(tool.__dict__)
                else:
                    logger.debug(f"Skipping unrecognized tool type: {type(tool)}")
                    continue

                if tool_dict.get("type") != "function":
                    continue
                fn = tool_dict.get("function") or {}
                action_name = fn.get("name") or ""
                if not action_name or action_name in seen:
                    continue
                seen.add(action_name)
                results.append({
                    "action_name": action_name,
                    "schema": tool_dict,  # OpenAI function-calling format
                })

        except Exception as e:
            logger.error(f"Composio semantic search failed (query={search_query!r}): {e}")

        logger.info(
            f"[ComposioClient] search_actions_for_step query={search_query!r} "
            f"apps={app_names} explicit={bool(explicit_actions)} → {len(results)} actions: "
            f"{[r['action_name'] for r in results]}"
        )
        return results

    def execute_action(
        self,
        action: str,
        params: Dict[str, Any],
        entity_id: str
    ) -> Dict[str, Any]:
        """
        Execute a Composio action using the new API.
        
        Args:
            action: Action name (e.g., "GITHUB_LIST_REPOS")
            params: Action parameters (arguments dict)
            entity_id: Entity identifier (user_id)
            
        Returns:
            Action execution result
        """
        if not self.composio:
            raise ValueError("Composio client not initialized.")
        
        try:
            # composio>=0.12 rejects manual tools.execute() with version="latest"
            # ("'latest' is not supported in manual execution"). We track the
            # latest schema at discovery, so skip the version check here — same
            # pattern the SDK uses internally for agentic tool execution.
            result = self.composio.tools.execute(
                slug=action,
                user_id=entity_id,
                arguments=params,
                dangerously_skip_version_check=True,
            )

            # Composio API may return errors in the response data without
            # raising an exception.  Detect {"successful": false} responses
            # so callers see a proper failure instead of a false success.
            api_success = True
            api_error = None
            if isinstance(result, dict):
                api_success = result.get("successful", result.get("success", True))
                api_error = result.get("error") or result.get("message")
            elif isinstance(result, list):
                # Batch responses: check first item
                for item in result:
                    if isinstance(item, dict) and item.get("successful") is False:
                        api_success = False
                        api_error = item.get("error") or item.get("message")
                        break

            return {
                "success": bool(api_success),
                "data": result,
                "error": api_error,
            }
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    def subscribe_to_trigger(
        self,
        entity_id: str,
        trigger_name: str,
        callback_url: str
    ) -> Dict[str, Any]:
        """
        Subscribe to a Composio trigger (webhook).
        
        Args:
            entity_id: Entity identifier (user_id)
            trigger_name: Trigger type (e.g., "github_pull_request_opened")
            callback_url: URL to receive webhook events
            
        Returns:
            Subscription details
        """
        if not self.composio:
            raise ValueError("Composio client not initialized.")
        
        try:
            # New API: composio.triggers
            subscription = self.composio.triggers.subscribe(
                user_id=entity_id,
                trigger_name=trigger_name,
                callback_url=callback_url
            )
            return {
                "id": subscription.id if hasattr(subscription, 'id') else str(subscription),
                "trigger_name": trigger_name,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Failed to subscribe to trigger {trigger_name}: {e}")
            raise
    
    def unsubscribe_trigger(self, subscription_id: str) -> bool:
        """
        Unsubscribe from a trigger.
        
        Args:
            subscription_id: Subscription to remove
            
        Returns:
            True if unsubscribed successfully
        """
        if not self.composio:
            return False
        
        try:
            self.composio.triggers.unsubscribe(subscription_id=subscription_id)
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe trigger: {e}")
            return False


# Singleton instance
_client_instance: Optional[ComposioClient] = None


def get_composio_client() -> ComposioClient:
    """Get or create the Composio client singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ComposioClient()
    return _client_instance
