"""
Composio Tool Executor (PRD-36)
===============================

Executes Composio actions with proper access validation.

This executor:
- Validates agent has permission for the action
- Resolves entity from workspace
- Executes action via Composio SDK
- Formats response for unified executor
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime, timezone
import time
import re
from difflib import SequenceMatcher

from sqlalchemy.orm import Session
from sqlalchemy import func

from core.composio.client import ComposioClient, get_composio_client

logger = logging.getLogger(__name__)


class ComposioToolExecutor:
    """
    Executes Composio tools with access validation.
    
    Integrates with the UnifiedToolExecutor to handle all Composio-based
    tool calls. Validates that agents have permission to use specific actions
    before execution.
    
    Usage:
        executor = ComposioToolExecutor(db)
        result = await executor.execute(
            action="github_list_repos",
            params={"per_page": 10},
            agent_id=123,
            workspace_id=workspace_uuid
        )
    """
    
    def __init__(self, db: Session, client: Optional[ComposioClient] = None):
        """
        Initialize tool executor.
        
        Args:
            db: Database session
            client: Optional Composio client (uses singleton if not provided)
        """
        self.db = db
        self._client = client
    
    @property
    def client(self) -> ComposioClient:
        """Get Composio client."""
        if self._client is None:
            self._client = get_composio_client()
        return self._client
    
    def validate_feature_access(
        self,
        agent_id: int,
        action_name: str,
        app_name: Optional[str] = None
    ) -> bool:
        """
        Check if an agent has access to a specific action.
        
        Args:
            agent_id: Agent ID
            action_name: Action to check
            app_name: Optional app name (extracted from action if not provided)
            
        Returns:
            True if agent can use this action
        """
        # Extract app name from action if not provided
        # Actions are typically formatted as: app_action_name (e.g., github_list_repos)
        if not app_name:
            parts = action_name.split("_", 1)
            app_name = parts[0].upper() if parts else None
        
        if not app_name:
            logger.warning(f"Could not determine app for action: {action_name}")
            return False
        
        try:
            from core.models.composio import AgentAppFeature
            
            # Check if there's an explicit disable for this action
            feature = self.db.query(AgentAppFeature).filter(
                AgentAppFeature.agent_id == agent_id,
                AgentAppFeature.app_name == app_name,
                AgentAppFeature.action_name == action_name
            ).first()
            
            if feature:
                return feature.enabled
            
            # Check if there's ANY feature config for this app
            # If no explicit config, allow by default (opt-out model)
            app_features = self.db.query(AgentAppFeature).filter(
                AgentAppFeature.agent_id == agent_id,
                AgentAppFeature.app_name == app_name
            ).count()
            
            # If app has no features configured, allow all (not yet set up)
            if app_features == 0:
                return True
            
            # If app has features but this action isn't in the list, 
            # it means it wasn't enabled
            return False
            
        except Exception as e:
            logger.error(f"Error checking feature access: {e}")
            # Default to allow if we can't check (fail open for now)
            return True
    
    def get_entity_for_workspace(self, workspace_id: UUID) -> Dict[str, Any]:
        """
        Get or create Composio entity for a workspace.
        
        Args:
            workspace_id: Workspace UUID
            
        Returns:
            Entity data with composio_entity_id
        """
        from core.composio.entity_manager import EntityManager
        
        manager = EntityManager(self.db)
        return manager.get_or_create_entity(workspace_id)

    _UPLOAD_ACTIONS = {
        "TWITTER_UPLOAD_MEDIA",
        "TWITTER_INITIALIZE_MEDIA_UPLOAD",
        "TWITTER_UPLOAD_LARGE_MEDIA",
        "TWITTER_APPEND_MEDIA_UPLOAD",
    }
    _FILE_PARAM_NAMES = {"media", "media_data", "media_url", "file", "image", "video", "media_file"}

    async def _resolve_file_uploads(
        self,
        action: str,
        params: Dict[str, Any],
        workspace_id: UUID,
    ) -> tuple[Dict[str, Any], list[Path]]:
        """Download workspace files and convert to FileUploadable for actions that need binary.

        Twitter's media upload requires binary data but agents pass workspace paths.
        Detects known upload actions, pulls the file from the workspace, writes a
        temp file, and uses Composio's FileUploadable to upload the binary to S3.

        Returns (modified params, list of temp files to clean up).
        """
        temp_files: list[Path] = []
        action_upper = action.upper()

        if action_upper not in self._UPLOAD_ACTIONS:
            return params, temp_files

        try:
            try:
                from composio.core.models._files import FileUploadable
            except ImportError:
                from composio.client.files import FileUploadable
            from core.workspace_client import WorkspaceClient

            http_client = self.client.composio.client
            action_slug = action_upper.lower().replace("_", "-")
            app_slug = action_upper.split("_", 1)[0].lower()
            ws_client = WorkspaceClient(workspace_id)

            for param_name in list(params.keys()):
                if param_name.lower() not in self._FILE_PARAM_NAMES:
                    continue
                value = params[param_name]
                if not value or not isinstance(value, str):
                    continue
                if isinstance(value, dict) and "s3key" in value:
                    continue

                if value.startswith(("http://", "https://")):
                    logger.info("[FileUpload] Uploading from URL for %s: %s", param_name, value[:120])
                    uploadable = FileUploadable.from_url(
                        client=http_client,
                        url=value,
                        tool=action_slug,
                        toolkit=app_slug,
                    )
                else:
                    logger.info("[FileUpload] Downloading workspace file %s for %s", value, param_name)
                    dl_result = await ws_client.download_file(value)
                    if not dl_result.get("success"):
                        logger.warning("[FileUpload] Failed to download %s: %s", value, dl_result.get("error"))
                        continue
                    file_bytes: bytes = dl_result["content"]
                    suffix = Path(value).suffix or ".png"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(file_bytes)
                    tmp.close()
                    tmp_path = Path(tmp.name)
                    temp_files.append(tmp_path)
                    uploadable = FileUploadable.from_path(
                        client=http_client,
                        file=tmp_path,
                        tool=action_slug,
                        toolkit=app_slug,
                        sensitive_file_upload_protection=False,
                    )

                params[param_name] = uploadable.model_dump()
                logger.info("[FileUpload] Resolved %s -> s3key=%s", param_name, uploadable.s3key)
        except ImportError:
            logger.warning("[FileUpload] FileUploadable not found in composio SDK — cannot resolve binary uploads")
        except Exception as e:
            logger.warning("[FileUpload] Failed to resolve file uploads for %s: %s", action, e, exc_info=True)

        return params, temp_files

    async def execute(
        self,
        action: str,
        params: Dict[str, Any],
        agent_id: int,
        workspace_id: UUID,
        app_name: Optional[str] = None,
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a Composio action.
        
        Args:
            action: Action name (e.g., "github_list_repos")
            params: Action parameters
            agent_id: Agent ID for access validation
            workspace_id: Workspace UUID for entity resolution
            skip_validation: Skip access validation (use carefully)
            
        Returns:
            Execution result in standard format
        """
        start_time = time.time()

        # Normalize action and derive app name
        action = str(action or "").strip()
        if not action:
            return {
                "success": False,
                "error": "Missing required Composio action name",
                "error_type": "invalid_parameters",
                "data": None,
                "execution_time_ms": int((time.time() - start_time) * 1000),
            }

        action_upper = action.upper()
        # Prefer explicit app_name passed from composio_execute() call;
        # then try ComposioActionCache (handles multi-word apps like COMPOSIO_SEARCH);
        # last resort: split on first underscore.
        app_name = (str(app_name).strip().upper() if app_name else None)
        if not app_name:
            # Look up in cache — this is the reliable source for app_name
            try:
                from core.models.composio_cache import ComposioActionCache
                cached = (
                    self.db.query(ComposioActionCache.app_name)
                    .filter(func.upper(ComposioActionCache.action_name) == action_upper)
                    .first()
                )
                if cached:
                    app_name = cached.app_name.upper()
            except Exception:
                pass  # Fall through to split heuristic
        if not app_name:
            app_name = action_upper.split("_", 1)[0].strip() if "_" in action_upper else None
        if not app_name:
            return {
                "success": False,
                "error": (
                    f"Invalid Composio action '{action}'. "
                    "Provide `app_name` (e.g. 'SLACK') or use a mapped action name formatted like 'APP_ACTION' "
                    "(for example: 'GMAIL_FETCH_EMAILS' or 'SLACK_SEND_MESSAGE')."
                ),
                "error_type": "invalid_parameters",
                "data": None,
                "execution_time_ms": int((time.time() - start_time) * 1000),
            }

        # ------------------------------------------------------------------
        # Hard requirements (per your rules):
        # - The app must be assigned to the agent (agent_app_assignments)
        # - The app must be connected for the workspace (composio_connections)
        # - The action must be mapped in composio_actions_cache (no "discover" calls)
        # When skip_validation=True (admin/system calls like bug reports), skip all checks.
        # ------------------------------------------------------------------
        if not skip_validation:
            try:
                from core.models.composio_cache import AgentAppAssignment, ComposioActionCache
                from core.composio.entity_manager import EntityManager

                # Assigned?
                assigned = (
                    self.db.query(AgentAppAssignment)
                    .filter(
                        AgentAppAssignment.agent_id == agent_id,
                        AgentAppAssignment.app_name == app_name,
                        AgentAppAssignment.is_active == True,  # noqa: E712
                    )
                    .first()
                )
                if not assigned:
                    # Inheritance fallback: agents with zero active assignments
                    # inherit workspace-connected apps. Mirrors the contract used
                    # by ComposioToolService, ComposioHintService and ToolRegistry
                    # at discovery time so an agent that can SEE an app can also
                    # EXECUTE it. The workspace connectivity check below still
                    # gates on whether the app is actually wired up.
                    has_any_assignment = (
                        self.db.query(AgentAppAssignment.id)
                        .filter(
                            AgentAppAssignment.agent_id == agent_id,
                            AgentAppAssignment.is_active == True,  # noqa: E712
                        )
                        .first()
                    )
                    if has_any_assignment:
                        return {
                            "success": False,
                            "error": f"'{app_name}' is not assigned to agent {agent_id}. Assign it to this agent before using it.",
                            "error_type": "composio_not_assigned",
                            "data": None,
                            "execution_time_ms": int((time.time() - start_time) * 1000),
                        }
                    logger.info(
                        "[ComposioToolExecutor] Agent %s has no app assignments — "
                        "inheriting workspace-connected apps for %s",
                        agent_id, app_name,
                    )

                # Connected?
                manager = EntityManager(self.db)
                connected_apps = {a.upper().strip() for a in manager.get_connected_apps(workspace_id)}
                if app_name not in connected_apps:
                    return {
                        "success": False,
                        "error": f"'{app_name}' is assigned to agent {agent_id} but is not connected for this workspace. Connect it first, then retry.",
                        "error_type": "composio_not_connected",
                        "data": None,
                        "execution_time_ms": int((time.time() - start_time) * 1000),
                    }

                # Mapped action?
                # Normalize incoming action so we can match either action_name or action_slug.
                raw_action = str(action).strip()
                action_lower = raw_action.lower()
                action_lower_snake = action_lower.replace("-", "_")
                action_lower_slug = action_lower.replace("_", "-")

                mapped = (
                    self.db.query(ComposioActionCache)
                    .filter(
                        ComposioActionCache.app_name == app_name,
                        (
                            (func.lower(ComposioActionCache.action_name).in_([action_lower, action_lower_snake]))
                            | (func.lower(ComposioActionCache.action_slug).in_([action_lower, action_lower_slug]))
                        ),
                    )
                    .first()
                )
                # Generic fallback: some models emit unprefixed action names (e.g., "send_message").
                # When app_name is known, try matching by suffix: "*_send_message" or "*-send-message".
                if not mapped and not action_upper.startswith(f"{app_name}_"):
                    suffix_snake = f"%_{action_lower_snake}"
                    suffix_slug = f"%-{action_lower_slug}"
                    mapped = (
                        self.db.query(ComposioActionCache)
                        .filter(ComposioActionCache.app_name == app_name)
                        .filter(
                            (func.lower(ComposioActionCache.action_name).like(suffix_snake))
                            | (func.lower(ComposioActionCache.action_slug).like(suffix_slug))
                        )
                        .first()
                    )
                if not mapped:
                    # Helpful error: tell the user whether cache is empty and suggest safe alternatives.
                    total_for_app = (
                        self.db.query(func.count(ComposioActionCache.id))
                        .filter(ComposioActionCache.app_name == app_name)
                        .scalar()
                        or 0
                    )

                    # Build ranked suggestions from cache (generic, no app-specific mapping).
                    dangerous_tokens = (
                        "archive",
                        "delete",
                        "remove",
                        "revoke",
                        "clear",
                        "close",
                        "disable",
                        "ban",
                        "kick",
                        "destroy",
                        "drop",
                        "purge",
                        "terminate",
                    )

                    def _tokenize(s: str) -> set[str]:
                        s = (s or "").lower()
                        # Split on underscores/hyphens too (GMAIL_LIST_EMAILS -> gmail list emails)
                        s = re.sub(r"[^a-z0-9_\\s\\-]", " ", s)
                        s = s.replace("-", " ").replace("_", " ")
                        return {t for t in s.split() if t}

                    req_tokens = _tokenize(raw_action)
                    read_verbs = {"list", "get", "fetch", "search", "read", "retrieve", "lookup"}
                    write_verbs = {"send", "post", "create", "update", "delete", "remove", "archive", "add", "set", "invite", "write"}
                    req_kind = "read" if (req_tokens & read_verbs) else ("write" if (req_tokens & write_verbs) else "unknown")

                    # Pull a bounded number of candidates for scoring.
                    rows = (
                        self.db.query(
                            ComposioActionCache.action_name,
                            ComposioActionCache.action_slug,
                            ComposioActionCache.description,
                        )
                        .filter(ComposioActionCache.app_name == app_name)
                        .limit(600)
                        .all()
                    )

                    scored: list[tuple[int, float, str, str]] = []
                    for name, slug, desc in rows:
                        name = str(name or "")
                        if not name:
                            continue
                        hay = f"{name} {slug or ''} {desc or ''}"
                        hay_l = hay.lower()
                        if any(tok in hay_l for tok in dangerous_tokens):
                            continue

                        # Combined similarity: token overlap + sequence ratio.
                        hay_tokens = _tokenize(hay)
                        overlap = len(req_tokens & hay_tokens)
                        ratio = SequenceMatcher(None, raw_action.lower(), name.lower()).ratio()
                        scored.append((overlap, ratio, name, (desc or "")[:160]))

                    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
                    suggestions = [
                        {"action": n, "description": d}
                        for _, _, n, d in scored[:10]
                    ]

                    examples_clause = ""
                    if suggestions:
                        examples = ", ".join(s.get("action") for s in suggestions[:8] if s.get("action"))
                        if examples:
                            examples_clause = f" Examples of mapped actions: {examples}"

                    # Auto-map if top suggestion has high confidence (ratio > 0.7 AND token overlap > 0)
                    auto_mapped_from: Optional[str] = None
                    auto_selected_action: Optional[str] = None

                    if scored and scored[0][0] > 0 and scored[0][1] > 0.7:
                        auto_selected_action = scored[0][2]
                        auto_mapped_from = raw_action
                        logger.info(
                            f"[composio_exec] Auto-mapped '{raw_action}' → '{auto_selected_action}' "
                            f"(overlap={scored[0][0]}, ratio={scored[0][1]:.2f})"
                        )
                        action_upper = auto_selected_action.upper()
                        # Re-query to get the mapped record for downstream use
                        mapped = (
                            self.db.query(ComposioActionCache)
                            .filter(
                                ComposioActionCache.app_name == app_name,
                                func.lower(ComposioActionCache.action_name) == auto_selected_action.lower(),
                            )
                            .first()
                        )
                        if not mapped:
                            return {
                                "success": False,
                                "error": f"Auto-mapped action '{auto_selected_action}' could not be resolved from cache for {app_name}.",
                                "error_type": "composio_action_not_mapped",
                                "data": None,
                                "execution_time_ms": int((time.time() - start_time) * 1000),
                            }
                    else:
                        return {
                            "success": False,
                            "error": (
                                f"Action '{raw_action}' is not mapped in composio_actions_cache for {app_name}. "
                                + (
                                    "The cache appears empty for this app. Run the cache sync."
                                    if total_for_app == 0
                                    else (
                                        "Use one of the suggested safe actions instead."
                                        if suggestions
                                        else "No safe suggestions found in cache; run the cache sync."
                                    )
                                )
                                + examples_clause
                            ),
                            "error_type": "composio_action_not_mapped",
                            "data": {
                                "app_name": app_name,
                                "requested_action": raw_action,
                                "suggested_actions": suggestions,
                            },
                            "execution_time_ms": int((time.time() - start_time) * 1000),
                        }

                # Normalize to canonical action_name
                original_action = str(mapped.action_name or action_upper).upper()
                action_upper = original_action

                # If action_name is a display name (no app prefix), reconstruct proper API format
                # e.g., "FETCH EMAILS" + app "GMAIL" → "GMAIL_FETCH_EMAILS"
                if app_name and not action_upper.startswith(f"{app_name}_"):
                    reconstructed = f"{app_name}_{action_upper.replace(' ', '_')}"
                    logger.info(f"Reconstructed Composio action: '{original_action}' -> '{reconstructed}'")
                    action_upper = reconstructed
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"Failed to validate Composio prerequisites: {exc}",
                    "error_type": "composio_action_not_allowed",
                    "data": None,
                    "execution_time_ms": int((time.time() - start_time) * 1000),
                }

            # Validate access — check both original and reconstructed action names
            has_access = self.validate_feature_access(agent_id, action_upper, app_name=app_name)
            if not has_access and original_action != action_upper:
                has_access = self.validate_feature_access(agent_id, original_action, app_name=app_name)
            if not has_access:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} does not have access to action: {action_upper}",
                    "error_type": "composio_action_not_allowed",
                    "data": None,
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }
        
        # Get entity
        try:
            entity = self.get_entity_for_workspace(workspace_id)
        except Exception as e:
            logger.error(f"Failed to get entity for workspace {workspace_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to resolve workspace entity: {str(e)}",
                "data": None,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
        
        composio_entity_id = entity.get("composio_entity_id")
        if not composio_entity_id:
            return {
                "success": False,
                "error": "No Composio entity ID found for workspace",
                "data": None,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
        
        # Resolve workspace file paths to binary uploads for actions that need them
        temp_files: list[Path] = []
        try:
            params, temp_files = await self._resolve_file_uploads(
                action=action_upper, params=params, workspace_id=workspace_id,
            )
        except Exception as e:
            logger.warning("[FileUpload] Pre-processing failed (continuing): %s", e)

        # Execute via Composio
        try:
            result = self.client.execute_action(
                action=action_upper,
                params=params,
                entity_id=composio_entity_id
            )

            execution_time = int((time.time() - start_time) * 1000)

            # PRD-41: Extract entities and store in Redis for context-aware suggestions
            # Only extract if execution was successful
            if result.get("success", False) and result.get("data"):
                try:
                    self._extract_and_store_entities(
                        app_name=app_name,
                        action_name=action_upper,
                        tool_result=result.get("data", {}),
                        workspace_id=workspace_id,
                        agent_id=agent_id
                    )
                except Exception as extract_error:
                    # Log but don't fail the execution
                    logger.warning(f"Failed to extract/store entities for {app_name}: {extract_error}")

            return {
                "success": result.get("success", False),
                "data": result.get("data"),
                "error": result.get("error"),
                "execution_time_ms": execution_time,
                "action": action_upper,
                "entity_id": composio_entity_id
            }

        except Exception as e:
            error_msg = str(e)
            error_type = None
            missing_dependency = None
            fatal = False

            if "Composio OpenAI SDK not available" in error_msg or "composio-openai" in error_msg:
                error_type = "dependency_missing"
                missing_dependency = "composio-openai"
                fatal = True

            logger.error(f"Composio action execution failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "error_type": error_type,
                "missing_dependency": missing_dependency,
                "fatal": fatal,
                "data": None,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
        finally:
            for tmp in temp_files:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_agent_enabled_actions(
        self,
        agent_id: int,
        app_name: str
    ) -> List[str]:
        """
        Get list of enabled actions for an agent-app combination.
        
        Args:
            agent_id: Agent ID
            app_name: App name
            
        Returns:
            List of enabled action names
        """
        try:
            from core.models.composio import AgentAppFeature
            
            features = self.db.query(AgentAppFeature).filter(
                AgentAppFeature.agent_id == agent_id,
                AgentAppFeature.app_name == app_name,
                AgentAppFeature.enabled == True
            ).all()
            
            return [f.action_name for f in features]
            
        except Exception as e:
            logger.error(f"Error getting enabled actions: {e}")
            return []
    
    def set_agent_action_enabled(
        self,
        agent_id: int,
        app_name: str,
        action_name: str,
        enabled: bool
    ) -> bool:
        """
        Enable or disable an action for an agent.
        
        Args:
            agent_id: Agent ID
            app_name: App name
            action_name: Action name
            enabled: Whether to enable or disable
            
        Returns:
            True if updated successfully
        """
        try:
            from core.models.composio import AgentAppFeature
            
            feature = self.db.query(AgentAppFeature).filter(
                AgentAppFeature.agent_id == agent_id,
                AgentAppFeature.app_name == app_name,
                AgentAppFeature.action_name == action_name
            ).first()
            
            if feature:
                feature.enabled = enabled
            else:
                feature = AgentAppFeature(
                    agent_id=agent_id,
                    app_name=app_name,
                    action_name=action_name,
                    enabled=enabled
                )
                self.db.add(feature)
            
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error setting action enabled: {e}")
            self.db.rollback()
            return False
    
    def batch_set_agent_actions(
        self,
        agent_id: int,
        app_name: str,
        actions: List[Dict[str, Any]]
    ) -> int:
        """
        Batch update action enabled states.
        
        Args:
            agent_id: Agent ID
            app_name: App name
            actions: List of {"action_name": str, "enabled": bool}
            
        Returns:
            Number of actions updated
        """
        count = 0
        for action in actions:
            if self.set_agent_action_enabled(
                agent_id=agent_id,
                app_name=app_name,
                action_name=action["action_name"],
                enabled=action["enabled"]
            ):
                count += 1
        return count

    def _extract_and_store_entities(
        self,
        app_name: str,
        action_name: str,
        tool_result: Dict[str, Any],
        workspace_id: UUID,
        agent_id: int
    ) -> None:
        """
        Extract entities from tool result and store in Redis (PRD-41: US-005, US-006).

        This enables context-aware suggestions by remembering what tools showed
        the user (e.g., "urgent email from Sarah", "PR #123").

        Context is stored with 10-minute TTL for ephemeral, short-lived suggestions.

        Args:
            app_name: Composio app name (e.g., "GMAIL")
            action_name: Action that was executed
            tool_result: Response data from tool execution
            workspace_id: Workspace UUID (used as Redis key component)
            agent_id: Agent ID (stored in context metadata)
        """
        from core.composio.entity_extractors import get_extractor
        import json

        # Get appropriate extractor for this app (always returns an extractor)
        extractor = get_extractor(app_name)

        # Extract entities
        entities = extractor.extract(tool_result)
        if not entities or not any(entities.values()):
            logger.debug(f"No entities extracted from {app_name} result")
            return

        # Store in Redis with 10-minute TTL (PRD-41: Context-aware suggestions)
        try:
            from core.redis.client import get_redis_client

            redis_client = get_redis_client()
            if not redis_client:
                logger.warning("Redis client not available, skipping context storage")
                return

            # Get actual Redis connection
            redis_conn = redis_client.get_redis()

            # Create context data with timestamp
            context_data = {
                "tool": app_name,
                "action": action_name,
                "entities": entities,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id
            }

            # Redis key: tool_context:{workspace_id}:{tool_name}
            redis_key = f"tool_context:{workspace_id}:{app_name}"

            # Store with 10-minute TTL (600 seconds)
            redis_conn.setex(
                redis_key,
                600,  # 10 minutes
                json.dumps(context_data)
            )
            redis_conn.close()

            logger.info(f"✅ Stored tool context in Redis: {app_name} for workspace {workspace_id}")

        except Exception as e:
            logger.error(f"Failed to store tool context in Redis: {e}")
