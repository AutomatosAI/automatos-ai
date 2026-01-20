"""
MCP Tool Executor Service
=========================
Phase 3: Skills & Tools Integration

Handles execution of MCP (Model Context Protocol) tools for agents.
Provides a clean interface for tool invocation, result handling, and usage tracking.
"""

import asyncio
import httpx
import logging
import json
import time
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

# Database models will be imported dynamically to avoid circular imports
from core.database.database import get_db
from core.models import MCPTool, ToolUsageLog, Agent, AgentToolAssignment
from core.models.tool_assignments import AgentToolAssignment
from core.credentials.service import CredentialStore, CredentialNotFoundError, CredentialValidationError
from core.credentials.encryption import EncryptionKeyError

logger = logging.getLogger(__name__)

class MCPToolExecutionError(Exception):
    """Raised when MCP tool execution fails"""
    pass

class MCPToolNotFoundError(Exception):
    """Raised when a tool is not found or not assigned to agent"""
    pass

# Simple cache for CF tool name lookups: {provider_method: cf_tool_name}
_cf_tool_name_cache: Dict[str, str] = {}

class MCPToolExecutor:
    """
    Executes MCP tools for agents with proper permission checking,
    logging, and error handling.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.http_client = httpx.AsyncClient(timeout=30.0)
        # Context Forge configuration (PRD-35: unified gateway)
        self.context_forge_url = os.getenv("CONTEXT_FORGE_URL", "https://mcp.automatos.app")
        self.context_forge_enabled = os.getenv("CONTEXT_FORGE_ENABLED", "false").lower() == "true"
        logger.info(f"MCPToolExecutor initialized (CF enabled: {self.context_forge_enabled})")

    def _get_available_methods(self, capabilities: Dict[str, Any]) -> List[str]:
        if not capabilities:
            return []
        methods = capabilities.get('methods') or capabilities.get('tools') or capabilities.get('actions') or []
        if isinstance(methods, dict):
            return list(methods.keys())
        if isinstance(methods, list):
            return methods
        return []

    def _resolve_tool_credentials(
        self,
        tool: MCPTool,
        assignment: Optional[AgentToolAssignment]
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        # AgentToolAssignment might not have configuration, default to empty dict
        config = getattr(assignment, 'configuration', {}) if assignment else {}
        metadata = tool.tool_metadata or {}

        credential_id = config.get("credential_id") or metadata.get("linked_credential_id")
        credential_name = config.get("credential_name") or metadata.get("linked_credential_name")
        environment = config.get("credential_environment", "production")

        store = CredentialStore(self.db)
        if not credential_id and not credential_name:
            # Fallback: resolve by credential type (if exactly one exists)
            credential_type = (
                metadata.get("credential_type")
                or metadata.get("auto_enable_on_credential")
                or metadata.get("n8n_credential_type")
            )
            if credential_type:
                from core.models.credentials import Credential, CredentialType
                candidate = self.db.query(Credential).join(CredentialType).filter(
                    Credential.is_active == True,
                    CredentialType.name == credential_type
                ).order_by(Credential.updated_at.desc()).first()
                if candidate:
                    credential_data = store.get_decrypted_credential(
                        credential_id=candidate.id,
                        service_name="mcp_tool_executor"
                    )
                    return credential_data, {
                        "credential_id": candidate.id,
                        "credential_type": credential_type,
                        "environment": candidate.environment,
                        "tenant_id": str(candidate.tenant_id) if hasattr(candidate, 'tenant_id') and candidate.tenant_id else None
                    }
            return None, None
        if credential_id:
            # [HOTFIX] Single Tenant Mode - No Tenant ID lookup
            return credential_data, {"credential_id": credential_id, "environment": environment, "tenant_id": "11111111-1111-1111-1111-111111111111"}

        credential_data = store.get_decrypted_credential_by_name(
            name=credential_name,
            environment=environment,
            service_name="mcp_tool_executor"
        )
        return credential_data, {"credential_name": credential_name, "environment": environment}
    
    async def execute_tool(
        self,
        agent_id: int,
        tool_id: int,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        execution_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool method for a specific agent.
        
        Args:
            agent_id: ID of the agent executing the tool
            tool_id: ID of the MCP tool to execute
            method: The tool method to call (e.g., "repos.list", "chat.postMessage")
            params: Parameters to pass to the method
            execution_id: Optional workflow execution ID for tracking
        
        Returns:
            Dict containing the execution result
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist or agent doesn't have access
            MCPToolExecutionError: If execution fails
        """
        start_time = time.time()
        success = False
        error_message = None
        output_data = None
        
        try:
            # 1. Verify tool exists and is active
            tool = self.db.query(MCPTool).filter(
                MCPTool.id == tool_id,
                MCPTool.status == 'active'
            ).first()
            
            if not tool:
                raise MCPToolNotFoundError(f"Tool {tool_id} not found or inactive")
            
            # 2. Verify agent has permission to use this tool
            # PRD-35: Enforce explicit agent-tool assignment (no auto-enable)
            # EXCEPTION: Chatbot (agent_id=0 or 1) doesn't need assignments
            assignment = None
            if agent_id > 1:  # Real agent, check assignment
                # CLEAN ARCHITECTURE: Use string tool_id, not numeric id
                string_tool_id = tool.tool_id  # This is the clean string ID ('slack', 'github')
                assignment = self.db.query(AgentToolAssignment).filter(
                    AgentToolAssignment.agent_id == agent_id,
                    AgentToolAssignment.tool_id == string_tool_id,
                    AgentToolAssignment.enabled == True
                ).first()
                
                if not assignment:
                     raise MCPToolNotFoundError(
                        f"Tool '{tool.name}' (tool_id: '{string_tool_id}') is not assigned to agent {agent_id}. "
                        f"Please assign the tool in agent configuration."
                    )
            else:
                # Chatbot system use - no assignment needed
                logger.info(f"✅ Chatbot using tool '{tool.name}' without assignment check")
            
            
            # 3. Permissions: V2 table doesn't have permissions column
            # Default to full access for enabled tools
            permissions = {"read": True, "write": True, "execute": True}
            
            # 4. Validate method exists in tool capabilities
            capabilities = tool.capabilities or {}
            available_methods = self._get_available_methods(capabilities)
            if available_methods and method not in available_methods:
                raise MCPToolExecutionError(
                    f"Method '{method}' not available for tool {tool.name}. "
                    f"Available methods: {available_methods}"
                )

            # Resolve credentials (if configured)
            credential_data = None
            credential_info = None
            try:
                credential_data, credential_info = self._resolve_tool_credentials(tool, assignment)
                if credential_info:
                    logger.info(
                        f"Resolved credentials for MCP tool '{tool.name}' "
                        f"({credential_info})"
                    )
            except (CredentialNotFoundError, CredentialValidationError, EncryptionKeyError) as e:
                raise MCPToolExecutionError(f"Credential resolution failed: {str(e)}")
            
            # 5. Execute the tool via Context Forge (PRD-35 architecture)
            
            # [HOTFIX] Force Hardcoded Tenant ID - User Request: NO SAAS/AUTH SETUP
            tenant_id = "11111111-1111-1111-1111-111111111111"
            logger.info(f"🚀 FINAL Tenant ID for Context Forge: {tenant_id}")
            
            # Route execution based on configuration
            if self.context_forge_enabled:
                # PRD-35: All executions route through Context Forge → Adapter
                logger.info(f"Routing execution through Context Forge: {self.context_forge_url}")
                
                # [HOTFIX] Inject credential token into params if available
                # Context Forge/Adapter requires 'token' in arguments for hosted mode
                cf_params = params.copy() if params else {}
                if credential_data:
                    # Log available keys (masked)
                    safe_keys = list(credential_data.keys())
                    logger.info(f"🔑 Credential keys available: {safe_keys}")
                    
                    if 'token' in credential_data:
                        cf_params['token'] = credential_data['token']
                        logger.info("  -> Injected 'token' from credentials")
                    elif 'bot_token' in credential_data:
                        cf_params['token'] = credential_data['bot_token']
                        logger.info("  -> Injected 'token' (mapped from bot_token) from credentials")
                    elif 'api_token' in credential_data:
                        cf_params['token'] = credential_data['api_token']
                        logger.info("  -> Injected 'token' (mapped from api_token) from credentials")
                    else:
                        # Fallback: Scan for Slack-like tokens (xoxb-*, xoxp-*)
                        found_token = None
                        for key, value in credential_data.items():
                            if isinstance(value, str) and (value.startswith("xoxb-") or value.startswith("xoxp-")):
                                found_token = value
                                logger.info(f"  -> Found Slack token in key '{key}', injecting as 'token'")
                                break
                        
                        if found_token:
                            cf_params['token'] = found_token
                        else:
                            if 'api_key' in credential_data:
                                cf_params['token'] = credential_data['api_key'] 
                                logger.info("  -> Injected 'token' (mapped from api_key) from credentials")
                
                # [HOTFIX] Dynamic credential mode
                # If we resolved credentials locally, use "byo" so CF accepts our injected token
                # Otherwise use "hosted" so CF looks up its own credentials
                final_credential_mode = "byo" if credential_data else "hosted"
                
                output_data = await self._call_context_forge(
                    tool=tool,
                    method=method,
                    params=cf_params,
                    tenant_id=tenant_id,
                    credential_mode=final_credential_mode
                )
            elif tool.mcp_server_url:
                # Legacy: Direct MCP server call (should be deprecated per PRD-35)
                configuration = dict(getattr(assignment, 'configuration', {}) or {})
                for key in ("credential_id", "credential_name", "credential_environment"):
                    configuration.pop(key, None)
                if credential_data:
                    configuration["credentials"] = credential_data

                output_data = await self._call_mcp_server(
                    tool.mcp_server_url,
                    method,
                    params or {},
                    configuration
                )
            else:
                 # No fallback - PRD conformance
                 raise MCPToolExecutionError(
                     f"Context Forge disabled and no MCP server URL for tool {tool.name}. "
                     "Cannot execute."
                 )
            
            success = True
            logger.info(
                f"Tool executed successfully: agent={agent_id}, tool={tool.name}, method={method}"
            )
            
            return {
                "success": True,
                "tool_id": tool_id,
                "tool_name": tool.name,
                "method": method,
                "output": output_data,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except MCPToolNotFoundError as e:
            error_message = str(e)
            logger.warning(error_message)
            raise
            
        except MCPToolExecutionError as e:
            error_message = str(e)
            logger.error(error_message)
            raise
            
        except Exception as e:
            error_message = f"Unexpected error executing tool: {str(e)}"
            logger.exception(error_message)
            raise MCPToolExecutionError(error_message)
            
        finally:
            # Log the execution attempt
            self._log_tool_usage(
                execution_id=execution_id,
                agent_id=agent_id,
                tool_id=tool_id,
                method=method,
                input_data=params,
                output_data=output_data,
                success=success,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_message
            )
    
    async def _call_context_forge(
        self,
        tool: MCPTool,
        method: str,
        params: Dict[str, Any],
        tenant_id: str,
        credential_mode: str = "hosted"
    ) -> Dict[str, Any]:
        """
        Route tool execution through Context Forge gateway.
        PRD-35: All executions must go through CF → Adapter for proper
        authentication, tenant isolation, and credential resolution.
        
        Args:
            tool: The MCPTool object
            method: Method to execute (e.g., "postMessage", "repos.list")
            params: Method parameters
            tenant_id: Tenant identifier for multi-tenant isolation
            credential_mode: "hosted" (Automatos manages) or "byo" (caller provides)
        
        Returns:
            Response data from the tool execution
        """
        try:
            # Build MCP JSON-RPC payload per PRD-35 spec
            # Docs: use /rpc endpoint with tools/call method
            
            # Smart naming for Context Forge routing
            # Adapter tools in CF are named: automatos-unified-adapter-mcp-{tool}-{method}
            # Native MCP tools are named: mcp_{tool}_{method}
            
            # Check if this is an adapter-backed tool
            # 1. Start with explicit provider check
            is_adapter = tool.provider in ("slack", "github", "google", "notion", "linear", "adapter", "unified_adapter")
            
            # 2. Or if configured with tags
            if not is_adapter and tool.tool_metadata and tool.tool_metadata.get("is_adapter"):
                is_adapter = True
                
            # 3. Or common known adapter tools by name
            if not is_adapter and tool.name.lower() in ("slack", "github", "google_drive", "notion", "linear"):
                is_adapter = True
            
            # 4. Or if tool_id is simple string like "slack"
            if not is_adapter and tool.tool_id and tool.tool_id.lower() in ("slack", "github"):
                is_adapter = True

            if is_adapter:
                # Use clean tool_id for the name part (e.g. "slack" from "Slack")
                short_name = tool.tool_id.replace("adapter_", "").lower()
                
                # HOTFIX: Slack methods need namespace prefix (e.g., postMessage → chat.postMessage)
                # TODO: Remove when metadata caching is implemented (PRD task)
                if short_name == "slack" and "." not in method:
                    # Common Slack API patterns - add proper namespace
                    slack_chat_methods = ["postMessage", "postEphemeral", "scheduleMessage", "deleteScheduledMessage", "meMessage", "getPermalink", "unfurl"]
                    slack_conversations_methods = ["archive", "close", "create", "history", "info", "invite", "join", "kick", "leave", "list", "mark", "open", "rename", "replies", "setPurpose", "setTopic", "unarchive"]
                    
                    if method in slack_chat_methods:
                        method = f"chat.{method}"
                    elif method in slack_conversations_methods:
                        method = f"conversations.{method}"
                    # If not in common lists, assume chat namespace (most common)
                    elif not method.startswith(("chat.", "conversations.", "users.", "team.", "admin.")):
                        method = f"chat.{method}"
                
                # Try to look up the actual CF tool name from metadata first
                # Metadata structure: tool_metadata["methods"][method_name]["cf_tool_name"]
                full_tool_name = None
                cache_key = f"{short_name}_{method}"
                
                # Check cache first
                if cache_key in _cf_tool_name_cache:
                    full_tool_name = _cf_tool_name_cache[cache_key]
                    self.logger.debug(f"✅ Using CF tool name from cache: {full_tool_name}")
                
                # Check metadata if not in cache
                if not full_tool_name and tool.tool_metadata and isinstance(tool.tool_metadata, dict):
                    methods_meta = tool.tool_metadata.get("methods", {})
                    if isinstance(methods_meta, dict) and method in methods_meta:
                        method_meta = methods_meta[method]
                        if isinstance(method_meta, dict) and "cf_tool_name" in method_meta:
                            full_tool_name = method_meta["cf_tool_name"]
                            _cf_tool_name_cache[cache_key] = full_tool_name  # Cache it
                            self.logger.info(f"✅ Using CF tool name from metadata: {full_tool_name}")
                
                # Fallback: Query Context Forge dynamically to find matching tool
                if not full_tool_name:
                    try:
                        full_tool_name = await self._find_cf_tool_name(short_name, method)
                        if full_tool_name:
                            _cf_tool_name_cache[cache_key] = full_tool_name  # Cache it
                            self.logger.info(f"✅ Found CF tool name via query: {full_tool_name}")
                        else:
                            # Last resort: construct (will likely fail, but gives us an error message)
                            method_clean = method.lower().replace(".", "-")
                            full_tool_name = f"automatos-unified-adapter-mcp-{short_name}-{method_clean}"
                            self.logger.warning(f"⚠️ Could not find CF tool, using constructed name: {full_tool_name}")
                    except Exception as e:
                        # If query fails, fall back to construction
                        method_clean = method.lower().replace(".", "-")
                        full_tool_name = f"automatos-unified-adapter-mcp-{short_name}-{method_clean}"
                        self.logger.warning(f"⚠️ CF tool lookup failed ({e}), using constructed name: {full_tool_name}")
    
    async def _find_cf_tool_name(self, provider: str, method: str) -> Optional[str]:
        """
        Query Context Forge to find the actual tool name that matches the method.
        
        This searches through all tools for the provider to find one that matches
        the method pattern. This avoids hardcoding 1200+ tool names.
        
        Args:
            provider: Tool provider (e.g., "github", "slack")
            method: Method name (e.g., "repos.list", "chat.postMessage")
            
        Returns:
            The actual CF tool name if found, None otherwise
        """
        cf_url = os.getenv("CONTEXT_FORGE_URL", "https://mcp.automatos.app")
        cf_token = os.getenv("CONTEXT_FORGE_TOKEN", "")
        
        if not cf_token:
            self.logger.warning("CONTEXT_FORGE_TOKEN not configured, cannot query for tool names")
            return None
        
        try:
            # Build search pattern: method parts (e.g., "repos.list" -> ["repos", "list"])
            method_parts = method.lower().replace(".", "-").split("-")
            prefix = f"automatos-unified-adapter-mcp-{provider}-"
            
            headers = {
                "Authorization": f"Bearer {cf_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Query all tools (we'll filter client-side for now)
                # TODO: If CF adds provider filtering, use that
                response = await client.get(f"{cf_url}/tools?limit=0", headers=headers)
                response.raise_for_status()
                tools = response.json()
                
                # Filter to tools for this provider
                provider_tools = [
                    t for t in tools 
                    if isinstance(t, dict) and t.get("name", "").startswith(prefix)
                ]
                
                if not provider_tools:
                    self.logger.warning(f"No tools found for provider '{provider}' in Context Forge")
                    return None
                
                # Score tools by how well they match the method
                # Look for tools that contain all method parts
                best_match = None
                best_score = 0
                
                for tool in provider_tools:
                    tool_name = tool.get("name", "")
                    # Extract the method part (everything after the prefix)
                    method_part = tool_name[len(prefix):] if len(tool_name) > len(prefix) else ""
                    
                    # Score: count how many method parts appear in the tool name
                    score = sum(1 for part in method_parts if part in method_part)
                    
                    # Bonus: exact match or starts with method parts
                    if method_part == "-".join(method_parts):
                        score += 10  # Exact match
                    elif method_part.startswith("-".join(method_parts[:2])):
                        score += 5  # Starts with first two parts
                    
                    if score > best_score:
                        best_score = score
                        best_match = tool_name
                
                if best_match:
                    self.logger.info(f"Matched method '{method}' to CF tool '{best_match}' (score: {best_score})")
                    return best_match
                else:
                    self.logger.warning(f"Could not find matching CF tool for method '{method}'")
                    return None
                    
        except httpx.HTTPError as e:
            self.logger.error(f"Failed to query Context Forge for tool names: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error querying Context Forge: {e}")
            return None
            else:
                # Default fallback for other MCP tools
                full_tool_name = f"mcp_{tool.name}_{method}"
            
            # Build params dict - Context Forge tool schema requires "payload" field
            # The Unified Adapter's FastMCP handler has parameter named "payload"
            # FastMCP generates inputSchema that requires "payload" field inside "arguments"
            if is_adapter:
                # Adapter tools: schema requires payload inside arguments
                # PRD-35: Credentials are now passed via meta field, not in arguments
                adapter_args = {"payload": params}  # Schema requires this structure
                
                # Note: Token remains in payload for backward compatibility with Adapter
                # The credentials will be passed via meta.credentials by Context Forge
                
                rpc_params = {
                    "name": full_tool_name,
                    "arguments": adapter_args
                }
            else:
                # Standard MCP tools: use arguments directly
                rpc_params = {
                    "name": full_tool_name,
                    "arguments": params
                }
            
            # Build the JSON-RPC payload
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": rpc_params,
                "id": str(uuid.uuid4()),
                "meta": {
                    "tenant_id": tenant_id,
                    "credential_mode": credential_mode
                }
            }
            
            # DEBUG: Log the exact payload being sent (redact sensitive data)
            logger.info(f"🔍 SENDING TO CF - is_adapter={is_adapter}, full_tool_name={full_tool_name}")
            logger.info(f"🔍 RPC params keys: {list(rpc_params.keys())}")
            
            # Redact token for logging
            safe_payload = json.loads(json.dumps(payload))  # Deep copy
            if "arguments" in safe_payload.get("params", {}) and "payload" in safe_payload["params"]["arguments"]:
                if "token" in safe_payload["params"]["arguments"]["payload"]:
                    safe_payload["params"]["arguments"]["payload"]["token"] = "***REDACTED***"
            logger.info(f"🔍 Payload (redacted): {json.dumps(safe_payload, indent=2)}")

            
            # Get JWT token for Context Forge authentication
            cf_token = os.getenv("CONTEXT_FORGE_TOKEN", "")
            
            headers = {"Content-Type": "application/json"}
            if cf_token:
                headers["Authorization"] = f"Bearer {cf_token}"
            
            logger.info(f"Calling Context Forge: POST {self.context_forge_url}/rpc")
            logger.debug(f"CF payload: tool={full_tool_name}, args={list(params.keys())}, tenant={tenant_id}")
            
            # Make the HTTP request to Context Forge
            response = await self.http_client.post(
                f"{self.context_forge_url}/rpc",
                json=payload,
                headers=headers,
                timeout=60.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for JSON-RPC error
            if "error" in result:
                raise MCPToolExecutionError(
                    f"Context Forge returned error: {result['error']}"
                )
            
            # Extract result from MCP response
            mcp_result = result.get("result", result)
            
            # DEBUG: Log actual response to diagnose Slack issues
            logger.info(f"CF Response: {mcp_result}")
            
            # Check for errors in the response FIRST - before processing
            if isinstance(mcp_result, dict):
                is_error = mcp_result.get("isError") or mcp_result.get("is_error", False)
                if is_error:
                    error_text = ""
                    if "content" in mcp_result:
                        content = mcp_result["content"]
                        if isinstance(content, list) and len(content) > 0:
                            first_content = content[0]
                            if isinstance(first_content, dict) and "text" in first_content:
                                error_text = first_content["text"]
                                logger.error(f"❌ Context Forge returned error: {error_text}")
                                # If error mentions 'payload', this is a schema validation issue
                                if "payload" in error_text.lower() and "required" in error_text.lower():
                                    logger.error(
                                        "⚠️ Schema validation error - Unified Adapter requires 'payload' field inside 'arguments'. "
                                        "Current structure may be incorrect."
                                    )
                                raise MCPToolExecutionError(f"Context Forge error: {error_text}")
                    # If no content but isError is True, still raise
                    raise MCPToolExecutionError("Context Forge returned error (no error message provided)")
            
            # MCP responses have content array, extract the actual data
            # Handle text responses explicitly for chat tools
            if isinstance(mcp_result, dict) and "content" in mcp_result:
                content = mcp_result["content"]
                if isinstance(content, list) and len(content) > 0:
                    first_content = content[0]
                    if isinstance(first_content, dict):
                        # Return JSON content if available
                        if "json" in first_content:
                            slack_response = first_content["json"]
                            logger.info(f"Slack API Response: {slack_response}")
                            return slack_response
                        elif "text" in first_content:
                            # If it looks like a Slack response, try to parse it
                            import json as json_lib
                            try:
                                txt = first_content["text"]
                                if txt.strip().startswith("{") and txt.strip().endswith("}"):
                                    slack_response = json_lib.loads(txt)
                                    logger.info(f"Slack API Response (parsed): {slack_response}")
                                    return slack_response
                            except:
                                pass
                            return {"message": first_content["text"], "ok": True}
            
            return mcp_result
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", str(e))
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else str(e)
            raise MCPToolExecutionError(
                f"Context Forge error ({e.response.status_code}): {error_detail}"
            )
        except httpx.HTTPError as e:
            raise MCPToolExecutionError(f"HTTP error calling Context Forge: {str(e)}")
        except json.JSONDecodeError as e:
            raise MCPToolExecutionError(f"Invalid JSON response from Context Forge: {str(e)}")
    
    async def _call_mcp_server(
        self,
        server_url: str,
        method: str,
        params: Dict[str, Any],
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call the MCP server to execute the tool method.
        
        Args:
            server_url: Base URL of the MCP server
            method: Method to call
            params: Method parameters
            configuration: Agent-specific tool configuration
        
        Returns:
            Response data from the MCP server
        """
        try:
            # Build the request payload according to MCP protocol
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": f"{int(time.time() * 1000)}"
            }
            
            # Add any agent-specific configuration
            if configuration:
                payload["config"] = configuration
            
            # Make the HTTP request
            response = await self.http_client.post(
                server_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for JSON-RPC error
            if "error" in result:
                raise MCPToolExecutionError(
                    f"MCP server returned error: {result['error']}"
                )
            
            return result.get("result", result)
            
        except httpx.HTTPError as e:
            raise MCPToolExecutionError(f"HTTP error calling MCP server: {str(e)}")
        except json.JSONDecodeError as e:
            raise MCPToolExecutionError(f"Invalid JSON response from MCP server: {str(e)}")
    
    async def _call_adapter_api(
        self,
        adapter_url: str,
        adapter_token: str,
        tool_name: str,
        method: str,
        params: Dict[str, Any],
        credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        PRD-35: Call the Unified Adapter's execute endpoint for REST tools.
        
        Args:
            adapter_url: Base URL of the Adapter admin API
            adapter_token: Bearer token for Adapter authentication
            tool_name: Name of the tool in the Adapter (cross-system compatible)
            method: Operation/method to execute
            params: Parameters for the operation
            credentials: Decrypted credentials to pass to the Adapter
            
        Returns:
            Response data from the Adapter
        """
        try:
            # Use name-based endpoint for cross-system compatibility
            # URL encode the tool name for safety
            import urllib.parse
            encoded_name = urllib.parse.quote(tool_name, safe='')
            execute_url = f"{adapter_url}/admin/tools/name/{encoded_name}/execute"
            
            # Build the request payload
            payload: Dict[str, Any] = {
                "operation": method,
                "params": params
            }
            
            # Include credentials for execution
            if credentials:
                payload["credentials"] = credentials
            
            headers = {"Content-Type": "application/json"}
            if adapter_token:
                headers["Authorization"] = f"Bearer {adapter_token}"
            
            logger.info(f"Calling Adapter API: POST {execute_url}")
            logger.debug(f"Adapter payload: operation={method}, params_keys={list(params.keys())}")
            
            response = await self.http_client.post(
                execute_url,
                json=payload,
                headers=headers,
                timeout=60.0  # Longer timeout for external API calls
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for success in response
            if not result.get("success", True):
                error_msg = result.get("error", "Unknown error from Adapter")
                raise MCPToolExecutionError(f"Adapter execution failed: {error_msg}")
            
            logger.info(f"Adapter execution successful for tool {tool_name}")
            return result.get("result", result)
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", str(e))
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else str(e)
            raise MCPToolExecutionError(f"Adapter API error ({e.response.status_code}): {error_detail}")
        except httpx.HTTPError as e:
            raise MCPToolExecutionError(f"HTTP error calling Adapter API: {str(e)}")
        except json.JSONDecodeError as e:
            raise MCPToolExecutionError(f"Invalid JSON response from Adapter: {str(e)}")
    
    def _check_method_permission(self, method: str, permissions: Dict[str, Any]) -> bool:
        """
        Check if the agent has permission to execute the method.
        
        Args:
            method: Method name (e.g., "repos.create")
            permissions: Agent's permissions dict (e.g., {"read": true, "write": false})
        
        Returns:
            True if permission granted, False otherwise
        """
        # If no permissions specified, allow all
        if not permissions:
            return True
        
        # Map method types to permission categories
        # Methods with "create", "update", "delete", "post" require write permission
        # Methods with "list", "get", "read" require read permission
        # Methods with "execute", "run", "trigger" require execute permission
        
        method_lower = method.lower()
        
        if any(verb in method_lower for verb in ['create', 'update', 'delete', 'post', 'put']):
            return permissions.get('write', False)
        elif any(verb in method_lower for verb in ['execute', 'run', 'trigger', 'start']):
            return permissions.get('execute', False)
        else:
            # Default to read permission for listing/getting
            return permissions.get('read', False)
    
    def _log_tool_usage(
        self,
        execution_id: Optional[int],
        agent_id: int,
        tool_id: int,
        method: str,
        input_data: Optional[Dict[str, Any]],
        output_data: Optional[Dict[str, Any]],
        success: bool,
        execution_time_ms: int,
        error_message: Optional[str]
    ):
        """
        Log tool usage to the database for analytics and auditing.
        """
        try:
            if not agent_id:
                logger.warning("Skipping tool usage log: missing agent_id")
                return

            # Refresh session to avoid stale queries
            self.db.expire_all()
            
            # Check if agent exists - skip logging if agent doesn't exist (prevents NOT NULL violation)
            agent_exists = self.db.query(Agent).filter(Agent.id == agent_id).first()
            if not agent_exists:
                logger.warning(
                    f"Agent {agent_id} not found for logging - skipping tool usage log"
                )
                return

            usage_log = ToolUsageLog(
                execution_id=execution_id,
                agent_id=agent_id,
                tool_id=tool_id,
                method_called=method,
                input_data=input_data,
                output_data=output_data,
                success=success,
                execution_time_ms=execution_time_ms,
                error_message=error_message
            )
            self.db.add(usage_log)
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to log tool usage: {e}")
            self.db.rollback()
            # Don't raise - logging failure shouldn't break tool execution
    
    async def get_agent_tools(self, agent_id: int, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all tools available to an agent.
        
        Args:
            agent_id: ID of the agent
            enabled_only: If True, only return enabled tools
        
        Returns:
            List of tool dictionaries with details and permissions
        """
        query = self.db.query(AgentToolAssignment, MCPTool).join(
            MCPTool, AgentToolAssignment.tool_id == MCPTool.tool_id
        ).filter(
            AgentToolAssignment.agent_id == agent_id
        )
        
        if enabled_only:
            query = query.filter(AgentToolAssignment.enabled == True)
        
        assignments = query.all()
        
        tools = []
        for assignment, tool in assignments:
            tools.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "provider": tool.provider,
                "category": tool.category,
                "icon": tool.icon,
                "capabilities": tool.capabilities,
                "permissions": assignment.permissions,
                "configuration": assignment.configuration,
                "enabled": assignment.enabled
            })
        
        return tools
    
    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
        logger.info("MCPToolExecutor closed")


# Convenience function for single-use executions
async def execute_tool_for_agent(
    db: Session,
    agent_id: int,
    tool_id: int,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    execution_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute a tool without managing the executor instance.
    
    Usage:
        result = await execute_tool_for_agent(
            db=db,
            agent_id=14,
            tool_id=1,
            method="repos.list",
            params={"org": "automatos-ai"}
        )
    """
    executor = MCPToolExecutor(db)
    try:
        return await executor.execute_tool(agent_id, tool_id, method, params, execution_id)
    finally:
        await executor.close()

