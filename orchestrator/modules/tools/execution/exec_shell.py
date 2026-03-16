"""
Shell, HTTP, and SSH executors.
Extracted from unified_executor.py.
"""

import logging
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


async def execute_shell(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """Execute shell commands via AgentActionExecutor."""
    success, output = executor.action_executor.execute_command(
        parameters.get('command', ''),
        timeout=parameters.get('timeout', 30)
    )
    return {
        "success": success,
        "action": "execute_command",
        "params": parameters,
        "result": output
    }


def _get_allowed_http_domains():
    from config import config
    return {
        config.INTERNAL_API_HOSTNAME,
        config.INTERNAL_FRONTEND_HOSTNAME,
        "api.automatos.app",
        "ui.automatos.app",
        "localhost",
        "127.0.0.1",
    }


async def execute_http_request(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an HTTP request to a whitelisted domain.

    Used by agents to test API endpoints, check health, and verify responses.
    Domain-restricted to prevent SSRF.
    """
    import httpx
    import time
    from urllib.parse import urlparse

    url = (parameters.get("url") or "").strip()
    method = (parameters.get("method") or "GET").upper()
    headers = parameters.get("headers") or {}
    body = parameters.get("body") or {}
    timeout = min(parameters.get("timeout", 30), 120)

    if not url:
        return {"success": False, "error": "Missing required parameter: url", "tool": tool_name}

    # Validate domain against whitelist
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return {"success": False, "error": "Invalid URL: no hostname", "tool": tool_name}

        allowed = _get_allowed_http_domains()
        if hostname not in allowed:
            return {
                "success": False,
                "error": (
                    f"Domain '{hostname}' is not whitelisted. "
                    f"Allowed: {', '.join(sorted(allowed))}"
                ),
                "tool": tool_name,
            }
    except Exception as e:
        return {"success": False, "error": f"Invalid URL: {e}", "tool": tool_name}

    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return {"success": False, "error": f"Invalid HTTP method: {method}", "tool": tool_name}

    logger.info(
        f"[http_request] {method} {url} agent={agent_id} "
        f"workspace={workspace_id} headers_keys={list(headers.keys())}"
    )

    t0 = time.time()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            verify=False,  # Internal Railway URLs use self-signed certs
        ) as client:
            # Build request kwargs
            kwargs: Dict[str, Any] = {"headers": headers}
            if method in {"POST", "PUT", "PATCH"} and body:
                kwargs["json"] = body

            resp = await client.request(method, url, **kwargs)

        duration_ms = int((time.time() - t0) * 1000)

        # Parse response body (try JSON first, fall back to text)
        resp_body: Any
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = resp.text[:50_000]
        else:
            resp_body = resp.text[:50_000]

        result = {
            "success": True,
            "tool": tool_name,
            "status_code": resp.status_code,
            "response_headers": dict(resp.headers),
            "body": resp_body,
            "duration_ms": duration_ms,
            "url": url,
            "method": method,
        }

        logger.info(
            f"[http_request] {method} {url} -> {resp.status_code} "
            f"in {duration_ms}ms (body_len={len(str(resp_body))})"
        )
        return result

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"Request timed out after {timeout}s",
            "tool": tool_name,
            "url": url,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except httpx.ConnectError as e:
        return {
            "success": False,
            "error": f"Connection failed: {e}",
            "tool": tool_name,
            "url": url,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"[http_request] Error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name,
            "url": url,
            "duration_ms": int((time.time() - t0) * 1000),
        }


async def execute_ssh(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a command on a remote server via SSH using paramiko.

    Supports password auth, private key auth, and stored credential lookup.
    """
    import io
    import time

    host = (parameters.get("host") or "").strip()
    command = (parameters.get("command") or "").strip()
    username = parameters.get("username", "automatos")
    port = int(parameters.get("port", 22))
    password = parameters.get("password")
    private_key = parameters.get("private_key")
    credential_id = parameters.get("credential_id")
    timeout = min(int(parameters.get("timeout", 60)), 300)

    if not host:
        return {"success": False, "error": "Missing required parameter: host", "tool": tool_name}
    if not command:
        return {"success": False, "error": "Missing required parameter: command", "tool": tool_name}

    # Resolve stored credential if provided
    if credential_id and not password and not private_key:
        try:
            from core.models.credentials import StoredCredential
            cred = executor.db.query(StoredCredential).filter(
                StoredCredential.id == credential_id,
                StoredCredential.workspace_id == workspace_id,
            ).first()
            if cred and cred.decrypted_data:
                cred_data = cred.decrypted_data
                password = cred_data.get("password")
                private_key = cred_data.get("private_key")
                username = cred_data.get("username", username)
                host = cred_data.get("host", host)
                port = int(cred_data.get("port", port))
                logger.info(f"[ssh_execute] Using stored credential {credential_id}")
            else:
                return {"success": False, "error": f"Credential {credential_id} not found", "tool": tool_name}
        except Exception as e:
            logger.warning(f"[ssh_execute] Credential lookup failed: {e}")
            return {"success": False, "error": f"Credential lookup failed: {e}", "tool": tool_name}

    if not password and not private_key:
        return {
            "success": False,
            "error": "SSH authentication required: provide password, private_key, or credential_id",
            "tool": tool_name,
        }

    logger.info(
        f"[ssh_execute] {username}@{host}:{port} command='{command[:80]}...' "
        f"agent={agent_id} timeout={timeout}s"
    )

    t0 = time.time()
    try:
        import paramiko

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": host,
            "port": port,
            "username": username,
            "timeout": min(timeout, 30),  # Connection timeout
        }

        if private_key:
            # Parse PEM key
            key_file = io.StringIO(private_key)
            try:
                pkey = paramiko.RSAKey.from_private_key(key_file)
            except paramiko.ssh_exception.SSHException:
                key_file.seek(0)
                pkey = paramiko.Ed25519Key.from_private_key(key_file)
            connect_kwargs["pkey"] = pkey
        elif password:
            connect_kwargs["password"] = password

        ssh.connect(**connect_kwargs)

        # Execute command
        _stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)

        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")

        ssh.close()

        duration_ms = int((time.time() - t0) * 1000)

        # Truncate large outputs
        max_output = 500_000
        if len(stdout_text) > max_output:
            stdout_text = stdout_text[:max_output] + "\n... (truncated)"
        if len(stderr_text) > max_output:
            stderr_text = stderr_text[:max_output] + "\n... (truncated)"

        result = {
            "success": exit_code == 0,
            "tool": tool_name,
            "exit_code": exit_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "host": host,
            "command": command,
            "duration_ms": duration_ms,
        }

        logger.info(
            f"[ssh_execute] {username}@{host} exit_code={exit_code} "
            f"in {duration_ms}ms stdout_len={len(stdout_text)} stderr_len={len(stderr_text)}"
        )
        return result

    except ImportError:
        return {"success": False, "error": "paramiko package not installed", "tool": tool_name}
    except paramiko.AuthenticationException:
        return {
            "success": False,
            "error": "SSH authentication failed -- check username/password/key",
            "tool": tool_name,
            "host": host,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except paramiko.SSHException as e:
        return {
            "success": False,
            "error": f"SSH error: {e}",
            "tool": tool_name,
            "host": host,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except TimeoutError:
        return {
            "success": False,
            "error": f"SSH command timed out after {timeout}s",
            "tool": tool_name,
            "host": host,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"[ssh_execute] Error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name,
            "host": host,
            "duration_ms": int((time.time() - t0) * 1000),
        }
