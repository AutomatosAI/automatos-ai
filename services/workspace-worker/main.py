"""
Workspace Worker — ARQ Consumer
================================
PRD-56 Phase 2: Physical Workspace Worker Service

Pulls tasks from Redis priority queues and executes them in isolated
workspace directories on the persistent volume. Each workspace gets
its own filesystem with cached repos, artifacts, and ephemeral task dirs.

Startup:
    python -m worker.main

Configuration (env vars):
    REDIS_URL              - Redis connection string (required)
    DATABASE_URL           - PostgreSQL connection string (required)
    WORKSPACE_VOLUME_PATH  - Mount path for persistent volume (default: /workspaces)
    WORKSPACE_DEFAULT_QUOTA_GB - Default storage per workspace (default: 5)
    WORKER_CONCURRENCY     - Max concurrent tasks (default: 3)
    WORKER_HEALTH_PORT     - Health check HTTP port (default: 8081)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

# Add orchestrator to path so we can import shared code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orchestrator"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("workspace-worker")

# Redis key patterns (must match queued.py)
QUEUE_NAMES = [
    "workspace:tasks:critical",
    "workspace:tasks:high",
    "workspace:tasks:normal",
    "workspace:tasks:low",
]
TASK_STATUS_KEY = "workspace:task:{task_id}:status"
TASK_RESULT_KEY = "workspace:task:{task_id}:result"
TASK_EVENTS_CHANNEL = "workspace:task:{task_id}:events"
WS_ACTIVE_TASKS_KEY = "workspace:ws:{workspace_id}:active_tasks"

RESULT_TTL_SECONDS = 3600
STATUS_TTL_SECONDS = 7200


class WorkspaceWorker:
    """ARQ-style worker that consumes tasks from Redis queues.

    Instead of using the ARQ library directly (which adds complexity),
    this implements a simple, reliable queue consumer with:
    - Priority queue polling (critical > high > normal > low)
    - Concurrent task execution (configurable)
    - Heartbeat reporting
    - Graceful shutdown
    """

    def __init__(self) -> None:
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.concurrency = int(os.environ.get("WORKER_CONCURRENCY", "3"))
        self.health_port = int(os.environ.get("WORKER_HEALTH_PORT", "8081"))
        self._redis = None
        self._db_engine = None
        self._running = True
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._worker_id = f"worker-{os.getpid()}-{int(time.time())}"

    async def start(self) -> None:
        """Start the worker — runs until SIGTERM/SIGINT."""
        logger.info(
            "Workspace Worker starting (id=%s, concurrency=%d, volume=%s)",
            self._worker_id, self.concurrency,
            os.environ.get("WORKSPACE_VOLUME_PATH", "/workspaces"),
        )

        import redis.asyncio as aioredis
        import redis.exceptions
        self._redis = aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            retry_on_error=[redis.exceptions.BusyLoadingError],
        )

        # Wait for Redis to be ready (handles BusyLoadingError on startup)
        for attempt in range(1, 16):
            try:
                await self._redis.ping()
                logger.info("Redis connection established")
                break
            except (redis.exceptions.BusyLoadingError, ConnectionError, OSError) as e:
                logger.warning("Redis not ready (attempt %d/15): %s", attempt, e)
                if attempt == 15:
                    raise RuntimeError("Redis not available after 15 attempts") from e
                await asyncio.sleep(2)

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Start health check server in background
        health_task = asyncio.create_task(self._health_server())

        # Start heartbeat reporter
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Main consumer loop
        try:
            await self._consume_loop()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down worker %s...", self._worker_id)
            # Wait for active tasks to finish
            if self._active_tasks:
                logger.info("Waiting for %d active tasks...", len(self._active_tasks))
                await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)

            health_task.cancel()
            heartbeat_task.cancel()

            if self._redis:
                await self._redis.aclose()

            if self._db_engine:
                self._db_engine.dispose()

            logger.info("Worker %s shutdown complete", self._worker_id)

    def _handle_shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._running = False

    async def _consume_loop(self) -> None:
        """Main loop: poll Redis queues for tasks."""
        poll_interval = 0.5  # seconds between polls when idle

        while self._running:
            # Try to acquire semaphore (limits concurrency)
            if self._semaphore._value == 0:
                await asyncio.sleep(poll_interval)
                continue

            # Poll queues in priority order
            task_payload = await self._dequeue_task()

            if task_payload is None:
                await asyncio.sleep(poll_interval)
                continue

            # Got a task — execute it
            task_id = task_payload["task_id"]
            await self._semaphore.acquire()

            async_task = asyncio.create_task(
                self._execute_task_wrapper(task_payload),
                name=f"task-{task_id[:8]}",
            )
            self._active_tasks[task_id] = async_task

            # Clean up completed tasks from tracking dict
            done = [tid for tid, t in self._active_tasks.items() if t.done()]
            for tid in done:
                del self._active_tasks[tid]

    async def _dequeue_task(self) -> Optional[Dict[str, Any]]:
        """Pop the highest-priority task from Redis queues.

        Checks queues in order: critical → high → normal → low.
        Returns None if all queues are empty or Redis is temporarily unavailable.
        """
        try:
            for queue_name in QUEUE_NAMES:
                result = await self._redis.rpop(queue_name)
                if result:
                    try:
                        payload = json.loads(result)
                        logger.info(
                            "Dequeued task %s from %s (workspace=%s)",
                            payload["task_id"][:8], queue_name, payload["workspace_id"][:8],
                        )
                        return payload
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error("Invalid task payload from %s: %s", queue_name, e)
                        continue
        except Exception as e:
            logger.warning("Redis dequeue error (will retry): %s", e)
        return None

    async def _execute_task_wrapper(self, payload: Dict[str, Any]) -> None:
        """Wrapper that handles task execution lifecycle + cleanup."""
        task_id = payload["task_id"]
        workspace_id = payload["workspace_id"]

        try:
            await self._execute_task(payload)
        except Exception as e:
            logger.error("Task %s failed with unhandled error: %s", task_id[:8], e, exc_info=True)
            await self._write_result(task_id, workspace_id, {
                "status": "failed",
                "error": f"Worker error: {str(e)}",
            })
            await self._update_status(task_id, "failed", error=str(e))
        finally:
            # Release semaphore
            self._semaphore.release()

            # Remove from workspace active tracking
            ws_key = WS_ACTIVE_TASKS_KEY.format(workspace_id=workspace_id)
            await self._redis.srem(ws_key, task_id)

    async def _execute_task(self, payload: Dict[str, Any]) -> None:
        """Execute a single workspace task."""
        from workspace_manager import WorkspaceManager
        from executor import WorkspaceToolExecutor

        task_id = payload["task_id"]
        workspace_id = payload["workspace_id"]
        start_time = time.monotonic()

        # Check if task was cancelled while queued
        status_key = TASK_STATUS_KEY.format(task_id=task_id)
        current_status = await self._redis.hget(status_key, "status")
        if current_status == "cancelled":
            logger.info("Task %s was cancelled while queued, skipping", task_id[:8])
            return

        # Transition: QUEUED → RUNNING
        await self._update_status(task_id, "running")
        await self._publish_event(task_id, "status_changed", {"status": "running"})

        # 1. Provision workspace
        ws_manager = WorkspaceManager(workspace_id)
        ws_manager.ensure_workspace_exists()

        # 2. Check storage quota
        if not ws_manager.check_quota():
            error_msg = (
                f"Workspace storage exceeds quota "
                f"({ws_manager.usage_human} / {ws_manager.quota_human}). "
                f"Free space by removing old repos or upgrade your plan."
            )
            await self._write_result(task_id, workspace_id, {
                "status": "failed",
                "error": error_msg,
            })
            await self._update_status(task_id, "failed", error=error_msg)
            await self._publish_event(task_id, "error", {"error": error_msg})
            return

        # 3. Create ephemeral task dir
        task_dir = ws_manager.create_task_dir(task_id)

        # 4. Inject credentials
        credentials = payload.get("credentials", {})
        if credentials:
            ws_manager.inject_credentials(task_id, credentials)

        # 5. Execute task steps
        executor = WorkspaceToolExecutor(ws_manager)
        steps = payload.get("steps", [])
        results = []

        try:
            for i, step in enumerate(steps):
                # Check for cancellation between steps
                current_status = await self._redis.hget(status_key, "status")
                if current_status == "cancelled":
                    logger.info("Task %s cancelled mid-execution at step %d", task_id[:8], i)
                    await self._publish_event(task_id, "status_changed", {"status": "cancelled"})
                    return

                # Publish progress
                step_desc = step.get("description", f"Step {i+1}: {step.get('action', 'unknown')}")
                await self._publish_event(task_id, "progress_update", {
                    "step": i + 1,
                    "total_steps": len(steps),
                    "description": step_desc,
                })

                # Execute step
                result = await executor.execute_step(step)
                results.append({"step": i + 1, "action": step.get("action"), "result": result})

                if result.get("error"):
                    # Step failed — report but continue (agent decides what to do)
                    await self._publish_event(task_id, "error", {
                        "step": i + 1,
                        "error": result["error"],
                    })

            # 6. All steps completed
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            final_result = {
                "status": "completed",
                "result": {
                    "steps": results,
                    "workspace_id": workspace_id,
                    "repos_cached": ws_manager.list_repos(),
                },
                "execution_time_ms": elapsed_ms,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            await self._write_result(task_id, workspace_id, final_result)
            await self._update_status(task_id, "completed")
            await self._publish_event(task_id, "status_changed", {
                "status": "completed",
                "execution_time_ms": elapsed_ms,
            })

            ws_manager.increment_task_count()
            logger.info("Task %s completed in %dms", task_id[:8], elapsed_ms)

        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            await self._write_result(task_id, workspace_id, {
                "status": "timed_out",
                "error": f"Task timed out after {payload.get('timeout_seconds', 300)}s",
                "result": {"steps": results},
                "execution_time_ms": elapsed_ms,
            })
            await self._update_status(task_id, "timed_out")
            await self._publish_event(task_id, "status_changed", {"status": "timed_out"})

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.error("Task %s execution error: %s", task_id[:8], e, exc_info=True)
            await self._write_result(task_id, workspace_id, {
                "status": "failed",
                "error": str(e),
                "result": {"steps": results},
                "execution_time_ms": elapsed_ms,
            })
            await self._update_status(task_id, "failed", error=str(e))
            await self._publish_event(task_id, "error", {"error": str(e)})
            await self._publish_event(task_id, "status_changed", {"status": "failed"})

        finally:
            # 7. Cleanup ephemeral task dir (workspace persists)
            ws_manager.cleanup_task(task_id)

    # =========================================================================
    # Redis helpers
    # =========================================================================

    async def _update_status(self, task_id: str, status: str, error: str = None) -> None:
        """Update task status in Redis hash AND Postgres task_executions."""
        status_key = TASK_STATUS_KEY.format(task_id=task_id)
        mapping = {"status": status, "worker_id": self._worker_id}
        if status == "running":
            mapping["started_at"] = datetime.now(timezone.utc).isoformat()
        if error:
            mapping["error"] = error
        await self._redis.hset(status_key, mapping=mapping)
        await self._redis.expire(status_key, STATUS_TTL_SECONDS)

        # Persist to Postgres in lockstep
        await self._update_db_execution(task_id, status, error=error)

    async def _update_db_execution(self, task_id: str, status: str, error: str = None) -> None:
        """Persist task lifecycle state to Postgres task_executions table."""
        try:
            from sqlalchemy import create_engine, text

            if self._db_engine is None:
                db_url = os.environ.get("DATABASE_URL", "")
                if not db_url:
                    logger.warning("DATABASE_URL not set — skipping DB update for task %s", task_id[:8])
                    return
                self._db_engine = create_engine(db_url, pool_pre_ping=True)

            now = datetime.now(timezone.utc)

            set_clauses = ["status = :status", "worker_id = :worker_id"]
            params: Dict[str, Any] = {
                "task_id": task_id,
                "status": status,
                "worker_id": self._worker_id,
            }

            if status == "running":
                set_clauses.append("started_at = :started_at")
                params["started_at"] = now
            elif status in ("completed", "failed", "timed_out", "cancelled"):
                set_clauses.append("completed_at = :completed_at")
                params["completed_at"] = now
            if error:
                set_clauses.append("error_message = :error")
                params["error"] = error

            sql = f"UPDATE task_executions SET {', '.join(set_clauses)} WHERE id = :task_id"

            with self._db_engine.connect() as conn:
                conn.execute(text(sql), params)
                conn.commit()

        except Exception as e:
            logger.error("DB update failed for task %s (%s): %s", task_id[:8], status, e)

    async def _write_result(self, task_id: str, workspace_id: str, result: Dict[str, Any]) -> None:
        """Write task result to Redis (for API polling)."""
        result_key = TASK_RESULT_KEY.format(task_id=task_id)
        await self._redis.set(result_key, json.dumps(result), ex=RESULT_TTL_SECONDS)

    async def _publish_event(self, task_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publish task event via Redis pub/sub."""
        channel = TASK_EVENTS_CHANNEL.format(task_id=task_id)
        event = {
            "event_type": event_type,
            "task_id": task_id,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await self._redis.publish(channel, json.dumps(event))
        except Exception as e:
            logger.warning("Failed to publish event for task %s: %s", task_id[:8], e)

    # =========================================================================
    # Health & heartbeat
    # =========================================================================

    async def _heartbeat_loop(self) -> None:
        """Report worker health to Redis every 30s."""
        heartbeat_key = f"workspace:worker:{self._worker_id}:heartbeat"
        tasks_key = f"workspace:worker:{self._worker_id}:tasks"

        while self._running:
            try:
                await self._redis.set(heartbeat_key, str(int(time.time())), ex=60)
                # Report active task IDs
                active_ids = list(self._active_tasks.keys())
                if active_ids:
                    await self._redis.delete(tasks_key)
                    await self._redis.sadd(tasks_key, *active_ids)
                    await self._redis.expire(tasks_key, 60)
                else:
                    await self._redis.delete(tasks_key)
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)

            await asyncio.sleep(30)

    async def _health_server(self) -> None:
        """HTTP server for health checks and file browsing endpoints."""
        from aiohttp import web
        from workspace_manager import WorkspaceManager, SecurityError

        volume_path = os.environ.get("WORKSPACE_VOLUME_PATH", "/workspaces")
        max_file_size = 2 * 1024 * 1024  # 2 MB
        max_dir_entries = 500
        internal_token = os.environ.get("WORKER_INTERNAL_TOKEN", "")
        bind_host = os.environ.get("WORKER_BIND_HOST", "0.0.0.0")

        # Sensitive paths that must never be exposed via file browsing
        _SENSITIVE_NAMES = {".ssh", ".gitconfig", ".aws", ".gcp", ".workspace_meta.json"}

        def _is_sensitive(name: str) -> bool:
            """Check if a file/dir name is sensitive and should be hidden."""
            if name in _SENSITIVE_NAMES:
                return True
            if name.startswith(".task_env_"):
                return True
            return False

        # Language detection map (Monaco-compatible)
        _lang_map = {
            ".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript", ".json": "json",
            ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
            ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
            ".sql": "sql", ".sh": "shell", ".bash": "shell", ".zsh": "shell",
            ".rs": "rust", ".go": "go", ".java": "java", ".c": "c",
            ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".rb": "ruby",
            ".php": "php", ".xml": "xml", ".toml": "toml", ".ini": "ini",
            ".cfg": "ini", ".env": "dotenv", ".dockerfile": "dockerfile",
            ".r": "r", ".swift": "swift", ".kt": "kotlin", ".lua": "lua",
        }

        def _guess_language(filename: str) -> str:
            from pathlib import Path as P
            return _lang_map.get(P(filename).suffix.lower(), "plaintext")

        # Auth middleware — reject requests without valid internal token
        @web.middleware
        async def internal_auth_middleware(request, handler):
            # Health endpoint is always public
            if request.path == "/health":
                return await handler(request)
            # If token is configured, enforce it
            if internal_token:
                req_token = request.headers.get("X-Internal-Token", "")
                if req_token != internal_token:
                    return web.json_response({"error": "Unauthorized"}, status=401)
            return await handler(request)

        app = web.Application(middlewares=[internal_auth_middleware])

        async def health_handler(request):
            return web.json_response({
                "status": "healthy",
                "worker_id": self._worker_id,
                "active_tasks": len(self._active_tasks),
                "concurrency": self.concurrency,
                "volume_path": volume_path,
            })

        async def list_files_handler(request):
            """GET /workspaces/{workspace_id}/files?path=. — directory listing."""
            from pathlib import Path as P

            workspace_id = request.match_info["workspace_id"]
            rel_path = request.query.get("path", ".")

            ws_manager = WorkspaceManager(workspace_id, volume_path)
            ws_dir = P(volume_path) / workspace_id

            if not ws_dir.is_dir():
                return web.json_response(
                    {"error": "Workspace directory not found"}, status=404
                )

            try:
                target = ws_manager.resolve_safe_path(rel_path)
            except SecurityError as exc:
                return web.json_response({"error": str(exc)}, status=403)

            # Block access to sensitive paths
            for part in target.relative_to(ws_dir.resolve()).parts:
                if _is_sensitive(part):
                    return web.json_response({"error": "Access denied"}, status=403)

            if not target.exists():
                return web.json_response({"error": "Path not found"}, status=404)
            if not target.is_dir():
                return web.json_response({"error": "Path is not a directory"}, status=400)

            entries = []
            truncated = False
            try:
                for i, item in enumerate(
                    sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                ):
                    # Skip sensitive entries
                    if _is_sensitive(item.name):
                        continue
                    if i >= max_dir_entries:
                        truncated = True
                        break
                    stat = item.stat()
                    rel = str(item.relative_to(ws_dir))
                    entries.append({
                        "name": item.name,
                        "path": rel,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else 0,
                        "modified_at": stat.st_mtime,
                    })
            except PermissionError:
                return web.json_response({"error": "Permission denied"}, status=403)

            return web.json_response({
                "path": rel_path,
                "entries": entries,
                "truncated": truncated,
            })

        async def file_content_handler(request):
            """GET /workspaces/{workspace_id}/files/content?path=file.py — file content."""
            from pathlib import Path as P
            import mimetypes

            workspace_id = request.match_info["workspace_id"]
            rel_path = request.query.get("path")
            if not rel_path:
                return web.json_response({"error": "path query param required"}, status=400)

            ws_manager = WorkspaceManager(workspace_id, volume_path)
            ws_dir = P(volume_path) / workspace_id

            if not ws_dir.is_dir():
                return web.json_response(
                    {"error": "Workspace directory not found"}, status=404
                )

            try:
                target = ws_manager.resolve_safe_path(rel_path)
            except SecurityError as exc:
                return web.json_response({"error": str(exc)}, status=403)

            # Block access to sensitive paths
            for part in target.relative_to(ws_dir.resolve()).parts:
                if _is_sensitive(part):
                    return web.json_response({"error": "Access denied"}, status=403)

            if not target.exists():
                return web.json_response({"error": "File not found"}, status=404)
            if not target.is_file():
                return web.json_response({"error": "Path is not a file"}, status=400)

            file_size = target.stat().st_size
            if file_size > max_file_size:
                return web.json_response(
                    {"error": f"File too large ({file_size} bytes, max {max_file_size})"},
                    status=413,
                )

            try:
                content = target.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                return web.json_response({"error": "Unable to read file as text"}, status=422)

            mime_type, _ = mimetypes.guess_type(target.name)

            return web.json_response({
                "path": rel_path,
                "name": target.name,
                "content": content,
                "size": file_size,
                "language": _guess_language(target.name),
                "mime_type": mime_type or "text/plain",
            })

        app.router.add_get("/health", health_handler)
        app.router.add_get("/workspaces/{workspace_id}/files", list_files_handler)
        app.router.add_get("/workspaces/{workspace_id}/files/content", file_content_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, bind_host, self.health_port)

        try:
            await site.start()
            logger.info("Health server listening on port %d", self.health_port)
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()


# =============================================================================
# Entry point
# =============================================================================

async def main():
    worker = WorkspaceWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
