"""F460 (night 18): a read reaches the workspace worker in the worker's own path form.

Auto could not open a file it had just listed: ``deliverables/sessions/3131/x.md`` was "File not
found" and ``/Users/<host>/Development/deliverables/sessions/3103/…`` was "Absolute path not
allowed", while ``sessions/3131/x.md`` opened. The path went to the worker as the model wrote it.
Now the read-only workspace tools map it first, the way a session's files are mapped
(``services.result_files.worker_read_path``): an absolute host path under the workspace root becomes
relative to it, and the root's own folder name in front is dropped. A path outside the root goes as
it came, and the worker refuses it as before. A write or a command is never touched.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Awaitable, Callable, Dict

READ_ONLY_TOOLS = frozenset({"workspace_read_file", "workspace_list_dir", "workspace_grep"})
PATH_PARAM = "path"

Execute = Callable[..., Awaitable[Any]]


def as_the_worker_reads(path: Any, workspace_id: Any) -> str:
    """``path`` as the worker reads it, for this workspace and the configured projects folder."""
    from config import config
    from services.result_files import worker_read_path

    return worker_read_path(str(path or ""), str(workspace_id), getattr(config, "LOCAL_PROJECTS_DIR", "") or None)


def _reads_a_path(call: Dict[str, Any]) -> bool:
    """A read-only tool's call in a workspace, with a path."""
    parameters = call.get("parameters")
    return (call.get("tool_name") in READ_ONLY_TOOLS and bool(call.get("workspace_id"))
            and isinstance(parameters, dict) and bool(parameters.get(PATH_PARAM)))


def reads_in_the_workers_form(execute: Execute) -> Execute:
    """Decorates ``exec_workspace.execute_workspace_action``: a read-only tool's ``path`` is sent
    in the worker's form (above); every other call passes as it came."""
    signature = inspect.signature(execute)

    @functools.wraps(execute)
    async def run(*args: Any, **kwargs: Any) -> Any:
        call = dict(signature.bind(*args, **kwargs).arguments)
        if _reads_a_path(call):
            parameters = call["parameters"]
            path = as_the_worker_reads(parameters[PATH_PARAM], call["workspace_id"])
            call = {**call, "parameters": {**parameters, PATH_PARAM: path}}
        return await execute(**call)
    return run


__all__ = ["READ_ONLY_TOOLS", "as_the_worker_reads", "reads_in_the_workers_form"]
