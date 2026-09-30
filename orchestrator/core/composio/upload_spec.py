"""
PRD-251 D8 (US-301) — a Composio call's OWN upload spec
=======================================================

The Socials publisher names the params of a channel step that take a file; exactly
those are converted to Composio FileUploadables, for that call only
(``ComposioToolExecutor.execute_with_uploads``). The executor's global
``UPLOAD_ACTIONS`` conversion is neither consulted nor widened, and it is not applied
to such a call at all: its approved copy (a bare link, say) is sent as approved,
never turned into a file.

Only a ``Path`` the platform staged is taken: a string (an agent's or a platform's
argument) never reads the local disk here.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence
from uuid import UUID

from core.composio.client import get_composio_client
from core.composio.deny_list import composio_action_denial_async, denied_result
from core.composio.post_gate import post_action_refusal, refused_result

logger = logging.getLogger(__name__)

ERROR_TYPE_FILE_UPLOAD = "file_upload_failed"

# Set while a call with its own upload spec runs, so the executor's global
# UPLOAD_ACTIONS conversion leaves it alone.
_OWN_SPEC: ContextVar[bool] = ContextVar("composio_own_upload_spec", default=False)


class FileUploadFailed(Exception):
    """A file the call's upload spec names could not be handed to Composio."""


@contextmanager
def own_upload_spec() -> Iterator[None]:
    """Mark the calls made inside as carrying their own upload spec."""
    token = _OWN_SPEC.set(True)
    try:
        yield
    finally:
        _OWN_SPEC.reset(token)


def has_own_upload_spec() -> bool:
    """Whether the running call carries its own upload spec (see :func:`own_upload_spec`)."""
    return _OWN_SPEC.get()


def _file_uploadable_class():
    try:
        from composio.core.models._files import FileUploadable
    except ImportError:
        from composio.client.files import FileUploadable
    return FileUploadable


def _uploaded(path: Any, param_name: str, action_upper: str, toolkit: str, http_client: Any) -> Dict[str, Any]:
    """One staged file as Composio's FileUploadable."""
    if not isinstance(path, Path) or not path.is_file():
        raise FileUploadFailed(f"{param_name} is not a file the platform staged for {action_upper}")
    try:
        uploadable = _file_uploadable_class().from_path(
            client=http_client,
            file=path,
            tool=action_upper.lower().replace("_", "-"),
            toolkit=toolkit,
            sensitive_file_upload_protection=False,
        )
    except Exception as exc:
        logger.exception("[FileUpload] %s: %s could not be uploaded to Composio", action_upper, param_name)
        raise FileUploadFailed(f"{param_name} could not be uploaded to Composio: {exc}") from exc
    return uploadable.model_dump()


def resolve_upload_spec(
    action: str, params: Dict[str, Any], upload_params: Sequence[str], toolkit: str,
) -> Dict[str, Any]:
    """``params`` with each of ``upload_params`` (a staged local file, or a list of
    them) converted to a Composio FileUploadable, and nothing else touched. Strict:
    a named param that is missing, not a staged file, or refused by Composio raises
    :class:`FileUploadFailed`. Blocking (the SDK uploads): run it in a thread."""
    action_upper = str(action).upper()
    http_client = get_composio_client().composio.client
    converted = dict(params)
    for name in upload_params:
        value = params.get(name)
        if value is None or value == []:
            raise FileUploadFailed(f"{action_upper} needs a file for {name}, and none was given")
        if isinstance(value, (list, tuple)):
            converted[name] = [_uploaded(item, f"{name}[{i}]", action_upper, toolkit, http_client) for i, item in enumerate(value)]
        else:
            converted[name] = _uploaded(value, name, action_upper, toolkit, http_client)
    return converted


def _result(result: Dict[str, Any], action_upper: str, start_time: float) -> Dict[str, Any]:
    """A refusal or failure in the executor's result shape."""
    return {**result, "action": action_upper, "execution_time_ms": int((time.time() - start_time) * 1000)}


async def execute_with_uploads(
    executor: Any,
    action: str,
    params: Dict[str, Any],
    *,
    agent_id: int,
    workspace_id: UUID,
    app_name: str,
    upload_params: Sequence[str] = (),
    way_through: Optional[object] = None,
) -> Dict[str, Any]:
    """Run ``action`` for the platform with the call's own upload spec: the Wave 0
    deny list, then the Socials post gate, then exactly ``upload_params`` as files
    (:func:`resolve_upload_spec`), then ``executor.execute`` on the workspace's own
    connection, with no global UPLOAD_ACTIONS conversion. A refused call uploads
    nothing. The LinkedIn image workaround reads the staged files itself (Composio
    cannot upload LinkedIn images), so a call it takes keeps them."""
    from core.composio.linkedin_image_workaround import IMAGE_POST_ACTION, has_image_params

    start_time = time.time()
    action_upper = str(action or "").strip().upper()
    denial = await composio_action_denial_async(action_upper)
    if denial:
        return _result(denied_result(denial), action_upper, start_time)
    refusal = await post_action_refusal(action_upper, workspace_id, way_through=way_through)
    if refusal:
        return _result(refused_result(refusal), action_upper, start_time)
    workaround = action_upper == IMAGE_POST_ACTION and has_image_params(params)
    if upload_params and not workaround:
        try:
            params = await asyncio.to_thread(resolve_upload_spec, action_upper, params, upload_params, app_name.lower())
        except FileUploadFailed as exc:
            failure = {"success": False, "data": None, "error": str(exc), "error_type": ERROR_TYPE_FILE_UPLOAD}
            return _result(failure, action_upper, start_time)
    with own_upload_spec():
        return await executor.execute(
            action=action_upper, params=params, agent_id=agent_id, workspace_id=workspace_id,
            app_name=app_name, skip_validation=True, way_through=way_through,
        )
