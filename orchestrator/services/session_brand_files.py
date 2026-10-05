"""F332 (night 10): a session gets the brand kit's logo as files.

A session agent runs on the operator's machine (the CLI host), where the platform's
document store is out of reach. Night 10's BA said "None of the tools I have here can
read an uploaded image from the Brand kit page"; Support had "no logo file". The claim
now carries the ticket's own workspace's uploaded logo and logo mark
(``modules.documents.brand_logo``), and the host writes them into the session's ticket
folder and names them in its ticket file (``automatos_cli_host/brand_files.py``).

Tenant isolation: the files are read from the CLAIMED ticket's workspace, which is the
host's own (the claim is filtered by it), and a stored path outside that workspace's
``<workspace_id>/brand/`` folder is never read.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The kit field of each stored file, and the name the session sees it under.
LOGO_FILES = (("logo_path", "logo"), ("logo_mark_path", "logo-mark"))


def _brand_file(workspace_id: str, path: str, stem: str) -> Optional[Dict[str, str]]:
    """One stored logo as ``{name, mime, data}`` (base64); None when it is not this
    workspace's or has no bytes anywhere."""
    from modules.documents.brand_logo import load_brand_logo, logo_mime

    if not path:
        return None
    if not path.startswith(f"{workspace_id}/brand/"):
        logger.warning("[SessionBrandFiles] workspace %s's kit names another folder's logo; not sent", workspace_id)
        return None
    data = load_brand_logo(path)
    if not data:
        return None
    return {"name": f"{stem}{os.path.splitext(path)[1].lower()}", "mime": logo_mime(path),
            "data": base64.b64encode(data).decode("ascii")}


def session_brand_files(db: Any, workspace_id: Any) -> List[Dict[str, str]]:
    """The uploaded logo and logo mark of ``workspace_id``'s brand kit, for its session's
    claim; empty when the workspace has no kit or uploaded neither."""
    from services.brand_rules import stored_kit

    kit = stored_kit(db, workspace_id)
    if not kit:
        return []
    found = (_brand_file(str(workspace_id), str(kit.get(field) or ""), stem) for field, stem in LOGO_FILES)
    return [entry for entry in found if entry is not None]


__all__ = ["LOGO_FILES", "session_brand_files"]
