"""F332 (night 10): a session gets the brand kit's logo as files.

A session agent runs on the operator's machine (the CLI host), where the platform's
document store is out of reach. Night 10's BA said "None of the tools I have here can
read an uploaded image from the Brand kit page"; Support had "no logo file". The claim
now carries the ticket's own workspace's uploaded logo and logo mark
(``modules.documents.brand_logo``), and the host writes them into the session's ticket
folder and names them in its ticket file (``automatos_cli_host/brand_files.py``).
PRD-255 (US-008): the logo's uploaded variants too, the logo for dark backgrounds and the
one-colour logo, which the rules block names by :func:`session_file_name`.

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
LOGO_FILES = (("logo_path", "logo"), ("logo_mark_path", "logo-mark"),
              ("logo_dark_path", "logo-dark"), ("logo_mono_path", "logo-mono"))
# The folder of the session's deliverables the host writes them into (the host's BRAND_FOLDER).
SESSION_BRAND_FOLDER = "brand"


def session_file_name(path: str, stem: str) -> str:
    """The name the session sees a stored brand file under: ``stem`` and the stored file's extension."""
    return f"{stem}{os.path.splitext(path)[1].lower()}"


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
    return {"name": session_file_name(path, stem), "mime": logo_mime(path),
            "data": base64.b64encode(data).decode("ascii")}


def session_brand_files(db: Any, workspace_id: Any) -> List[Dict[str, str]]:
    """The uploaded logo, logo mark and logo variants of ``workspace_id``'s brand kit, for
    its session's claim; empty when the workspace has no kit or uploaded none."""
    from services.brand_rules import stored_kit

    kit = stored_kit(db, workspace_id)
    if not kit:
        return []
    found = (_brand_file(str(workspace_id), str(kit.get(field) or ""), stem) for field, stem in LOGO_FILES)
    return [entry for entry in found if entry is not None]


__all__ = ["LOGO_FILES", "SESSION_BRAND_FOLDER", "session_brand_files", "session_file_name"]
