"""The media-render HTTP client the seeded-template checks use (scripts/ci).

``social_template_previews.py`` posts each bundle here and waits for the finished
job, outputs fetched. Standard library only: CI runs it with the runner's python3.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional, Tuple

POLL_SECONDS = 2.0
JOB_WAIT_SECONDS = 900
TOKEN_HEADER = "X-Internal-Token"


class PreviewFailure(RuntimeError):
    """A template that did not check, preview or render cleanly."""


class Renderer:
    def __init__(self, url: str, token: str) -> None:
        self.url, self.token = url.rstrip("/"), token

    def _request(self, method: str, path: str, body: Optional[bytes] = None) -> Tuple[int, bytes]:
        request = urllib.request.Request(self.url + path, data=body, method=method)
        request.add_header(TOKEN_HEADER, self.token)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=JOB_WAIT_SECONDS) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def render(self, bundle: Mapping[str, Any]) -> Dict[str, Any]:
        """Post the bundle; the finished job (outputs fetched), or PreviewFailure with the service's answer."""
        started = time.monotonic()
        status, body = self._request("POST", "/render", json.dumps(bundle).encode("utf-8"))
        answer = json.loads(body or b"{}")
        if status != 202:
            findings = answer.get("findings") or []
            for finding in findings[:40]:
                partner = f" with {finding['containerSelector']}" if finding.get("containerSelector") else ""
                print(f"    {finding.get('severity')}: {finding.get('section')}/{finding.get('code')}: {finding.get('message')} "
                      f"{finding.get('selector') or ''}{partner} t={finding.get('time')}")
            raise PreviewFailure(f"POST /render answered {status}: {answer.get('message') or answer}")
        print(f"    checked in {time.monotonic() - started:.1f} s (staged, spoken, mixed, checked); job {answer['id']}")
        job = answer
        while job.get("status") not in ("done", "failed", "rejected"):
            if time.monotonic() - started > JOB_WAIT_SECONDS:
                raise PreviewFailure(f"job {answer['id']} did not finish in {JOB_WAIT_SECONDS} s")
            time.sleep(POLL_SECONDS)
            status, body = self._request("GET", f"/render/{answer['id']}")
            job = json.loads(body)
        if job["status"] != "done":
            raise PreviewFailure(f"job {job['id']} {job['status']}: {json.dumps(job.get('error'))}")
        for output in job["outputs"]:
            status, data = self._request("GET", output["path"])
            if status != 200:
                raise PreviewFailure(f"GET {output['path']} answered {status}")
            output["data"] = data
        job["seconds"] = round(time.monotonic() - started, 1)
        return job
