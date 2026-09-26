#!/usr/bin/env python3
"""Render one bundle through the media-render API (the media-render CI job).

Stdlib only: it runs on the runner, not in the image. POST /render with the
internal token, poll GET /render/{id} until the job ends, then fetch the output
named ``--output``. It writes ``job.json`` (the job as GET /render/{id} last
returned it) and the output into ``--out``, and prints the timings, so the job
log is the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TERMINAL = ("done", "failed", "rejected")


def _call(url: str, token: str, *, data: Optional[bytes] = None, timeout: float) -> Tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    request.add_header("X-Internal-Token", token)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="the service, e.g. http://127.0.0.1:8092")
    parser.add_argument("--token", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output", default="render.mp4", help="the output file to fetch")
    parser.add_argument("--save-as", default=None, help="the fetched file's name in --out (default: --output)")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds for the whole render")
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    code, body = _call(f"{args.url}/render", args.token, data=args.bundle.read_bytes(), timeout=args.timeout)
    print(f"POST /render answered {code} after {time.monotonic() - started:.1f} s (staged, spoken, fitted, mixed, checked)")
    print(body.decode("utf-8", "replace")[:4000])
    if code != 202:
        return 1
    job_id = json.loads(body)["id"]

    job: Dict[str, Any] = {}
    deadline = started + args.timeout
    while time.monotonic() < deadline:
        code, body = _call(f"{args.url}/render/{job_id}", args.token, timeout=args.timeout)
        job = json.loads(body) if code == 200 else {"status": f"http {code}", "body": body.decode("utf-8", "replace")}
        if job.get("status") in TERMINAL or code != 200:
            break
        time.sleep(args.poll)
    (args.out / "job.json").write_text(json.dumps(job, indent=1))
    print(f"render {job_id}: {job.get('status')}, {time.monotonic() - started:.1f} s from POST to finished")
    if job.get("status") != "done":
        print(json.dumps(job, indent=1)[:8000])
        return 1

    code, data = _call(f"{args.url}/render/{job_id}/output/{args.output}", args.token, timeout=args.timeout)
    if code != 200 or not data:
        print(f"GET /render/{job_id}/output/{args.output} answered {code} with {len(data)} bytes")
        return 1
    target = args.out / (args.save_as or args.output)
    target.write_bytes(data)
    print(f"fetched {target.name}: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
