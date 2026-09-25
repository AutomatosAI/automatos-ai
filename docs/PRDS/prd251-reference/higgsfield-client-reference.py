"""Minimal Higgsfield API client: estimate / run jobs with a spend cap.

Usage:
  python hf.py estimate jobs.json
  python hf.py run jobs.json --cap 10 [--only name1,name2]

The key is read from ~/.config/higgsfield/demo.conf (HF_KEY=id:secret) and never printed.
"""
import json
import os
import sys
import time
from pathlib import Path

import httpx

API = "https://api.higgsfield.ai"
KEY_FILE = Path.home() / ".config/higgsfield/demo.conf"
TERMINAL = {"completed", "failed", "nsfw", "canceled"}


def load_key() -> str:
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("HF_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_KEY not found in key file")


def headers() -> dict:
    return {"Authorization": f"Key {load_key()}", "Content-Type": "application/json"}


def estimate(client: httpx.Client, job: dict) -> float:
    r = client.post(f"{API}/estimate/{job['endpoint']}", json=job["args"], headers=headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"estimate {job['name']}: HTTP {r.status_code} {r.text[:300]}")
    body = r.json()
    if "usd" in body:
        return float(body["usd"])
    if "cinema-studio" not in job["endpoint"]:
        # token-metered image models (e.g. marketing-studio flare/sunburst): no up-front number;
        # budget a conservative ceiling so the spend cap still holds.
        return float(job.get("budget_usd", 0.30))
    # Token-metered models (Cinema Studio) return a pricing description instead of a number:
    # tokens = ceil(seconds x width x height x 24 / 1024); $0.0214 per 1,000 tokens at 480p/720p.
    dims = {("720p", "9:16"): (720, 1280), ("720p", "16:9"): (1280, 720), ("720p", "1:1"): (720, 720),
            ("480p", "9:16"): (480, 854), ("480p", "16:9"): (854, 480), ("480p", "1:1"): (480, 480)}
    a = job["args"]
    w, h = dims[(a.get("resolution", "720p"), a.get("aspect_ratio", "16:9"))]
    tokens = -(-(a.get("duration", 5) * w * h * 24) // 1024)
    return tokens / 1000 * 0.0214


def submit(client: httpx.Client, job: dict) -> dict:
    r = client.post(f"{API}/{job['endpoint']}", json=job["args"], headers=headers(), timeout=60)
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"submit {job['name']}: HTTP {r.status_code} {r.text[:300]}")
    return r.json()


def poll(client: httpx.Client, status_url: str, timeout_s: int = 900) -> dict:
    delay, start = 2.0, time.time()
    while True:
        r = client.get(status_url, headers=headers(), timeout=30)
        if r.status_code >= 500:
            time.sleep(delay)
            continue
        r.raise_for_status()
        res = r.json()
        if res.get("status") in TERMINAL:
            return res
        if time.time() - start > timeout_s:
            raise TimeoutError(f"poll timeout {status_url}")
        time.sleep(delay)
        delay = min(delay * 1.4, 10.0)


def download(client: httpx.Client, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def main() -> None:
    mode, jobs_path = sys.argv[1], Path(sys.argv[2])
    cap = float(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 10.0
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    jobs = [j for j in json.loads(jobs_path.read_text()) if not only or j["name"] in only]
    log_path = jobs_path.with_suffix(".results.json")
    results = json.loads(log_path.read_text()) if log_path.exists() else {}
    with httpx.Client() as client:
        ests = {j["name"]: estimate(client, j) for j in jobs}
        for n, usd in ests.items():
            print(f"estimate  {n:<14} ${usd:.3f}")
        total = sum(ests.values())
        print(f"estimate  TOTAL          ${total:.3f}  (cap ${cap:.2f})")
        if mode == "estimate":
            return
        if total > cap:
            raise SystemExit(f"refusing: estimated ${total:.2f} exceeds cap ${cap:.2f}")
        pending = {}
        for j in jobs:
            sub = submit(client, j)
            pending[j["name"]] = (j, sub)
            print(f"submitted {j['name']:<14} {sub.get('request_id')}")
        for name, (j, sub) in pending.items():
            res = poll(client, sub["status_url"])
            status = res.get("status")
            outs = []
            if status == "completed":
                media = []
                if res.get("video"):
                    media.append(res["video"]["url"])
                media += [im["url"] for im in res.get("images") or []]
                for i, url in enumerate(media):
                    ext = ".mp4" if res.get("video") and i == 0 else Path(url.split("?")[0]).suffix or ".png"
                    dest = Path(j["out"]).with_suffix("") if len(media) > 1 else Path(j["out"])
                    dest = Path(f"{dest}-{i}{ext}") if len(media) > 1 else dest
                    download(client, url, dest)
                    outs.append(str(dest))
            results[name] = {"status": status, "error": res.get("error"), "request_id": sub.get("request_id"),
                             "estimate_usd": ests[name], "outputs": outs}
            print(f"done      {name:<14} {status} {outs}")
            log_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
