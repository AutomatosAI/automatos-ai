"""What the image carries, recorded once at build and served by /health.

``python -m media_render.versions --browser <path> --write <file>`` runs in the
Dockerfile, so the build log shows every version and a missing tool fails the
build. The browser version is best effort: the pin that matters is Hyperframes'.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

NODE_PACKAGES = ("hyperframes", "gsap")
PYTHON_PACKAGES = ("aiohttp", "kokoro-onnx", "espeakng-loader", "phonemizer", "onnxruntime", "numpy", "soundfile")
TOOL_TIMEOUT_SECONDS = 30


def _run(argv: Sequence[str]) -> str:
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=True, timeout=TOOL_TIMEOUT_SECONDS
    ).stdout.strip()


def _browser_version(browser_path: str) -> str:
    try:
        output = _run([browser_path, "--version"])
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"\d+(?:\.\d+){3}", output)
    return match.group(0) if match else "unknown"


def collect_versions(browser_path: str) -> Dict[str, str]:
    npm_root = Path(_run(["npm", "root", "-g"]))
    versions = {
        name: json.loads((npm_root / name / "package.json").read_text())["version"] for name in NODE_PACKAGES
    }
    versions["node"] = _run(["node", "--version"])
    versions["ffmpeg"] = _run(["ffmpeg", "-version"]).splitlines()[0].split()[2]
    versions["chrome-headless-shell"] = _browser_version(browser_path)
    for name in PYTHON_PACKAGES:
        versions[name] = importlib.metadata.version(name)
    return versions


def read_versions(path: str) -> Dict[str, str]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m media_render.versions")
    parser.add_argument("--browser", required=True, help="the chrome-headless-shell binary")
    parser.add_argument("--write", required=True, type=Path, help="where to record the versions")
    args = parser.parse_args(argv)
    versions = collect_versions(args.browser)
    text = json.dumps(versions, indent=2, sort_keys=True) + "\n"
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
