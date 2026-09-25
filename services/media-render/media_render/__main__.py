"""``python -m media_render [serve | boot-check | fixture --out DIR]``

The image's entrypoint. Every command boots through the assertions first
(boot.py): a container that would render wrong exits with code 2 instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from automatos_logging import setup_logging

from .boot import BOOT_FAILURE_EXIT_CODE, BootError, assert_boot_environment
from .config import TOKEN_ENV, ConfigError, load_settings

logger = logging.getLogger("media_render")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m media_render")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the HTTP service (the default)")
    commands.add_parser("boot-check", help="run the boot assertions and exit")
    fixture = commands.add_parser("fixture", help="render the committed fixture composition")
    fixture.add_argument("--out", required=True, type=Path, help="directory for the MP4 and its report")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging(service="media-render")
    try:
        settings = load_settings()
        assert_boot_environment(settings, os.environ)
    except (ConfigError, BootError) as exc:
        logger.error("%s", exc)
        print(f"media-render: {exc}", file=sys.stderr)
        return BOOT_FAILURE_EXIT_CODE
    if not settings.internal_token:
        logger.warning("%s is not set: every route is open (development only)", TOKEN_ENV)

    command = args.command or "serve"
    if command == "boot-check":
        print("media-render: boot checks passed")
        return 0
    if command == "fixture":
        from .fixture import FixtureError, render_fixture

        try:
            report = render_fixture(args.out, settings)
        except FixtureError as exc:
            print(f"media-render: fixture render failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0

    from .server import serve

    serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
