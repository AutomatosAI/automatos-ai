"""``python -m media_render [serve | boot-check | fixture-bundle [fixture | script | music]]``

The image's entrypoint. Every command boots through the assertions first
(boot.py): a container that would render wrong exits with code 2 instead.
``fixture-bundle`` prints a committed fixture (``fixture`` by default, the
``script`` of US-111, or the ``music`` of US-112) as a POST /render body,
which the media-render CI job renders through the API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
    fixture = commands.add_parser("fixture-bundle", help="print a committed fixture as a POST /render body")
    fixture.add_argument("name", nargs="?", default="fixture", choices=("fixture", "script", "music"))
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
    if command == "fixture-bundle":
        from .fixture import BUNDLES

        print(json.dumps(BUNDLES[args.name]()))
        return 0

    from .server import serve

    serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
