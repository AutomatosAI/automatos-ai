"""``python -m media_render [serve | boot-check | fixture-bundle]``

The image's entrypoint. Every command boots through the assertions first
(boot.py): a container that would render wrong exits with code 2 instead.
``fixture-bundle`` prints the committed fixture as a POST /render body, which
the media-render CI job renders through the API.
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
    commands.add_parser("fixture-bundle", help="print the committed fixture as a POST /render body")
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
        from .fixture import fixture_bundle

        print(json.dumps(fixture_bundle()))
        return 0

    from .server import serve

    serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
