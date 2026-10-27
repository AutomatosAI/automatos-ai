"""``python -m voice_tts [serve | boot-check]``

The image's entrypoint. Every command boots through the assertions first
(boot.py): a container that could not speak exits with code 2 instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from automatos_logging import setup_logging

from .boot import BOOT_FAILURE_EXIT_CODE, BootError, assert_boot_environment
from .config import ALLOW_NO_TOKEN_ENV, TOKEN_ENV, ConfigError, load_settings

logger = logging.getLogger("voice_tts")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voice_tts")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the HTTP service (the default)")
    commands.add_parser("boot-check", help="run the boot assertions and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging(service="voice-tts")
    try:
        settings = load_settings()
        assert_boot_environment(settings)
    except (ConfigError, BootError) as exc:
        logger.error("%s", exc)
        print(f"voice-tts: {exc}", file=sys.stderr)
        return BOOT_FAILURE_EXIT_CODE
    if not settings.token:
        logger.warning("%s is not set and %s=true: /speak is open (development only)", TOKEN_ENV,
                       ALLOW_NO_TOKEN_ENV)
    if (args.command or "serve") == "boot-check":
        print("voice-tts: boot checks passed")
        return 0

    from .server import serve

    serve(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
