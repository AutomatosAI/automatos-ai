"""``python -m automatos_cli_host`` — run the Automatos CLI host."""
from __future__ import annotations

import sys

UNSUPPORTED_PLATFORM_MESSAGE = (
    "Session mode needs macOS, Linux or WSL2; native Windows is not supported "
    "(the host drives sessions through a Unix pty and installs as a launchd or "
    "systemd service). On Windows, run the host inside WSL2 — see "
    "docs/getting-started/self-hosting.md, 'Session mode' → 'Before you start'.\n"
)


def platform_supported(platform: str = sys.platform) -> bool:
    """True where the host can run: it needs ``pty`` and launchd or systemd."""
    return platform != "win32"


if __name__ == "__main__":
    # Checked before importing the host: on Windows that import itself fails on ``pty``.
    if not platform_supported():
        sys.stderr.write(UNSUPPORTED_PLATFORM_MESSAGE)
        sys.exit(2)

    from .host import main

    sys.exit(main())
