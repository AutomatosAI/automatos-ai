"""``python -m automatos_cli_host`` — run the Automatos CLI host."""
from __future__ import annotations

import sys
from typing import Optional

# ConPTY, the Windows pseudo console the host drives sessions through (#818),
# arrived in Windows 10 1809.
CONPTY_FIRST_BUILD = 17763
UNSUPPORTED_PLATFORM_MESSAGE = (
    "Session mode on Windows needs Windows 10 version 1809 or later (the host drives "
    "sessions through ConPTY, which older Windows lacks). On this machine, run the host "
    "inside WSL2: see docs/getting-started/self-hosting.md, 'Session mode' → 'Before you start'.\n"
)


def _windows_build() -> int:
    return sys.getwindowsversion().build if hasattr(sys, "getwindowsversion") else 0


def platform_supported(platform: str = sys.platform, windows_build: Optional[int] = None) -> bool:
    """True where the host can run: macOS, Linux, and Windows with ConPTY."""
    if platform != "win32":
        return True
    build = _windows_build() if windows_build is None else windows_build
    return build >= CONPTY_FIRST_BUILD


if __name__ == "__main__":
    # Checked before importing the host, so an old Windows gets the sentence, not a ctypes error.
    if not platform_supported():
        sys.stderr.write(UNSUPPORTED_PLATFORM_MESSAGE)
        sys.exit(2)

    from .host import main

    sys.exit(main())
