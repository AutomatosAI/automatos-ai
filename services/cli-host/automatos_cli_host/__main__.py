"""``python -m automatos_cli_host`` — run the Automatos CLI host."""
from __future__ import annotations

import sys

from .host import main

if __name__ == "__main__":
    sys.exit(main())
