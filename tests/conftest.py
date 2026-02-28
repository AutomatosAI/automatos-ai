"""Root conftest — load .env before any test imports."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load tests/.env so every test module sees the variables
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)
