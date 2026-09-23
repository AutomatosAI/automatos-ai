"""F112 (night 3 refresh) — the Telegram channel can start: its library is installed.

#751 made Channels a local-edition setting: a local install answers agent
questions over Telegram by polling, with no public URL. Both Telegram paths —
the polling driver (channels/drivers/telegram.py) and the channel manager's
adapter (channels/telegram_adapter.py) — import telegram.ext, and
python-telegram-bot was never in requirements.txt, so every boot logged
"python-telegram-bot not installed" and polling never started.
"""
from __future__ import annotations

import re
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def test_requirements_install_python_telegram_bot():
    lines = [line.split("#", 1)[0].strip() for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()]
    assert any(re.match(r"python-telegram-bot\b", line) for line in lines)


def test_the_polling_driver_and_the_adapter_can_import_what_they_need():
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters  # noqa: F401
