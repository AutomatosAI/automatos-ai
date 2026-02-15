"""
Channel Adapters (PRD-55 US-019)
=================================
Base infrastructure for platform channel adapters (Telegram, Slack, Discord, etc.).
"""

from .base import BaseChannelAdapter
from .manager import ChannelManager, get_channel_manager

__all__ = [
    "BaseChannelAdapter",
    "ChannelManager",
    "get_channel_manager",
]
