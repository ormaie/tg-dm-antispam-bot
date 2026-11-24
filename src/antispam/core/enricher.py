from __future__ import annotations

import logging
import time

from antispam.core.interfaces import TelegramActionsProtocol
from antispam.models import UserProfile

logger = logging.getLogger(__name__)


class ProfileEnricher:
    """Fetches and caches Telegram user profiles & common chats with a TTL."""

    def __init__(self, tg_actions: TelegramActionsProtocol, ttl_seconds: int = 1800) -> None:
        self._tg = tg_actions
        self._ttl = ttl_seconds
        self._cache: dict[int, UserProfile] = {}

    async def get_profile(self, user_id: int, *, force_refresh: bool = False) -> UserProfile | None:
        now = time.monotonic()
        cached = self._cache.get(user_id)
        if not force_refresh and cached is not None and (now - cached.fetched_at) < self._ttl:
            return cached
        try:
            profile = await self._tg.fetch_user_profile(user_id)
        except Exception as exc:
            logger.warning("Failed to fetch profile for user %s: %s", user_id, exc)
            return cached
        if profile is not None:
            profile.fetched_at = now
            self._cache[user_id] = profile
        return profile

    def invalidate(self, user_id: int) -> None:
        self._cache.pop(user_id, None)
