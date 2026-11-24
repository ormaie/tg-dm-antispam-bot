from __future__ import annotations

from typing import Protocol

from antispam.models import IncomingMessage, LLMVerdict, RunMode, UserProfile


class TelegramActionsProtocol(Protocol):
    async def has_owner_history(self, user_id: int, limit: int) -> bool: ...

    async def fetch_user_profile(self, user_id: int) -> UserProfile | None: ...

    async def resolve_user_id(self, identifier: str) -> tuple[int, str, str | None] | None: ...

    async def archive_chat(self, user_id: int, archive: bool) -> None: ...

    async def mute_chat(self, user_id: int, mute: bool) -> None: ...

    async def mark_unread(self, user_id: int) -> None: ...

    async def send_probe_reply(self, user_id: int, text: str) -> int | None: ...

    async def block_and_clean(
        self,
        user_id: int,
        *,
        delete_history: bool,
        revoke: bool,
        report_spam: bool,
    ) -> None: ...

    async def unblock_user(self, user_id: int) -> None: ...

    async def report_spam(self, user_id: int) -> None: ...


class NotifierProtocol(Protocol):
    async def send_review_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
        images: list[bytes],
        mode: RunMode,
        extra_note: str = "",
    ) -> None: ...

    async def send_blocked_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
        images: list[bytes],
    ) -> None: ...

    async def send_passed_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
    ) -> None: ...

    async def send_dryrun_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
        images: list[bytes],
        would_action: str,
    ) -> None: ...

    async def send_pending_followup(
        self,
        *,
        user_id: int,
        display_name: str,
        username: str | None,
        message: IncomingMessage,
    ) -> None: ...
