from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from telethon import TelegramClient, events
from telethon.errors import FloodWaitError, RPCError
from telethon.sessions import StringSession
from telethon.tl.functions.account import UpdateNotifySettingsRequest
from telethon.tl.functions.contacts import BlockRequest, UnblockRequest
from telethon.tl.functions.folders import EditPeerFoldersRequest
from telethon.tl.functions.messages import (
    DeleteHistoryRequest,
    GetCommonChatsRequest,
    MarkDialogUnreadRequest,
    ReportSpamRequest,
)
from telethon.tl.functions.users import GetFullUserRequest
from telethon.tl.types import (
    DocumentAttributeAudio,
    DocumentAttributeFilename,
    DocumentAttributeSticker,
    DocumentAttributeVideo,
    InputDialogPeer,
    InputFolderPeer,
    InputNotifyPeer,
    InputPeerNotifySettings,
    MessageMediaContact,
    MessageMediaDocument,
    MessageMediaGeo,
    MessageMediaGeoLive,
    MessageMediaPhoto,
    MessageMediaPoll,
    User,
)

from antispam.config import Settings
from antispam.core.media import compress_image_to_jpeg, is_high_risk_filename
from antispam.models import IncomingMessage, MediaItem, UserProfile

if TYPE_CHECKING:
    from antispam.core.pipeline import ModerationPipeline

logger = logging.getLogger(__name__)

T = TypeVar("T")

TELEGRAM_SERVICE_IDS = {777000, 42777}
MUTE_FOREVER_UNTIL = 2147483647


class UserbotClient:
    """Telethon MTProto userbot adapter implementing TelegramActionsProtocol."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cfg = settings.cfg
        self._pipeline: ModerationPipeline | None = None
        self._me_id: int | None = None
        self._helper_bot_id: int | None = None

        # Anti-race tracking to distinguish bot probe replies from manual owner messages
        self._in_flight_replies: dict[int, list[tuple[str, float]]] = {}
        self._bot_sent_msg_ids: set[int] = set()

        session: str | StringSession
        if settings.env.tg_session_string:
            session = StringSession(settings.env.tg_session_string.strip())
        else:
            session = str(settings.user_session_path)

        self.client = TelegramClient(
            session,
            settings.env.tg_api_id,
            settings.env.tg_api_hash,
        )

    def bind_pipeline(self, pipeline: ModerationPipeline) -> None:
        self._pipeline = pipeline

    def set_helper_bot_id(self, bot_id: int) -> None:
        self._helper_bot_id = bot_id

    async def start(self) -> User:
        phone = self._settings.env.tg_phone
        if phone:
            await self.client.start(phone=phone)
        else:
            await self.client.start()
        me = await self.client.get_me()
        assert isinstance(me, User)
        self._me_id = me.id
        self._register_handlers()
        logger.info("Userbot signed in as %s (id=%s)", self._format_user_name(me), me.id)
        return me

    async def start_login_only(self, export_session: bool = False) -> None:
        phone = self._settings.env.tg_phone
        if phone:
            await self.client.start(phone=phone)
        else:
            await self.client.start()
        me = await self.client.get_me()
        assert isinstance(me, User)
        print(f"登录成功：{self._format_user_name(me)} (id={me.id})")
        if export_session:
            session_str = StringSession.save(self.client.session)
            print("\nTG_SESSION_STRING (请妥善保密，切勿泄露):")
            print(session_str)
        else:
            print(f"Session 已保存到：{self._settings.user_session_path}.session")
        await self.client.disconnect()

    async def stop(self) -> None:
        if self.client.is_connected():
            await self.client.disconnect()

    # ---------- Event Registration ----------

    def _register_handlers(self) -> None:
        self.client.add_event_handler(
            self._on_incoming_dm,
            events.NewMessage(incoming=True),
        )
        self.client.add_event_handler(
            self._on_outgoing_dm,
            events.NewMessage(outgoing=True),
        )

    @staticmethod
    def _format_user_name(user: User) -> str:
        full = f"{user.first_name or ''} {user.last_name or ''}".strip()
        if full:
            return full
        if user.username:
            return f"@{user.username}"
        return str(user.id)

    def _should_ignore_peer_id(self, peer_id: int) -> bool:
        if peer_id in TELEGRAM_SERVICE_IDS:
            return True
        if self._me_id is not None and peer_id == self._me_id:
            return True
        return self._helper_bot_id is not None and peer_id == self._helper_bot_id

    async def _on_incoming_dm(self, event: events.NewMessage.Event) -> None:
        if self._pipeline is None or not event.is_private:
            return
        sender_id = event.sender_id
        if not sender_id or self._should_ignore_peer_id(sender_id):
            return

        sender = await event.get_sender()
        if not isinstance(sender, User):
            return
        if sender.is_self or sender.deleted or sender.support:
            return
        if sender.contact:
            return
        if self._cfg.screening.ignore_bots and sender.bot:
            return

        forward_source = await self._extract_forward_source(event.message)
        media_item = await self._extract_media(event.message)
        incoming = IncomingMessage(
            user_id=sender.id,
            tg_msg_id=event.id,
            text=(event.raw_text or "").strip(),
            display_name=self._format_user_name(sender),
            username=sender.username,
            forward_source=forward_source,
            media=media_item,
            timestamp=event.date.timestamp() if event.date else time.time(),
        )
        await self._pipeline.handle_incoming_message(incoming)

    async def _on_outgoing_dm(self, event: events.NewMessage.Event) -> None:
        if self._pipeline is None or not event.is_private:
            return
        chat_id = event.chat_id
        if not chat_id or self._should_ignore_peer_id(chat_id):
            return

        # Ignore messages sent by the anti-spam bot itself (anti-race protection)
        if self._is_bot_outgoing_message(chat_id, event.id, event.raw_text or ""):
            return

        peer = await event.get_chat()
        if isinstance(peer, User):
            if peer.contact:
                return
            if peer.bot and self._cfg.screening.ignore_bots:
                return

        await self._pipeline.handle_owner_outgoing(chat_id)

    def _register_in_flight_reply(self, user_id: int, text: str) -> None:
        now = time.monotonic()
        entries = self._in_flight_replies.setdefault(user_id, [])
        entries[:] = [(t, ts) for t, ts in entries if now - ts < 30.0]
        entries.append((text.strip(), now))

    def _is_bot_outgoing_message(self, user_id: int, msg_id: int, text: str) -> bool:
        if msg_id in self._bot_sent_msg_ids:
            self._bot_sent_msg_ids.discard(msg_id)
            return True

        entries = self._in_flight_replies.get(user_id)
        if not entries:
            return False
        now = time.monotonic()
        cleaned = text.strip()
        for idx, (expected_text, ts) in enumerate(entries):
            if now - ts < 30.0 and expected_text == cleaned:
                entries.pop(idx)
                return True
        entries[:] = [(t, ts) for t, ts in entries if now - ts < 30.0]
        return False

    # ---------- Extraction Helpers ----------

    async def _extract_forward_source(self, message: Any) -> str | None:
        fwd = getattr(message, "fwd_from", None)
        if fwd is None:
            return None
        if getattr(fwd, "from_name", None):
            return str(fwd.from_name)
        from_id = getattr(fwd, "from_id", None)
        if from_id is None:
            return "匿名转发"
        try:
            entity = await self.client.get_entity(from_id)
            title = getattr(entity, "title", None)
            username = getattr(entity, "username", None)
            if title and username:
                return f"{title} (@{username})"
            if title:
                return str(title)
            if isinstance(entity, User):
                return self._format_user_name(entity)
        except Exception:
            return "未知频道/用户转发"
        return "未知来源转发"

    async def _extract_media(self, message: Any) -> MediaItem | None:
        media = getattr(message, "media", None)
        if media is None:
            return None

        media_cfg = self._cfg.media

        if isinstance(media, MessageMediaPhoto):
            image_bytes: bytes | None = None
            if media_cfg.enable_vision:
                raw = await self._safe_download(message)
                if raw:
                    image_bytes = compress_image_to_jpeg(
                        raw,
                        max_dimension=media_cfg.image_max_dimension,
                        quality=media_cfg.image_jpeg_quality,
                    )
            return MediaItem(
                media_type="photo",
                image_bytes=image_bytes,
                mime_type="image/jpeg" if image_bytes else None,
                description="图片",
            )

        if isinstance(media, MessageMediaDocument) and media.document:
            doc = media.document
            mime_type = getattr(doc, "mime_type", None) or ""
            file_size = getattr(doc, "size", None)
            file_name: str | None = None
            is_sticker = False
            sticker_alt = ""
            is_video = False
            is_voice = False
            is_audio = False

            for attr in getattr(doc, "attributes", []) or []:
                if isinstance(attr, DocumentAttributeFilename):
                    file_name = attr.file_name
                elif isinstance(attr, DocumentAttributeSticker):
                    is_sticker = True
                    sticker_alt = attr.alt or ""
                elif isinstance(attr, DocumentAttributeVideo):
                    is_video = True
                elif isinstance(attr, DocumentAttributeAudio):
                    if attr.voice:
                        is_voice = True
                    else:
                        is_audio = True

            if is_sticker:
                return MediaItem(
                    media_type="sticker",
                    mime_type=mime_type,
                    file_name=file_name,
                    file_size=file_size,
                    description=f"贴纸 {sticker_alt}".strip(),
                )

            if is_video or mime_type == "image/gif":
                thumb_bytes: bytes | None = None
                if media_cfg.enable_vision and getattr(doc, "thumbs", None):
                    raw_thumb = await self._safe_download(message, thumb=-1)
                    if raw_thumb:
                        thumb_bytes = compress_image_to_jpeg(
                            raw_thumb,
                            max_dimension=media_cfg.image_max_dimension,
                            quality=media_cfg.image_jpeg_quality,
                        )
                kind = "gif_thumb" if mime_type == "image/gif" else "video_thumb"
                label = "GIF 动图" if mime_type == "image/gif" else "视频(已提取缩略图)"
                return MediaItem(
                    media_type=kind,
                    image_bytes=thumb_bytes,
                    mime_type=mime_type,
                    file_name=file_name,
                    file_size=file_size,
                    description=label,
                )

            if is_voice:
                return MediaItem(
                    media_type="voice",
                    mime_type=mime_type,
                    file_size=file_size,
                    description="语音消息",
                )

            if is_audio:
                return MediaItem(
                    media_type="audio",
                    mime_type=mime_type,
                    file_name=file_name,
                    file_size=file_size,
                    description="音频文件",
                )

            # Image sent uncompressed as document
            if mime_type.startswith("image/") and (file_size or 0) <= 10 * 1024 * 1024:
                img_bytes: bytes | None = None
                if media_cfg.enable_vision:
                    raw = await self._safe_download(message)
                    if raw:
                        img_bytes = compress_image_to_jpeg(
                            raw,
                            max_dimension=media_cfg.image_max_dimension,
                            quality=media_cfg.image_jpeg_quality,
                        )
                return MediaItem(
                    media_type="photo",
                    image_bytes=img_bytes,
                    mime_type=mime_type,
                    file_name=file_name,
                    file_size=file_size,
                    description="原图文件",
                )

            high_risk = is_high_risk_filename(file_name, media_cfg.high_risk_extensions)
            return MediaItem(
                media_type="document",
                mime_type=mime_type,
                file_name=file_name,
                file_size=file_size,
                is_high_risk=high_risk,
                description="高危文件附件" if high_risk else "普通文件附件",
            )

        if isinstance(media, MessageMediaContact):
            return MediaItem(
                media_type="contact",
                description=f"联系人名片 ({media.first_name or ''} {media.phone_number or ''})".strip(),
            )
        if isinstance(media, (MessageMediaGeo, MessageMediaGeoLive)):
            return MediaItem(media_type="geo", description="地理位置")
        if isinstance(media, MessageMediaPoll):
            return MediaItem(media_type="other", description="投票")
        return MediaItem(media_type="other", description=type(media).__name__)

    async def _safe_download(self, message: Any, *, thumb: int | None = None) -> bytes | None:
        try:
            kwargs: dict[str, Any] = {"file": bytes}
            if thumb is not None:
                kwargs["thumb"] = thumb
            data = await self.client.download_media(message, **kwargs)
            return data if isinstance(data, bytes) else None
        except Exception as exc:
            logger.debug("Media download skipped/failed: %s", exc)
            return None

    # ---------- FloodWait-Safe RPC Wrapper ----------

    async def _safe_rpc(self, label: str, fn: Callable[[], Awaitable[T]]) -> T | None:
        try:
            return await fn()
        except FloodWaitError as exc:
            if exc.seconds <= 15:
                logger.warning("FloodWait (%ss) on %s, sleeping and retrying once", exc.seconds, label)
                await asyncio.sleep(exc.seconds + 1)
                try:
                    return await fn()
                except Exception as retry_exc:
                    logger.error("Retry failed on %s: %s", label, retry_exc)
                    return None
            logger.warning("FloodWait too long (%ss) on %s, skipping action", exc.seconds, label)
            return None
        except RPCError as exc:
            logger.warning("Telegram RPC error on %s: %s", label, exc)
            return None
        except Exception:
            logger.exception("Unexpected error on %s", label)
            return None

    # ---------- TelegramActionsProtocol Implementation ----------

    async def has_owner_history(self, user_id: int, limit: int) -> bool:
        async def _check() -> bool:
            messages = await self.client.get_messages(user_id, limit=limit)
            return any(bool(getattr(m, "out", False)) for m in messages)

        res = await self._safe_rpc(f"has_owner_history({user_id})", _check)
        return bool(res)

    async def fetch_user_profile(self, user_id: int) -> UserProfile | None:
        async def _fetch() -> UserProfile | None:
            input_user = await self.client.get_input_entity(user_id)
            full_res = await self.client(GetFullUserRequest(input_user))
            user_obj: User | None = None
            for u in getattr(full_res, "users", []) or []:
                if isinstance(u, User) and u.id == user_id:
                    user_obj = u
                    break
            if user_obj is None:
                entity = await self.client.get_entity(user_id)
                if isinstance(entity, User):
                    user_obj = entity
            if user_obj is None:
                return None

            bio = getattr(full_res.full_user, "about", None) or ""
            common_titles: list[str] = []
            try:
                common_res = await self.client(
                    GetCommonChatsRequest(user_id=input_user, max_id=0, limit=15)
                )
                for chat in getattr(common_res, "chats", []) or []:
                    title = getattr(chat, "title", None)
                    if title:
                        common_titles.append(str(title))
            except Exception as exc:
                logger.debug("GetCommonChats failed for %s: %s", user_id, exc)

            return UserProfile(
                user_id=user_obj.id,
                first_name=user_obj.first_name or "",
                last_name=user_obj.last_name or "",
                username=user_obj.username,
                bio=bio,
                is_premium=bool(getattr(user_obj, "premium", False)),
                is_verified=bool(getattr(user_obj, "verified", False)),
                is_scam=bool(getattr(user_obj, "scam", False)),
                is_fake=bool(getattr(user_obj, "fake", False)),
                has_photo=user_obj.photo is not None,
                common_chats=common_titles,
            )

        return await self._safe_rpc(f"fetch_user_profile({user_id})", _fetch)

    async def resolve_user_id(self, identifier: str) -> tuple[int, str, str | None] | None:
        cleaned = identifier.strip()
        if not cleaned:
            return None

        async def _resolve() -> tuple[int, str, str | None] | None:
            target: int | str = int(cleaned) if cleaned.lstrip("-").isdigit() else cleaned
            entity = await self.client.get_entity(target)
            if isinstance(entity, User):
                return (entity.id, self._format_user_name(entity), entity.username)
            return None

        return await self._safe_rpc(f"resolve_user_id({cleaned})", _resolve)

    async def archive_chat(self, user_id: int, archive: bool) -> None:
        async def _op() -> None:
            peer = await self.client.get_input_entity(user_id)
            await self.client(
                EditPeerFoldersRequest(
                    folder_peers=[InputFolderPeer(peer=peer, folder_id=1 if archive else 0)]
                )
            )

        await self._safe_rpc(f"archive_chat({user_id}, {archive})", _op)

    async def mute_chat(self, user_id: int, mute: bool) -> None:
        async def _op() -> None:
            peer = await self.client.get_input_entity(user_id)
            await self.client(
                UpdateNotifySettingsRequest(
                    peer=InputNotifyPeer(peer=peer),
                    settings=InputPeerNotifySettings(
                        mute_until=MUTE_FOREVER_UNTIL if mute else 0,
                        silent=mute,
                    ),
                )
            )

        await self._safe_rpc(f"mute_chat({user_id}, {mute})", _op)

    async def mark_unread(self, user_id: int) -> None:
        async def _op() -> None:
            peer = await self.client.get_input_entity(user_id)
            await self.client(
                MarkDialogUnreadRequest(peer=InputDialogPeer(peer=peer), unread=True)
            )

        await self._safe_rpc(f"mark_unread({user_id})", _op)

    async def send_probe_reply(self, user_id: int, text: str) -> int | None:
        async def _op() -> int | None:
            peer = await self.client.get_input_entity(user_id)
            delay_min = self._cfg.actions.reply_delay_min_seconds
            delay_max = max(delay_min, self._cfg.actions.reply_delay_max_seconds)
            delay = random.uniform(delay_min, delay_max)

            if self._cfg.actions.simulate_typing and delay > 0:
                async with self.client.action(peer, "typing"):
                    await asyncio.sleep(delay)
            elif delay > 0:
                await asyncio.sleep(delay)

            self._register_in_flight_reply(user_id, text)
            msg = await self.client.send_message(
                peer,
                text,
                parse_mode=None,
                link_preview=False,
            )
            if len(self._bot_sent_msg_ids) > 1000:
                self._bot_sent_msg_ids.clear()
            self._bot_sent_msg_ids.add(msg.id)
            return msg.id

        return await self._safe_rpc(f"send_probe_reply({user_id})", _op)

    async def block_and_clean(
        self,
        user_id: int,
        *,
        delete_history: bool,
        revoke: bool,
        report_spam: bool,
    ) -> None:
        async def _op() -> None:
            peer = await self.client.get_input_entity(user_id)
            if report_spam:
                await self.client(ReportSpamRequest(peer=peer))
            await self.client(BlockRequest(id=peer))
            if delete_history:
                await self.client(
                    DeleteHistoryRequest(
                        peer=peer,
                        max_id=0,
                        just_clear=False,
                        revoke=revoke,
                    )
                )

        await self._safe_rpc(f"block_and_clean({user_id})", _op)

    async def unblock_user(self, user_id: int) -> None:
        async def _op() -> None:
            peer = await self.client.get_input_entity(user_id)
            await self.client(UnblockRequest(id=peer))

        await self._safe_rpc(f"unblock_user({user_id})", _op)

    async def report_spam(self, user_id: int) -> None:
        async def _op() -> None:
            peer = await self.client.get_input_entity(user_id)
            await self.client(ReportSpamRequest(peer=peer))

        await self._safe_rpc(f"report_spam({user_id})", _op)
