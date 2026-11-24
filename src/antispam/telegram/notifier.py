from __future__ import annotations

import html
import io
import logging
from typing import TYPE_CHECKING, Any

from telethon import Button, TelegramClient, events
from telethon.extensions import html as tg_html
from telethon.tl.types import User

from antispam.config import Settings
from antispam.core.rules import RuleEngine
from antispam.models import IncomingMessage, LLMVerdict, RunMode, UserProfile, UserStatus

if TYPE_CHECKING:
    from antispam.core.pipeline import ModerationPipeline

logger = logging.getLogger(__name__)


def _esc(text: str | None) -> str:
    return html.escape(text or "", quote=False)


def _truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class HelperBotNotifier:
    """Independent Telegram Bot client for interactive review cards and owner commands."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._owner_id = settings.env.owner_id
        self._pipeline: ModerationPipeline | None = None
        self.bot_id: int | None = None

        self.client = TelegramClient(
            str(settings.bot_session_path),
            settings.env.tg_api_id,
            settings.env.tg_api_hash,
            flood_sleep_threshold=5,
        )

    def bind_pipeline(self, pipeline: ModerationPipeline) -> None:
        self._pipeline = pipeline

    async def start(self) -> User:
        await self.client.start(bot_token=self._settings.env.bot_token)
        me = await self.client.get_me()
        assert isinstance(me, User)
        self.bot_id = me.id
        self._register_handlers()
        logger.info("Helper Bot started as @%s (id=%s)", me.username, me.id)
        return me

    async def stop(self) -> None:
        if self.client.is_connected():
            await self.client.disconnect()

    async def send_startup_banner(self, mode: RunMode, userbot_name: str) -> None:
        text = (
            "🛡️ <b>TG 私聊防广告助手已启动</b>\n"
            f"• 保护账号：{_esc(userbot_name)}\n"
            f"• 当前模式：<code>{mode.value}</code>\n"
            "• 发送 /help 查看全部管理指令，或 /mode 切换运行模式。"
        )
        try:
            await self.client.send_message(
                self._owner_id, text, parse_mode="html", link_preview=False, silent=True
            )
        except Exception as exc:
            logger.warning(
                "Could not send startup banner to OWNER_ID=%s (make sure you pressed /start on the helper bot first): %s",
                self._owner_id,
                exc,
            )

    # ---------- Handlers ----------

    def _register_handlers(self) -> None:
        self.client.add_event_handler(
            self._on_callback_query,
            events.CallbackQuery(),
        )
        self.client.add_event_handler(
            self._on_owner_command,
            events.NewMessage(incoming=True),
        )

    async def _on_callback_query(self, event: events.CallbackQuery.Event) -> None:
        if event.sender_id != self._owner_id or self._pipeline is None:
            await event.answer("无权限", alert=True)
            return

        data = (event.data or b"").decode("utf-8", errors="ignore")
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != "act":
            await event.answer("无效操作")
            return

        action, raw_uid = parts[1], parts[2]
        if not raw_uid.isdigit():
            await event.answer("无效用户 ID")
            return
        user_id = int(raw_uid)

        try:
            if action in ("allow", "unblock"):
                result = await self._pipeline.manual_allow(user_id)
                new_buttons = [[Button.inline("🚫 改为拉黑", f"act:block:{user_id}".encode())]]
            elif action == "block":
                result = await self._pipeline.manual_block(user_id, report_spam=False)
                new_buttons = [
                    [
                        Button.inline("↩️ 误杀恢复", f"act:unblock:{user_id}".encode()),
                        Button.inline("🚨 补点举报", f"act:report_only:{user_id}".encode()),
                    ]
                ]
            elif action == "report":
                result = await self._pipeline.manual_block(user_id, report_spam=True)
                new_buttons = [[Button.inline("↩️ 误杀恢复", f"act:unblock:{user_id}".encode())]]
            elif action == "report_only":
                result = await self._pipeline.manual_report_only(user_id)
                new_buttons = [[Button.inline("↩️ 误杀恢复", f"act:unblock:{user_id}".encode())]]
            elif action == "probe":
                result = await self._pipeline.manual_probe_again(user_id)
                new_buttons = [
                    [
                        Button.inline("✅ 放行", f"act:allow:{user_id}".encode()),
                        Button.inline("🚫 拉黑+删", f"act:block:{user_id}".encode()),
                    ]
                ]
            else:
                await event.answer("未知操作")
                return
        except Exception as exc:
            logger.exception("Callback action %s failed for %s", action, user_id)
            await event.answer(f"执行失败：{exc}", alert=True)
            return

        await event.answer(result)
        try:
            msg = await event.get_message()
            original = tg_html.unparse(msg.raw_text or "", msg.entities or [])
            marker = "\n\n━━━━━━━━"
            base = original.split(marker)[0]
            updated = f"{base}{marker}\n<b>最新处理：</b>{_esc(result)}"
            await event.edit(updated, parse_mode="html", buttons=new_buttons, link_preview=False)
        except Exception as exc:
            logger.debug("Could not edit card message after callback: %s", exc)

    async def _on_owner_command(self, event: events.NewMessage.Event) -> None:
        if not event.is_private or event.sender_id != self._owner_id or self._pipeline is None:
            return
        text = (event.raw_text or "").strip()
        if not text.startswith("/"):
            return

        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        args = parts[1:]

        if cmd in ("/start", "/help"):
            await self._cmd_help(event)
        elif cmd == "/status":
            await self._cmd_status(event)
        elif cmd == "/mode":
            await self._cmd_mode(event, args)
        elif cmd == "/allow":
            await self._cmd_allow(event, args)
        elif cmd == "/block":
            await self._cmd_block(event, args)
        elif cmd == "/pending":
            await self._cmd_pending(event)
        elif cmd == "/rules":
            await self._cmd_rules(event, args)
        else:
            await event.reply("未知指令，发送 /help 查看可用命令。")

    async def _cmd_help(self, event: events.NewMessage.Event) -> None:
        assert self._pipeline is not None
        mode = await self._pipeline.get_run_mode()
        reply = (
            "🛡️ <b>TG 私聊防广告管理指令</b>\n\n"
            f"当前模式：<code>{mode.value}</code>\n\n"
            "• /status — 查看运行状态、今日拦截统计与 LLM 预算\n"
            "• /mode <code>dryrun|review_only|auto</code> — 实时切换运行模式\n"
            "• /allow <code>&lt;user_id|@username&gt;</code> — 手动放行/解封并加白\n"
            "• /block <code>&lt;user_id|@username&gt;</code> — 手动拉黑并双向删记录\n"
            "• /pending — 列出待人工确认的陌生人\n"
            "• /rules list — 查看本地前置拦截规则\n"
            "• /rules add <code>keyword|regex</code> <code>spam|review</code> <code>规则内容</code> — 新增规则\n"
            "• /rules del <code>&lt;id&gt;</code> — 删除规则"
        )
        await event.reply(reply, parse_mode="html")

    async def _cmd_status(self, event: events.NewMessage.Event) -> None:
        assert self._pipeline is not None
        mode = await self._pipeline.get_run_mode()
        stats = await self._pipeline.db.get_today_stats()
        budget = self._settings.cfg.rate_limit.daily_llm_budget_calls
        reply = (
            "📊 <b>今日运行状态（UTC）</b>\n\n"
            f"• 运行模式：<code>{mode.value}</code>\n"
            f"• 当前待人工审核：<b>{stats['pending_now']}</b> 人\n"
            f"• 今日自动/手动拦截：<b>{stats['blocked']}</b> 次\n"
            f"• 今日放行通过：<b>{stats['passed']}</b> 次\n"
            f"• 今日自动追问：<b>{stats['probed']}</b> 次\n"
            f"• 今日 LLM 调用：<b>{stats['llm_calls']}</b> / {budget}\n"
            f"• 今日 Token 消耗：in={stats['prompt_tokens']} / out={stats['completion_tokens']}\n"
            f"• 当前分钟剩余配额：回复 {self._pipeline.reply_limiter.remaining()} / 拉黑 {self._pipeline.block_limiter.remaining()}"
        )
        await event.reply(reply, parse_mode="html")

    async def _cmd_mode(self, event: events.NewMessage.Event, args: list[str]) -> None:
        assert self._pipeline is not None
        if not args:
            mode = await self._pipeline.get_run_mode()
            await event.reply(
                f"当前运行模式：<code>{mode.value}</code>\n可选：<code>dryrun</code> / <code>review_only</code> / <code>auto</code>",
                parse_mode="html",
            )
            return
        raw = args[0].strip().lower()
        try:
            new_mode = RunMode(raw)
        except ValueError:
            await event.reply("无效模式，请使用：dryrun / review_only / auto")
            return
        await self._pipeline.set_run_mode(new_mode)
        desc = {
            RunMode.DRYRUN: "演习模式（仅推送建议，不回复、不归档、不拉黑）",
            RunMode.REVIEW_ONLY: "半自动模式（自动静音归档与追问，放行/拉黑全由你按钮确认）",
            RunMode.AUTO: "全自动分级模式（高置信度广告自动拉黑删记录，中置信度人工审）",
        }[new_mode]
        await event.reply(f"✅ 已切换模式为 <code>{new_mode.value}</code>：{desc}", parse_mode="html")

    async def _resolve_target(self, raw_target: str) -> tuple[int, str, str | None] | None:
        assert self._pipeline is not None
        cleaned = raw_target.strip()
        if cleaned.isdigit():
            uid = int(cleaned)
            stored = await self._pipeline.db.get_user(uid)
            if stored:
                return (stored.user_id, stored.display_name, stored.username)
        else:
            stored = await self._pipeline.db.find_user_by_username(cleaned)
            if stored:
                return (stored.user_id, stored.display_name, stored.username)
        return await self._pipeline.tg.resolve_user_id(cleaned)

    async def _cmd_allow(self, event: events.NewMessage.Event, args: list[str]) -> None:
        assert self._pipeline is not None
        if not args:
            await event.reply("用法：/allow <user_id|@username>")
            return
        resolved = await self._resolve_target(args[0])
        if not resolved:
            await event.reply(f"无法解析用户：{_esc(args[0])}", parse_mode="html")
            return
        uid, name, username = resolved
        res = await self._pipeline.manual_allow(uid, display_name=name, username=username)
        await event.reply(f"用户 {_esc(name)} (<code>{uid}</code>)：{_esc(res)}", parse_mode="html")

    async def _cmd_block(self, event: events.NewMessage.Event, args: list[str]) -> None:
        assert self._pipeline is not None
        if not args:
            await event.reply("用法：/block <user_id|@username>")
            return
        resolved = await self._resolve_target(args[0])
        if not resolved:
            await event.reply(f"无法解析用户：{_esc(args[0])}", parse_mode="html")
            return
        uid, name, username = resolved
        res = await self._pipeline.manual_block(
            uid, report_spam=False, display_name=name, username=username
        )
        await event.reply(f"用户 {_esc(name)} (<code>{uid}</code>)：{_esc(res)}", parse_mode="html")

    async def _cmd_pending(self, event: events.NewMessage.Event) -> None:
        assert self._pipeline is not None
        users = await self._pipeline.db.list_users_by_status(UserStatus.PENDING_REVIEW, limit=15)
        if not users:
            await event.reply("🎉 当前没有待人工确认的陌生人。")
            return
        lines = ["📋 <b>待人工审核列表</b>\n"]
        buttons: list[list[Any]] = []
        for u in users:
            uname = f" (@{_esc(u.username)})" if u.username else ""
            reason = _esc(_truncate(u.last_reason or "无说明", 60))
            lines.append(f"• <b>{_esc(u.display_name)}</b>{uname} [<code>{u.user_id}</code>]\n  └ {reason}")
            buttons.append(
                [
                    Button.inline(f"✅ 放行 {u.display_name[:10]}", f"act:allow:{u.user_id}".encode()),
                    Button.inline(f"🚫 拉黑 {u.display_name[:10]}", f"act:block:{u.user_id}".encode()),
                ]
            )
        await event.reply("\n".join(lines), parse_mode="html", buttons=buttons[:8])

    async def _cmd_rules(self, event: events.NewMessage.Event, args: list[str]) -> None:
        assert self._pipeline is not None
        sub = args[0].lower() if args else "list"
        if sub == "list":
            rules = await self._pipeline.db.list_rules(enabled_only=False)
            if not rules:
                await event.reply("当前没有本地前置规则。")
                return
            lines = ["📜 <b>本地前置规则列表</b>\n"]
            for r in rules:
                status = "🟢" if r.enabled else "⚪"
                note = f" — {_esc(r.note)}" if r.note else ""
                lines.append(
                    f"{status} <code>#{r.id}</code> [{r.rule_type} -> {r.action}] <code>{_esc(r.pattern)}</code>{note}"
                )
            await event.reply("\n".join(lines), parse_mode="html")
            return

        if sub == "add":
            if len(args) < 4:
                await event.reply(
                    "用法：/rules add <keyword|regex> <spam|review> <规则内容>\n"
                    "示例：/rules add keyword spam 稳定日赚"
                )
                return
            rule_type = args[1].lower()
            action = args[2].lower()
            pattern = " ".join(args[3:]).strip()
            if action not in ("spam", "review"):
                await event.reply("动作仅支持 spam 或 review")
                return
            err = RuleEngine.validate_pattern(rule_type, pattern)
            if err:
                await event.reply(f"❌ {err}")
                return
            created = await self._pipeline.db.add_rule(rule_type, pattern, action, note="Bot 指令添加")
            await event.reply(
                f"✅ 已新增规则 <code>#{created.id}</code> [{created.rule_type} -> {created.action}] <code>{_esc(created.pattern)}</code>",
                parse_mode="html",
            )
            return

        if sub == "del":
            if len(args) < 2 or not args[1].isdigit():
                await event.reply("用法：/rules del <rule_id>")
                return
            deleted = await self._pipeline.db.delete_rule(int(args[1]))
            await event.reply("✅ 规则已删除。" if deleted else "❌ 未找到对应 ID 的规则。")
            return

        await event.reply("用法：/rules list | /rules add <keyword|regex> <spam|review> <内容> | /rules del <id>")

    # ---------- Card Formatting & Sending (NotifierProtocol) ----------

    def _format_user_block(
        self,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
    ) -> str:
        if profile:
            name = profile.display_name
            username = f"@{profile.username}" if profile.username else "无"
            bio = _truncate(profile.bio, 160) if profile.bio else "空"
            chats = ", ".join(profile.common_chats[:5]) if profile.common_chats else "无"
            flags: list[str] = []
            if profile.is_premium:
                flags.append("Premium")
            if profile.is_scam:
                flags.append("⚠️SCAM")
            if profile.is_fake:
                flags.append("⚠️FAKE")
            if not profile.has_photo:
                flags.append("无头像")
            flag_str = f" ({', '.join(flags)})" if flags else ""
        else:
            first = batch[0] if batch else None
            name = first.display_name if first else str(user_id)
            username = f"@{first.username}" if first and first.username else "无"
            bio = "未知"
            chats = "未知"
            flag_str = ""

        msg_lines: list[str] = []
        for msg in batch[:6]:
            segs: list[str] = []
            if msg.forward_source:
                segs.append(f"[转发: {_esc(msg.forward_source)}]")
            if msg.text:
                segs.append(_esc(_truncate(msg.text, 220)))
            if msg.media:
                m_txt = msg.media.description or msg.media.media_type
                if msg.media.file_name:
                    m_txt += f": {msg.media.file_name}"
                if msg.media.is_high_risk:
                    m_txt = f"⚠️高危 {m_txt}"
                segs.append(f"[{_esc(m_txt)}]")
            msg_lines.append("• " + (" ".join(segs) or "(空)"))
        if len(batch) > 6:
            msg_lines.append(f"• ... 共 {len(batch)} 条消息")

        return (
            f"👤 <b>用户</b>：{_esc(name)}{_esc(flag_str)}\n"
            f"🆔 <b>ID / 用户名</b>：<code>{user_id}</code> / {_esc(username)}\n"
            f"📝 <b>Bio</b>：{_esc(bio)}\n"
            f"👥 <b>共同群</b>：{_esc(chats)}\n"
            f"💬 <b>消息快照</b>：\n" + "\n".join(msg_lines)
        )

    async def _send_card(
        self,
        *,
        text: str,
        buttons: list[list[Any]] | None,
        images: list[bytes] | None = None,
        silent: bool = False,
    ) -> None:
        imgs = images or []
        try:
            if len(imgs) == 1 and len(text) <= 950:
                buf = io.BytesIO(imgs[0])
                buf.name = "snapshot.jpg"
                await self.client.send_file(
                    self._owner_id,
                    buf,
                    caption=text,
                    parse_mode="html",
                    buttons=buttons,
                    silent=silent,
                )
                return

            if imgs:
                files = []
                for idx, raw in enumerate(imgs[:3], start=1):
                    buf = io.BytesIO(raw)
                    buf.name = f"snapshot_{idx}.jpg"
                    files.append(buf)
                await self.client.send_file(
                    self._owner_id,
                    files,
                    caption=f"🖼️ 上述审核对象附带的 {len(files)} 张图片/缩略图快照",
                    silent=True,
                )

            await self.client.send_message(
                self._owner_id,
                text,
                parse_mode="html",
                buttons=buttons,
                link_preview=False,
                silent=silent,
            )
        except Exception:
            logger.exception("Failed to send card via Helper Bot")

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
    ) -> None:
        note_line = f"\n{_esc(extra_note)}" if extra_note else ""
        text = (
            f"🕵️ <b>待人工确认</b>（模式：<code>{mode.value}</code>）\n\n"
            f"{self._format_user_block(user_id, profile, batch)}\n\n"
            f"🤖 <b>AI 建议</b>：<code>{verdict.decision.value}</code> "
            f"({verdict.confidence:.0%}) · <code>{_esc(verdict.category)}</code>\n"
            f"📌 <b>理由</b>：{_esc(verdict.reason)}{note_line}"
        )
        buttons = [
            [
                Button.inline("✅ 放行并移出归档", f"act:allow:{user_id}".encode()),
                Button.inline("🚫 拉黑+删记录", f"act:block:{user_id}".encode()),
            ],
            [
                Button.inline("🚨 拉黑+删+举报", f"act:report:{user_id}".encode()),
                Button.inline("💬 再追问一次", f"act:probe:{user_id}".encode()),
            ],
        ]
        await self._send_card(text=text, buttons=buttons, images=images, silent=False)

    async def send_blocked_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
        images: list[bytes],
    ) -> None:
        text = (
            "🚫 <b>已自动拦截广告并双向删记录</b>\n\n"
            f"{self._format_user_block(user_id, profile, batch)}\n\n"
            f"🤖 <b>判定</b>：<code>{verdict.category}</code> ({verdict.confidence:.0%})\n"
            f"📌 <b>理由</b>：{_esc(verdict.reason)}"
        )
        buttons = [
            [
                Button.inline("↩️ 误杀恢复（解封+加白）", f"act:unblock:{user_id}".encode()),
                Button.inline("🚨 补点举报", f"act:report_only:{user_id}".encode()),
            ]
        ]
        await self._send_card(text=text, buttons=buttons, images=images, silent=True)

    async def send_passed_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
    ) -> None:
        text = (
            "✅ <b>已自动放行陌生私聊</b>（已移出归档并标未读）\n\n"
            f"{self._format_user_block(user_id, profile, batch)}\n\n"
            f"📌 <b>AI 摘要</b>：{_esc(verdict.reason)} ({verdict.confidence:.0%})"
        )
        buttons = [[Button.inline("🚫 其实是广告，立即拉黑", f"act:block:{user_id}".encode())]]
        await self._send_card(text=text, buttons=buttons, silent=False)

    async def send_dryrun_card(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        verdict: LLMVerdict,
        images: list[bytes],
        would_action: str,
    ) -> None:
        reply_preview = (
            f"\n💬 <b>拟发送追问</b>：{_esc(verdict.reply_to_user)}"
            if verdict.reply_to_user
            else ""
        )
        text = (
            f"🧪 <b>演习模式（Dry-Run）审核日志</b>\n"
            f"<i>未执行任何拉黑/回复操作，若在 auto 模式下拟执行：<code>{would_action}</code></i>\n\n"
            f"{self._format_user_block(user_id, profile, batch)}\n\n"
            f"🤖 <b>AI 判定</b>：<code>{verdict.decision.value}</code> "
            f"({verdict.confidence:.0%}) · <code>{_esc(verdict.category)}</code>\n"
            f"📌 <b>理由</b>：{_esc(verdict.reason)}{reply_preview}"
        )
        buttons = [
            [
                Button.inline("✅ 手动放行加白", f"act:allow:{user_id}".encode()),
                Button.inline("🚫 手动拉黑+删", f"act:block:{user_id}".encode()),
            ]
        ]
        await self._send_card(text=text, buttons=buttons, images=images, silent=True)

    async def send_pending_followup(
        self,
        *,
        user_id: int,
        display_name: str,
        username: str | None,
        message: IncomingMessage,
    ) -> None:
        uname = f" (@{_esc(username)})" if username else ""
        content = _esc(_truncate(message.text, 250)) if message.text else ""
        media_note = f" [{_esc(message.media.description)}]" if message.media else ""
        text = (
            f"💬 <b>待审核对象补充消息</b>\n"
            f"👤 {_esc(display_name)}{uname} (<code>{user_id}</code>)\n"
            f"• {content}{media_note}"
        )
        buttons = [
            [
                Button.inline("✅ 放行", f"act:allow:{user_id}".encode()),
                Button.inline("🚫 拉黑+删", f"act:block:{user_id}".encode()),
                Button.inline("🚨 举报", f"act:report:{user_id}".encode()),
            ]
        ]
        images = [message.media.image_bytes] if message.media and message.media.image_bytes else []
        await self._send_card(text=text, buttons=buttons, images=images, silent=True)
