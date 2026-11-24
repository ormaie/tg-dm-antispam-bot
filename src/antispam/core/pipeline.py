from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from antispam.config import AppConfig
from antispam.core.enricher import ProfileEnricher
from antispam.core.interfaces import NotifierProtocol, TelegramActionsProtocol
from antispam.core.llm import LLMModerator, sanitize_probe_reply
from antispam.core.ratelimit import SlidingWindowRateLimiter
from antispam.core.rules import RuleEngine
from antispam.models import (
    DecisionType,
    IncomingMessage,
    LLMVerdict,
    RunMode,
    UserProfile,
    UserStatus,
)
from antispam.storage.db import Database, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass
class _PendingBatch:
    messages: list[IncomingMessage] = field(default_factory=list)
    first_at: float = 0.0
    timer_task: asyncio.Task[None] | None = None


class ModerationPipeline:
    """Coordinates debouncing, state transitions, rule checks, LLM verdicts, and actions."""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        db: Database,
        tg: TelegramActionsProtocol,
        notifier: NotifierProtocol,
        llm: LLMModerator,
        rule_engine: RuleEngine | None = None,
        enricher: ProfileEnricher | None = None,
        reply_limiter: SlidingWindowRateLimiter | None = None,
        block_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.tg = tg
        self.notifier = notifier
        self.llm = llm
        self.rule_engine = rule_engine or RuleEngine()
        self.enricher = enricher or ProfileEnricher(
            tg, ttl_seconds=cfg.screening.profile_cache_ttl_seconds
        )
        self.reply_limiter = reply_limiter or SlidingWindowRateLimiter(
            cfg.rate_limit.max_replies_per_minute, window_seconds=60.0
        )
        self.block_limiter = block_limiter or SlidingWindowRateLimiter(
            cfg.rate_limit.max_blocks_per_minute, window_seconds=60.0
        )

        self._pending: dict[int, _PendingBatch] = {}
        self._user_locks: dict[int, asyncio.Lock] = {}
        self._maintenance_task: asyncio.Task[None] | None = None

    def _lock_for(self, user_id: int) -> asyncio.Lock:
        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    async def get_run_mode(self) -> RunMode:
        return await self.db.get_run_mode(self.cfg.mode)

    async def set_run_mode(self, mode: RunMode) -> None:
        await self.db.set_run_mode(mode)
        logger.info("Runtime mode switched to: %s", mode.value)

    def _is_static_whitelisted(self, user_id: int, username: str | None) -> bool:
        if user_id in self.cfg.screening.whitelist_user_ids:
            return True
        return bool(
            username
            and username.lstrip("@").strip().lower() in self.cfg.screening.whitelist_usernames
        )

    def _cancel_pending(self, user_id: int) -> None:
        batch = self._pending.pop(user_id, None)
        if batch and batch.timer_task and not batch.timer_task.done():
            batch.timer_task.cancel()

    def start_background_tasks(self) -> None:
        if self._maintenance_task is None or self._maintenance_task.done():
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self) -> None:
        interval = max(60.0, self.cfg.privacy.maintenance_interval_minutes * 60.0)
        try:
            await asyncio.sleep(10.0)
            while True:
                try:
                    summary = await self.run_maintenance_once()
                    if any(summary.values()):
                        logger.info("Periodic maintenance finished: %s", summary)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Error during periodic maintenance sweep")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    async def run_maintenance_once(
        self,
        *,
        message_days: int | None = None,
        audit_days: int | None = None,
        probe_timeout_hours: float | None = None,
    ) -> dict[str, int]:
        now = datetime.now(UTC)
        expired_count = 0

        timeout_h = (
            self.cfg.screening.probe_timeout_hours
            if probe_timeout_hours is None
            else probe_timeout_hours
        )
        if timeout_h > 0:
            cutoff_iso = (now - timedelta(hours=timeout_h)).isoformat(timespec="seconds")
            stale_users = await self.db.list_stale_probing_users(cutoff_iso)
            run_mode = await self.get_run_mode()
            for stale in stale_users:
                if stale.user_id in self._pending:
                    continue
                async with self._lock_for(stale.user_id):
                    current = await self.db.get_user(stale.user_id)
                    if (
                        current is None
                        or current.status != UserStatus.PROBING
                        or current.updated_at > cutoff_iso
                    ):
                        continue
                    if run_mode != RunMode.DRYRUN:
                        if (
                            self.cfg.screening.auto_archive
                            and not current.archived
                            and await self.tg.archive_chat(stale.user_id, True)
                        ):
                            await self.db.set_user_flags(stale.user_id, archived=True)
                        if (
                            self.cfg.screening.auto_mute
                            and not current.muted
                            and await self.tg.mute_chat(stale.user_id, True)
                        ):
                            await self.db.set_user_flags(stale.user_id, muted=True)
                    await self.db.update_user_status(
                        stale.user_id,
                        UserStatus.EXPIRED,
                        reason=f"追问超过 {timeout_h:g} 小时未回复，已自动静默归档",
                    )
                    await self.db.add_audit_log(
                        user_id=stale.user_id,
                        decision="review",
                        confidence=0.5,
                        category="unknown_intent",
                        reason=f"追问超时（>{timeout_h:g}h）自动归档收尾",
                        action_taken="probe_expired",
                        model_used="maintenance",
                    )
                    expired_count += 1

        eff_msg_days = (
            self.cfg.privacy.message_retention_days if message_days is None else message_days
        )
        eff_audit_days = (
            self.cfg.privacy.audit_log_retention_days if audit_days is None else audit_days
        )
        msg_cutoff = (
            (now - timedelta(days=eff_msg_days)).isoformat(timespec="seconds")
            if eff_msg_days > 0
            else None
        )
        audit_cutoff = (
            (now - timedelta(days=eff_audit_days)).isoformat(timespec="seconds")
            if eff_audit_days > 0
            else None
        )
        cleaned = await self.db.cleanup_expired_records(
            message_cutoff_iso=msg_cutoff,
            audit_cutoff_iso=audit_cutoff,
        )
        return {
            "expired_probes": expired_count,
            "messages_deleted": cleaned["messages_deleted"],
            "audits_deleted": cleaned["audits_deleted"],
        }

    async def shutdown(self) -> None:
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
        for user_id in list(self._pending.keys()):
            self._cancel_pending(user_id)

    # ---------- Inbound Events ----------

    async def handle_incoming_message(self, msg: IncomingMessage) -> None:
        user_id = msg.user_id

        if self._is_static_whitelisted(user_id, msg.username):
            logger.debug("Skipping static whitelisted user %s", user_id)
            return

        async with self._lock_for(user_id):
            user = await self.db.get_user(user_id)
            if user is not None:
                if user.status == UserStatus.WHITELISTED:
                    return
                if user.status == UserStatus.BLOCKED:
                    logger.debug("Ignoring message from already blocked user %s", user_id)
                    return
            else:
                # First time seeing this user: check if owner has ever messaged them before
                has_history = await self.tg.has_owner_history(
                    user_id, limit=self.cfg.screening.history_check_limit
                )
                if has_history:
                    await self.db.get_or_create_user(
                        user_id,
                        msg.display_name,
                        msg.username,
                        default_status=UserStatus.WHITELISTED,
                    )
                    logger.info(
                        "Auto-whitelisted user %s (%s) due to existing owner message history",
                        user_id,
                        msg.display_name,
                    )
                    return
                user = await self.db.get_or_create_user(
                    user_id,
                    msg.display_name,
                    msg.username,
                    default_status=UserStatus.NEW,
                )

            # Refresh display name / username if changed
            user = await self.db.get_or_create_user(user_id, msg.display_name, msg.username)

            # Persist incoming message for multi-turn context & audit trail
            await self.db.add_message(
                user_id=user_id,
                direction="in",
                text=msg.text,
                tg_msg_id=msg.tg_msg_id,
                media_type=msg.media.media_type if msg.media else None,
                media_meta=msg.media.to_meta_dict() if msg.media else None,
            )

            # If already awaiting owner decision, stop AI probing and just relay a brief follow-up
            if user.status == UserStatus.PENDING_REVIEW:
                await self.notifier.send_pending_followup(
                    user_id=user_id,
                    display_name=msg.display_name,
                    username=msg.username,
                    message=msg,
                )
                return

            run_mode = await self.get_run_mode()
            if run_mode != RunMode.DRYRUN:
                if (
                    self.cfg.screening.auto_archive
                    and not user.archived
                    and await self.tg.archive_chat(user_id, True)
                ):
                    await self.db.set_user_flags(user_id, archived=True)
                if (
                    self.cfg.screening.auto_mute
                    and not user.muted
                    and await self.tg.mute_chat(user_id, True)
                ):
                    await self.db.set_user_flags(user_id, muted=True)

            self._enqueue_for_debounce(msg)

    def _enqueue_for_debounce(self, msg: IncomingMessage) -> None:
        user_id = msg.user_id
        now = time.monotonic()
        batch = self._pending.get(user_id)
        if batch is None:
            batch = _PendingBatch(messages=[msg], first_at=now)
            self._pending[user_id] = batch
        else:
            batch.messages.append(msg)
            if batch.timer_task and not batch.timer_task.done():
                batch.timer_task.cancel()

        elapsed = now - batch.first_at
        remaining_cap = max(0.0, self.cfg.screening.max_wait_seconds - elapsed)
        delay = min(self.cfg.screening.debounce_seconds, remaining_cap)
        batch.timer_task = asyncio.create_task(self._debounced_flush(user_id, delay))

    async def _debounced_flush(self, user_id: int, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await self.flush_user(user_id)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Unhandled error while flushing batch for user %s", user_id)

    async def flush_user(self, user_id: int) -> None:
        """Immediately evaluate any buffered messages for `user_id`."""
        async with self._lock_for(user_id):
            pending = self._pending.pop(user_id, None)
            if not pending or not pending.messages:
                return
            if pending.timer_task and pending.timer_task is not asyncio.current_task():
                pending.timer_task.cancel()
            await self._process_batch_locked(user_id, pending.messages)

    def _is_within_edit_probation(self, auto_passed_at: str | None) -> bool:
        probation_minutes = self.cfg.screening.edit_probation_minutes
        if probation_minutes <= 0 or not auto_passed_at:
            return False
        try:
            passed_dt = datetime.fromisoformat(auto_passed_at)
            if passed_dt.tzinfo is None:
                passed_dt = passed_dt.replace(tzinfo=UTC)
        except ValueError:
            return False
        elapsed_seconds = (datetime.now(UTC) - passed_dt).total_seconds()
        return 0 <= elapsed_seconds <= probation_minutes * 60

    async def handle_edited_message(self, msg: IncomingMessage) -> None:
        """Re-evaluate edited messages during screening or post-auto-pass probation."""
        user_id = msg.user_id
        if self._is_static_whitelisted(user_id, msg.username):
            return

        async with self._lock_for(user_id):
            user = await self.db.get_user(user_id)
            if user is None or user.status == UserStatus.BLOCKED:
                return

            if user.status == UserStatus.WHITELISTED:
                if not self._is_within_edit_probation(user.auto_passed_at):
                    return
                logger.info(
                    "Auto-passed user %s edited a message within probation window; re-screening",
                    user_id,
                )
                await self.db.update_user_status(
                    user_id,
                    UserStatus.PROBING,
                    reason="自动放行观察期内编辑消息，重新触发审核",
                    clear_auto_passed=True,
                )
                user = await self.db.get_user(user_id)
                assert user is not None

            meta = msg.media.to_meta_dict() if msg.media else {}
            meta["edited"] = True
            await self.db.add_message(
                user_id=user_id,
                direction="in",
                text=msg.text,
                tg_msg_id=msg.tg_msg_id,
                media_type=msg.media.media_type if msg.media else None,
                media_meta=meta,
            )

            if user.status == UserStatus.PENDING_REVIEW:
                await self.notifier.send_pending_followup(
                    user_id=user_id,
                    display_name=msg.display_name,
                    username=msg.username,
                    message=msg,
                )
                return

            self.enricher.invalidate(user_id)
            run_mode = await self.get_run_mode()
            if run_mode != RunMode.DRYRUN:
                if (
                    self.cfg.screening.auto_archive
                    and not user.archived
                    and await self.tg.archive_chat(user_id, True)
                ):
                    await self.db.set_user_flags(user_id, archived=True)
                if (
                    self.cfg.screening.auto_mute
                    and not user.muted
                    and await self.tg.mute_chat(user_id, True)
                ):
                    await self.db.set_user_flags(user_id, muted=True)

            self._enqueue_for_debounce(msg)

    # ---------- Human Takeover ----------

    async def handle_owner_outgoing(
        self,
        user_id: int,
        text: str = "",
        *,
        display_name: str | None = None,
        username: str | None = None,
        tg_msg_id: int | None = None,
    ) -> None:
        """Called when the account owner manually sends a DM to `user_id`.

        Only takes over users currently tracked in non-whitelisted states and never
        persists the owner's outgoing plaintext messages to SQLite.
        """
        del text, display_name, username, tg_msg_id
        async with self._lock_for(user_id):
            self._cancel_pending(user_id)
            user = await self.db.get_user(user_id)
            if user is None or user.status == UserStatus.WHITELISTED:
                return

            await self.db.update_user_status(
                user_id,
                UserStatus.WHITELISTED,
                reason="机主已手动回复接管对话",
                clear_auto_passed=True,
            )
            if self.cfg.privacy.purge_messages_on_whitelist:
                await self.db.delete_user_messages(user_id)
            logger.info("Human takeover: user %s marked WHITELISTED", user_id)

            if user.archived and await self.tg.archive_chat(user_id, False):
                await self.db.set_user_flags(user_id, archived=False)
            if user.muted and await self.tg.mute_chat(user_id, False):
                await self.db.set_user_flags(user_id, muted=False)

    # ---------- Core Evaluation Logic ----------

    async def _process_batch_locked(self, user_id: int, batch: list[IncomingMessage]) -> None:
        user = await self.db.get_user(user_id)
        if user is None or user.status in (
            UserStatus.WHITELISTED,
            UserStatus.BLOCKED,
            UserStatus.PENDING_REVIEW,
        ):
            return

        run_mode = await self.get_run_mode()
        profile = await self.enricher.get_profile(user_id)
        images = [
            msg.media.image_bytes
            for msg in batch
            if msg.media and msg.media.image_bytes
        ][: self.cfg.media.max_images_per_batch]
        has_high_risk_file = any(msg.media and msg.media.is_high_risk for msg in batch)

        # Build prior conversation history excluding messages from the current batch
        batch_msg_ids = {msg.tg_msg_id for msg in batch if msg.tg_msg_id}
        recent_messages = await self.db.get_recent_messages(user_id, limit=25)
        prior_history = [
            item
            for item in recent_messages
            if not (item.direction == "in" and item.tg_msg_id in batch_msg_ids)
        ]

        verdict = await self._evaluate_verdict(
            profile=profile,
            prior_history=prior_history,
            batch=batch,
            probe_count=user.probe_count,
        )

        # Safety guard 1: never auto-pass high-risk executable attachments
        if has_high_risk_file and verdict.decision == DecisionType.PASS:
            verdict = LLMVerdict(
                decision=DecisionType.REVIEW,
                confidence=min(verdict.confidence, 0.5),
                category="high_risk_file",
                reason=f"包含高危可执行附件，已禁止自动放行并转人工审核（原判定：{verdict.reason}）",
                reply_to_user="",
                model_used=verdict.model_used,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
                raw_response=verdict.raw_response,
            )

        # Safety guard 2: enforce max probe rounds to prevent infinite bot-to-bot loops
        max_probes = self.cfg.screening.max_probe_rounds
        if verdict.decision == DecisionType.ASK and user.probe_count >= max_probes:
            verdict = LLMVerdict(
                decision=DecisionType.REVIEW,
                confidence=verdict.confidence,
                category=verdict.category,
                reason=f"已达到最大追问轮次（{max_probes} 轮）仍未明确来意，已停止自动回复并转人工审核（原理由：{verdict.reason}）",
                reply_to_user="",
                model_used=verdict.model_used,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
                raw_response=verdict.raw_response,
            )

        await self._execute_verdict(
            user_id=user_id,
            profile=profile,
            batch=batch,
            images=images,
            verdict=verdict,
            run_mode=run_mode,
        )

    async def _evaluate_verdict(
        self,
        *,
        profile: UserProfile | None,
        prior_history: list,
        batch: list[IncomingMessage],
        probe_count: int,
    ) -> LLMVerdict:
        # Stage 1: Local fast rules
        rules = await self.db.list_rules(enabled_only=True)
        match = self.rule_engine.evaluate(rules, profile, batch)
        if match is not None:
            note_desc = match.note or match.pattern
            if match.action == "spam":
                return LLMVerdict(
                    decision=DecisionType.SPAM,
                    confidence=0.98,
                    category="local_rule",
                    reason=f"命中本地强拦截规则 #{match.rule_id}（{note_desc}，匹配位置：{match.matched_in}）",
                    model_used="local_rule",
                )
            return LLMVerdict(
                decision=DecisionType.REVIEW,
                confidence=0.75,
                category="local_rule",
                reason=f"命中本地复核规则 #{match.rule_id}（{note_desc}，匹配位置：{match.matched_in}）",
                model_used="local_rule",
            )

        # Stage 2: Daily LLM budget circuit breaker (atomic check + reserve)
        budget = self.cfg.rate_limit.daily_llm_budget_calls
        reserved, used_today = await self.db.try_reserve_daily_llm_call(budget)
        if not reserved:
            return LLMVerdict(
                decision=DecisionType.REVIEW,
                confidence=0.5,
                category="unknown_intent",
                reason=f"今日 LLM 调用已达预算上限（{used_today}/{budget}），已自动熔断并转人工审核",
                model_used="budget_guard",
            )

        return await self.llm.evaluate(
            profile=profile,
            history=prior_history,
            batch=batch,
            probe_count=probe_count,
        )

    def _classify_effective_action(self, verdict: LLMVerdict, run_mode: RunMode) -> str:
        th = self.cfg.thresholds
        if verdict.decision == DecisionType.SPAM and verdict.confidence >= th.spam_auto_block:
            return "auto_block" if run_mode == RunMode.AUTO else "review"
        if verdict.decision == DecisionType.PASS and verdict.confidence >= th.pass_auto_allow:
            return "auto_pass" if run_mode == RunMode.AUTO else "review"
        if verdict.decision == DecisionType.ASK:
            return "probe"
        return "review"

    async def _execute_verdict(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        images: list[bytes],
        verdict: LLMVerdict,
        run_mode: RunMode,
    ) -> None:
        if run_mode == RunMode.DRYRUN:
            would_action = self._classify_effective_action(verdict, RunMode.AUTO)
            await self.db.add_audit_log(
                user_id=user_id,
                decision=verdict.decision.value,
                confidence=verdict.confidence,
                category=verdict.category,
                reason=verdict.reason,
                reply_text=verdict.reply_to_user,
                action_taken=f"dryrun:{would_action}",
                model_used=verdict.model_used,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
                dry_run=True,
            )
            await self.notifier.send_dryrun_card(
                user_id=user_id,
                profile=profile,
                batch=batch,
                verdict=verdict,
                images=images,
                would_action=would_action,
            )
            return

        effective_action = self._classify_effective_action(verdict, run_mode)

        if effective_action == "auto_block":
            if not self.block_limiter.allow():
                await self._send_to_review(
                    user_id=user_id,
                    profile=profile,
                    batch=batch,
                    images=images,
                    verdict=verdict,
                    run_mode=run_mode,
                    extra_note="⚠️ 触发每分钟拉黑限流保护，已降级为人工审核。",
                )
                return

            blocked_ok = await self.tg.block_and_clean(
                user_id,
                delete_history=self.cfg.actions.auto_block_delete_history,
                revoke=self.cfg.actions.auto_block_revoke_both,
                report_spam=self.cfg.actions.auto_block_report_spam,
            )
            if not blocked_ok:
                await self._send_to_review(
                    user_id=user_id,
                    profile=profile,
                    batch=batch,
                    images=images,
                    verdict=verdict,
                    run_mode=run_mode,
                    extra_note="⚠️ Telegram 自动拉黑执行失败（接口异常或限流），未修改拦截状态，已降级为人工审核。",
                )
                return

            await self.db.update_user_status(
                user_id,
                UserStatus.BLOCKED,
                category=verdict.category,
                confidence=verdict.confidence,
                reason=verdict.reason,
            )
            await self.db.add_audit_log(
                user_id=user_id,
                decision=verdict.decision.value,
                confidence=verdict.confidence,
                category=verdict.category,
                reason=verdict.reason,
                action_taken="auto_block",
                model_used=verdict.model_used,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
            )
            await self.notifier.send_blocked_card(
                user_id=user_id,
                profile=profile,
                batch=batch,
                verdict=verdict,
                images=images,
            )
            return

        if effective_action == "auto_pass":
            await self._release_chat_to_owner(user_id)
            await self.db.update_user_status(
                user_id,
                UserStatus.WHITELISTED,
                category=verdict.category,
                confidence=verdict.confidence,
                reason=verdict.reason,
                auto_passed_at=utc_now_iso(),
            )
            if self.cfg.privacy.purge_messages_on_whitelist:
                await self.db.delete_user_messages(user_id)
            await self.db.add_audit_log(
                user_id=user_id,
                decision=verdict.decision.value,
                confidence=verdict.confidence,
                category=verdict.category,
                reason=verdict.reason,
                action_taken="auto_pass",
                model_used=verdict.model_used,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
            )
            await self.notifier.send_passed_card(
                user_id=user_id,
                profile=profile,
                batch=batch,
                verdict=verdict,
            )
            return

        if effective_action == "probe":
            if not self.reply_limiter.allow():
                await self._send_to_review(
                    user_id=user_id,
                    profile=profile,
                    batch=batch,
                    images=images,
                    verdict=verdict,
                    run_mode=run_mode,
                    extra_note="⚠️ 触发每分钟自动回复限流保护，已暂停自动追问并转人工审核。",
                )
                return

            if self.cfg.actions.use_fixed_probe_message:
                reply_text = self.cfg.llm.fallback_probe_message
            else:
                reply_text = sanitize_probe_reply(
                    verdict.reply_to_user,
                    fallback=self.cfg.llm.fallback_probe_message,
                    max_length=self.cfg.actions.max_probe_reply_chars,
                )
            sent_msg_id = await self.tg.send_probe_reply(user_id, reply_text)
            if sent_msg_id is None:
                await self._send_to_review(
                    user_id=user_id,
                    profile=profile,
                    batch=batch,
                    images=images,
                    verdict=verdict,
                    run_mode=run_mode,
                    extra_note="⚠️ 自动追问消息发送失败（Telegram 接口异常或限流），未扣减追问次数，已降级为人工审核。",
                )
                return

            await self.db.add_message(
                user_id=user_id,
                direction="out_bot",
                text=reply_text,
                tg_msg_id=sent_msg_id,
            )
            await self.db.increment_probe_count(user_id)
            await self.db.update_user_status(
                user_id,
                UserStatus.PROBING,
                category=verdict.category,
                confidence=verdict.confidence,
                reason=verdict.reason,
            )
            await self.db.add_audit_log(
                user_id=user_id,
                decision=verdict.decision.value,
                confidence=verdict.confidence,
                category=verdict.category,
                reason=verdict.reason,
                reply_text=reply_text,
                action_taken="probe",
                model_used=verdict.model_used,
                prompt_tokens=verdict.prompt_tokens,
                completion_tokens=verdict.completion_tokens,
            )
            return

        await self._send_to_review(
            user_id=user_id,
            profile=profile,
            batch=batch,
            images=images,
            verdict=verdict,
            run_mode=run_mode,
        )

    async def _send_to_review(
        self,
        *,
        user_id: int,
        profile: UserProfile | None,
        batch: list[IncomingMessage],
        images: list[bytes],
        verdict: LLMVerdict,
        run_mode: RunMode,
        extra_note: str = "",
    ) -> None:
        await self.db.update_user_status(
            user_id,
            UserStatus.PENDING_REVIEW,
            category=verdict.category,
            confidence=verdict.confidence,
            reason=verdict.reason,
        )
        await self.db.add_audit_log(
            user_id=user_id,
            decision=verdict.decision.value,
            confidence=verdict.confidence,
            category=verdict.category,
            reason=verdict.reason,
            action_taken="review",
            model_used=verdict.model_used,
            prompt_tokens=verdict.prompt_tokens,
            completion_tokens=verdict.completion_tokens,
        )
        await self.notifier.send_review_card(
            user_id=user_id,
            profile=profile,
            batch=batch,
            verdict=verdict,
            images=images,
            mode=run_mode,
            extra_note=extra_note,
        )

    async def _release_chat_to_owner(self, user_id: int) -> None:
        archived_cleared = not self.cfg.screening.auto_archive or await self.tg.archive_chat(
            user_id, False
        )
        muted_cleared = not self.cfg.screening.auto_mute or await self.tg.mute_chat(user_id, False)
        if self.cfg.screening.mark_unread_on_pass:
            await self.tg.mark_unread(user_id)
        await self.db.set_user_flags(
            user_id,
            archived=False if archived_cleared else None,
            muted=False if muted_cleared else None,
        )

    # ---------- Owner Manual Controls (Helper Bot Buttons & Commands) ----------

    async def manual_allow(
        self,
        user_id: int,
        *,
        display_name: str | None = None,
        username: str | None = None,
    ) -> str:
        async with self._lock_for(user_id):
            self._cancel_pending(user_id)
            user = await self.db.get_user(user_id)
            was_blocked = user is not None and user.status == UserStatus.BLOCKED
            if was_blocked:
                unblocked = await self.tg.unblock_user(user_id)
                if not unblocked:
                    return "❌ 解封失败（Telegram 接口异常或限流），状态未变更，请稍后重试"
            if user is None:
                await self.db.get_or_create_user(
                    user_id,
                    display_name or str(user_id),
                    username,
                    default_status=UserStatus.WHITELISTED,
                )
            await self._release_chat_to_owner(user_id)
            await self.db.update_user_status(
                user_id,
                UserStatus.WHITELISTED,
                reason="机主手动放行/加白",
                clear_auto_passed=True,
            )
            if self.cfg.privacy.purge_messages_on_whitelist:
                await self.db.delete_user_messages(user_id)
            await self.db.add_audit_log(
                user_id=user_id,
                decision="pass",
                confidence=1.0,
                category="normal",
                reason="机主手动放行/误杀恢复",
                action_taken="manual_allow",
                model_used="owner",
            )
            return "✅ 已解封并加白（已移出归档、取消静音）" if was_blocked else "✅ 已放行并加入白名单（已移出归档、取消静音）"

    async def manual_block(
        self,
        user_id: int,
        *,
        report_spam: bool = False,
        display_name: str | None = None,
        username: str | None = None,
    ) -> str:
        async with self._lock_for(user_id):
            self._cancel_pending(user_id)
            blocked_ok = await self.tg.block_and_clean(
                user_id,
                delete_history=self.cfg.actions.auto_block_delete_history,
                revoke=self.cfg.actions.auto_block_revoke_both,
                report_spam=report_spam,
            )
            if not blocked_ok:
                return "❌ 拉黑失败（Telegram 接口异常或限流），状态未变更，请稍后重试"
            user = await self.db.get_user(user_id)
            if user is None:
                await self.db.get_or_create_user(
                    user_id,
                    display_name or str(user_id),
                    username,
                    default_status=UserStatus.BLOCKED,
                )
            await self.db.update_user_status(
                user_id,
                UserStatus.BLOCKED,
                reason="机主手动拉黑并举报" if report_spam else "机主手动拉黑",
            )
            await self.db.add_audit_log(
                user_id=user_id,
                decision="spam",
                confidence=1.0,
                category="manual_block",
                reason="机主手动操作",
                action_taken="manual_report" if report_spam else "manual_block",
                model_used="owner",
            )
            return "🚨 已拉黑、双向删记录并举报垃圾信息" if report_spam else "🚫 已拉黑并删除聊天记录"

    async def manual_report_only(self, user_id: int) -> str:
        async with self._lock_for(user_id):
            user = await self.db.get_user(user_id)
            if user is not None and user.status == UserStatus.WHITELISTED:
                return "ℹ️ 该用户当前已放行加白，已忽略旧卡片举报操作"
            reported_ok = await self.tg.report_spam(user_id)
            if not reported_ok:
                return "❌ 举报失败（Telegram 接口异常或限流），请稍后重试"
            await self.db.add_audit_log(
                user_id=user_id,
                decision="spam",
                confidence=1.0,
                category="manual_report",
                reason="机主手动补点举报",
                action_taken="manual_report",
                model_used="owner",
            )
            return "🚨 已向 Telegram 补交垃圾信息举报"

    async def manual_probe_again(self, user_id: int) -> str:
        async with self._lock_for(user_id):
            user = await self.db.get_user(user_id)
            if user is None:
                return "ℹ️ 未找到该用户的审核记录，无法追问"
            if user.status not in (
                UserStatus.NEW,
                UserStatus.PROBING,
                UserStatus.PENDING_REVIEW,
            ):
                status_label = "已放行加白" if user.status == UserStatus.WHITELISTED else "已拉黑"
                return f"ℹ️ 该用户已处理（当前状态：{status_label}），无法再追问"

            self._cancel_pending(user_id)
            reply_text = self.cfg.llm.fallback_probe_message
            sent_msg_id = await self.tg.send_probe_reply(user_id, reply_text)
            if sent_msg_id is None:
                return "❌ 追问发送失败（Telegram 接口异常或限流），状态未变更"
            await self.db.add_message(
                user_id=user_id,
                direction="out_bot",
                text=reply_text,
                tg_msg_id=sent_msg_id,
            )
            await self.db.increment_probe_count(user_id)
            await self.db.update_user_status(
                user_id,
                UserStatus.PROBING,
                reason="机主触发再次追问来意",
            )
            await self.db.add_audit_log(
                user_id=user_id,
                decision="ask",
                confidence=0.5,
                category="unknown_intent",
                reason="机主手动触发再追问一次",
                reply_text=reply_text,
                action_taken="probe",
                model_used="owner",
            )
            return "💬 已向对方发送礼貌追问，等待回复"
