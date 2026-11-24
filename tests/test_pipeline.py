from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from antispam.config import AppConfig, DefaultRuleConfig
from antispam.core.pipeline import ModerationPipeline
from antispam.core.ratelimit import SlidingWindowRateLimiter
from antispam.models import (
    DecisionType,
    IncomingMessage,
    LLMVerdict,
    MediaItem,
    RunMode,
    UserProfile,
    UserStatus,
)
from antispam.storage.db import Database


class FakeTelegramActions:
    def __init__(self) -> None:
        self.owner_history_users: set[int] = set()
        self.profiles: dict[int, UserProfile] = {}
        self.archived: list[tuple[int, bool]] = []
        self.muted: list[tuple[int, bool]] = []
        self.marked_unread: list[int] = []
        self.probes_sent: list[tuple[int, str]] = []
        self.blocked: list[tuple[int, bool, bool, bool]] = []
        self.unblocked: list[int] = []
        self.reported: list[int] = []
        self._msg_seq = 5000

    async def has_owner_history(self, user_id: int, limit: int) -> bool:
        return user_id in self.owner_history_users

    async def fetch_user_profile(self, user_id: int) -> UserProfile | None:
        return self.profiles.get(
            user_id, UserProfile(user_id=user_id, first_name=f"User{user_id}")
        )

    async def resolve_user_id(self, identifier: str) -> tuple[int, str, str | None] | None:
        return None

    async def archive_chat(self, user_id: int, archive: bool) -> None:
        self.archived.append((user_id, archive))

    async def mute_chat(self, user_id: int, mute: bool) -> None:
        self.muted.append((user_id, mute))

    async def mark_unread(self, user_id: int) -> None:
        self.marked_unread.append(user_id)

    async def send_probe_reply(self, user_id: int, text: str) -> int | None:
        self._msg_seq += 1
        self.probes_sent.append((user_id, text))
        return self._msg_seq

    async def block_and_clean(
        self,
        user_id: int,
        *,
        delete_history: bool,
        revoke: bool,
        report_spam: bool,
    ) -> None:
        self.blocked.append((user_id, delete_history, revoke, report_spam))

    async def unblock_user(self, user_id: int) -> None:
        self.unblocked.append(user_id)

    async def report_spam(self, user_id: int) -> None:
        self.reported.append(user_id)


class FakeNotifier:
    def __init__(self) -> None:
        self.review_cards: list[dict[str, Any]] = []
        self.blocked_cards: list[dict[str, Any]] = []
        self.passed_cards: list[dict[str, Any]] = []
        self.dryrun_cards: list[dict[str, Any]] = []
        self.followups: list[dict[str, Any]] = []

    async def send_review_card(self, **kwargs: Any) -> None:
        self.review_cards.append(kwargs)

    async def send_blocked_card(self, **kwargs: Any) -> None:
        self.blocked_cards.append(kwargs)

    async def send_passed_card(self, **kwargs: Any) -> None:
        self.passed_cards.append(kwargs)

    async def send_dryrun_card(self, **kwargs: Any) -> None:
        self.dryrun_cards.append(kwargs)

    async def send_pending_followup(self, **kwargs: Any) -> None:
        self.followups.append(kwargs)


class FakeLLM:
    def __init__(self, verdicts: list[LLMVerdict] | None = None) -> None:
        self.verdicts = list(verdicts or [])
        self.calls: list[dict[str, Any]] = []

    async def evaluate(self, **kwargs: Any) -> LLMVerdict:
        self.calls.append(kwargs)
        if self.verdicts:
            return self.verdicts.pop(0)
        return LLMVerdict(
            decision=DecisionType.REVIEW,
            confidence=0.5,
            category="unknown_intent",
            reason="default",
            model_used="fake",
        )


def _fast_cfg(mode: RunMode = RunMode.AUTO) -> AppConfig:
    cfg = AppConfig(mode=mode)
    cfg.screening.debounce_seconds = 0.05
    cfg.screening.max_wait_seconds = 0.2
    cfg.screening.max_probe_rounds = 2
    return cfg


@pytest.mark.asyncio
async def test_existing_owner_history_auto_whitelists_without_llm(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    tg.owner_history_users.add(1001)
    notifier = FakeNotifier()
    llm = FakeLLM()

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=1001, tg_msg_id=1, text="老朋友在吗", display_name="Friend")
    )
    await asyncio.sleep(0.1)

    user = await db.get_user(1001)
    assert user is not None and user.status == UserStatus.WHITELISTED
    assert len(llm.calls) == 0
    assert tg.archived == []
    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_debounce_merges_multiple_messages_and_auto_blocks_high_spam(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.SPAM,
                confidence=0.95,
                category="crypto_scam",
                reason="币圈带单诈骗",
                model_used="fake-vision",
            )
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=2001, tg_msg_id=10, text="你好", display_name="Spammer")
    )
    await asyncio.sleep(0.01)
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=2001, tg_msg_id=11, text="高收益合约带单加群", display_name="Spammer")
    )
    await asyncio.sleep(0.15)

    assert len(llm.calls) == 1
    assert len(llm.calls[0]["batch"]) == 2
    assert tg.archived == [(2001, True)]
    assert tg.muted == [(2001, True)]
    assert tg.blocked == [(2001, True, True, False)]
    assert len(notifier.blocked_cards) == 1

    user = await db.get_user(2001)
    assert user is not None and user.status == UserStatus.BLOCKED

    # Owner clicks false-positive restore button
    msg = await pipeline.manual_allow(2001)
    assert "解封" in msg
    assert tg.unblocked == [2001]
    assert (2001, False) in tg.archived
    assert 2001 in tg.marked_unread
    restored = await db.get_user(2001)
    assert restored is not None and restored.status == UserStatus.WHITELISTED

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_review_only_mode_does_not_auto_block_or_auto_pass(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.SPAM,
                confidence=0.99,
                category="promo_ad",
                reason="广告",
            )
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.REVIEW_ONLY), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=2002, tg_msg_id=1, text="广告链接", display_name="Ad")
    )
    await asyncio.sleep(0.12)

    assert tg.blocked == []
    assert len(notifier.review_cards) == 1
    user = await db.get_user(2002)
    assert user is not None and user.status == UserStatus.PENDING_REVIEW

    # Extra message while pending review should only trigger follow-up, NOT another LLM call
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=2002, tg_msg_id=2, text="还在吗", display_name="Ad")
    )
    assert len(llm.calls) == 1
    assert len(notifier.followups) == 1

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_dryrun_mode_has_zero_telegram_side_effects(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.SPAM,
                confidence=0.95,
                category="promo_ad",
                reason="明显推广",
            )
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.DRYRUN), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=2003, tg_msg_id=1, text="推广消息", display_name="Dry")
    )
    await asyncio.sleep(0.12)

    assert tg.archived == []
    assert tg.muted == []
    assert tg.blocked == []
    assert tg.probes_sent == []
    assert len(notifier.dryrun_cards) == 1
    assert notifier.dryrun_cards[0]["would_action"] == "auto_block"

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_probe_limit_prevents_infinite_bot_loop(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    ask_verdict = LLMVerdict(
        decision=DecisionType.ASK,
        confidence=0.6,
        category="unknown_intent",
        reason="只打招呼",
        reply_to_user="请问哪位？请说明来意。",
    )
    llm = FakeLLM([ask_verdict, ask_verdict, ask_verdict])

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    # Round 1 -> probe 1
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=3001, tg_msg_id=1, text="你好", display_name="BotLoop")
    )
    await asyncio.sleep(0.12)
    assert len(tg.probes_sent) == 1

    # Round 2 -> probe 2
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=3001, tg_msg_id=2, text="在吗", display_name="BotLoop")
    )
    await asyncio.sleep(0.12)
    assert len(tg.probes_sent) == 2

    # Round 3 -> exceeds max_probe_rounds (2), must stop replying and escalate to review!
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=3001, tg_msg_id=3, text="交个朋友", display_name="BotLoop")
    )
    await asyncio.sleep(0.12)
    assert len(tg.probes_sent) == 2
    assert len(notifier.review_cards) == 1
    user = await db.get_user(3001)
    assert user is not None and user.status == UserStatus.PENDING_REVIEW

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_human_takeover_cancels_debounce_and_whitelists(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM()

    cfg = _fast_cfg(RunMode.AUTO)
    cfg.screening.debounce_seconds = 0.5
    pipeline = ModerationPipeline(
        cfg=cfg, db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=4001, tg_msg_id=1, text="哥们我是小王", display_name="Wang")
    )
    await pipeline.handle_owner_outgoing(4001, "哈哈你好！")
    await asyncio.sleep(0.6)

    assert len(llm.calls) == 0
    assert (4001, False) in tg.archived
    assert (4001, False) in tg.muted
    user = await db.get_user(4001)
    assert user is not None and user.status == UserStatus.WHITELISTED

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_high_risk_attachment_blocks_auto_pass(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.PASS,
                confidence=0.95,
                category="normal",
                reason="自称发文档",
            )
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await pipeline.handle_incoming_message(
        IncomingMessage(
            user_id=5001,
            tg_msg_id=1,
            text="这是资料请查收",
            display_name="Phish",
            media=MediaItem(
                media_type="document",
                file_name="资料.apk",
                is_high_risk=True,
                description="高危文件附件",
            ),
        )
    )
    await asyncio.sleep(0.12)

    assert len(notifier.passed_cards) == 0
    assert len(notifier.review_cards) == 1
    user = await db.get_user(5001)
    assert user is not None and user.status == UserStatus.PENDING_REVIEW

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_local_rule_short_circuits_and_budget_breaker_and_rate_limit(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    await db.seed_default_rules_if_empty(
        [DefaultRuleConfig(rule_type="keyword", pattern="必中彩票", action="spam", note="博彩")]
    )
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM()

    cfg = _fast_cfg(RunMode.AUTO)
    cfg.rate_limit.daily_llm_budget_calls = 1

    # Block limiter exhausted -> should degrade local spam rule to review instead of blocking!
    exhausted_block_limiter = SlidingWindowRateLimiter(max_events=0)
    pipeline = ModerationPipeline(
        cfg=cfg,
        db=db,
        tg=tg,
        notifier=notifier,
        llm=llm,  # type: ignore[arg-type]
        block_limiter=exhausted_block_limiter,
    )

    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=6001, tg_msg_id=1, text="私聊领取必中彩票", display_name="Bet")
    )
    await asyncio.sleep(0.12)

    assert len(llm.calls) == 0  # Local rule short-circuited
    assert tg.blocked == []  # Rate limiter degraded to review
    assert len(notifier.review_cards) == 1
    assert "限流" in notifier.review_cards[0]["extra_note"]

    # Budget circuit breaker check: consume the 1 allowed call, then next user shouldn't call LLM
    await db.increment_daily_llm_calls()
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=6002, tg_msg_id=2, text="你好咨询一下", display_name="User2")
    )
    await asyncio.sleep(0.12)

    assert len(llm.calls) == 0
    assert len(notifier.review_cards) == 2
    assert "预算上限" in notifier.review_cards[1]["verdict"].reason

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_auto_pass_releases_chat_and_marks_unread(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.PASS,
                confidence=0.92,
                category="normal",
                reason="在 Python 群看到你的开源项目，想请教部署问题",
            )
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await pipeline.handle_incoming_message(
        IncomingMessage(
            user_id=7001,
            tg_msg_id=1,
            text="你好，我在 Python 群看到你的项目，想问下 Docker 怎么配",
            display_name="Dev",
        )
    )
    await asyncio.sleep(0.12)

    assert (7001, True) in tg.archived
    assert (7001, False) in tg.archived
    assert (7001, False) in tg.muted
    assert 7001 in tg.marked_unread
    assert len(notifier.passed_cards) == 1
    user = await db.get_user(7001)
    assert user is not None and user.status == UserStatus.WHITELISTED

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_probe_reply_sanitized_and_fixed_mode_enforced(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.ASK,
                confidence=0.6,
                category="unknown_intent",
                reason="问来意",
                reply_to_user="请点击 https://evil.com 或联系 @scam 核实",
            ),
            LLMVerdict(
                decision=DecisionType.ASK,
                confidence=0.6,
                category="unknown_intent",
                reason="问来意",
                reply_to_user="这是一条正常的定制追问",
            ),
        ]
    )

    cfg = _fast_cfg(RunMode.AUTO)
    pipeline = ModerationPipeline(
        cfg=cfg, db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # Unsafe LLM reply containing URL & @ must fall back to static probe message
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=8001, tg_msg_id=1, text="你好", display_name="UserA")
    )
    await asyncio.sleep(0.12)
    assert tg.probes_sent[-1] == (8001, cfg.llm.fallback_probe_message)

    # Fixed probe mode forces static fallback even for clean LLM replies
    cfg.actions.use_fixed_probe_message = True
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=8002, tg_msg_id=2, text="你好", display_name="UserB")
    )
    await asyncio.sleep(0.12)
    assert tg.probes_sent[-1] == (8002, cfg.llm.fallback_probe_message)

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_userbot_send_probe_reply_disables_parse_mode_and_preview(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from antispam.config import AppConfig, EnvSettings, Settings
    from antispam.telegram.userbot import UserbotClient

    env = EnvSettings(
        tg_api_id=12345,
        tg_api_hash="hash",
        bot_token="123:abc",
        owner_id=999,
        llm_api_key="sk-test",
        tg_session_string="",
        data_dir=tmp_path,
    )
    cfg = AppConfig()
    cfg.actions.reply_delay_min_seconds = 0.0
    cfg.actions.reply_delay_max_seconds = 0.0
    cfg.actions.simulate_typing = False

    userbot = UserbotClient(Settings(env=env, cfg=cfg))
    userbot.client.get_input_entity = AsyncMock(return_value="peer_obj")  # type: ignore[method-assign]
    userbot.client.send_message = AsyncMock(return_value=SimpleNamespace(id=777))  # type: ignore[method-assign]

    msg_id = await userbot.send_probe_reply(123456, "**纯文本** 不渲染")
    assert msg_id == 777
    userbot.client.send_message.assert_awaited_once_with(
        "peer_obj",
        "**纯文本** 不渲染",
        parse_mode=None,
        link_preview=False,
    )
    userbot.client.session.close()
