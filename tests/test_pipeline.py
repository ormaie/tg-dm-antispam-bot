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
        self.fail_block: bool = False
        self.fail_probe: bool = False
        self.fail_unblock: bool = False
        self.fail_report: bool = False
        self._msg_seq = 5000

    async def has_owner_history(self, user_id: int, limit: int) -> bool:
        return user_id in self.owner_history_users

    async def fetch_user_profile(self, user_id: int) -> UserProfile | None:
        return self.profiles.get(
            user_id, UserProfile(user_id=user_id, first_name=f"User{user_id}")
        )

    async def resolve_user_id(self, identifier: str) -> tuple[int, str, str | None] | None:
        return None

    async def archive_chat(self, user_id: int, archive: bool) -> bool:
        self.archived.append((user_id, archive))
        return True

    async def mute_chat(self, user_id: int, mute: bool) -> bool:
        self.muted.append((user_id, mute))
        return True

    async def mark_unread(self, user_id: int) -> bool:
        self.marked_unread.append(user_id)
        return True

    async def send_probe_reply(self, user_id: int, text: str) -> int | None:
        if self.fail_probe:
            return None
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
    ) -> bool:
        if self.fail_block:
            return False
        self.blocked.append((user_id, delete_history, revoke, report_spam))
        return True

    async def unblock_user(self, user_id: int) -> bool:
        if self.fail_unblock:
            return False
        self.unblocked.append(user_id)
        return True

    async def report_spam(self, user_id: int) -> bool:
        if self.fail_report:
            return False
        self.reported.append(user_id)
        return True


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


@pytest.mark.asyncio
async def test_daily_llm_budget_atomic_under_50_concurrent_calls(tmp_path: Path) -> None:
    db = Database(tmp_path / "race.db")
    await db.connect()

    # 1. 50 concurrent increments must equal exactly 50 (no lost updates)
    results = await asyncio.gather(
        *[db.increment_daily_llm_calls(date_str="2026-01-01") for _ in range(50)]
    )
    assert len(results) == 50
    assert await db.get_daily_llm_calls(date_str="2026-01-01") == 50

    # 2. 50 concurrent spammer flushes against budget=10 must invoke LLM exactly 10 times
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM()
    cfg = _fast_cfg(RunMode.AUTO)
    cfg.screening.debounce_seconds = 0.02
    cfg.rate_limit.daily_llm_budget_calls = 10

    pipeline = ModerationPipeline(
        cfg=cfg, db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )
    await asyncio.gather(
        *[
            pipeline.handle_incoming_message(
                IncomingMessage(
                    user_id=9000 + i,
                    tg_msg_id=i + 1,
                    text=f"广告消息 {i}",
                    display_name=f"Spammer{i}",
                )
            )
            for i in range(50)
        ]
    )
    await asyncio.sleep(0.25)

    assert len(llm.calls) == 10
    assert await db.get_daily_llm_calls() == 10
    budget_tripped = [
        c for c in notifier.review_cards if c["verdict"].model_used == "budget_guard"
    ]
    assert len(budget_tripped) == 40

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_owner_outgoing_never_persists_friends_or_plaintext_messages(tmp_path: Path) -> None:
    db = Database(tmp_path / "privacy.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM()

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # Messaging 3 normal friends must create zero user rows and zero message rows
    for friend_id in (11, 12, 13):
        await pipeline.handle_owner_outgoing(friend_id, f"给朋友 {friend_id} 的私密消息")
        assert await db.get_user(friend_id) is None
        assert await db.get_recent_messages(friend_id) == []

    # Stranger enters screening, then owner manually takes over
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=14, tg_msg_id=1, text="你好我是同事", display_name="Colleague")
    )
    await pipeline.handle_owner_outgoing(14, "私密回复内容")
    await pipeline.handle_owner_outgoing(14, "第二条私密回复")

    user = await db.get_user(14)
    assert user is not None and user.status == UserStatus.WHITELISTED
    # Default purge_messages_on_whitelist=True wipes temporary screening messages too
    assert await db.get_recent_messages(14) == []

    # Even with purge disabled, owner outgoing plaintext is never stored
    pipeline.cfg.privacy.purge_messages_on_whitelist = False
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=15, tg_msg_id=2, text="我是另一个同事", display_name="Colleague2")
    )
    await pipeline.handle_owner_outgoing(15, "绝不落盘的私密回复")
    stored_msgs = await db.get_recent_messages(15)
    assert len(stored_msgs) == 1
    assert stored_msgs[0].direction == "in"
    assert all(m.direction != "out_owner" for m in stored_msgs)

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_telegram_rpc_failures_degrade_to_review_without_corrupting_state(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "rpc_fail.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.SPAM,
                confidence=0.98,
                category="promo_ad",
                reason="广告",
            ),
            LLMVerdict(
                decision=DecisionType.ASK,
                confidence=0.6,
                category="unknown_intent",
                reason="问来意",
                reply_to_user="请问哪位？",
            ),
        ]
    )
    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # 1. Auto-block fails -> must NOT mark BLOCKED or send blocked card; degrades to review
    tg.fail_block = True
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9501, tg_msg_id=1, text="广告", display_name="AdFail")
    )
    await asyncio.sleep(0.12)
    u1 = await db.get_user(9501)
    assert u1 is not None and u1.status == UserStatus.PENDING_REVIEW
    assert len(notifier.blocked_cards) == 0
    assert len(notifier.review_cards) == 1
    assert "自动拉黑执行失败" in notifier.review_cards[0]["extra_note"]

    # Manual block failure also leaves status unchanged
    res_block = await pipeline.manual_block(9501)
    assert "失败" in res_block
    u1_after = await db.get_user(9501)
    assert u1_after is not None and u1_after.status == UserStatus.PENDING_REVIEW

    # 2. Auto-probe send fails -> must NOT increment probe_count or store out_bot message
    tg.fail_probe = True
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9502, tg_msg_id=2, text="你好", display_name="ProbeFail")
    )
    await asyncio.sleep(0.12)
    u2 = await db.get_user(9502)
    assert u2 is not None and u2.status == UserStatus.PENDING_REVIEW
    assert u2.probe_count == 0
    msgs2 = await db.get_recent_messages(9502)
    assert all(m.direction != "out_bot" for m in msgs2)
    assert "自动追问消息发送失败" in notifier.review_cards[1]["extra_note"]

    # 3. Unblock failure on manual_allow leaves BLOCKED status intact
    tg.fail_block = False
    await pipeline.manual_block(9501)
    tg.fail_unblock = True
    res_unblock = await pipeline.manual_allow(9501)
    assert "解封失败" in res_unblock
    u1_blocked = await db.get_user(9501)
    assert u1_blocked is not None and u1_blocked.status == UserStatus.BLOCKED

    await pipeline.shutdown()
    await db.close()


def test_userbot_sets_explicit_flood_sleep_threshold(tmp_path: Path) -> None:
    from antispam.config import AppConfig, EnvSettings, Settings
    from antispam.telegram.notifier import HelperBotNotifier
    from antispam.telegram.userbot import UserbotClient

    env = EnvSettings(
        tg_api_id=12345,
        tg_api_hash="hash",
        bot_token="123:abc",
        owner_id=999,
        llm_api_key="sk-test",
        data_dir=tmp_path,
    )
    settings = Settings(env=env, cfg=AppConfig())
    userbot = UserbotClient(settings)
    notifier = HelperBotNotifier(settings)
    try:
        assert userbot.client.flood_sleep_threshold == 5
        assert notifier.client.flood_sleep_threshold == 5
    finally:
        userbot.client.session.close()
        notifier.client.session.close()


@pytest.mark.asyncio
async def test_stale_card_buttons_cannot_reprobe_or_report_resolved_users(tmp_path: Path) -> None:
    db = Database(tmp_path / "stale_cards.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM()

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.REVIEW_ONLY), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # 1. Unknown user cannot be probed
    res_unknown = await pipeline.manual_probe_again(99999)
    assert "未找到" in res_unknown
    assert tg.probes_sent == []

    # 2. Pending user can be probed cleanly
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9601, tg_msg_id=1, text="你好", display_name="FriendA")
    )
    await asyncio.sleep(0.12)
    res_ok = await pipeline.manual_probe_again(9601)
    assert "已向对方发送" in res_ok
    assert len(tg.probes_sent) == 1

    # 3. Allow user -> clicking stale "probe again" or "report_only" must be rejected!
    await pipeline.manual_allow(9601)
    res_stale_probe = await pipeline.manual_probe_again(9601)
    assert "已处理" in res_stale_probe
    assert len(tg.probes_sent) == 1  # no extra message sent to whitelisted friend
    user_after_probe = await db.get_user(9601)
    assert user_after_probe is not None and user_after_probe.status == UserStatus.WHITELISTED

    res_stale_report = await pipeline.manual_report_only(9601)
    assert "已放行加白" in res_stale_report
    assert tg.reported == []

    # 4. Blocked user -> clicking stale "probe again" must also be rejected!
    await pipeline.manual_block(9601)
    res_blocked_probe = await pipeline.manual_probe_again(9601)
    assert "已处理" in res_blocked_probe
    assert len(tg.probes_sent) == 1
    user_after_block = await db.get_user(9601)
    assert user_after_block is not None and user_after_block.status == UserStatus.BLOCKED

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_edited_message_catches_bait_and_switch_during_probation(tmp_path: Path) -> None:
    db = Database(tmp_path / "edit_probation.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.PASS,
                confidence=0.92,
                category="normal",
                reason="自称群友请教问题",
            ),
            LLMVerdict(
                decision=DecisionType.SPAM,
                confidence=0.97,
                category="promo_ad",
                reason="放行后立刻编辑原消息为博彩广告链接",
            ),
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # 1. Stranger passes initial screening -> WHITELISTED with auto_passed_at set
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9701, tg_msg_id=10, text="我在群里看到你的项目想请教", display_name="Bait")
    )
    await asyncio.sleep(0.12)
    u = await db.get_user(9701)
    assert u is not None and u.status == UserStatus.WHITELISTED and u.auto_passed_at is not None

    # 2. Stranger edits message to spam within probation window -> re-screened & auto-blocked!
    await pipeline.handle_edited_message(
        IncomingMessage(
            user_id=9701,
            tg_msg_id=10,
            text="澳门赌场高返水点击链接",
            display_name="Bait",
            is_edited=True,
        )
    )
    await asyncio.sleep(0.12)
    u_after = await db.get_user(9701)
    assert u_after is not None and u_after.status == UserStatus.BLOCKED
    assert len(tg.blocked) == 1
    assert len(notifier.blocked_cards) == 1

    # 3. Manually whitelisted friend (auto_passed_at=None) editing a typo must NOT trigger LLM
    await pipeline.manual_allow(9702, display_name="RealFriend")
    await pipeline.handle_edited_message(
        IncomingMessage(
            user_id=9702,
            tg_msg_id=20,
            text="修正一个错别字",
            display_name="RealFriend",
            is_edited=True,
        )
    )
    await asyncio.sleep(0.12)
    assert len(llm.calls) == 2  # still only the 2 calls for 9701
    friend = await db.get_user(9702)
    assert friend is not None and friend.status == UserStatus.WHITELISTED

    await pipeline.shutdown()
    await db.close()


def test_parse_duration_seconds() -> None:
    from antispam.core.pipeline import format_remaining, parse_duration_seconds

    assert parse_duration_seconds("30m") == 1800
    assert parse_duration_seconds("2h") == 7200
    assert parse_duration_seconds("1h30m") == 5400
    assert parse_duration_seconds("1d") == 86400
    assert parse_duration_seconds("1m") is None  # below 5m min
    assert parse_duration_seconds("30d") is None  # above 7d max
    assert parse_duration_seconds("abc") is None
    assert format_remaining(5400) == "1小时30分钟"


@pytest.mark.asyncio
async def test_lockdown_relax_budget_override_and_reset_user(tmp_path: Path) -> None:
    db = Database(tmp_path / "dynamic_modes.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            # Confidence 0.68 is below default pass_auto_allow (0.75),
            # but >= relaxed threshold (0.65), so it should auto-pass during /relax!
            LLMVerdict(
                decision=DecisionType.PASS,
                confidence=0.68,
                category="normal",
                reason="符合临时招聘场景的求职联系",
            )
        ]
    )

    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # 1. Lockdown mode: holds stranger silently with ZERO LLM calls and ZERO review cards
    await pipeline.set_lockdown(3600)
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9801, tg_msg_id=1, text="广告轰炸", display_name="Flood1")
    )
    await asyncio.sleep(0.12)
    assert len(llm.calls) == 0
    assert len(notifier.review_cards) == 0
    assert (9801, True) in tg.archived
    u_locked = await db.get_user(9801)
    assert u_locked is not None and u_locked.status == UserStatus.PENDING_REVIEW

    # 2. Enabling relax automatically clears lockdown and injects relax_note + lowers pass threshold
    await pipeline.set_relax(3600, "刚在前端群发了招聘，放行求职咨询")
    assert await pipeline.get_lockdown_state() is None
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9802, tg_msg_id=2, text="你好看到你的招聘贴", display_name="Candidate")
    )
    await asyncio.sleep(0.12)
    assert len(llm.calls) == 1
    assert llm.calls[0]["relax_note"] == "刚在前端群发了招聘，放行求职咨询"
    u_relaxed = await db.get_user(9802)
    assert u_relaxed is not None and u_relaxed.status == UserStatus.WHITELISTED

    # 3. Hot budget override takes effect immediately
    assert await pipeline.set_daily_budget_override(1) == 1
    # Budget is already 1/1 -> next stranger hits budget guard without calling LLM
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9803, tg_msg_id=3, text="你好", display_name="BudgetHit")
    )
    await asyncio.sleep(0.12)
    assert len(llm.calls) == 1
    assert any(c["verdict"].model_used == "budget_guard" for c in notifier.review_cards)

    # 4. Whitelist removal & reset_user clean DB rows cleanly
    removed, in_static = await pipeline.remove_from_whitelist(9802)
    assert removed is True and in_static is False
    assert await db.get_user(9802) is None
    assert await pipeline.reset_user(9801) is True
    assert await db.get_user(9801) is None
    assert await db.get_recent_messages(9801) == []

    await pipeline.shutdown()
    await db.close()


@pytest.mark.asyncio
async def test_privacy_purge_retention_cleanup_and_probe_timeout(tmp_path: Path) -> None:
    db = Database(tmp_path / "maintenance.db")
    await db.connect()
    tg = FakeTelegramActions()
    notifier = FakeNotifier()
    llm = FakeLLM(
        [
            LLMVerdict(
                decision=DecisionType.PASS,
                confidence=0.95,
                category="normal",
                reason="开源项目用户咨询",
            ),
            LLMVerdict(
                decision=DecisionType.ASK,
                confidence=0.6,
                category="unknown_intent",
                reason="只打招呼",
                reply_to_user="请说明来意",
            ),
            LLMVerdict(
                decision=DecisionType.PASS,
                confidence=0.92,
                category="normal",
                reason="几天后回来清晰说明了来意",
            ),
        ]
    )
    pipeline = ModerationPipeline(
        cfg=_fast_cfg(RunMode.AUTO), db=db, tg=tg, notifier=notifier, llm=llm  # type: ignore[arg-type]
    )

    # 1. Auto-pass purges temporary screening messages automatically
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9901, tg_msg_id=1, text="你好咨询部署问题", display_name="User1")
    )
    await asyncio.sleep(0.12)
    assert await db.get_recent_messages(9901) == []

    # 2. Stale probing user gets swept to EXPIRED after timeout, and can still resume later
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9902, tg_msg_id=2, text="你好", display_name="Ghost")
    )
    await asyncio.sleep(0.12)
    u_probing = await db.get_user(9902)
    assert u_probing is not None and u_probing.status == UserStatus.PROBING

    # Backdate user updated_at + old message + old audit log to 40 days ago
    old_ts = "2020-01-01T00:00:00+00:00"
    async with db._lock:
        await db.conn.execute(
            "UPDATE users SET updated_at = ? WHERE user_id = ?", (old_ts, 9902)
        )
        await db.conn.execute(
            "INSERT INTO messages (user_id, tg_msg_id, direction, text, created_at) VALUES (?, ?, 'in', 'old', ?)",
            (9999, 99, old_ts),
        )
        await db.conn.execute(
            """
            INSERT INTO audit_logs (user_id, decision, confidence, category, reason, action_taken, created_at)
            VALUES (?, 'review', 0.5, 'unknown_intent', 'old', 'review', ?)
            """,
            (9999, old_ts),
        )
        await db.conn.commit()

    summary = await pipeline.run_maintenance_once()
    assert summary["expired_probes"] == 1
    assert summary["messages_deleted"] >= 1
    assert summary["audits_deleted"] >= 1

    u_expired = await db.get_user(9902)
    assert u_expired is not None and u_expired.status == UserStatus.EXPIRED
    assert await db.get_recent_messages(9999) == []

    # Returning EXPIRED user sends real explanation -> resumes screening & passes
    await pipeline.handle_incoming_message(
        IncomingMessage(user_id=9902, tg_msg_id=3, text="不好意思前几天忙，我在开源群看到你", display_name="Ghost")
    )
    await asyncio.sleep(0.12)
    u_returned = await db.get_user(9902)
    assert u_returned is not None and u_returned.status == UserStatus.WHITELISTED

    await pipeline.shutdown()
    await db.close()
