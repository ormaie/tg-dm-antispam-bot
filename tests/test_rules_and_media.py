from __future__ import annotations

import io

from PIL import Image

from antispam.core.media import bytes_to_data_url, compress_image_to_jpeg, is_high_risk_filename
from antispam.core.ratelimit import SlidingWindowRateLimiter
from antispam.core.rules import RuleEngine
from antispam.models import IncomingMessage, LocalRule, UserProfile


def test_rule_engine_spam_overrides_review_and_checks_profile() -> None:
    engine = RuleEngine()
    rules = [
        LocalRule(1, "keyword", "代开", "review", "复核", True, "2026-01-01"),
        LocalRule(2, "regex", r"(?i)usdt.*搬砖", "spam", "币圈诈骗", True, "2026-01-01"),
        LocalRule(3, "regex", "[invalid(", "spam", "坏正则", True, "2026-01-01"),
    ]
    profile = UserProfile(user_id=100, first_name="客服", bio="专业代开会员")
    messages = [
        IncomingMessage(user_id=100, tg_msg_id=1, text="你好，USDT高息搬砖了解下", display_name="客服")
    ]

    match = engine.evaluate(rules, profile, messages)
    assert match is not None
    assert match.rule_id == 2
    assert match.action == "spam"
    assert match.matched_in == "message"


def test_rule_engine_matches_bio_and_forward_source() -> None:
    engine = RuleEngine()
    rules = [
        LocalRule(1, "keyword", "博彩频道", "review", "频道引流", True, "2026-01-01"),
    ]
    msg = IncomingMessage(
        user_id=101,
        tg_msg_id=2,
        text="看看这个",
        display_name="Bob",
        forward_source="澳门博彩频道 (@bet123)",
    )
    match = engine.evaluate(rules, None, [msg])
    assert match is not None
    assert match.matched_in == "forward_source"
    assert match.action == "review"


def test_rule_engine_matches_personal_channel_title_and_about() -> None:
    engine = RuleEngine()
    rules = [
        LocalRule(1, "keyword", "合约带单", "spam", "个人频道推广", True, "2026-01-01"),
    ]
    profile = UserProfile(
        user_id=102,
        first_name="普通人",
        bio="",
        personal_channel_title="VIP 交流",
        personal_channel_about="专注加密货币合约带单，私聊进群",
    )
    msg = IncomingMessage(user_id=102, tg_msg_id=3, text="你好", display_name="普通人")
    match = engine.evaluate(rules, profile, [msg])
    assert match is not None
    assert match.matched_in == "personal_channel_about"
    assert match.action == "spam"


def test_rule_engine_validate_pattern() -> None:
    assert RuleEngine.validate_pattern("keyword", "推广") is None
    assert RuleEngine.validate_pattern("regex", r"(?i)spam\d+") is None
    assert RuleEngine.validate_pattern("regex", "[unclosed") is not None
    assert RuleEngine.validate_pattern("unknown", "abc") is not None


def test_compress_image_to_jpeg_resizes_rgba_png() -> None:
    img = Image.new("RGBA", (1600, 800), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    compressed = compress_image_to_jpeg(buf.getvalue(), max_dimension=512, quality=75)
    assert compressed is not None

    with Image.open(io.BytesIO(compressed)) as out_img:
        assert out_img.format == "JPEG"
        assert out_img.mode == "RGB"
        assert max(out_img.size) == 512
        assert out_img.size == (512, 256)

    data_url = bytes_to_data_url(compressed)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_compress_image_handles_invalid_bytes() -> None:
    assert compress_image_to_jpeg(b"not an image") is None
    assert compress_image_to_jpeg(b"") is None


def test_is_high_risk_filename() -> None:
    exts = [".apk", ".exe", ".scr"]
    assert is_high_risk_filename("wallet_update.APK", exts) is True
    assert is_high_risk_filename("invoice.pdf.exe", exts) is True
    assert is_high_risk_filename("report.pdf", exts) is False
    assert is_high_risk_filename(None, exts) is False


def test_sliding_window_rate_limiter() -> None:
    now = 100.0
    limiter = SlidingWindowRateLimiter(max_events=2, window_seconds=60.0, clock=lambda: now)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False
    assert limiter.remaining() == 0

    now = 161.0
    assert limiter.remaining() == 2
    assert limiter.allow() is True


def test_telethon_tl_objects_serialize() -> None:
    from telethon.extensions import html as tg_html
    from telethon.tl.functions.account import UpdateNotifySettingsRequest
    from telethon.tl.functions.folders import EditPeerFoldersRequest
    from telethon.tl.functions.messages import MarkDialogUnreadRequest
    from telethon.tl.types import (
        InputDialogPeer,
        InputFolderPeer,
        InputNotifyPeer,
        InputPeerNotifySettings,
        InputPeerUser,
    )

    peer = InputPeerUser(user_id=123456, access_hash=0)
    assert bytes(
        UpdateNotifySettingsRequest(
            peer=InputNotifyPeer(peer=peer),
            settings=InputPeerNotifySettings(mute_until=2147483647, silent=True),
        )
    )
    assert bytes(
        UpdateNotifySettingsRequest(
            peer=InputNotifyPeer(peer=peer),
            settings=InputPeerNotifySettings(mute_until=0, silent=False),
        )
    )
    assert bytes(
        EditPeerFoldersRequest(folder_peers=[InputFolderPeer(peer=peer, folder_id=1)])
    )
    assert bytes(
        MarkDialogUnreadRequest(peer=InputDialogPeer(peer=peer), unread=True)
    )
    parsed_text, entities = tg_html.parse("<b>测试</b> <code>123</code>")
    assert tg_html.unparse(parsed_text, entities) == "<strong>测试</strong> <code>123</code>"
