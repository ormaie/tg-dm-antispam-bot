from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunMode(str, Enum):
    DRYRUN = "dryrun"
    REVIEW_ONLY = "review_only"
    AUTO = "auto"


class UserStatus(str, Enum):
    NEW = "NEW"
    PROBING = "PROBING"
    WHITELISTED = "WHITELISTED"
    PENDING_REVIEW = "PENDING_REVIEW"
    BLOCKED = "BLOCKED"


class DecisionType(str, Enum):
    PASS = "pass"
    ASK = "ask"
    REVIEW = "review"
    SPAM = "spam"


class SpamCategory(str, Enum):
    NORMAL = "normal"
    UNKNOWN_INTENT = "unknown_intent"
    PROMO_AD = "promo_ad"
    CRYPTO_SCAM = "crypto_scam"
    ADULT_SPAM = "adult_spam"
    ROMANCE_SCAM = "romance_scam"
    PHISHING = "phishing"
    PROMPT_INJECTION = "prompt_injection"
    HIGH_RISK_FILE = "high_risk_file"
    LOCAL_RULE = "local_rule"


@dataclass(slots=True)
class UserProfile:
    user_id: int
    first_name: str = ""
    last_name: str = ""
    username: str | None = None
    bio: str = ""
    is_premium: bool = False
    is_verified: bool = False
    is_scam: bool = False
    is_fake: bool = False
    has_photo: bool = False
    common_chats: list[str] = field(default_factory=list)
    fetched_at: float = 0.0

    @property
    def display_name(self) -> str:
        full = f"{self.first_name} {self.last_name}".strip()
        if full:
            return full
        if self.username:
            return f"@{self.username}"
        return str(self.user_id)


@dataclass(slots=True)
class MediaItem:
    media_type: str  # photo | video_thumb | gif_thumb | sticker | document | voice | audio | contact | geo | other
    image_bytes: bytes | None = None
    mime_type: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    is_high_risk: bool = False
    description: str = ""

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "mime_type": self.mime_type,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "is_high_risk": self.is_high_risk,
            "description": self.description,
            "has_image": self.image_bytes is not None,
        }


@dataclass(slots=True)
class IncomingMessage:
    user_id: int
    tg_msg_id: int
    text: str
    display_name: str
    username: str | None = None
    forward_source: str | None = None
    media: MediaItem | None = None
    timestamp: float = 0.0


@dataclass(slots=True)
class StoredUser:
    user_id: int
    username: str | None
    display_name: str
    status: UserStatus
    probe_count: int
    archived: bool
    muted: bool
    last_category: str | None
    last_confidence: float | None
    last_reason: str | None
    first_seen_at: str
    updated_at: str


@dataclass(slots=True)
class StoredMessage:
    id: int
    user_id: int
    tg_msg_id: int | None
    direction: str  # in | out_bot | out_owner
    text: str
    media_type: str | None
    media_meta: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class LocalRule:
    id: int
    rule_type: str  # keyword | regex
    pattern: str
    action: str  # spam | review
    note: str
    enabled: bool
    created_at: str


@dataclass(slots=True)
class LLMVerdict:
    decision: DecisionType
    confidence: float
    category: str
    reason: str
    reply_to_user: str = ""
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
