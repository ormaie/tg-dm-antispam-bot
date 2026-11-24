from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from antispam.models import RunMode


class ScreeningConfig(BaseModel):
    debounce_seconds: float = Field(default=5.0, ge=0.0, le=60.0)
    max_wait_seconds: float = Field(default=15.0, ge=0.0, le=120.0)
    max_probe_rounds: int = Field(default=2, ge=0, le=10)
    auto_archive: bool = True
    auto_mute: bool = True
    mark_unread_on_pass: bool = True
    ignore_bots: bool = True
    history_check_limit: int = Field(default=10, ge=1, le=50)
    profile_cache_ttl_seconds: int = Field(default=1800, ge=60)
    edit_probation_minutes: int = Field(default=30, ge=0, le=1440)
    whitelist_user_ids: list[int] = Field(default_factory=list)
    whitelist_usernames: list[str] = Field(default_factory=list)

    @field_validator("whitelist_usernames")
    @classmethod
    def normalize_usernames(cls, values: list[str]) -> list[str]:
        return [v.lstrip("@").strip().lower() for v in values if v and v.strip()]


class ThresholdConfig(BaseModel):
    spam_auto_block: float = Field(default=0.85, ge=0.5, le=1.0)
    spam_review_min: float = Field(default=0.55, ge=0.0, le=1.0)
    pass_auto_allow: float = Field(default=0.75, ge=0.5, le=1.0)


class ActionConfig(BaseModel):
    auto_block_delete_history: bool = True
    auto_block_revoke_both: bool = True
    auto_block_report_spam: bool = False
    use_fixed_probe_message: bool = False
    max_probe_reply_chars: int = Field(default=200, ge=40, le=500)
    reply_delay_min_seconds: float = Field(default=2.0, ge=0.0, le=30.0)
    reply_delay_max_seconds: float = Field(default=5.0, ge=0.0, le=60.0)
    simulate_typing: bool = True


class RateLimitConfig(BaseModel):
    max_replies_per_minute: int = Field(default=6, ge=1, le=60)
    max_blocks_per_minute: int = Field(default=10, ge=1, le=60)
    daily_llm_budget_calls: int = Field(default=200, ge=1, le=100000)


class MediaConfig(BaseModel):
    enable_vision: bool = True
    max_images_per_batch: int = Field(default=3, ge=0, le=10)
    image_max_dimension: int = Field(default=768, ge=128, le=2048)
    image_jpeg_quality: int = Field(default=75, ge=30, le=95)
    high_risk_extensions: list[str] = Field(
        default_factory=lambda: [
            ".apk",
            ".exe",
            ".scr",
            ".bat",
            ".cmd",
            ".msi",
            ".vbs",
            ".ps1",
            ".lnk",
            ".dmg",
            ".iso",
        ]
    )

    @field_validator("high_risk_extensions")
    @classmethod
    def normalize_exts(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for ext in values:
            cleaned = ext.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            normalized.append(cleaned)
        return normalized


class LLMConfig(BaseModel):
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=180.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=100, le=4096)
    owner_context: str = (
        "机主平时在 Telegram 上进行日常社交、技术交流与正常项目合作。"
        "放行对象：明确说明来意的朋友/熟人、共同群组里的正常群友交流、合理的技术咨询或正规商务联系。"
        "重点拦截：加密货币/合约带单/U商兑换、博彩棋牌/体育盘口、色情约炮/同城空降/裸聊、"
        "刷单兼职/灰产引流、代开会员/卖号/机场推广、冒充客服或美女搭讪的杀猪盘、发送可疑安装包或钓鱼链接。"
    )
    assistant_persona: str = (
        "你是机主的私聊自动接待防广告助手。请礼貌、简明地表明身份，请对方简单说明来意"
        "（例如从哪个群找到、有什么事情），并说明确认后会立刻通知机主。使用与对方相同的语言回复"
        "（若对方仅发贴纸/表情则默认中文），长度控制在 1~2 句话以内，不与对方闲聊或争辩。"
    )
    fallback_probe_message: str = (
        "你好，这是自动接待助手。为了防止广告打扰，请简单说明一下你的来意"
        "（例如从哪个群找到的、有什么事），确认后我会立刻通知机主。"
    )


class DefaultRuleConfig(BaseModel):
    rule_type: Literal["keyword", "regex"] = "keyword"
    pattern: str
    action: Literal["spam", "review"] = "review"
    note: str = ""


class AppConfig(BaseModel):
    mode: RunMode = RunMode.REVIEW_ONLY
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    actions: ActionConfig = Field(default_factory=ActionConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    default_rules: list[DefaultRuleConfig] = Field(default_factory=list)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tg_api_id: int
    tg_api_hash: str
    tg_phone: str | None = None
    tg_session_string: str | None = None

    bot_token: str
    owner_id: int

    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_text_model: str = "gpt-4o-mini"
    llm_vision_model: str = "gpt-4o-mini"

    config_path: Path = Path("config.yaml")
    data_dir: Path = Path("data")
    log_level: str = "INFO"


@dataclass(slots=True)
class Settings:
    env: EnvSettings
    cfg: AppConfig

    @property
    def data_dir(self) -> Path:
        return self.env.data_dir

    @property
    def db_path(self) -> Path:
        return self.env.data_dir / "antispam.db"

    @property
    def user_session_path(self) -> Path:
        return self.env.data_dir / "userbot"

    @property
    def bot_session_path(self) -> Path:
        return self.env.data_dir / "helper_bot"


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def load_settings(config_path: Path | None = None) -> Settings:
    env = EnvSettings()  # type: ignore[call-arg]
    cfg_path = config_path or env.config_path
    cfg = load_app_config(cfg_path)
    env.data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(env=env, cfg=cfg)
