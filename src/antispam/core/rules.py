from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from antispam.models import IncomingMessage, LocalRule, UserProfile

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuleMatch:
    rule_id: int
    action: str  # spam | review
    pattern: str
    note: str
    matched_in: str


class RuleEngine:
    """Evaluates local keyword/regex rules against profile metadata and messages."""

    @staticmethod
    def validate_pattern(rule_type: str, pattern: str) -> str | None:
        if not pattern or not pattern.strip():
            return "规则内容不能为空"
        if rule_type == "regex":
            try:
                re.compile(pattern)
            except re.error as exc:
                return f"无效的正则表达式：{exc}"
        elif rule_type != "keyword":
            return f"不支持的规则类型：{rule_type}（仅支持 keyword 或 regex）"
        return None

    @staticmethod
    def _matches(rule: LocalRule, text: str) -> bool:
        if not text:
            return False
        if rule.rule_type == "keyword":
            return rule.pattern.lower() in text.lower()
        if rule.rule_type == "regex":
            try:
                return re.search(rule.pattern, text) is not None
            except re.error as exc:
                logger.warning("Skipping invalid regex rule #%s (%s): %s", rule.id, rule.pattern, exc)
                return False
        return False

    def evaluate(
        self,
        rules: list[LocalRule],
        profile: UserProfile | None,
        messages: list[IncomingMessage],
    ) -> RuleMatch | None:
        """Return highest-severity match (`spam` takes precedence over `review`)."""
        targets: list[tuple[str, str]] = []
        if profile:
            if profile.display_name:
                targets.append(("display_name", profile.display_name))
            if profile.bio:
                targets.append(("bio", profile.bio))
        for msg in messages:
            if msg.text:
                targets.append(("message", msg.text))
            if msg.forward_source:
                targets.append(("forward_source", msg.forward_source))

        review_match: RuleMatch | None = None
        for rule in rules:
            if not rule.enabled:
                continue
            for field_name, content in targets:
                if self._matches(rule, content):
                    match = RuleMatch(
                        rule_id=rule.id,
                        action=rule.action,
                        pattern=rule.pattern,
                        note=rule.note,
                        matched_in=field_name,
                    )
                    if rule.action == "spam":
                        return match
                    if review_match is None:
                        review_match = match
                    break
        return review_match
