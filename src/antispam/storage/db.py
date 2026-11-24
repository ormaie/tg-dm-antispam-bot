from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from antispam.config import DefaultRuleConfig
from antispam.models import (
    LocalRule,
    RunMode,
    StoredMessage,
    StoredUser,
    UserStatus,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def utc_today_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL,
    probe_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    muted INTEGER NOT NULL DEFAULT 0,
    last_category TEXT,
    last_confidence REAL,
    last_reason TEXT,
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    auto_passed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(LOWER(username));

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    tg_msg_id INTEGER,
    direction TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    media_type TEXT,
    media_meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id, id DESC);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    decision TEXT NOT NULL,
    confidence REAL NOT NULL,
    category TEXT NOT NULL,
    reason TEXT NOT NULL,
    reply_text TEXT NOT NULL DEFAULT '',
    action_taken TEXT NOT NULL,
    model_used TEXT NOT NULL DEFAULT '',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    dry_run INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,
    pattern TEXT NOT NULL,
    action TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected")
        return self._conn

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(SCHEMA_SQL)
        await self._ensure_migrations()
        await self._conn.commit()

    async def _ensure_migrations(self) -> None:
        async with self.conn.execute("PRAGMA table_info(users)") as cur:
            cols = {row["name"] for row in await cur.fetchall()}
        if "auto_passed_at" not in cols:
            await self.conn.execute("ALTER TABLE users ADD COLUMN auto_passed_at TEXT")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ---------- Users ----------

    @staticmethod
    def _row_to_user(row: aiosqlite.Row) -> StoredUser:
        return StoredUser(
            user_id=row["user_id"],
            username=row["username"],
            display_name=row["display_name"],
            status=UserStatus(row["status"]),
            probe_count=row["probe_count"],
            archived=bool(row["archived"]),
            muted=bool(row["muted"]),
            last_category=row["last_category"],
            last_confidence=row["last_confidence"],
            last_reason=row["last_reason"],
            first_seen_at=row["first_seen_at"],
            updated_at=row["updated_at"],
            auto_passed_at=row["auto_passed_at"],
        )

    async def _get_user_unlocked(self, user_id: int) -> StoredUser | None:
        async with self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
            return self._row_to_user(row) if row else None

    async def get_user(self, user_id: int) -> StoredUser | None:
        async with self._lock:
            return await self._get_user_unlocked(user_id)

    async def find_user_by_username(self, username: str) -> StoredUser | None:
        clean = username.lstrip("@").strip().lower()
        async with self._lock, self.conn.execute(
            "SELECT * FROM users WHERE LOWER(username) = ? LIMIT 1", (clean,)
        ) as cur:
            row = await cur.fetchone()
            return self._row_to_user(row) if row else None

    async def get_or_create_user(
        self,
        user_id: int,
        display_name: str,
        username: str | None = None,
        default_status: UserStatus = UserStatus.NEW,
    ) -> StoredUser:
        now = utc_now_iso()
        clean_username = username.lstrip("@") if username else None
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO users (user_id, username, display_name, status, first_seen_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    display_name = excluded.display_name,
                    updated_at = excluded.updated_at
                WHERE users.username IS NOT excluded.username
                   OR users.display_name IS NOT excluded.display_name
                """,
                (user_id, clean_username, display_name, default_status.value, now, now),
            )
            await self.conn.commit()
            user = await self._get_user_unlocked(user_id)
            assert user is not None
            return user

    async def update_user_status(
        self,
        user_id: int,
        status: UserStatus,
        *,
        category: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        reset_probes: bool = False,
        auto_passed_at: str | None = None,
        clear_auto_passed: bool = False,
    ) -> None:
        now = utc_now_iso()
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, now]
        if category is not None:
            fields.append("last_category = ?")
            params.append(category)
        if confidence is not None:
            fields.append("last_confidence = ?")
            params.append(confidence)
        if reason is not None:
            fields.append("last_reason = ?")
            params.append(reason)
        if reset_probes:
            fields.append("probe_count = 0")
        if auto_passed_at is not None:
            fields.append("auto_passed_at = ?")
            params.append(auto_passed_at)
        elif clear_auto_passed:
            fields.append("auto_passed_at = NULL")
        params.append(user_id)
        async with self._lock:
            await self.conn.execute(
                f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?", params
            )
            await self.conn.commit()

    async def increment_probe_count(self, user_id: int) -> int:
        now = utc_now_iso()
        async with self._lock:
            await self.conn.execute(
                "UPDATE users SET probe_count = probe_count + 1, updated_at = ? WHERE user_id = ?",
                (now, user_id),
            )
            await self.conn.commit()
            user = await self._get_user_unlocked(user_id)
            return user.probe_count if user else 0

    async def set_user_flags(
        self,
        user_id: int,
        *,
        archived: bool | None = None,
        muted: bool | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []
        if archived is not None:
            fields.append("archived = ?")
            params.append(int(archived))
        if muted is not None:
            fields.append("muted = ?")
            params.append(int(muted))
        if not fields:
            return
        fields.append("updated_at = ?")
        params.append(utc_now_iso())
        params.append(user_id)
        async with self._lock:
            await self.conn.execute(
                f"UPDATE users SET {', '.join(fields)} WHERE user_id = ?", params
            )
            await self.conn.commit()

    async def list_users_by_status(self, status: UserStatus, limit: int = 20) -> list[StoredUser]:
        async with self._lock, self.conn.execute(
            "SELECT * FROM users WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
            (status.value, limit),
        ) as cur:
            rows = await cur.fetchall()
            return [self._row_to_user(row) for row in rows]

    async def delete_user_and_messages(self, user_id: int) -> bool:
        async with self._lock:
            await self.conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            async with self.conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,)) as cur:
                deleted = cur.rowcount > 0
            await self.conn.commit()
            return deleted

    async def delete_user_messages(self, user_id: int) -> int:
        async with self._lock:
            async with self.conn.execute(
                "DELETE FROM messages WHERE user_id = ?", (user_id,)
            ) as cur:
                count = cur.rowcount or 0
            await self.conn.commit()
            return count

    async def list_stale_probing_users(self, cutoff_iso: str, limit: int = 100) -> list[StoredUser]:
        async with self._lock, self.conn.execute(
            "SELECT * FROM users WHERE status = ? AND updated_at <= ? ORDER BY updated_at ASC LIMIT ?",
            (UserStatus.PROBING.value, cutoff_iso, limit),
        ) as cur:
            rows = await cur.fetchall()
            return [self._row_to_user(row) for row in rows]

    async def cleanup_expired_records(
        self,
        *,
        message_cutoff_iso: str | None = None,
        audit_cutoff_iso: str | None = None,
    ) -> dict[str, int]:
        messages_deleted = 0
        audits_deleted = 0
        async with self._lock:
            if message_cutoff_iso is not None:
                async with self.conn.execute(
                    "DELETE FROM messages WHERE created_at <= ?", (message_cutoff_iso,)
                ) as cur:
                    messages_deleted = cur.rowcount or 0
            if audit_cutoff_iso is not None:
                async with self.conn.execute(
                    "DELETE FROM audit_logs WHERE created_at <= ?", (audit_cutoff_iso,)
                ) as cur:
                    audits_deleted = cur.rowcount or 0
            await self.conn.commit()
        return {"messages_deleted": messages_deleted, "audits_deleted": audits_deleted}

    # ---------- Messages ----------

    async def add_message(
        self,
        user_id: int,
        direction: str,
        text: str,
        *,
        tg_msg_id: int | None = None,
        media_type: str | None = None,
        media_meta: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now_iso()
        meta_str = json.dumps(media_meta or {}, ensure_ascii=False)
        async with self._lock:
            async with self.conn.execute(
                """
                INSERT INTO messages (user_id, tg_msg_id, direction, text, media_type, media_meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, tg_msg_id, direction, text, media_type, meta_str, now),
            ) as cur:
                lastrowid = cur.lastrowid or 0
            await self.conn.commit()
            return lastrowid

    async def get_recent_messages(self, user_id: int, limit: int = 20) -> list[StoredMessage]:
        async with self._lock, self.conn.execute(
            "SELECT * FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        result: list[StoredMessage] = []
        for row in reversed(rows):
            try:
                meta = json.loads(row["media_meta_json"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            result.append(
                StoredMessage(
                    id=row["id"],
                    user_id=row["user_id"],
                    tg_msg_id=row["tg_msg_id"],
                    direction=row["direction"],
                    text=row["text"],
                    media_type=row["media_type"],
                    media_meta=meta,
                    created_at=row["created_at"],
                )
            )
        return result

    # ---------- Audit Logs ----------

    async def add_audit_log(
        self,
        *,
        user_id: int,
        decision: str,
        confidence: float,
        category: str,
        reason: str,
        action_taken: str,
        reply_text: str = "",
        model_used: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        dry_run: bool = False,
    ) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO audit_logs (
                    user_id, decision, confidence, category, reason, reply_text,
                    action_taken, model_used, prompt_tokens, completion_tokens, dry_run, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    decision,
                    confidence,
                    category,
                    reason,
                    reply_text,
                    action_taken,
                    model_used,
                    prompt_tokens,
                    completion_tokens,
                    int(dry_run),
                    utc_now_iso(),
                ),
            )
            await self.conn.commit()

    # ---------- Rules ----------

    @staticmethod
    def _row_to_rule(row: aiosqlite.Row) -> LocalRule:
        return LocalRule(
            id=row["id"],
            rule_type=row["rule_type"],
            pattern=row["pattern"],
            action=row["action"],
            note=row["note"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
        )

    async def seed_default_rules_if_empty(self, default_rules: list[DefaultRuleConfig]) -> None:
        if not default_rules:
            return
        async with self._lock:
            seeded = await self._get_runtime_state_unlocked("rules_seeded")
            if seeded == "1":
                return
            async with self.conn.execute("SELECT COUNT(*) AS cnt FROM rules") as cur:
                row = await cur.fetchone()
            if row and row["cnt"] == 0:
                now = utc_now_iso()
                for rule in default_rules:
                    await self.conn.execute(
                        """
                        INSERT INTO rules (rule_type, pattern, action, note, enabled, created_at)
                        VALUES (?, ?, ?, ?, 1, ?)
                        """,
                        (rule.rule_type, rule.pattern, rule.action, rule.note, now),
                    )
            await self.conn.execute(
                """
                INSERT INTO runtime_state (key, value) VALUES ('rules_seeded', '1')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )
            await self.conn.commit()

    async def list_rules(self, enabled_only: bool = True) -> list[LocalRule]:
        query = "SELECT * FROM rules"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY id ASC"
        async with self._lock, self.conn.execute(query) as cur:
            rows = await cur.fetchall()
            return [self._row_to_rule(row) for row in rows]

    async def add_rule(
        self,
        rule_type: str,
        pattern: str,
        action: str = "review",
        note: str = "",
    ) -> LocalRule:
        now = utc_now_iso()
        async with self._lock:
            async with self.conn.execute(
                """
                INSERT INTO rules (rule_type, pattern, action, note, enabled, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (rule_type, pattern, action, note, now),
            ) as cur:
                rule_id = cur.lastrowid
            await self.conn.commit()
            assert rule_id is not None
            async with self.conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)) as row_cur:
                row = await row_cur.fetchone()
            assert row is not None
            return self._row_to_rule(row)

    async def delete_rule(self, rule_id: int) -> bool:
        async with self._lock:
            async with self.conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,)) as cur:
                deleted = cur.rowcount > 0
            await self.conn.commit()
            return deleted

    # ---------- Runtime State & Stats ----------

    async def _get_runtime_state_unlocked(
        self, key: str, default: str | None = None
    ) -> str | None:
        async with self.conn.execute(
            "SELECT value FROM runtime_state WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
            return row["value"] if row else default

    async def _get_daily_llm_calls_unlocked(self, date_str: str | None = None) -> int:
        day = date_str or utc_today_str()
        raw = await self._get_runtime_state_unlocked(f"llm_calls:{day}", "0")
        try:
            return int(raw or "0")
        except ValueError:
            return 0

    async def get_runtime_state(self, key: str, default: str | None = None) -> str | None:
        async with self._lock:
            return await self._get_runtime_state_unlocked(key, default)

    async def set_runtime_state(self, key: str, value: str) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO runtime_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            await self.conn.commit()

    async def delete_runtime_state(self, key: str) -> None:
        async with self._lock:
            await self.conn.execute("DELETE FROM runtime_state WHERE key = ?", (key,))
            await self.conn.commit()

    async def get_daily_budget_override(self) -> int | None:
        raw = await self.get_runtime_state("daily_llm_budget_override")
        if raw is None:
            return None
        try:
            val = int(raw)
            return val if val >= 1 else None
        except ValueError:
            return None

    async def set_daily_budget_override(self, budget: int | None) -> None:
        if budget is None:
            await self.delete_runtime_state("daily_llm_budget_override")
        else:
            await self.set_runtime_state("daily_llm_budget_override", str(budget))

    async def get_timed_state(self, key: str, now_ts: float) -> dict[str, Any] | None:
        raw = await self.get_runtime_state(key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self.delete_runtime_state(key)
            return None
        if not isinstance(data, dict):
            await self.delete_runtime_state(key)
            return None
        until = float(data.get("until", 0) or 0)
        if until <= now_ts:
            await self.delete_runtime_state(key)
            return None
        data["remaining_seconds"] = max(0.0, until - now_ts)
        return data

    async def set_timed_state(self, key: str, payload: dict[str, Any] | None) -> None:
        if payload is None:
            await self.delete_runtime_state(key)
        else:
            await self.set_runtime_state(key, json.dumps(payload, ensure_ascii=False))

    async def get_run_mode(self, fallback: RunMode) -> RunMode:
        raw = await self.get_runtime_state("run_mode")
        if raw:
            try:
                return RunMode(raw)
            except ValueError:
                pass
        return fallback

    async def set_run_mode(self, mode: RunMode) -> None:
        await self.set_runtime_state("run_mode", mode.value)

    async def get_daily_llm_calls(self, date_str: str | None = None) -> int:
        async with self._lock:
            return await self._get_daily_llm_calls_unlocked(date_str)

    async def increment_daily_llm_calls(self, date_str: str | None = None) -> int:
        day = date_str or utc_today_str()
        key = f"llm_calls:{day}"
        async with self._lock:
            async with self.conn.execute(
                """
                INSERT INTO runtime_state (key, value) VALUES (?, '1')
                ON CONFLICT(key) DO UPDATE
                    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                RETURNING CAST(value AS INTEGER) AS new_val
                """,
                (key,),
            ) as cur:
                row = await cur.fetchone()
            await self.conn.commit()
            return int(row["new_val"]) if row else 1

    async def try_reserve_daily_llm_call(
        self, budget: int, date_str: str | None = None
    ) -> tuple[bool, int]:
        """Atomically check daily budget and reserve one LLM call slot."""
        day = date_str or utc_today_str()
        key = f"llm_calls:{day}"
        async with self._lock:
            if budget <= 0:
                current = await self._get_daily_llm_calls_unlocked(day)
                return False, current
            async with self.conn.execute(
                """
                INSERT INTO runtime_state (key, value) VALUES (?, '1')
                ON CONFLICT(key) DO UPDATE
                    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                    WHERE CAST(value AS INTEGER) < ?
                RETURNING CAST(value AS INTEGER) AS new_val
                """,
                (key, budget),
            ) as cur:
                row = await cur.fetchone()
            await self.conn.commit()
            if row is not None:
                return True, int(row["new_val"])
            current = await self._get_daily_llm_calls_unlocked(day)
            return False, current

    async def get_today_stats(self, date_str: str | None = None) -> dict[str, int]:
        day = date_str or utc_today_str()
        prefix = f"{day}%"
        async with self._lock:
            async with self.conn.execute(
                """
                SELECT
                    COUNT(*) AS total_audits,
                    SUM(CASE WHEN action_taken IN ('auto_block', 'manual_block', 'manual_report') THEN 1 ELSE 0 END) AS blocked,
                    SUM(CASE WHEN action_taken IN ('auto_pass', 'manual_allow') THEN 1 ELSE 0 END) AS passed,
                    SUM(CASE WHEN action_taken = 'probe' THEN 1 ELSE 0 END) AS probed,
                    SUM(CASE WHEN action_taken = 'review' THEN 1 ELSE 0 END) AS sent_to_review,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM audit_logs
                WHERE created_at LIKE ?
                """,
                (prefix,),
            ) as cur:
                row = await cur.fetchone()
            async with self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE status = ?",
                (UserStatus.PENDING_REVIEW.value,),
            ) as pending_cur:
                pending_row = await pending_cur.fetchone()
            llm_calls = await self._get_daily_llm_calls_unlocked(day)
        return {
            "total_audits": int(row["total_audits"] or 0) if row else 0,
            "blocked": int(row["blocked"] or 0) if row else 0,
            "passed": int(row["passed"] or 0) if row else 0,
            "probed": int(row["probed"] or 0) if row else 0,
            "sent_to_review": int(row["sent_to_review"] or 0) if row else 0,
            "prompt_tokens": int(row["prompt_tokens"] or 0) if row else 0,
            "completion_tokens": int(row["completion_tokens"] or 0) if row else 0,
            "pending_now": int(pending_row["cnt"] or 0) if pending_row else 0,
            "llm_calls": llm_calls,
        }
