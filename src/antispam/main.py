from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from antispam.config import load_settings
from antispam.core.llm import LLMModerator
from antispam.core.pipeline import ModerationPipeline
from antispam.storage.db import Database
from antispam.telegram.notifier import HelperBotNotifier
from antispam.telegram.userbot import UserbotClient

logger = logging.getLogger(__name__)


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def _run_service(config_path: Path | None) -> None:
    settings = load_settings(config_path)
    _setup_logging(settings.env.log_level)

    db = Database(settings.db_path)
    await db.connect()
    await db.seed_default_rules_if_empty(settings.cfg.default_rules)

    llm = LLMModerator(
        api_key=settings.env.llm_api_key,
        base_url=settings.env.llm_base_url,
        text_model=settings.env.llm_text_model,
        vision_model=settings.env.llm_vision_model,
        llm_cfg=settings.cfg.llm,
        media_cfg=settings.cfg.media,
        max_probe_rounds=settings.cfg.screening.max_probe_rounds,
    )

    userbot = UserbotClient(settings)
    notifier = HelperBotNotifier(settings)
    pipeline = ModerationPipeline(
        cfg=settings.cfg,
        db=db,
        tg=userbot,
        notifier=notifier,
        llm=llm,
    )
    userbot.bind_pipeline(pipeline)
    notifier.bind_pipeline(pipeline)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows event loop does not support add_signal_handler for SIGTERM
            pass

    try:
        bot_me = await notifier.start()
        userbot.set_helper_bot_id(bot_me.id)
        user_me = await userbot.start()

        current_mode = await pipeline.get_run_mode()
        user_label = f"{user_me.first_name or ''} {user_me.last_name or ''}".strip() or str(user_me.id)
        await notifier.send_startup_banner(current_mode, user_label)
        logger.info("Service is running (mode=%s). Press Ctrl+C to stop.", current_mode.value)

        user_task = asyncio.create_task(userbot.client.run_until_disconnected())
        bot_task = asyncio.create_task(notifier.client.run_until_disconnected())
        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {user_task, bot_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()  # type: ignore[misc]
    finally:
        logger.info("Shutting down gracefully...")
        await pipeline.shutdown()
        await userbot.stop()
        await notifier.stop()
        await llm.close()
        await db.close()
        logger.info("Shutdown complete.")


async def _run_login_only(config_path: Path | None, export_session: bool) -> None:
    settings = load_settings(config_path)
    _setup_logging(settings.env.log_level)
    userbot = UserbotClient(settings)
    await userbot.start_login_only(export_session=export_session)


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="antispam",
        description="Telegram DM Anti-Spam Userbot with multimodal LLM moderation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to CONFIG_PATH env var or ./config.yaml)",
    )
    parser.add_argument(
        "--login-only",
        action="store_true",
        help="Run interactive Telegram login to generate the userbot session file and exit.",
    )
    parser.add_argument(
        "--export-session",
        action="store_true",
        help="With --login-only, print a TG_SESSION_STRING for headless deployment.",
    )
    args = parser.parse_args(argv)

    try:
        if args.login_only:
            asyncio.run(_run_login_only(args.config, export_session=args.export_session))
        else:
            asyncio.run(_run_service(args.config))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli()
