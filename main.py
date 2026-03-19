#!/usr/bin/env python3
"""
AI Trading Bot - Main Entry Point

Connects to Interactive Brokers via IB Gateway, analyzes stocks using
TradingView indicators and AI chart analysis, executes trades with
automated risk management, and provides a Telegram bot interface.

Usage:
    python main.py              # Run the full trading bot
    python main.py --dashboard  # Run only the Streamlit dashboard
    python main.py --backtest AAPL  # Run a quick backtest
"""

import argparse
import asyncio
import signal
import subprocess
import sys
from datetime import datetime

# ib_insync's sync methods (qualifyContracts, placeOrder, etc.) internally
# run the event loop, which fails if asyncio already has a loop running.
# nest_asyncio patches asyncio to allow nested event loop calls.
import nest_asyncio
nest_asyncio.apply()

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.utils.config import get_config, TRADING_MODE
from src.utils.database import init_db
from src.utils.logger import setup_logger
from src.strategy.trading_engine import TradingEngine
from src.telegram_bot.bot import TradingBot
from src.backtest.backtester import Backtester

logger = setup_logger("main")


async def run_bot():
    """Run the full trading bot with all components."""
    config = get_config()
    logger.info("="*60)
    logger.info("AI TRADING BOT STARTING")
    logger.info("Mode: %s", TRADING_MODE)
    logger.info("Watchlist: %s", config["watchlist"])
    logger.info("="*60)

    # Initialize database
    init_db()

    # Initialize trading engine
    engine = TradingEngine()

    # Initialize Telegram bot
    telegram_bot = TradingBot(trading_engine=engine)
    telegram_app = telegram_bot.build_app()
    engine.set_telegram_bot(telegram_bot)

    # Connect to broker
    try:
        await engine.connect()
    except Exception as e:
        logger.error("Failed to connect to broker: %s", e)
        logger.info("Bot will run in analysis-only mode (no order execution)")

    # Start the engine
    engine.start()

    # Setup scheduler
    scheduler = AsyncIOScheduler(timezone=config["schedule"]["timezone"])
    scan_interval = config["schedule"]["scan_interval_minutes"]

    # Scan watchlist every N minutes during market hours
    scheduler.add_job(
        engine.scan_watchlist,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute=f"*/{scan_interval}",
            timezone=config["schedule"]["timezone"],
        ),
        id="scan_watchlist",
        name="Scan Watchlist",
    )

    # Pre-market scan
    pre_market = config["schedule"]["pre_market_scan"].split(":")
    scheduler.add_job(
        engine.scan_watchlist,
        CronTrigger(
            day_of_week="mon-fri",
            hour=int(pre_market[0]),
            minute=int(pre_market[1]),
            timezone=config["schedule"]["timezone"],
        ),
        id="pre_market_scan",
        name="Pre-Market Scan",
    )

    # Manage positions every minute during market hours
    scheduler.add_job(
        engine.manage_open_positions,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="*",
            timezone=config["schedule"]["timezone"],
        ),
        id="manage_positions",
        name="Manage Positions",
    )

    # Check for closed positions every minute
    scheduler.add_job(
        engine.check_closed_positions,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-16",
            minute="*",
            timezone=config["schedule"]["timezone"],
        ),
        id="check_closed",
        name="Check Closed Positions",
    )

    # Daily summary at market close
    market_close = config["schedule"]["market_close"].split(":")
    scheduler.add_job(
        engine.generate_daily_summary,
        CronTrigger(
            day_of_week="mon-fri",
            hour=int(market_close[0]),
            minute=int(market_close[1]) + 5,  # 5 min after close
            timezone=config["schedule"]["timezone"],
        ),
        id="daily_summary",
        name="Daily Summary",
    )

    scheduler.start()
    logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))

    # Run Telegram bot
    try:
        await telegram_app.initialize()
        await telegram_app.start()
        await telegram_app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

        # Keep running
        stop_event = asyncio.Event()

        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received")
            stop_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        await stop_event.wait()

    finally:
        logger.info("Shutting down...")
        scheduler.shutdown()
        engine.stop()
        engine.broker.disconnect()
        try:
            if telegram_app.updater and telegram_app.updater.running:
                await telegram_app.updater.stop()
            if telegram_app.running:
                await telegram_app.stop()
            await telegram_app.shutdown()
        except Exception as e:
            logger.warning("Cleanup error (safe to ignore): %s", e)
        logger.info("Bot shutdown complete")


def run_dashboard():
    """Launch the Streamlit dashboard."""
    logger.info("Starting dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


def run_backtest(symbol: str, period: str = "1y"):
    """Run a quick backtest from the command line."""
    init_db()
    backtester = Backtester()
    report = backtester.run_backtest(symbol, period=period)
    print(backtester.format_report(report))


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--dashboard", action="store_true", help="Run dashboard only")
    parser.add_argument("--backtest", type=str, help="Run backtest for symbol")
    parser.add_argument("--period", type=str, default="1y", help="Backtest period (default: 1y)")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    elif args.backtest:
        run_backtest(args.backtest, args.period)
    else:
        asyncio.run(run_bot())


if __name__ == "__main__":
    main()
