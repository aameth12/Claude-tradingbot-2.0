import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, date

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.utils.logger import setup_logger
from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PROJECT_ROOT, get_config, save_config
from src.utils.database import get_session, Trade
from src.utils.performance_tracker import PerformanceTracker
from src.backtest.backtester import Backtester


logger = setup_logger("telegram")


class TradingBot:
    """Telegram bot for controlling the trading bot and receiving notifications."""

    def __init__(self, trading_engine=None):
        self.app = None
        self.trading_engine = trading_engine
        self.backtester = Backtester()
        self.performance_tracker = PerformanceTracker()
        self.authorized_chat_id = TELEGRAM_CHAT_ID

    def _is_authorized(self, update: Update) -> bool:
        if not self.authorized_chat_id:
            return True
        return str(update.effective_chat.id) == str(self.authorized_chat_id)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await update.message.reply_text("Unauthorized.")
            return
        msg = (
            "AI Trading Bot\n"
            f"{'='*30}\n\n"
            "📊 Dashboard & Info\n"
            "  /dashboard  - Account overview & positions\n"
            "  /market     - Market hours & status\n\n"
            "📈 Analysis\n"
            "  /regime          - Market regime & adjustments\n"
            "  /sentiment <SYM> - Sentiment & earnings\n"
            "  /review [ID]     - Trade review\n"
            "  /accuracy        - Indicator accuracy\n"
            "  /backtest <SYM>  - Backtest a symbol\n\n"
            "📋 Watchlist\n"
            "  /watchlist       - View watchlist\n"
            "  /add <SYM>       - Add symbol\n"
            "  /remove <SYM>    - Remove symbol\n\n"
            "⚙️ Controls\n"
            "  /startbot  - Start trading engine\n"
            "  /stopbot   - Stop trading engine\n"
            "  /sellall   - Close all positions\n"
            "  /sync      - Sync DB with IBKR\n"
            "  /update    - Git pull & restart"
        )
        await update.message.reply_text(msg)

    async def dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comprehensive account dashboard — pulls data from IBKR directly."""
        if not self._is_authorized(update):
            return

        config = get_config()
        broker = self.trading_engine.broker if self.trading_engine else None
        broker_connected = broker and broker.connected

        # --- Header: Market status + bot mode ---
        market_line = ""
        if self.trading_engine:
            ms = self.trading_engine.get_market_status()
            if ms["is_open"]:
                close_str = ""
                if ms["next_close"]:
                    close_str = f" (closes {ms['next_close'].strftime('%I:%M %p ET')})"
                market_line = f"Market: OPEN{close_str}"
            else:
                if ms["is_holiday"]:
                    market_line = f"Market: CLOSED ({ms['holiday_name'] or 'Holiday'})"
                else:
                    market_line = "Market: CLOSED"
                if ms["next_open"]:
                    market_line += f"\nNext Open: {ms['next_open'].strftime('%a %b %d, %I:%M %p ET')}"

        engine_status = "RUNNING" if (self.trading_engine and self.trading_engine.running) else "STOPPED"
        msg = (
            f"Dashboard\n"
            f"{'='*30}\n"
            f"{market_line}\n"
            f"Mode: {config['trading']['mode']} | Engine: {engine_status}\n"
        )

        # --- Account section (from IBKR) ---
        account = {}
        if broker_connected:
            try:
                account = await broker.get_account_pnl()
            except Exception:
                account = {}

            if account and account.get('NetLiquidation', 0) > 0:
                msg += (
                    f"\nAccount\n"
                    f"{'-'*30}\n"
                    f"Total Balance: ${account.get('NetLiquidation', 0):,.2f}\n"
                    f"Cash: ${account.get('TotalCashValue', 0):,.2f}\n"
                    f"Buying Power: ${account.get('BuyingPower', 0):,.2f}\n"
                )
            else:
                # Show whatever tags we did get for debugging
                tags = ", ".join(f"{k}={v}" for k, v in account.items()) if account else "none"
                msg += f"\nAccount: (IBKR connected, waiting for data — tags: {tags})\n"
        else:
            msg += "\nAccount: (IBKR not connected — restart bot with IB Gateway running)\n"

        # --- Today's P&L (combined IBKR + DB) ---
        today_content = ""
        if account and account.get('NetLiquidation', 0) > 0:
            realized = account.get('RealizedPnL', 0)
            unrealized = account.get('UnrealizedPnL', 0)
            today_content += (
                f"IBKR Realized: ${realized:+,.2f}\n"
                f"IBKR Unrealized: ${unrealized:+,.2f}\n"
                f"IBKR Net: ${realized + unrealized:+,.2f}\n"
            )

        session = get_session()
        try:
            today_str = date.today().isoformat()
            today_closed = session.query(Trade).filter(
                Trade.status == "CLOSED",
                Trade.exit_time >= today_str,
            ).all()
            if today_closed:
                day_winners = [t for t in today_closed if (t.pnl or 0) > 0]
                day_pnl = sum(t.pnl or 0 for t in today_closed)
                day_win_rate = len(day_winners) / len(today_closed) * 100
                today_content += (
                    f"Bot Trades: {len(today_closed)}"
                    f" | Win Rate: {day_win_rate:.1f}%\n"
                    f"Bot P&L: ${day_pnl:+,.2f}\n"
                )

            if today_content:
                msg += f"\nToday\n{'-'*30}\n" + today_content
            else:
                msg += f"\nToday\n{'-'*30}\nNo activity today.\n"

            # --- All-time stats (from bot DB) ---
            all_closed = session.query(Trade).filter(Trade.status == "CLOSED").all()
            if all_closed:
                total_pnl = sum(t.pnl or 0 for t in all_closed)
                winners = [t for t in all_closed if (t.pnl or 0) > 0]
                losers = [t for t in all_closed if (t.pnl or 0) <= 0]
                total_wins = sum(t.pnl or 0 for t in winners)
                total_losses = abs(sum(t.pnl or 0 for t in losers))

                msg += (
                    f"\nAll-Time (Bot Tracked)\n"
                    f"{'-'*30}\n"
                    f"Total Trades: {len(all_closed)}"
                    f" | Win Rate: {len(winners)/len(all_closed)*100:.1f}%\n"
                    f"Total P&L: ${total_pnl:+,.2f}\n"
                )
                if total_losses > 0:
                    msg += f"Profit Factor: {total_wins/total_losses:.2f}\n"

            # --- Open positions (from IBKR portfolio) ---
            portfolio = []
            if broker_connected:
                try:
                    portfolio = await broker.get_portfolio()
                    portfolio = [p for p in portfolio if p["position"] != 0]
                except Exception:
                    pass

            if portfolio:
                total_unrealized = sum(p["unrealized_pnl"] for p in portfolio)
                msg += (
                    f"\nOpen Positions ({len(portfolio)})\n"
                    f"{'-'*30}\n"
                )
                for p in portfolio:
                    side = "LONG" if p["position"] > 0 else "SHORT"
                    qty = abs(int(p["position"]))
                    pnl_val = p["unrealized_pnl"]
                    pnl_pct = (pnl_val / (p["average_cost"] * qty)) * 100 if p["average_cost"] and qty else 0
                    msg += (
                        f"  {side} {p['symbol']} x{qty} @ ${p['average_cost']:.2f}\n"
                        f"    Now: ${p['market_price']:.2f}"
                        f" | P&L: ${pnl_val:+,.2f} ({pnl_pct:+.1f}%)\n"
                    )
                msg += f"  Total Unrealized: ${total_unrealized:+,.2f}\n"
            else:
                # Fallback to DB if IBKR not available
                open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
                if open_trades:
                    msg += (
                        f"\nOpen Positions ({len(open_trades)})\n"
                        f"{'-'*30}\n"
                    )
                    for t in open_trades:
                        msg += f"  {t.side} {t.symbol} x{t.quantity} @ ${t.entry_price:.2f}\n"
                else:
                    msg += "\nNo open positions.\n"

            # Split message if too long for Telegram (4096 char limit)
            if len(msg) > 3800:
                await update.message.reply_text(msg)
                msg = ""

            # --- Today's executions (from IBKR) ---
            if broker_connected:
                try:
                    executions = await broker.get_executions()
                    today_dt = datetime.utcnow().date()
                    today_execs = [
                        e for e in executions
                        if hasattr(e["time"], "date") and e["time"].date() == today_dt
                    ]
                    if today_execs:
                        exec_msg = (
                            f"\nToday's Executions ({len(today_execs)})\n"
                            f"{'-'*30}\n"
                        )
                        for e in today_execs[-10:]:  # Last 10
                            time_str = e["time"].strftime("%H:%M") if hasattr(e["time"], "strftime") else ""
                            exec_msg += (
                                f"  {e['side']} {int(e['quantity'])} {e['symbol']}"
                                f" @ ${e['price']:.2f} [{time_str}]\n"
                            )
                        msg += exec_msg
                except Exception:
                    pass

            # --- Performance targets ---
            try:
                today_targets = self.performance_tracker.get_today_targets()
                targets_msg = (
                    f"\nTargets\n"
                    f"{'-'*30}\n"
                )
                if today_targets.get("win_rate_actual") is not None:
                    wr_icon = "HIT" if today_targets["win_rate_hit"] else "MISS"
                    pnl_icon = "HIT" if today_targets["pnl_pct_hit"] else "MISS"
                    targets_msg += (
                        f"  Win Rate: {today_targets['win_rate_actual']:.1f}%"
                        f" / {today_targets['win_rate_target']:.1f}% [{wr_icon}]\n"
                        f"  P&L: {today_targets['pnl_pct_actual']:+.2f}%"
                        f" / {today_targets['pnl_pct_target']:+.2f}% [{pnl_icon}]\n"
                    )
                else:
                    targets_msg += (
                        f"  Win Rate Target: {today_targets['win_rate_target']:.1f}%\n"
                        f"  P&L Target: {today_targets['pnl_pct_target']:+.2f}%\n"
                    )
                streak = today_targets.get("streak", 0)
                targets_msg += f"  Streak: {streak} day(s)\n"
                msg += targets_msg
            except Exception:
                pass

            if msg:
                await update.message.reply_text(msg)
        finally:
            session.close()

    async def market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current market status with holiday awareness."""
        if not self._is_authorized(update):
            return

        if not self.trading_engine:
            await update.message.reply_text("Trading engine not initialized.")
            return

        ms = self.trading_engine.get_market_status()
        now = ms["current_time"]

        msg = (
            f"Market Status\n"
            f"{'='*30}\n"
        )

        if ms["is_open"]:
            msg += f"US Stock Market: OPEN\n"
            msg += f"Current Time: {now.strftime('%I:%M %p ET')}\n"
            if ms["next_close"]:
                delta = ms["next_close"] - now
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes = remainder // 60
                msg += f"\nCloses: {ms['next_close'].strftime('%I:%M %p ET')} ({hours}h {minutes}m)\n"
        else:
            msg += f"US Stock Market: CLOSED\n"
            if ms["is_holiday"]:
                msg += f"Reason: {ms['holiday_name'] or 'Market Holiday'}\n"
            msg += f"Current Time: {now.strftime('%I:%M %p ET')}\n"
            if ms["next_open"]:
                delta = ms["next_open"] - now
                total_hours = int(delta.total_seconds()) // 3600
                minutes = (int(delta.total_seconds()) % 3600) // 60
                msg += (
                    f"\nNext Open: {ms['next_open'].strftime('%a %b %d, %I:%M %p ET')}"
                    f" (in {total_hours}h {minutes}m)\n"
                )

        await update.message.reply_text(msg)

    async def watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        config = get_config()
        symbols = config["watchlist"]
        msg = f"Watchlist ({len(symbols)} stocks)\n{'='*30}\n"
        msg += "\n".join(f"  {s}" for s in symbols)
        await update.message.reply_text(msg)

    async def add_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /add SYMBOL")
            return
        symbol = context.args[0].upper()
        config = get_config()
        if symbol in config["watchlist"]:
            await update.message.reply_text(f"{symbol} already in watchlist.")
            return
        config["watchlist"].append(symbol)
        save_config()
        await update.message.reply_text(f"Added {symbol} to watchlist.")

    async def remove_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /remove SYMBOL")
            return
        symbol = context.args[0].upper()
        config = get_config()
        if symbol not in config["watchlist"]:
            await update.message.reply_text(f"{symbol} not in watchlist.")
            return
        config["watchlist"].remove(symbol)
        save_config()
        await update.message.reply_text(f"Removed {symbol} from watchlist.")

    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /backtest SYMBOL [period]\nExample: /backtest AAPL 6mo")
            return

        symbol = context.args[0].upper()
        period = context.args[1] if len(context.args) > 1 else "1y"

        await update.message.reply_text(f"Running backtest for {symbol} ({period})...")

        try:
            report = self.backtester.run_backtest(symbol, period=period)
            msg = self.backtester.format_report(report)
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"Backtest failed: {e}")

    async def startbot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if self.trading_engine:
            self.trading_engine.start()
            await update.message.reply_text("Trading bot STARTED.")
        else:
            await update.message.reply_text("Trading engine not initialized.")

    async def stopbot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if self.trading_engine:
            self.trading_engine.stop()
            await update.message.reply_text("Trading bot STOPPED.")
        else:
            await update.message.reply_text("Trading engine not initialized.")

    async def sellall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all open positions at market price."""
        if not self._is_authorized(update):
            return
        if not self.trading_engine:
            await update.message.reply_text("Trading engine not initialized.")
            return

        broker = self.trading_engine.broker

        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            if not open_trades:
                await update.message.reply_text("No open positions to close.")
                return

            await update.message.reply_text(f"Closing {len(open_trades)} position(s)...\nStep 1: Cancelling all open orders...")

            # Step 1: Cancel ALL open orders first to free up order slots
            # This prevents Error 201 (too many orders) and Error 10148 (already cancelled)
            try:
                ib_trades = broker.ib.openTrades()
                cancelled_count = 0
                for t in ib_trades:
                    if t.orderStatus.status not in ("Cancelled", "Filled", "Inactive"):
                        broker.ib.cancelOrder(t.order)
                        cancelled_count += 1
                if cancelled_count:
                    await asyncio.sleep(1)  # Let cancellations propagate
                    await update.message.reply_text(f"Cancelled {cancelled_count} pending orders.")
            except Exception as e:
                logger.warning("Error cancelling open orders: %s", e)

            await update.message.reply_text("Step 2: Closing positions at market...")

            # Step 2: Close each position with a market order
            closed = 0
            for trade in open_trades:
                try:
                    action = "SELL" if trade.side == "LONG" else "BUY"
                    contract = await broker._get_qualified_contract(trade.symbol)

                    # Use MarketOrder directly instead of place_entry_order
                    # to avoid issues with order presets overriding TIF
                    from ib_insync import MarketOrder as MktOrder
                    order = MktOrder(action, trade.quantity)
                    order.tif = "GTC"  # Avoid Error 10349 (DAY preset rejection)
                    ib_trade = broker.ib.placeOrder(contract, order)

                    # Wait for fill (up to 15s)
                    fill_price = None
                    for _ in range(30):
                        await asyncio.sleep(0.5)
                        if ib_trade.orderStatus.status == "Filled":
                            fill_price = ib_trade.orderStatus.avgFillPrice
                            break

                    if not fill_price:
                        # Try to get last price as fallback
                        broker.ib.cancelOrder(ib_trade.order)
                        market_data = await broker.get_market_data(trade.symbol)
                        fill_price = market_data.get("last") or trade.entry_price
                        logger.warning("Sell order not filled for %s, using market price %.2f", trade.symbol, fill_price)

                    # Calculate P&L
                    if trade.side == "LONG":
                        pnl = (fill_price - trade.entry_price) * trade.quantity
                    else:
                        pnl = (trade.entry_price - fill_price) * trade.quantity
                    pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100

                    # Update trade in DB
                    trade.exit_price = fill_price
                    trade.exit_time = datetime.utcnow()
                    trade.pnl = round(pnl, 2)
                    trade.pnl_pct = round(pnl_pct, 2)
                    trade.status = "CLOSED"
                    trade.exit_reason = "MANUAL"
                    session.commit()
                    closed += 1

                    await update.message.reply_text(
                        f"Closed {trade.side} {trade.symbol} @ ${fill_price:.2f} | P&L: ${pnl:+,.2f}"
                    )

                except Exception as e:
                    logger.error("Failed to close %s %s: %s", trade.side, trade.symbol, e)
                    await update.message.reply_text(f"Failed to close {trade.symbol}: {e}")

            # Clean up price tracking
            self.trading_engine._price_extremes.clear()

            total_pnl = sum(t.pnl or 0 for t in open_trades if t.status == "CLOSED")
            await update.message.reply_text(
                f"Sell All Complete\n"
                f"{'='*30}\n"
                f"Closed: {closed}/{len(open_trades)} positions\n"
                f"Total P&L: ${total_pnl:+,.2f}"
            )
        finally:
            session.close()

    async def update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pull latest code from git and restart the bot."""
        if not self._is_authorized(update):
            return

        bot_dir = str(PROJECT_ROOT)
        await update.message.reply_text("Pulling latest code...")

        # Run git pull in a thread (SelectorEventLoop on Windows doesn't support subprocesses)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: subprocess.run(
                ["git", "pull", "origin", "claude/ai-trading-bot-a1jC4"],
                capture_output=True, text=True, timeout=30, cwd=bot_dir,
            ))
            git_output = (result.stdout or result.stderr or "").strip()
            await update.message.reply_text(f"Git pull:\n{git_output}")
            if result.returncode != 0:
                await update.message.reply_text("Git pull failed (non-zero exit).")
                return
        except Exception as e:
            await update.message.reply_text(f"Git pull failed: {e}")
            return

        # Install any new/updated dependencies
        try:
            result = await loop.run_in_executor(None, lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                capture_output=True, text=True, timeout=120, cwd=bot_dir,
            ))
            if result.returncode != 0:
                await update.message.reply_text(
                    f"pip install warning:\n{(result.stderr or '')[:500]}"
                )
        except Exception as e:
            await update.message.reply_text(f"pip install failed: {e} — restarting anyway")

        await update.message.reply_text("Restarting bot...")

        # Clean up broker connection before restart
        try:
            if self.trading_engine:
                self.trading_engine.stop()
                self.trading_engine.broker.disconnect()
        except Exception:
            pass

        # Spawn new process then exit
        subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=bot_dir,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        for handler in logging.root.handlers:
            handler.flush()
        os._exit(0)

    async def regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current market regime."""
        if not self._is_authorized(update):
            return
        if not self.trading_engine or not self.trading_engine.agent_manager:
            await update.message.reply_text("Agents not initialized.")
            return

        regime = self.trading_engine._current_regime
        if not regime:
            await update.message.reply_text("No regime data yet. Wait for next scan cycle.")
            return

        msg = (
            f"Market Regime\n{'='*30}\n"
            f"Regime: {regime.regime} ({regime.confidence:.0%} confidence)\n"
            f"VIX: {regime.vix_level:.1f} ({regime.vix_trend})\n"
            f"Breadth: {regime.breadth_pct:.1f}% above EMA50\n\n"
            f"Adjustments:\n"
            f"  SL Multiplier: {regime.recommended_sl_multiplier}x ATR\n"
            f"  TP Multiplier: {regime.recommended_tp_multiplier}x ATR\n"
            f"  Confidence Threshold: {regime.recommended_confidence_threshold}\n"
            f"  Position Scale: {regime.recommended_position_scale:.0%}\n"
            f"  Updated: {regime.timestamp[:19]}"
        )
        await update.message.reply_text(msg)

    async def sentiment_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show sentiment for a symbol."""
        if not self._is_authorized(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /sentiment SYMBOL")
            return
        if not self.trading_engine or not self.trading_engine.agent_manager:
            await update.message.reply_text("Agents not initialized.")
            return

        symbol = context.args[0].upper()
        await update.message.reply_text(f"Getting sentiment for {symbol}...")

        try:
            sentiment = await self.trading_engine.agent_manager.get_sentiment(symbol)
            if not sentiment:
                await update.message.reply_text("Sentiment agent disabled or failed.")
                return

            earnings_str = sentiment.earnings_date or "Unknown"
            block_str = " BLOCKED" if sentiment.should_block_trade else " OK to trade"

            msg = (
                f"Sentiment: {symbol}\n{'='*30}\n"
                f"Score: {sentiment.sentiment_score:+.2f} "
                f"({'bullish' if sentiment.sentiment_score > 0.1 else 'bearish' if sentiment.sentiment_score < -0.1 else 'neutral'})\n"
                f"Earnings: {earnings_str}{block_str}\n"
            )
            if sentiment.news_events:
                msg += "Events:\n"
                for event in sentiment.news_events[:5]:
                    msg += f"  - {event}\n"

            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"Failed: {e}")

    async def review_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show trade review for a specific trade or the last closed trade."""
        if not self._is_authorized(update):
            return

        session = get_session()
        try:
            if context.args:
                trade_id = int(context.args[0])
            else:
                # Get last closed trade
                last = session.query(Trade).filter(Trade.status == "CLOSED").order_by(Trade.exit_time.desc()).first()
                if not last:
                    await update.message.reply_text("No closed trades to review.")
                    return
                trade_id = last.id

            from src.utils.database import TradeReviewRecord
            review = session.query(TradeReviewRecord).filter(
                TradeReviewRecord.trade_id == trade_id
            ).first()

            if not review:
                await update.message.reply_text(f"No review found for trade #{trade_id}.")
                return

            import json
            correct = json.loads(review.correct_indicators or "[]")
            incorrect = json.loads(review.incorrect_indicators or "[]")

            msg = (
                f"Trade Review #{trade_id}\n{'='*30}\n"
                f"Correct: {', '.join(correct) if correct else 'None'}\n"
                f"Incorrect: {', '.join(incorrect) if incorrect else 'None'}\n\n"
                f"{review.review_text or ''}"
            )
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"Failed: {e}")
        finally:
            session.close()

    async def accuracy_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show indicator accuracy stats."""
        if not self._is_authorized(update):
            return
        if not self.trading_engine or not self.trading_engine.agent_manager:
            await update.message.reply_text("Agents not initialized.")
            return

        stats = self.trading_engine.agent_manager.get_accuracy_stats()
        if not stats:
            await update.message.reply_text("No accuracy data yet. Trade reviews build this over time.")
            return

        msg = f"Indicator Accuracy\n{'='*30}\n"
        for s in stats:
            msg += f"  {s['name']}: {s['accuracy']:.1f}% ({s['correct']}/{s['total']})\n"

        await update.message.reply_text(msg)

    async def sync(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reconcile bot's database with IBKR's actual state."""
        if not self._is_authorized(update):
            return
        if not self.trading_engine:
            await update.message.reply_text("Trading engine not initialized.")
            return
        if not self.trading_engine.broker.connected:
            await update.message.reply_text("Broker not connected.")
            return

        await update.message.reply_text("Syncing with IBKR...")
        try:
            result = await self.trading_engine.reconcile_with_broker()
            msg = (
                f"IBKR Sync Complete\n"
                f"{'='*30}\n"
                f"Trades closed by IBKR: {result['closed_count']}\n"
                f"Untracked IBKR positions: {result['orphan_count']}\n"
            )
            if result["closed_trades"]:
                msg += "\nReconciled Trades:\n"
                for t in result["closed_trades"]:
                    msg += f"  {t['side']} {t['symbol']}: ${t['pnl'] or 0:+,.2f}\n"
            if result["orphans"]:
                msg += "\nAdopted IBKR Positions:\n"
                for o in result["orphans"]:
                    msg += (
                        f"  {o['symbol']}: {o['position']:.0f} shares"
                        f" | P&L: ${o['unrealized_pnl']:+,.2f}"
                        f" (now tracked)\n"
                    )

            # Show final state after sync
            sync_session = get_session()
            try:
                open_count = sync_session.query(Trade).filter(Trade.status == "OPEN").count()
                closed_count = sync_session.query(Trade).filter(Trade.status == "CLOSED").count()
                msg += (
                    f"\nBot DB State:\n"
                    f"  Open positions: {open_count}\n"
                    f"  Closed trades: {closed_count}\n"
                )
            finally:
                sync_session.close()

            # Append current account state
            try:
                account = await self.trading_engine.broker.get_account_pnl()
                portfolio = await self.trading_engine.broker.get_portfolio()
                ibkr_open = len([p for p in portfolio if p["position"] != 0])
                msg += (
                    f"\nIBKR Account:\n"
                    f"  Total Balance: ${account.get('NetLiquidation', 0):,.2f}\n"
                    f"  Cash: ${account.get('TotalCashValue', 0):,.2f}\n"
                    f"  Unrealized P&L: ${account.get('UnrealizedPnL', 0):+,.2f}\n"
                    f"  Realized P&L: ${account.get('RealizedPnL', 0):+,.2f}\n"
                    f"  Buying Power: ${account.get('BuyingPower', 0):,.2f}\n"
                    f"  IBKR Open Positions: {ibkr_open}\n"
                )
            except Exception:
                pass

            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"Sync failed: {e}")

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)

    def build_app(self) -> Application:
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Dashboard & Info
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("dashboard", self.dashboard))
        self.app.add_handler(CommandHandler("market", self.market))

        # Analysis
        self.app.add_handler(CommandHandler("regime", self.regime))
        self.app.add_handler(CommandHandler("sentiment", self.sentiment_cmd))
        self.app.add_handler(CommandHandler("review", self.review_cmd))
        self.app.add_handler(CommandHandler("accuracy", self.accuracy_cmd))
        self.app.add_handler(CommandHandler("backtest", self.backtest))

        # Watchlist
        self.app.add_handler(CommandHandler("watchlist", self.watchlist))
        self.app.add_handler(CommandHandler("add", self.add_symbol))
        self.app.add_handler(CommandHandler("remove", self.remove_symbol))

        # Controls
        self.app.add_handler(CommandHandler("startbot", self.startbot))
        self.app.add_handler(CommandHandler("stopbot", self.stopbot))
        self.app.add_handler(CommandHandler("sellall", self.sellall))
        self.app.add_handler(CommandHandler("sync", self.sync))
        self.app.add_handler(CommandHandler("update", self.update))
        self.app.add_handler(CommandHandler("help", self.help_cmd))

        return self.app

    async def send_notification(self, message: str):
        """Send a notification message to the configured chat."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured, skipping notification")
            return
        if self.app and self.app.bot:
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

    async def send_trade_alert(self, trade_data: dict):
        """Send a formatted trade alert."""
        side = trade_data.get("side", "")
        symbol = trade_data.get("symbol", "")
        entry = trade_data.get("entry_price", 0)
        sl = trade_data.get("stop_loss", 0)
        tp = trade_data.get("take_profit", 0)
        qty = trade_data.get("quantity", 0)
        confidence = trade_data.get("confidence", 0)

        msg = (
            f"NEW TRADE ALERT\n"
            f"{'='*30}\n"
            f"Side: {side}\n"
            f"Symbol: {symbol}\n"
            f"Entry: ${entry:.2f}\n"
            f"Stop Loss: ${sl:.2f}\n"
            f"Take Profit: ${tp:.2f}\n"
            f"Quantity: {qty}\n"
            f"Confidence: {confidence:.1%}\n"
        )
        await self.send_notification(msg)
