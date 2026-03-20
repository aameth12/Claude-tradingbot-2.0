import asyncio
import json
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
from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, get_config, save_config
from src.utils.database import get_session, Trade, DailySummary, BacktestResult
from src.utils.performance_tracker import PerformanceTracker
from src.backtest.backtester import Backtester
from src.utils.version import VERSION, VERSION_NAME, CHANGELOG, get_version_string

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
            "AI Trading Bot Commands:\n\n"
            "/status - Bot status & open positions\n"
            "/pnl - Today's P&L summary\n"
            "/trades - Recent trades\n"
            "/watchlist - View watchlist\n"
            "/add <SYMBOL> - Add to watchlist\n"
            "/remove <SYMBOL> - Remove from watchlist\n"
            "/backtest <SYMBOL> [period] - Backtest a stock\n"
            "/summary - Daily summary\n"
            "/positions - Open positions\n"
            "/sellall - Close ALL open positions\n"
            "/startbot - Start trading\n"
            "/stopbot - Stop trading\n"
            "/update - Git pull & restart bot\n"
            "/performance - Overall performance stats\n"
            "/regime - Current market regime & adjustments\n"
            "/sentiment <SYMBOL> - Sentiment & earnings check\n"
            "/review [ID] - Trade review (last trade if no ID)\n"
            "/accuracy - Indicator accuracy stats\n"
            "/targets - Daily performance targets & history\n"
            "/version - Version info & changelog\n"
            "/help - Show this help"
        )
        await update.message.reply_text(msg)

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            today_str = date.today().isoformat()
            today_closed = session.query(Trade).filter(
                Trade.status == "CLOSED", Trade.exit_time >= today_str
            ).all()

            config = get_config()
            msg = (
                f"Bot Status\n"
                f"{'='*30}\n"
                f"Mode: {config['trading']['mode']}\n"
                f"Open Positions: {len(open_trades)}/{config['trading']['max_open_positions']}\n"
                f"Today's Trades: {len(today_closed)}/{config['trading']['max_daily_trades']}\n"
                f"Watchlist: {', '.join(config['watchlist'])}\n"
            )

            if open_trades:
                msg += f"\nOpen Positions:\n"
                total_unrealized = 0
                for t in open_trades:
                    # Fetch current price from broker, fallback to yfinance
                    current_price = None
                    if self.trading_engine and self.trading_engine.broker:
                        try:
                            market_data = await self.trading_engine.broker.get_market_data(t.symbol)
                            current_price = market_data.get("last") or market_data.get("close")
                        except Exception:
                            pass
                    if not current_price and self.trading_engine:
                        try:
                            current_price = self.trading_engine.tv_analyzer.get_current_price(t.symbol)
                        except Exception:
                            pass

                    if current_price:
                        if t.side == "LONG":
                            pnl = (current_price - t.entry_price) * t.quantity
                        else:
                            pnl = (t.entry_price - current_price) * t.quantity
                        pnl_pct = (pnl / (t.entry_price * t.quantity)) * 100
                        total_unrealized += pnl
                        msg += (
                            f"  {t.side} {t.symbol} @ ${t.entry_price:.2f}\n"
                            f"    Now: ${current_price:.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
                            f"    SL: ${t.stop_loss:.2f} | TP: ${t.take_profit:.2f}\n"
                        )
                    else:
                        msg += f"  {t.side} {t.symbol} @ ${t.entry_price:.2f} | SL: ${t.stop_loss:.2f} | TP: ${t.take_profit:.2f}\n"

                if total_unrealized != 0:
                    msg += f"\n  Total Unrealized P&L: ${total_unrealized:+,.2f}\n"

            await update.message.reply_text(msg)
        finally:
            session.close()

    async def pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        session = get_session()
        try:
            today_str = date.today().isoformat()
            trades = session.query(Trade).filter(
                Trade.status == "CLOSED", Trade.exit_time >= today_str
            ).all()

            total_pnl = sum(t.pnl or 0 for t in trades)
            winners = [t for t in trades if (t.pnl or 0) > 0]
            losers = [t for t in trades if (t.pnl or 0) <= 0]

            msg = (
                f"Today's P&L\n"
                f"{'='*30}\n"
                f"Total P&L: ${total_pnl:+,.2f}\n"
                f"Trades: {len(trades)}\n"
                f"Winners: {len(winners)}\n"
                f"Losers: {len(losers)}\n"
                f"Win Rate: {len(winners)/len(trades)*100:.1f}%\n" if trades else
                f"Today's P&L\n{'='*30}\nNo trades today.\n"
            )

            if trades:
                msg += "\nTrade Details:\n"
                for t in trades:
                    emoji = "+" if (t.pnl or 0) > 0 else ""
                    msg += f"  {t.side} {t.symbol}: ${t.pnl or 0:{emoji},.2f}\n"

            await update.message.reply_text(msg)
        finally:
            session.close()

    async def trades_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        session = get_session()
        try:
            recent = session.query(Trade).order_by(Trade.entry_time.desc()).limit(10).all()
            if not recent:
                await update.message.reply_text("No trades recorded yet.")
                return

            msg = "Recent Trades (last 10)\n" + "=" * 30 + "\n"
            for t in recent:
                pnl_str = f"${t.pnl:+,.2f}" if t.pnl is not None else "Open"
                msg += f"{t.side} {t.symbol} | Entry: ${t.entry_price:.2f} | {pnl_str} | {t.status}\n"

            await update.message.reply_text(msg)
        finally:
            session.close()

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

    async def summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        session = get_session()
        try:
            today_str = date.today().isoformat()
            trades = session.query(Trade).filter(Trade.exit_time >= today_str).all()
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()

            closed = [t for t in trades if t.status == "CLOSED"]
            total_pnl = sum(t.pnl or 0 for t in closed)
            winners = [t for t in closed if (t.pnl or 0) > 0]

            msg = (
                f"Daily Summary - {today_str}\n"
                f"{'='*40}\n\n"
                f"Closed Trades: {len(closed)}\n"
                f"Open Positions: {len(open_trades)}\n"
                f"Total P&L: ${total_pnl:+,.2f}\n"
                f"Win Rate: {len(winners)/len(closed)*100:.1f}%\n" if closed else
                f"Daily Summary - {today_str}\n{'='*40}\n\nNo closed trades today.\n"
                f"Open Positions: {len(open_trades)}\n"
            )

            if closed:
                best = max(closed, key=lambda t: t.pnl or 0)
                worst = min(closed, key=lambda t: t.pnl or 0)
                msg += f"\nBest Trade: {best.side} {best.symbol} ${best.pnl:+,.2f}\n"
                msg += f"Worst Trade: {worst.side} {worst.symbol} ${worst.pnl:+,.2f}\n"

            if open_trades:
                msg += f"\nOpen Positions:\n"
                for t in open_trades:
                    msg += f"  {t.side} {t.symbol} @ ${t.entry_price:.2f}\n"

            await update.message.reply_text(msg)
        finally:
            session.close()

    async def positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            if not open_trades:
                await update.message.reply_text("No open positions.")
                return

            msg = f"Open Positions ({len(open_trades)})\n{'='*40}\n"
            for t in open_trades:
                msg += (
                    f"\n{t.side} {t.symbol}\n"
                    f"  Entry: ${t.entry_price:.2f} | Qty: {t.quantity}\n"
                    f"  SL: ${t.stop_loss:.2f} | TP: ${t.take_profit:.2f}\n"
                    f"  Entered: {t.entry_time}\n"
                )
            await update.message.reply_text(msg)
        finally:
            session.close()

    async def performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        session = get_session()
        try:
            all_closed = session.query(Trade).filter(Trade.status == "CLOSED").all()
            if not all_closed:
                await update.message.reply_text("No completed trades yet.")
                return

            total_pnl = sum(t.pnl or 0 for t in all_closed)
            winners = [t for t in all_closed if (t.pnl or 0) > 0]
            losers = [t for t in all_closed if (t.pnl or 0) <= 0]
            total_wins = sum(t.pnl or 0 for t in winners)
            total_losses = abs(sum(t.pnl or 0 for t in losers))

            msg = (
                f"Overall Performance\n"
                f"{'='*40}\n\n"
                f"Total Trades: {len(all_closed)}\n"
                f"Win Rate: {len(winners)/len(all_closed)*100:.1f}%\n"
                f"Total P&L: ${total_pnl:+,.2f}\n"
                f"Profit Factor: {total_wins/total_losses:.2f}\n" if total_losses > 0 else ""
                f"Avg Win: ${total_wins/len(winners):,.2f}\n" if winners else ""
                f"Avg Loss: ${total_losses/len(losers):,.2f}\n" if losers else ""
            )
            await update.message.reply_text(msg)
        finally:
            session.close()

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

        await update.message.reply_text("Pulling latest code...")

        # Run git pull
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "claude/ai-trading-bot-a1jC4"],
                capture_output=True, text=True, timeout=30,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            git_output = result.stdout.strip() or result.stderr.strip()
            await update.message.reply_text(f"Git pull:\n{git_output}")
        except Exception as e:
            await update.message.reply_text(f"Git pull failed: {e}")
            return

        await update.message.reply_text("Restarting bot...")

        # Restart the bot process
        os.execv(sys.executable, [sys.executable, "main.py"])

    async def version(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current version and recent changelog."""
        if not self._is_authorized(update):
            return

        msg = f"AI Trading Bot {get_version_string()}\n{'='*30}\n"

        # Show last 3 versions
        for entry in CHANGELOG[:3]:
            msg += f"\nv{entry['version']} \"{entry['name']}\"\n"
            for change in entry["changes"]:
                msg += f"  - {change}\n"

        await update.message.reply_text(msg)

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

    async def targets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show today's performance targets and recent history."""
        if not self._is_authorized(update):
            return

        today = self.performance_tracker.get_today_targets()
        history = self.performance_tracker.get_recent_history(days=5)

        msg = (
            f"Performance Targets\n"
            f"{'='*30}\n\n"
            f"Today ({today['date']}):\n"
            f"  Win Rate Target: {today['win_rate_target']:.1f}%\n"
            f"  P&L Target: {today['pnl_pct_target']:+.2f}%\n"
        )

        if today["win_rate_actual"] is not None:
            wr_icon = "HIT" if today["win_rate_hit"] else "MISS"
            pnl_icon = "HIT" if today["pnl_pct_hit"] else "MISS"
            msg += (
                f"\n  Actual Win Rate: {today['win_rate_actual']:.1f}% [{wr_icon}]\n"
                f"  Actual P&L: {today['pnl_pct_actual']:+.2f}% [{pnl_icon}]\n"
            )

        if today.get("streak", 0) > 0:
            msg += f"\n  Streak: {today['streak']} day(s) hitting both targets!\n"

        # Recent days
        past = [h for h in history if h["date"] != today["date"] and h["win_rate_actual"] is not None]
        if past:
            msg += f"\nRecent History:\n"
            for h in past[:5]:
                wr_status = "HIT" if h["win_rate_hit"] else "MISS"
                pnl_status = "HIT" if h["pnl_pct_hit"] else "MISS"
                msg += (
                    f"  {h['date']}: WR {h['win_rate_actual']:.0f}%/{h['win_rate_target']:.0f}% [{wr_status}]"
                    f" | P&L {h['pnl_pct_actual']:+.1f}%/{h['pnl_pct_target']:+.1f}% [{pnl_status}]\n"
                )

        await update.message.reply_text(msg)

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)

    def build_app(self) -> Application:
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("pnl", self.pnl))
        self.app.add_handler(CommandHandler("trades", self.trades_cmd))
        self.app.add_handler(CommandHandler("watchlist", self.watchlist))
        self.app.add_handler(CommandHandler("add", self.add_symbol))
        self.app.add_handler(CommandHandler("remove", self.remove_symbol))
        self.app.add_handler(CommandHandler("backtest", self.backtest))
        self.app.add_handler(CommandHandler("summary", self.summary))
        self.app.add_handler(CommandHandler("positions", self.positions))
        self.app.add_handler(CommandHandler("performance", self.performance))
        self.app.add_handler(CommandHandler("sellall", self.sellall))
        self.app.add_handler(CommandHandler("startbot", self.startbot))
        self.app.add_handler(CommandHandler("stopbot", self.stopbot))
        self.app.add_handler(CommandHandler("update", self.update))
        self.app.add_handler(CommandHandler("version", self.version))
        self.app.add_handler(CommandHandler("regime", self.regime))
        self.app.add_handler(CommandHandler("sentiment", self.sentiment_cmd))
        self.app.add_handler(CommandHandler("review", self.review_cmd))
        self.app.add_handler(CommandHandler("accuracy", self.accuracy_cmd))
        self.app.add_handler(CommandHandler("targets", self.targets))
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
