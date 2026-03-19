import asyncio
import json
from datetime import datetime, date
from typing import Optional

import pandas as pd

from src.broker.ibkr_client import IBKRClient
from src.analysis.tradingview_data import TradingViewAnalyzer
from src.analysis.chart_analyzer import ChartAnalyzer
from src.strategy.signal_combiner import SignalCombiner, TradeSignal
from src.risk.risk_manager import RiskManager
from src.utils.database import get_session, Trade, DailySummary, init_db
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("engine")


class TradingEngine:
    """Main trading engine that orchestrates the full trading pipeline."""

    def __init__(self):
        self.broker = IBKRClient()
        self.tv_analyzer = TradingViewAnalyzer()
        self.chart_analyzer = ChartAnalyzer()
        self.signal_combiner = SignalCombiner()
        self.risk_manager = RiskManager()
        self.config = get_config()
        self.running = False
        self.telegram_bot = None  # Set externally after init

        # Track highest/lowest prices for trailing stops
        self._price_extremes = {}  # {symbol: highest_or_lowest_price}
        self._cached_portfolio_value = None  # Cached per scan cycle

        init_db()

    def set_telegram_bot(self, bot):
        self.telegram_bot = bot

    async def connect(self):
        await self.broker.connect()
        logger.info("Trading engine connected to broker")

    def start(self):
        self.running = True
        logger.info("Trading engine STARTED")

    def stop(self):
        self.running = False
        logger.info("Trading engine STOPPED")

    async def scan_watchlist(self):
        """Scan all watchlist symbols for trade opportunities."""
        if not self.running:
            logger.info("Engine not running, skipping scan")
            return

        watchlist = self.config["watchlist"]
        logger.info("Scanning watchlist: %s", watchlist)

        # Cancel stale unfilled orders to avoid Error 201 (max orders per side)
        try:
            await self.broker.cancel_stale_orders(symbols=watchlist, max_age_seconds=300)
        except Exception as e:
            logger.warning("Failed to cancel stale orders: %s", e)

        # Clear caches from previous scan cycle
        self.tv_analyzer.clear_cache()
        self._cached_portfolio_value = None

        # Analyze symbols in parallel (with per-symbol timeout)
        async def _analyze_with_timeout(symbol: str):
            try:
                return await asyncio.wait_for(self.analyze_symbol(symbol), timeout=60)
            except asyncio.TimeoutError:
                logger.warning("Analysis timeout for %s (>60s), skipping", symbol)
                return None
            except Exception as e:
                logger.error("Error analyzing %s: %s", symbol, e)
                return None

        signals = await asyncio.gather(*[_analyze_with_timeout(s) for s in watchlist])

        # Execute signals sequentially (order placement should be serial)
        for signal in signals:
            if signal:
                try:
                    await self.execute_signal(signal)
                except Exception as e:
                    logger.error("Error executing signal for %s: %s", signal.symbol, e)

    async def analyze_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Run full analysis pipeline on a single symbol."""
        logger.info("Analyzing %s...", symbol)

        # Check if we can open new trades
        can_trade, reason = self.risk_manager.can_open_new_trade()
        if not can_trade:
            logger.info("Cannot open new trade: %s", reason)
            return None

        # Check if already in a position for this symbol
        session = get_session()
        try:
            existing = session.query(Trade).filter(
                Trade.symbol == symbol, Trade.status == "OPEN"
            ).first()
            if existing:
                logger.info("Already in position for %s, skipping", symbol)
                return None
        finally:
            session.close()

        # 1. Get TradingView analysis (primary timeframe)
        tv_analysis = self.tv_analyzer.get_analysis(symbol, "1h")

        # 2. Get individual indicator signals
        tv_signals = self.tv_analyzer.check_indicator_signals(tv_analysis)

        # 3. Multi-timeframe analysis
        multi_tf = self.tv_analyzer.get_multi_timeframe_analysis(symbol)

        # 4. Get ATR for risk calculations
        atr = tv_analysis.get("indicators", {}).get("atr")
        current_price = tv_analysis.get("indicators", {}).get("close")

        if not atr or not current_price:
            logger.warning("Missing ATR or price for %s", symbol)
            return None

        # 5. AI chart analysis (if enabled)
        ai_analysis = {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": "Disabled"}
        if self.config["ai"]["chart_analysis_enabled"]:
            try:
                # Get historical bars from broker for chart generation
                bars = await self.broker.get_historical_bars(
                    symbol, duration="5 D", bar_size="5 mins"
                )
                if bars:
                    df = pd.DataFrame(bars)
                    # Run sync AI analysis in executor to avoid event loop conflicts
                    loop = asyncio.get_event_loop()
                    ai_analysis = await loop.run_in_executor(
                        None, self.chart_analyzer.analyze_from_data, df, symbol, "5m"
                    )
            except Exception as e:
                logger.warning("AI chart analysis skipped for %s: %s", symbol, e)

        # 6. Determine signal direction from all sources FIRST
        direction = self.signal_combiner.evaluate_direction(
            symbol=symbol,
            tv_analysis=tv_analysis,
            tv_indicator_signals=tv_signals,
            ai_analysis=ai_analysis,
            multi_tf_analyses=multi_tf,
        )

        if not direction:
            return None

        # 7. Now compute trade levels for the ACTUAL signal side
        if self._cached_portfolio_value is None:
            self._cached_portfolio_value = 100000  # Default
            try:
                account = await self.broker.get_account_summary()
                self._cached_portfolio_value = float(account.get("NetLiquidation", 100000))
            except Exception:
                pass
        portfolio_value = self._cached_portfolio_value

        trade_levels = self.risk_manager.get_trade_levels(
            current_price, atr, direction["side"], portfolio_value
        )

        # 8. Build final signal with correctly matched trade levels
        signal = self.signal_combiner.build_signal(
            symbol=symbol,
            direction=direction,
            trade_levels=trade_levels,
        )

        return signal

    async def execute_signal(self, signal: TradeSignal):
        """Execute a trade signal through the broker."""
        logger.info("Executing signal: %s %s", signal.side, signal.symbol)

        action = "BUY" if signal.side == "LONG" else "SELL"

        try:
            # Place bracket order (entry + SL + TP)
            trades = await self.broker.place_bracket_order(
                symbol=signal.symbol,
                action=action,
                quantity=signal.quantity,
                limit_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit,
            )

            # Record trade in database
            session = get_session()
            try:
                db_trade = Trade(
                    symbol=signal.symbol,
                    side=signal.side,
                    entry_price=signal.entry_price,
                    quantity=signal.quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    status="OPEN",
                    strategy=signal.strategy,
                    timeframe=signal.timeframe,
                    signals=json.dumps(signal.signals_detail),
                    order_id=trades[0].order.orderId if trades else None,
                )
                session.add(db_trade)
                session.commit()
                logger.info("Trade recorded in DB: %s %s", signal.side, signal.symbol)
            finally:
                session.close()

            # Initialize price tracking for trailing stop
            self._price_extremes[signal.symbol] = signal.entry_price

            # Send Telegram alert
            if self.telegram_bot:
                await self.telegram_bot.send_trade_alert({
                    "side": signal.side,
                    "symbol": signal.symbol,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "quantity": signal.quantity,
                    "confidence": signal.confidence,
                })

        except Exception as e:
            logger.error("Failed to execute trade %s %s: %s", signal.side, signal.symbol, e)

    async def manage_open_positions(self):
        """Update trailing stops and manage existing positions.

        Only adjusts trailing stop when the new level moves significantly
        (>0.10 from current stop) to avoid micro-adjustments that cause
        premature exits via bid/ask spread noise.
        """
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()

            for trade in open_trades:
                try:
                    market_data = await self.broker.get_market_data(trade.symbol)
                    current_price = market_data.get("last", 0)
                    if not current_price:
                        continue

                    # Update price extremes
                    extreme = self._price_extremes.get(trade.symbol, trade.entry_price)
                    if trade.side == "LONG":
                        extreme = max(extreme, current_price)
                    else:
                        extreme = min(extreme, current_price)
                    self._price_extremes[trade.symbol] = extreme

                    # Calculate new trailing stop
                    new_sl = self.risk_manager.calculate_trailing_stop(
                        entry_price=trade.entry_price,
                        current_price=current_price,
                        current_stop=trade.stop_loss,
                        highest_price=extreme,
                        side=trade.side,
                    )

                    # Only update if the change is meaningful (>$0.10)
                    # to avoid micro-adjustments from bid/ask noise
                    sl_diff = abs(new_sl - trade.stop_loss)
                    if sl_diff > 0.10 and new_sl != trade.stop_loss and trade.order_id:
                        success = await self.broker.modify_stop_loss(trade.order_id, new_sl)
                        if success:
                            trade.stop_loss = new_sl
                            session.commit()
                            logger.info(
                                "Trailing stop updated for %s: %.2f (moved $%.2f)",
                                trade.symbol, new_sl, sl_diff,
                            )

                except Exception as e:
                    logger.error("Error managing position %s: %s", trade.symbol, e)

        finally:
            session.close()

    async def check_closed_positions(self):
        """Check broker for filled orders and update trade records."""
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            broker_positions = await self.broker.get_positions()
            position_symbols = {p["symbol"] for p in broker_positions}

            for trade in open_trades:
                if trade.symbol not in position_symbols:
                    # Position was closed (SL or TP hit)
                    market_data = await self.broker.get_market_data(trade.symbol)
                    exit_price = market_data.get("last") or trade.entry_price

                    if trade.side == "LONG":
                        pnl = (exit_price - trade.entry_price) * trade.quantity
                    else:
                        pnl = (trade.entry_price - exit_price) * trade.quantity

                    pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100

                    trade.exit_price = exit_price
                    trade.exit_time = datetime.utcnow()
                    trade.pnl = round(pnl, 2)
                    trade.pnl_pct = round(pnl_pct, 2)
                    trade.status = "CLOSED"
                    session.commit()

                    logger.info(
                        "Trade closed: %s %s | PnL: $%.2f (%.2f%%)",
                        trade.side, trade.symbol, pnl, pnl_pct,
                    )

                    # Notify via Telegram
                    if self.telegram_bot:
                        msg = (
                            f"TRADE CLOSED\n"
                            f"{trade.side} {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
                            f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
                        )
                        await self.telegram_bot.send_notification(msg)

                    # Clean up tracking
                    self._price_extremes.pop(trade.symbol, None)

        finally:
            session.close()

    async def generate_daily_summary(self):
        """Generate and store daily trading summary."""
        session = get_session()
        try:
            today_str = date.today().isoformat()
            trades = session.query(Trade).filter(
                Trade.status == "CLOSED",
                Trade.exit_time >= today_str,
            ).all()

            if not trades:
                return

            winners = [t for t in trades if (t.pnl or 0) > 0]
            total_pnl = sum(t.pnl or 0 for t in trades)

            summary = DailySummary(
                date=today_str,
                total_trades=len(trades),
                winning_trades=len(winners),
                losing_trades=len(trades) - len(winners),
                total_pnl=round(total_pnl, 2),
                win_rate=round(len(winners) / len(trades) * 100, 1) if trades else 0,
                best_trade_pnl=max(t.pnl or 0 for t in trades),
                worst_trade_pnl=min(t.pnl or 0 for t in trades),
            )

            # Update or insert
            existing = session.query(DailySummary).filter(DailySummary.date == today_str).first()
            if existing:
                for key, value in {
                    "total_trades": summary.total_trades,
                    "winning_trades": summary.winning_trades,
                    "losing_trades": summary.losing_trades,
                    "total_pnl": summary.total_pnl,
                    "win_rate": summary.win_rate,
                    "best_trade_pnl": summary.best_trade_pnl,
                    "worst_trade_pnl": summary.worst_trade_pnl,
                }.items():
                    setattr(existing, key, value)
            else:
                session.add(summary)

            session.commit()

            # Send summary via Telegram
            if self.telegram_bot:
                msg = (
                    f"DAILY SUMMARY - {today_str}\n"
                    f"{'='*30}\n"
                    f"Trades: {len(trades)}\n"
                    f"Winners: {len(winners)}\n"
                    f"Win Rate: {summary.win_rate:.1f}%\n"
                    f"Total P&L: ${total_pnl:+,.2f}\n"
                    f"Best: ${summary.best_trade_pnl:+,.2f}\n"
                    f"Worst: ${summary.worst_trade_pnl:+,.2f}\n"
                )
                await self.telegram_bot.send_notification(msg)

        finally:
            session.close()
