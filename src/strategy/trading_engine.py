import asyncio
import json
from datetime import datetime, date, time as dt_time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.broker.ibkr_client import IBKRClient
from src.analysis.tradingview_data import TradingViewAnalyzer
from src.analysis.pattern_analyzer import PatternAnalyzer
from src.strategy.signal_combiner import SignalCombiner, TradeSignal
from src.risk.risk_manager import RiskManager
from src.risk.correlation_filter import CorrelationFilter
from src.agents.agent_manager import AgentManager
from src.utils.database import get_session, Trade, DailySummary, IndicatorAccuracy, init_db
from src.utils.performance_tracker import PerformanceTracker
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("engine")


class TradingEngine:
    """Main trading engine that orchestrates the full trading pipeline."""

    def __init__(self):
        self.broker = IBKRClient()
        self.tv_analyzer = TradingViewAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.signal_combiner = SignalCombiner()
        self.risk_manager = RiskManager()
        self.correlation_filter = CorrelationFilter()
        self.agent_manager = AgentManager()
        self.config = get_config()
        self.running = False
        self.telegram_bot = None  # Set externally after init

        # Track highest/lowest prices for trailing stops
        self._price_extremes = {}  # {symbol: highest_or_lowest_price}
        self._cached_portfolio_value = None  # Cached per scan cycle
        self._current_regime = None  # Cached market regime
        self.performance_tracker = PerformanceTracker()

        init_db()

        # Market hours config for gating expensive operations
        tz_name = self.config["schedule"].get("timezone", "US/Eastern")
        self._market_tz = ZoneInfo(tz_name)
        open_parts = self.config["schedule"].get("market_open", "09:30").split(":")
        close_parts = self.config["schedule"].get("market_close", "16:00").split(":")
        self._market_open = dt_time(int(open_parts[0]), int(open_parts[1]))
        self._market_close = dt_time(int(close_parts[0]), int(close_parts[1]))

    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (Mon-Fri, open-close)."""
        now = datetime.now(self._market_tz)
        # Weekday: 0=Mon, 4=Fri
        if now.weekday() > 4:
            return False
        current_time = now.time()
        return self._market_open <= current_time <= self._market_close

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

        if not self._is_market_hours():
            logger.info("Outside market hours, skipping scan (no API calls)")
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
        self.correlation_filter.clear_cache()
        self._cached_portfolio_value = None

        # Refresh market regime (cached, only calls API if stale)
        try:
            self._current_regime = await self.agent_manager.get_market_regime(watchlist)
            if self._current_regime:
                logger.info("Market regime: %s (VIX=%.1f, scale=%.1f)",
                            self._current_regime.regime,
                            self._current_regime.vix_level,
                            self._current_regime.recommended_position_scale)
        except Exception as e:
            logger.warning("Failed to get market regime: %s", e)

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
            # Get open position symbols for correlation check
            open_symbols = [
                t.symbol for t in session.query(Trade).filter(Trade.status == "OPEN").all()
            ]
        finally:
            session.close()

        # Correlation filter — block highly correlated or sector-concentrated trades
        can_trade_corr, corr_reason, _ = self.correlation_filter.check(symbol, open_symbols)
        if not can_trade_corr:
            logger.info("%s blocked by correlation filter: %s", symbol, corr_reason)
            return None

        # Earnings/sentiment check — block if earnings too close
        sentiment_score = 0.0
        try:
            sentiment = await self.agent_manager.get_sentiment(symbol)
            if sentiment:
                if sentiment.should_block_trade:
                    logger.info("%s blocked: earnings within %d days (%s)",
                                symbol, self.config.get("agents", {}).get("news_sentiment", {}).get("earnings_block_days", 2),
                                sentiment.earnings_date)
                    return None
                sentiment_score = sentiment.sentiment_score
        except Exception as e:
            logger.warning("Sentiment check failed for %s: %s", symbol, e)

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

        # 5. Local pattern analysis (candlestick patterns, S/R, trend — no API call)
        ai_analysis = {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": "Disabled"}
        if self.config["ai"].get("chart_analysis_enabled", True):
            try:
                # Get historical bars from broker for pattern analysis
                bars = await self.broker.get_historical_bars(
                    symbol, duration="5 D", bar_size="5 mins"
                )
                if bars:
                    df = pd.DataFrame(bars)
                    ai_analysis = self.pattern_analyzer.analyze(df, symbol, "5m")
            except Exception as e:
                logger.warning("Pattern analysis skipped for %s: %s", symbol, e)

        # 6. Determine signal direction from all sources FIRST
        regime_confidence = None
        regime_name = None
        if self._current_regime:
            regime_confidence = self._current_regime.recommended_confidence_threshold
            regime_name = self._current_regime.regime

        direction = self.signal_combiner.evaluate_direction(
            symbol=symbol,
            tv_analysis=tv_analysis,
            tv_indicator_signals=tv_signals,
            ai_analysis=ai_analysis,
            multi_tf_analyses=multi_tf,
            sentiment_score=sentiment_score,
            confidence_threshold_override=regime_confidence,
            regime=regime_name,
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

        # Use regime-adjusted multipliers and position scaling
        sl_override = None
        tp_override = None
        scale = 1.0
        if self._current_regime:
            sl_override = self._current_regime.recommended_sl_multiplier
            tp_override = self._current_regime.recommended_tp_multiplier
            scale = self._current_regime.recommended_position_scale

        trade_levels = self.risk_manager.get_trade_levels(
            current_price, atr, direction["side"], portfolio_value,
            sl_multiplier_override=sl_override,
            tp_multiplier_override=tp_override,
            scale_factor=scale,
        )

        # 8. Build final signal with correctly matched trade levels
        signal = self.signal_combiner.build_signal(
            symbol=symbol,
            direction=direction,
            trade_levels=trade_levels,
        )

        return signal

    async def execute_signal(self, signal: TradeSignal):
        """Execute a trade signal through the broker.

        Uses market order for immediate entry, then places SL/TP only after
        entry is confirmed filled. Recalculates SL/TP from actual fill price.
        """
        logger.info("Executing signal: %s %s", signal.side, signal.symbol)

        action = "BUY" if signal.side == "LONG" else "SELL"
        exit_action = "SELL" if action == "BUY" else "BUY"

        try:
            # 1. Market entry — wait for fill
            entry_trade, fill_price = await self.broker.place_entry_order(
                symbol=signal.symbol,
                action=action,
                quantity=signal.quantity,
            )

            # 2. Recalculate SL/TP from actual fill price (not stale analysis price)
            atr = abs(signal.take_profit - signal.entry_price) / self.config["risk"]["take_profit"]["atr_multiplier"]
            trade_levels = self.risk_manager.get_trade_levels(
                fill_price, atr, signal.side,
                self._cached_portfolio_value or 100000,
            )
            stop_loss = trade_levels["stop_loss"]
            take_profit = trade_levels["take_profit"]

            # 3. Place SL/TP only after entry is confirmed
            tp_trade, sl_trade = await self.broker.place_exit_orders(
                symbol=signal.symbol,
                exit_action=exit_action,
                quantity=signal.quantity,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )

            # Record trade in database with actual fill price
            session = get_session()
            try:
                db_trade = Trade(
                    symbol=signal.symbol,
                    side=signal.side,
                    entry_price=fill_price,
                    quantity=signal.quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status="OPEN",
                    strategy=signal.strategy,
                    timeframe=signal.timeframe,
                    signals=json.dumps(signal.signals_detail),
                    order_id=sl_trade.order.orderId,
                )
                session.add(db_trade)
                session.commit()
                logger.info(
                    "Trade recorded in DB: %s %s | fill=%.2f | SL=%.2f | TP=%.2f",
                    signal.side, signal.symbol, fill_price, stop_loss, take_profit,
                )
            finally:
                session.close()

            # Initialize price tracking for trailing stop
            self._price_extremes[signal.symbol] = fill_price

            # Send Telegram alert
            if self.telegram_bot:
                await self.telegram_bot.send_trade_alert({
                    "side": signal.side,
                    "symbol": signal.symbol,
                    "entry_price": fill_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
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

                    # Get ATR for volatility-adaptive trailing stop
                    atr = None
                    try:
                        tv_analysis = self.tv_analyzer.get_analysis(trade.symbol, "1h")
                        atr = tv_analysis.get("indicators", {}).get("atr")
                    except Exception:
                        pass

                    # Calculate new trailing stop (ATR-based if available)
                    new_sl = self.risk_manager.calculate_trailing_stop(
                        entry_price=trade.entry_price,
                        current_price=current_price,
                        current_stop=trade.stop_loss,
                        highest_price=extreme,
                        side=trade.side,
                        atr=atr,
                    )

                    # Only update if the change is meaningful (>$0.10)
                    # to avoid micro-adjustments from bid/ask noise
                    sl_diff = abs(new_sl - trade.stop_loss)
                    if sl_diff > 0.10 and new_sl != trade.stop_loss:
                        if trade.order_id:
                            success = await self.broker.modify_stop_loss(trade.order_id, new_sl)
                        else:
                            success = True  # Paper mode — no broker order to modify
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
        """Check broker for filled orders and update trade records.

        Uses actual broker fill prices instead of stale market data for
        accurate P&L calculation. Tracks exit reason (SL/TP/trailing).
        """
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            broker_positions = await self.broker.get_positions()
            position_symbols = {p["symbol"] for p in broker_positions}

            for trade in open_trades:
                if trade.symbol not in position_symbols:
                    # Position was closed — get actual fill price from broker
                    exit_price = None
                    try:
                        fills = await self.broker.get_recent_fills(trade.symbol)
                        # Find the exit fill (opposite side of entry)
                        exit_side = "SLD" if trade.side == "LONG" else "BOT"
                        exit_fills = [f for f in fills if f["side"] == exit_side]
                        if exit_fills:
                            exit_price = exit_fills[-1]["price"]
                    except Exception as e:
                        logger.warning("Could not get fills for %s: %s", trade.symbol, e)

                    # Fallback to market data if fills unavailable
                    if not exit_price:
                        market_data = await self.broker.get_market_data(trade.symbol)
                        exit_price = market_data.get("last") or trade.entry_price

                    if trade.side == "LONG":
                        pnl = (exit_price - trade.entry_price) * trade.quantity
                    else:
                        pnl = (trade.entry_price - exit_price) * trade.quantity

                    pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100

                    # Determine exit reason from price proximity to SL/TP
                    exit_reason = self._infer_exit_reason(trade, exit_price)

                    trade.exit_price = exit_price
                    trade.exit_time = datetime.utcnow()
                    trade.pnl = round(pnl, 2)
                    trade.pnl_pct = round(pnl_pct, 2)
                    trade.status = "CLOSED"
                    trade.exit_reason = exit_reason
                    session.commit()

                    logger.info(
                        "Trade closed: %s %s | PnL: $%.2f (%.2f%%) | reason: %s",
                        trade.side, trade.symbol, pnl, pnl_pct, exit_reason,
                    )

                    # Notify via Telegram
                    if self.telegram_bot:
                        msg = (
                            f"TRADE CLOSED ({exit_reason})\n"
                            f"{trade.side} {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.2f} -> Exit: ${exit_price:.2f}\n"
                            f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
                        )
                        await self.telegram_bot.send_notification(msg)

                    # Fire trade review agent (async, don't block)
                    if self.config.get("agents", {}).get("trade_review", {}).get("enabled"):
                        asyncio.create_task(self.agent_manager.review_trade(trade.id))

                    # Clean up tracking
                    self._price_extremes.pop(trade.symbol, None)

        finally:
            session.close()

    @staticmethod
    def _infer_exit_reason(trade: Trade, exit_price: float) -> str:
        """Infer why a trade was closed based on exit price vs SL/TP levels."""
        sl_tolerance = abs(trade.stop_loss * 0.005)  # 0.5% tolerance
        tp_tolerance = abs(trade.take_profit * 0.005)

        if trade.side == "LONG":
            if exit_price <= trade.stop_loss + sl_tolerance:
                return "SL_HIT"
            if exit_price >= trade.take_profit - tp_tolerance:
                return "TP_HIT"
        else:  # SHORT
            if exit_price >= trade.stop_loss - sl_tolerance:
                return "SL_HIT"
            if exit_price <= trade.take_profit + tp_tolerance:
                return "TP_HIT"

        return "TRAILING_STOP"

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

            # Evaluate daily performance against targets
            try:
                portfolio_value = self._cached_portfolio_value or 100000
                result = self.performance_tracker.evaluate_day(portfolio_value)
                if result.get("trades", 0) > 0 and self.telegram_bot:
                    await self.telegram_bot.send_notification(result["message"])
            except Exception as e:
                logger.error("Performance target evaluation failed: %s", e)

            # Generate AI improvement suggestions
            try:
                await self._generate_eod_suggestions(session, trades, today_str)
            except Exception as e:
                logger.error("EOD suggestions failed: %s", e)

        finally:
            session.close()

    async def _generate_eod_suggestions(self, session, today_trades, today_str):
        """Generate end-of-day AI suggestions for bot improvement."""
        if not self.telegram_bot:
            return

        # Gather data for analysis
        # 1. Today's closed trades with details
        trade_details = []
        for t in today_trades:
            signals = {}
            try:
                signals = json.loads(t.signals or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            trade_details.append({
                "symbol": t.symbol,
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "signals": signals,
            })

        # 2. Open positions still held
        open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
        open_details = []
        for t in open_trades:
            open_details.append({
                "symbol": t.symbol,
                "side": t.side,
                "entry": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "entry_time": str(t.entry_time),
            })

        # 3. Indicator accuracy stats
        accuracy_records = session.query(IndicatorAccuracy).all()
        accuracy_stats = [
            {"name": r.indicator_name, "accuracy": r.accuracy_pct,
             "correct": r.correct_signals, "total": r.total_signals}
            for r in accuracy_records
        ]

        # 4. Recent daily summaries for trend context
        recent_summaries = session.query(DailySummary).order_by(
            DailySummary.date.desc()
        ).limit(5).all()
        summary_history = [
            {"date": s.date, "trades": s.total_trades, "win_rate": s.win_rate,
             "pnl": s.total_pnl}
            for s in recent_summaries
        ]

        # 5. Current config parameters
        config = self.config
        params = {
            "confidence_threshold": config["ai"]["confidence_threshold"],
            "max_open_positions": config["trading"]["max_open_positions"],
            "max_daily_trades": config["trading"]["max_daily_trades"],
            "risk_per_trade_pct": config["trading"]["risk_per_trade_pct"],
            "weights": {
                "tradingview_summary": 0.25,
                "tradingview_indicators": 0.25,
                "ai_chart": 0.15,
                "multi_timeframe": 0.35,
            },
            "volume_filter": "0.8x SMA20",
            "adx_min": 15,
        }

        system_prompt = (
            "You are a trading bot performance analyst. Analyze today's trading session "
            "and provide specific, actionable suggestions to improve the bot's code and parameters. "
            "The user will copy-paste your suggestions to a developer to implement.\n\n"
            "Focus on:\n"
            "1. Parameter adjustments (confidence threshold, signal weights, risk per trade, SL/TP ratios)\n"
            "2. Filter improvements (volume filter, ADX threshold, correlation limits)\n"
            "3. Strategy observations (which indicators are working/failing, pattern recognition issues)\n"
            "4. Risk management (position sizing, trailing stop behavior, drawdown patterns)\n"
            "5. Any code changes that could improve performance\n\n"
            "Be specific with numbers. Say 'change confidence_threshold from 0.55 to 0.50' not 'lower the threshold'.\n"
            "If there were no trades, focus on why (filters too strict?) and suggest loosening specific parameters.\n\n"
            "Respond with JSON: {\"suggestions\": [\"suggestion 1\", \"suggestion 2\", ...]}"
        )

        user_prompt = (
            f"Date: {today_str}\n\n"
            f"TODAY'S CLOSED TRADES ({len(trade_details)}):\n"
            f"{json.dumps(trade_details, indent=2)}\n\n"
            f"OPEN POSITIONS ({len(open_details)}):\n"
            f"{json.dumps(open_details, indent=2)}\n\n"
            f"INDICATOR ACCURACY (all-time):\n"
            f"{json.dumps(accuracy_stats, indent=2)}\n\n"
            f"RECENT DAILY SUMMARIES:\n"
            f"{json.dumps(summary_history, indent=2)}\n\n"
            f"CURRENT BOT PARAMETERS:\n"
            f"{json.dumps(params, indent=2)}\n\n"
            f"Provide 3-7 specific suggestions to improve the bot."
        )

        # Use the trade review agent's Claude access
        review_agent = self.agent_manager.review_agent
        result = review_agent._call_claude(system_prompt, user_prompt, max_tokens=1500)

        if "error" in result:
            logger.error("EOD suggestions Claude call failed: %s", result["error"])
            return

        # Format suggestions
        suggestions = result.get("suggestions", [])
        if isinstance(suggestions, list) and suggestions:
            suggestions_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))
        else:
            suggestions_text = json.dumps(result, indent=2)

        msg = (
            f"EOD IMPROVEMENT SUGGESTIONS\n"
            f"{'='*30}\n"
            f"{today_str} | {len(today_trades)} trades\n\n"
            f"{suggestions_text}\n\n"
            f"Copy these suggestions to Claude Code to implement them."
        )

        # Split long messages for Telegram (4096 char limit)
        if len(msg) > 4000:
            parts = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
            for part in parts:
                await self.telegram_bot.send_notification(part)
        else:
            await self.telegram_bot.send_notification(msg)

        logger.info("EOD suggestions sent via Telegram")
