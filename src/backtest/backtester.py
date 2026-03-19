import json
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import yfinance as yf
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.utils.database import get_session, BacktestResult

logger = setup_logger("backtest")


@dataclass
class BacktestTrade:
    entry_date: str
    exit_date: str
    side: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # "TP", "SL", "TRAILING_SL", "END"


@dataclass
class BacktestReport:
    symbol: str
    strategy: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    avg_rr: float
    trades: list = field(default_factory=list)


class Backtester:
    """Backtests trading strategies on historical data."""

    def __init__(self):
        self.config = get_config()
        self.risk_config = self.config["risk"]
        self.indicator_config = self.config["indicators"]

    def fetch_historical_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.dropna(inplace=True)
        logger.info("Fetched %d bars for %s (%s, %s)", len(df), symbol, period, interval)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        cfg = self.indicator_config

        # RSI
        rsi = RSIIndicator(df["close"], window=cfg["rsi"]["period"])
        df["rsi"] = rsi.rsi()

        # MACD
        macd = MACD(
            df["close"],
            window_fast=cfg["macd"]["fast"],
            window_slow=cfg["macd"]["slow"],
            window_sign=cfg["macd"]["signal"],
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # EMAs
        df["ema_short"] = EMAIndicator(df["close"], window=cfg["ema"]["short"]).ema_indicator()
        df["ema_medium"] = EMAIndicator(df["close"], window=cfg["ema"]["medium"]).ema_indicator()
        df["ema_long"] = EMAIndicator(df["close"], window=cfg["ema"]["long"]).ema_indicator()

        # Bollinger Bands
        bb = BollingerBands(df["close"], window=cfg["bollinger"]["period"], window_dev=cfg["bollinger"]["std_dev"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()

        # ATR
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"] = atr.average_true_range()

        # Volume SMA
        df["volume_sma"] = df["volume"].rolling(cfg["volume"]["sma_period"]).mean()

        df.dropna(inplace=True)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell/short signals based on indicator confluence."""
        cfg = self.indicator_config
        df["signal"] = 0  # 0=none, 1=long, -1=short

        for i in range(1, len(df)):
            buy_score = 0
            sell_score = 0

            # RSI
            if df["rsi"].iloc[i] < cfg["rsi"]["oversold"]:
                buy_score += 1
            elif df["rsi"].iloc[i] > cfg["rsi"]["overbought"]:
                sell_score += 1

            # MACD crossover (strong signal)
            if df["macd"].iloc[i] > df["macd_signal"].iloc[i] and df["macd"].iloc[i - 1] <= df["macd_signal"].iloc[i - 1]:
                buy_score += 1.5
            elif df["macd"].iloc[i] < df["macd_signal"].iloc[i] and df["macd"].iloc[i - 1] >= df["macd_signal"].iloc[i - 1]:
                sell_score += 1.5

            # MACD direction (weaker but more frequent)
            if df["macd"].iloc[i] > df["macd_signal"].iloc[i]:
                buy_score += 0.5
            elif df["macd"].iloc[i] < df["macd_signal"].iloc[i]:
                sell_score += 0.5

            # EMA alignment (full)
            if df["ema_short"].iloc[i] > df["ema_medium"].iloc[i] > df["ema_long"].iloc[i]:
                buy_score += 1
            elif df["ema_short"].iloc[i] < df["ema_medium"].iloc[i] < df["ema_long"].iloc[i]:
                sell_score += 1

            # EMA short/medium crossover
            if df["ema_short"].iloc[i] > df["ema_medium"].iloc[i] and df["ema_short"].iloc[i - 1] <= df["ema_medium"].iloc[i - 1]:
                buy_score += 0.5
            elif df["ema_short"].iloc[i] < df["ema_medium"].iloc[i] and df["ema_short"].iloc[i - 1] >= df["ema_medium"].iloc[i - 1]:
                sell_score += 0.5

            # Bollinger Band touch
            if df["close"].iloc[i] <= df["bb_lower"].iloc[i]:
                buy_score += 0.5
            elif df["close"].iloc[i] >= df["bb_upper"].iloc[i]:
                sell_score += 0.5

            # Volume confirmation
            if df["volume"].iloc[i] > df["volume_sma"].iloc[i] * 1.5:
                if buy_score > sell_score:
                    buy_score += 0.5
                elif sell_score > buy_score:
                    sell_score += 0.5

            # Need at least 1.0 score for a signal
            if buy_score >= 1.0 and buy_score > sell_score:
                df.iloc[i, df.columns.get_loc("signal")] = 1
            elif sell_score >= 1.0 and sell_score > buy_score:
                df.iloc[i, df.columns.get_loc("signal")] = -1

        return df

    def run_backtest(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        initial_capital: float = 10000.0,
    ) -> BacktestReport:
        """Run a full backtest for a symbol."""
        logger.info("Starting backtest: %s | period=%s | interval=%s", symbol, period, interval)

        df = self.fetch_historical_data(symbol, period, interval)
        df = self.add_indicators(df)
        df = self.generate_signals(df)

        trades = []
        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0.0
        # Support multiple concurrent positions
        open_positions = []  # list of dicts with trade info

        for i in range(len(df)):
            row = df.iloc[i]

            # Check exits for all open positions
            closed_indices = []
            for idx, pos in enumerate(open_positions):
                exit_price = None
                exit_reason = None

                if pos["side"] == "LONG":
                    if row["high"] > pos["highest_price"]:
                        pos["highest_price"] = row["high"]
                    profit_pct = ((row["close"] - pos["entry"]) / pos["entry"]) * 100
                    if profit_pct >= self.risk_config["trailing_stop"]["activation_pct"]:
                        new_sl = pos["highest_price"] * (1 - self.risk_config["trailing_stop"]["trail_pct"] / 100)
                        if new_sl > pos["sl"]:
                            pos["sl"] = new_sl

                    if row["low"] <= pos["sl"]:
                        exit_price = pos["sl"]
                        original_sl = pos["entry"] - df["atr"].iloc[i] * self.risk_config["stop_loss"]["atr_multiplier"]
                        exit_reason = "TRAILING_SL" if pos["sl"] > original_sl else "SL"
                    elif row["high"] >= pos["tp"]:
                        exit_price = pos["tp"]
                        exit_reason = "TP"

                else:  # SHORT
                    if row["low"] < pos["highest_price"]:
                        pos["highest_price"] = row["low"]
                    profit_pct = ((pos["entry"] - row["close"]) / pos["entry"]) * 100
                    if profit_pct >= self.risk_config["trailing_stop"]["activation_pct"]:
                        new_sl = pos["highest_price"] * (1 + self.risk_config["trailing_stop"]["trail_pct"] / 100)
                        if new_sl < pos["sl"]:
                            pos["sl"] = new_sl

                    if row["high"] >= pos["sl"]:
                        exit_price = pos["sl"]
                        exit_reason = "SL"
                    elif row["low"] <= pos["tp"]:
                        exit_price = pos["tp"]
                        exit_reason = "TP"

                if exit_price is not None:
                    if pos["side"] == "LONG":
                        pnl = (exit_price - pos["entry"]) * pos["qty"]
                    else:
                        pnl = (pos["entry"] - exit_price) * pos["qty"]

                    pnl_pct = (pnl / (pos["entry"] * pos["qty"])) * 100
                    capital += pnl

                    trades.append(BacktestTrade(
                        entry_date=str(pos["entry_date"]),
                        exit_date=str(row.name),
                        side=pos["side"],
                        entry_price=pos["entry"],
                        exit_price=exit_price,
                        stop_loss=pos["sl"],
                        take_profit=pos["tp"],
                        quantity=pos["qty"],
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        exit_reason=exit_reason,
                    ))

                    peak_capital = max(peak_capital, capital)
                    dd = ((peak_capital - capital) / peak_capital) * 100
                    max_drawdown = max(max_drawdown, dd)
                    closed_indices.append(idx)

            # Remove closed positions (reverse order to preserve indices)
            for idx in reversed(closed_indices):
                open_positions.pop(idx)

            # Open new trade on signal (allow multiple concurrent)
            if row["signal"] != 0:
                atr = row["atr"]
                entry = row["close"]

                if row["signal"] == 1:
                    side = "LONG"
                    sl = entry - atr * self.risk_config["stop_loss"]["atr_multiplier"]
                    tp = entry + atr * self.risk_config["take_profit"]["atr_multiplier"]
                    highest = entry
                else:
                    side = "SHORT"
                    sl = entry + atr * self.risk_config["stop_loss"]["atr_multiplier"]
                    tp = entry - atr * self.risk_config["take_profit"]["atr_multiplier"]
                    highest = entry

                # Ensure minimum RR
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                if risk > 0 and reward / risk < self.risk_config["risk_reward_ratio"]:
                    tp = entry + (risk * self.risk_config["risk_reward_ratio"]) if side == "LONG" \
                        else entry - (risk * self.risk_config["risk_reward_ratio"])

                # Position sizing
                risk_per_share = abs(entry - sl)
                max_risk = capital * (self.risk_config["max_risk_per_trade_pct"] / 100)
                qty = max(1, int(max_risk / risk_per_share)) if risk_per_share > 0 else 0

                if qty > 0:
                    open_positions.append({
                        "entry": entry,
                        "entry_date": row.name,
                        "side": side,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "highest_price": highest,
                    })

        # Close any remaining open positions at last price
        if open_positions:
            last_row = df.iloc[-1]
            for pos in open_positions:
                exit_price = last_row["close"]
                if pos["side"] == "LONG":
                    pnl = (exit_price - pos["entry"]) * pos["qty"]
                else:
                    pnl = (pos["entry"] - exit_price) * pos["qty"]
                pnl_pct = (pnl / (pos["entry"] * pos["qty"])) * 100
                capital += pnl
                trades.append(BacktestTrade(
                    entry_date=str(pos["entry_date"]),
                    exit_date=str(last_row.name),
                    side=pos["side"],
                    entry_price=pos["entry"],
                    exit_price=exit_price,
                    stop_loss=pos["sl"],
                    take_profit=pos["tp"],
                    quantity=pos["qty"],
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    exit_reason="END",
                ))

        # Calculate stats
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))

        report = BacktestReport(
            symbol=symbol,
            strategy="indicator_confluence",
            timeframe=interval,
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            initial_capital=initial_capital,
            final_capital=round(capital, 2),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(len(winning) / len(trades) * 100, 1) if trades else 0,
            profit_factor=round(total_wins / total_losses, 2) if total_losses > 0 else float("inf"),
            total_return_pct=round(((capital - initial_capital) / initial_capital) * 100, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            sharpe_ratio=self._calculate_sharpe(trades, initial_capital),
            avg_win=round(total_wins / len(winning), 2) if winning else 0,
            avg_loss=round(total_losses / len(losing), 2) if losing else 0,
            avg_rr=round((total_wins / len(winning)) / (total_losses / len(losing)), 2) if winning and losing else 0,
            trades=trades,
        )

        logger.info(
            "Backtest complete: %s | trades=%d | win_rate=%.1f%% | return=%.2f%% | max_dd=%.2f%%",
            symbol, report.total_trades, report.win_rate, report.total_return_pct, report.max_drawdown_pct,
        )

        self._save_result(report)
        return report

    def _calculate_sharpe(self, trades: list, initial_capital: float) -> float:
        if len(trades) < 2:
            return 0.0
        returns = [t.pnl_pct / 100 for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        # Annualize (approximate)
        return round(avg_return / std_return * np.sqrt(252), 2)

    def _save_result(self, report: BacktestReport):
        session = get_session()
        try:
            result = BacktestResult(
                symbol=report.symbol,
                strategy=report.strategy,
                timeframe=report.timeframe,
                start_date=report.start_date,
                end_date=report.end_date,
                total_trades=report.total_trades,
                win_rate=report.win_rate,
                profit_factor=report.profit_factor,
                total_return_pct=report.total_return_pct,
                max_drawdown_pct=report.max_drawdown_pct,
                sharpe_ratio=report.sharpe_ratio,
                details=json.dumps([
                    {
                        "entry": t.entry_date, "exit": t.exit_date,
                        "side": t.side, "entry_price": t.entry_price,
                        "exit_price": t.exit_price, "pnl": t.pnl,
                        "exit_reason": t.exit_reason,
                    }
                    for t in report.trades
                ]),
            )
            session.add(result)
            session.commit()
        finally:
            session.close()

    def format_report(self, report: BacktestReport) -> str:
        """Format backtest report as readable text."""
        lines = [
            f"{'='*50}",
            f"BACKTEST REPORT: {report.symbol}",
            f"{'='*50}",
            f"Strategy:       {report.strategy}",
            f"Timeframe:      {report.timeframe}",
            f"Period:         {report.start_date} to {report.end_date}",
            f"",
            f"Initial Capital: ${report.initial_capital:,.2f}",
            f"Final Capital:   ${report.final_capital:,.2f}",
            f"Total Return:    {report.total_return_pct:+.2f}%",
            f"",
            f"Total Trades:    {report.total_trades}",
            f"Winning Trades:  {report.winning_trades}",
            f"Losing Trades:   {report.losing_trades}",
            f"Win Rate:        {report.win_rate:.1f}%",
            f"",
            f"Profit Factor:   {report.profit_factor:.2f}",
            f"Max Drawdown:    {report.max_drawdown_pct:.2f}%",
            f"Sharpe Ratio:    {report.sharpe_ratio:.2f}",
            f"Avg Win:         ${report.avg_win:,.2f}",
            f"Avg Loss:        ${report.avg_loss:,.2f}",
            f"Avg R:R:         {report.avg_rr:.2f}",
            f"{'='*50}",
        ]
        return "\n".join(lines)
