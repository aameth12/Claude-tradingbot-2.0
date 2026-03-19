"""Market data and technical analysis using yfinance + ta library.

Replaces TradingView's unofficial API which was heavily rate-limited (HTTP 429).
Uses Yahoo Finance for price data and the `ta` library to compute indicators locally.
"""
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("analysis")

# Map config interval strings to yfinance intervals and lookback periods
YF_INTERVAL_MAP = {
    "1m": ("1m", "1d"),
    "5m": ("5m", "5d"),
    "15m": ("15m", "5d"),
    "30m": ("30m", "10d"),
    "1h": ("1h", "30d"),
    "2h": ("2h", "60d"),
    "4h": ("4h", "60d"),
    "1d": ("1d", "6mo"),
    "1w": ("1wk", "2y"),
    "1M": ("1mo", "5y"),
}


def _fetch_dataframe(symbol: str, interval: str = "1h", timeout: int = 15) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with timeout."""
    import concurrent.futures

    yf_interval, period = YF_INTERVAL_MAP.get(interval, ("1h", "30d"))

    def _download():
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, interval=yf_interval)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_download)
            df = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("Yahoo Finance timeout for %s (%s) after %ds", symbol, interval, timeout)
        return pd.DataFrame()
    except Exception as e:
        logger.warning("Yahoo Finance error for %s (%s): %s", symbol, interval, e)
        return pd.DataFrame()

    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(inplace=True)
    return df


def _compute_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute all technical indicators on a DataFrame."""
    if len(df) < 30:
        return df

    # RSI
    rsi_period = config["rsi"]["period"]
    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()

    # MACD
    macd = MACD(
        df["close"],
        window_fast=config["macd"]["fast"],
        window_slow=config["macd"]["slow"],
        window_sign=config["macd"]["signal"],
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=config["bollinger"]["period"], window_dev=config["bollinger"]["std_dev"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()

    # ADX
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()

    # Stochastic
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # EMAs
    for period in [10, 20, 50, 100, 200]:
        if len(df) >= period:
            df[f"ema{period}"] = EMAIndicator(df["close"], window=period).ema_indicator()
        else:
            df[f"ema{period}"] = None

    # SMAs
    for period in [10, 20, 50, 100, 200]:
        if len(df) >= period:
            df[f"sma{period}"] = SMAIndicator(df["close"], window=period).sma_indicator()
        else:
            df[f"sma{period}"] = None

    return df


def _generate_recommendation(df: pd.DataFrame, config: dict | None = None) -> str:
    """Generate BUY/SELL/NEUTRAL recommendation from latest indicators."""
    if df.empty or len(df) < 2:
        return "NEUTRAL"

    last = df.iloc[-1]
    buy_signals = 0
    sell_signals = 0
    total = 0

    # RSI — use config thresholds if available
    oversold = config["rsi"]["oversold"] if config else 35
    overbought = config["rsi"]["overbought"] if config else 65
    if pd.notna(last.get("rsi")):
        total += 1
        if last["rsi"] < oversold:
            buy_signals += 1
        elif last["rsi"] > overbought:
            sell_signals += 1

    # MACD
    if pd.notna(last.get("macd")) and pd.notna(last.get("macd_signal")):
        total += 1
        if last["macd"] > last["macd_signal"]:
            buy_signals += 1
        else:
            sell_signals += 1

    # EMA trend — single consolidated vote based on majority of EMAs
    ema_buy = 0
    ema_sell = 0
    ema_count = 0
    for ema in ["ema10", "ema20", "ema50"]:
        if pd.notna(last.get(ema)):
            ema_count += 1
            if last["close"] > last[ema]:
                ema_buy += 1
            else:
                ema_sell += 1
    if ema_count > 0:
        total += 1
        if ema_buy > ema_sell:
            buy_signals += 1
        elif ema_sell > ema_buy:
            sell_signals += 1

    # Stochastic
    if pd.notna(last.get("stoch_k")) and pd.notna(last.get("stoch_d")):
        total += 1
        if last["stoch_k"] < 20 and last["stoch_k"] > last["stoch_d"]:
            buy_signals += 1
        elif last["stoch_k"] > 80 and last["stoch_k"] < last["stoch_d"]:
            sell_signals += 1

    # Bollinger Bands
    if pd.notna(last.get("bb_lower")) and pd.notna(last.get("bb_upper")):
        total += 1
        if last["close"] <= last["bb_lower"]:
            buy_signals += 1
        elif last["close"] >= last["bb_upper"]:
            sell_signals += 1

    if total == 0:
        return "NEUTRAL"

    ratio = (buy_signals - sell_signals) / total
    if ratio >= 0.6:
        return "STRONG_BUY"
    elif ratio >= 0.2:
        return "BUY"
    elif ratio <= -0.6:
        return "STRONG_SELL"
    elif ratio <= -0.2:
        return "SELL"
    return "NEUTRAL"


class TradingViewAnalyzer:
    """Fetch technical indicators using yfinance + ta library.

    Maintains the same interface as the old TradingView-based analyzer
    so the rest of the bot works without changes.
    """

    def __init__(self):
        self.config = get_config()["indicators"]
        self._df_cache = {}  # Cache DataFrames to avoid redundant fetches

    def _get_df(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get DataFrame with indicators, using cache for same scan cycle."""
        cache_key = f"{symbol}_{interval}"
        if cache_key in self._df_cache:
            return self._df_cache[cache_key]

        df = _fetch_dataframe(symbol, interval)
        if not df.empty:
            df = _compute_indicators(df, self.config)
        self._df_cache[cache_key] = df
        return df

    def clear_cache(self):
        """Clear the DataFrame cache between scan cycles."""
        self._df_cache.clear()

    def get_analysis(self, symbol: str, interval: str = "1h") -> dict:
        """Get full analysis for a symbol — compatible with old TradingView format."""
        try:
            df = self._get_df(symbol, interval)

            if df.empty or len(df) < 2:
                logger.error("No data from Yahoo Finance for %s (%s)", symbol, interval)
                return {"symbol": symbol, "error": "No data available"}

            last = df.iloc[-1]
            recommendation = _generate_recommendation(df, self.config)

            # Count buy/sell/neutral from indicators
            buy_count = 0
            sell_count = 0
            neutral_count = 0

            checks = [
                ("rsi", lambda v: v < 40, lambda v: v > 60),
                ("macd", lambda v: v > (last.get("macd_signal") or 0), lambda v: v < (last.get("macd_signal") or 0)),
            ]
            for key, is_buy, is_sell in checks:
                val = last.get(key)
                if pd.notna(val):
                    if is_buy(val):
                        buy_count += 1
                    elif is_sell(val):
                        sell_count += 1
                    else:
                        neutral_count += 1

            # EMA checks
            for ema in ["ema10", "ema20", "ema50"]:
                val = last.get(ema)
                if pd.notna(val):
                    if last["close"] > val:
                        buy_count += 1
                    else:
                        sell_count += 1

            return {
                "symbol": symbol,
                "interval": interval,
                "summary": {
                    "recommendation": recommendation,
                    "buy_signals": buy_count,
                    "sell_signals": sell_count,
                    "neutral_signals": neutral_count,
                },
                "oscillators": {
                    "recommendation": recommendation,
                    "rsi": _safe_float(last.get("rsi")),
                    "stoch_k": _safe_float(last.get("stoch_k")),
                    "stoch_d": _safe_float(last.get("stoch_d")),
                    "cci": None,
                    "adx": _safe_float(last.get("adx")),
                    "ao": None,
                    "momentum": None,
                    "macd": _safe_float(last.get("macd")),
                    "macd_signal": _safe_float(last.get("macd_signal")),
                },
                "moving_averages": {
                    "recommendation": recommendation,
                    "ema10": _safe_float(last.get("ema10")),
                    "ema20": _safe_float(last.get("ema20")),
                    "ema50": _safe_float(last.get("ema50")),
                    "ema100": _safe_float(last.get("ema100")),
                    "ema200": _safe_float(last.get("ema200")),
                    "sma10": _safe_float(last.get("sma10")),
                    "sma20": _safe_float(last.get("sma20")),
                    "sma50": _safe_float(last.get("sma50")),
                    "sma100": _safe_float(last.get("sma100")),
                    "sma200": _safe_float(last.get("sma200")),
                },
                "indicators": {
                    "rsi": _safe_float(last.get("rsi")),
                    "macd": _safe_float(last.get("macd")),
                    "macd_signal": _safe_float(last.get("macd_signal")),
                    "bb_upper": _safe_float(last.get("bb_upper")),
                    "bb_lower": _safe_float(last.get("bb_lower")),
                    "atr": _safe_float(last.get("atr")),
                    "adx": _safe_float(last.get("adx")),
                    "volume": _safe_float(last.get("volume")),
                    "close": _safe_float(last.get("close")),
                    "open": _safe_float(last.get("open")),
                    "high": _safe_float(last.get("high")),
                    "low": _safe_float(last.get("low")),
                },
            }
        except Exception as e:
            logger.error("Analysis failed for %s: %s", symbol, e)
            return {"symbol": symbol, "error": str(e)}

    def get_multi_timeframe_analysis(self, symbol: str) -> dict:
        """Get analysis across multiple timeframes."""
        config = get_config()["timeframes"]
        results = {}

        for style_name, style_tf in config.items():
            results[style_name] = {}
            for tf_role, tf_value in style_tf.items():
                results[style_name][tf_role] = self.get_analysis(symbol, tf_value)

        return results

    def get_signal_score(self, analysis: dict) -> float:
        """Convert analysis to a -1.0 to 1.0 score."""
        if "error" in analysis:
            return 0.0

        summary = analysis["summary"]
        total = summary["buy_signals"] + summary["sell_signals"] + summary["neutral_signals"]
        if total == 0:
            return 0.0

        score = (summary["buy_signals"] - summary["sell_signals"]) / total
        return round(score, 3)

    def check_indicator_signals(self, analysis: dict) -> dict:
        """Check individual indicator signals against configured thresholds."""
        if "error" in analysis:
            return {"overall": "NEUTRAL", "signals": []}

        indicators = analysis["indicators"]
        config = self.config
        signals = []

        # RSI
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < config["rsi"]["oversold"]:
                signals.append({"indicator": "RSI", "signal": "BUY", "value": rsi, "reason": "Oversold"})
            elif rsi > config["rsi"]["overbought"]:
                signals.append({"indicator": "RSI", "signal": "SELL", "value": rsi, "reason": "Overbought"})
            else:
                signals.append({"indicator": "RSI", "signal": "NEUTRAL", "value": rsi, "reason": "Normal range"})

        # MACD
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                signals.append({"indicator": "MACD", "signal": "BUY", "value": macd, "reason": "MACD above signal"})
            else:
                signals.append({"indicator": "MACD", "signal": "SELL", "value": macd, "reason": "MACD below signal"})

        # Bollinger Bands
        close = indicators.get("close")
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        if all(v is not None for v in [close, bb_upper, bb_lower]):
            if close <= bb_lower:
                signals.append({"indicator": "BB", "signal": "BUY", "value": close, "reason": "At lower band"})
            elif close >= bb_upper:
                signals.append({"indicator": "BB", "signal": "SELL", "value": close, "reason": "At upper band"})
            else:
                signals.append({"indicator": "BB", "signal": "NEUTRAL", "value": close, "reason": "Within bands"})

        # EMA crossover
        ema_short = analysis["moving_averages"].get("ema10")
        ema_medium = analysis["moving_averages"].get("ema20")
        if ema_short is not None and ema_medium is not None:
            if ema_short > ema_medium:
                signals.append({"indicator": "EMA_CROSS", "signal": "BUY", "value": ema_short, "reason": "EMA10 > EMA20"})
            else:
                signals.append({"indicator": "EMA_CROSS", "signal": "SELL", "value": ema_short, "reason": "EMA10 < EMA20"})

        buy_count = sum(1 for s in signals if s["signal"] == "BUY")
        sell_count = sum(1 for s in signals if s["signal"] == "SELL")

        if buy_count > sell_count:
            overall = "BUY"
        elif sell_count > buy_count:
            overall = "SELL"
        else:
            overall = "NEUTRAL"

        return {"overall": overall, "signals": signals, "buy_count": buy_count, "sell_count": sell_count}


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 6)
    except (TypeError, ValueError):
        return None
