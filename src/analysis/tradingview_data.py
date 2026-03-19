from tradingview_ta import TA_Handler, Interval, Exchange
from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("analysis")

INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "30m": Interval.INTERVAL_30_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "2h": Interval.INTERVAL_2_HOURS,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1M": Interval.INTERVAL_1_MONTH,
}


EXCHANGE_MAP = {
    "SPY": "AMEX", "QQQ": "NASDAQ", "IWM": "AMEX", "DIA": "AMEX",
    "GLD": "AMEX", "SLV": "AMEX", "TLT": "NASDAQ", "XLF": "AMEX",
    "XLE": "AMEX", "XLK": "AMEX", "VTI": "AMEX", "VOO": "AMEX",
}

EXCHANGE_FALLBACKS = ["NASDAQ", "NYSE", "AMEX"]


class TradingViewAnalyzer:
    """Fetch technical indicators and recommendations from TradingView."""

    def __init__(self):
        self.config = get_config()["indicators"]
        self._exchange_cache = {}

    def _get_exchange(self, symbol: str) -> str:
        """Get the exchange for a symbol, with caching."""
        if symbol in self._exchange_cache:
            return self._exchange_cache[symbol]
        if symbol in EXCHANGE_MAP:
            return EXCHANGE_MAP[symbol]
        return "NASDAQ"

    def get_analysis(self, symbol: str, interval: str = "1h") -> dict:
        """Get full TradingView analysis for a symbol."""
        try:
            tv_interval = INTERVAL_MAP.get(interval, Interval.INTERVAL_1_HOUR)

            # Try cached/mapped exchange first, then fallbacks
            exchanges_to_try = [self._get_exchange(symbol)]
            for ex in EXCHANGE_FALLBACKS:
                if ex not in exchanges_to_try:
                    exchanges_to_try.append(ex)

            analysis = None
            for exchange in exchanges_to_try:
                try:
                    handler = TA_Handler(
                        symbol=symbol,
                        screener="america",
                        exchange=exchange,
                        interval=tv_interval,
                    )
                    analysis = handler.get_analysis()
                    if analysis and analysis.indicators.get("close") is not None:
                        self._exchange_cache[symbol] = exchange
                        logger.info("Using exchange %s for %s", exchange, symbol)
                        break
                except Exception:
                    continue

            if analysis is None:
                logger.error("No valid exchange found for %s", symbol)
                return {"symbol": symbol, "error": "No valid exchange found"}

            return {
                "symbol": symbol,
                "interval": interval,
                "summary": {
                    "recommendation": analysis.summary["RECOMMENDATION"],
                    "buy_signals": analysis.summary["BUY"],
                    "sell_signals": analysis.summary["SELL"],
                    "neutral_signals": analysis.summary["NEUTRAL"],
                },
                "oscillators": {
                    "recommendation": analysis.oscillators["RECOMMENDATION"],
                    "rsi": analysis.indicators.get("RSI"),
                    "stoch_k": analysis.indicators.get("Stoch.K"),
                    "stoch_d": analysis.indicators.get("Stoch.D"),
                    "cci": analysis.indicators.get("CCI20"),
                    "adx": analysis.indicators.get("ADX"),
                    "ao": analysis.indicators.get("AO"),
                    "momentum": analysis.indicators.get("Mom"),
                    "macd": analysis.indicators.get("MACD.macd"),
                    "macd_signal": analysis.indicators.get("MACD.signal"),
                },
                "moving_averages": {
                    "recommendation": analysis.moving_averages["RECOMMENDATION"],
                    "ema10": analysis.indicators.get("EMA10"),
                    "ema20": analysis.indicators.get("EMA20"),
                    "ema50": analysis.indicators.get("EMA50"),
                    "ema100": analysis.indicators.get("EMA100"),
                    "ema200": analysis.indicators.get("EMA200"),
                    "sma10": analysis.indicators.get("SMA10"),
                    "sma20": analysis.indicators.get("SMA20"),
                    "sma50": analysis.indicators.get("SMA50"),
                    "sma100": analysis.indicators.get("SMA100"),
                    "sma200": analysis.indicators.get("SMA200"),
                },
                "indicators": {
                    "rsi": analysis.indicators.get("RSI"),
                    "macd": analysis.indicators.get("MACD.macd"),
                    "macd_signal": analysis.indicators.get("MACD.signal"),
                    "bb_upper": analysis.indicators.get("BB.upper"),
                    "bb_lower": analysis.indicators.get("BB.lower"),
                    "atr": analysis.indicators.get("ATR"),
                    "adx": analysis.indicators.get("ADX"),
                    "volume": analysis.indicators.get("volume"),
                    "close": analysis.indicators.get("close"),
                    "open": analysis.indicators.get("open"),
                    "high": analysis.indicators.get("high"),
                    "low": analysis.indicators.get("low"),
                },
            }
        except Exception as e:
            logger.error("TradingView analysis failed for %s: %s", symbol, e)
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
        """Convert TradingView analysis to a -1.0 to 1.0 score.
        Positive = bullish, Negative = bearish.
        """
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

        # EMA crossover (short vs medium)
        ema_short = analysis["moving_averages"].get("ema10")
        ema_medium = analysis["moving_averages"].get("ema20")
        if ema_short is not None and ema_medium is not None:
            if ema_short > ema_medium:
                signals.append({"indicator": "EMA_CROSS", "signal": "BUY", "value": ema_short, "reason": "EMA10 > EMA20"})
            else:
                signals.append({"indicator": "EMA_CROSS", "signal": "SELL", "value": ema_short, "reason": "EMA10 < EMA20"})

        # Determine overall
        buy_count = sum(1 for s in signals if s["signal"] == "BUY")
        sell_count = sum(1 for s in signals if s["signal"] == "SELL")

        if buy_count > sell_count:
            overall = "BUY"
        elif sell_count > buy_count:
            overall = "SELL"
        else:
            overall = "NEUTRAL"

        return {"overall": overall, "signals": signals, "buy_count": buy_count, "sell_count": sell_count}
