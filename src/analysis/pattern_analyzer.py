"""Local chart pattern analysis — replaces Claude Vision chart analyzer.

Detects candlestick patterns, support/resistance levels, and trend structure
from OHLCV data. Returns the same format as ChartAnalyzer so the signal
combiner works without changes. Runs locally, $0 API cost, instant.
"""
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("pattern_analyzer")


class PatternAnalyzer:
    """Analyzes OHLCV data for chart patterns, S/R levels, and trend structure."""

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Run full pattern analysis. Returns same format as ChartAnalyzer."""
        try:
            if df.empty or len(df) < 20:
                return _neutral("Insufficient data")

            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            patterns = self._detect_candlestick_patterns(df)
            support, resistance = self._find_support_resistance(df)
            trend, trend_strength = self._determine_trend(df)
            volume_confirms = self._check_volume_confirmation(df, trend)
            momentum = self._assess_momentum(df)

            # Combine into recommendation
            score = self._compute_score(trend, trend_strength, patterns, volume_confirms, momentum)
            recommendation = _score_to_recommendation(score)
            confidence = min(abs(score), 1.0)

            current_price = float(df["close"].iloc[-1])
            atr = self._compute_atr(df)

            result = {
                "trend": trend,
                "trend_strength": round(trend_strength, 2),
                "patterns_detected": [p["name"] for p in patterns],
                "support_levels": support[:3],
                "resistance_levels": resistance[:3],
                "recommendation": recommendation,
                "confidence": round(confidence, 2),
                "entry_zone": {
                    "low": round(current_price - atr * 0.5, 2),
                    "high": round(current_price + atr * 0.5, 2),
                },
                "stop_loss_suggestion": round(current_price - atr * 2 if trend == "BULLISH" else current_price + atr * 2, 2),
                "target_suggestion": round(current_price + atr * 4 if trend == "BULLISH" else current_price - atr * 4, 2),
                "reasoning": self._build_reasoning(trend, patterns, volume_confirms, momentum),
                "long_opportunity": score > 0.1,
                "long_reasoning": f"Trend: {trend}, momentum: {momentum}",
                "short_opportunity": score < -0.1,
                "short_reasoning": f"Trend: {trend}, momentum: {momentum}",
            }

            logger.info(
                "Pattern analysis for %s: %s (confidence: %.2f, patterns: %s)",
                symbol, recommendation, confidence,
                [p["name"] for p in patterns] or "none",
            )
            return result

        except Exception as e:
            logger.error("Pattern analysis failed for %s: %s", symbol, e)
            return _neutral(str(e))

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> list[dict]:
        """Detect common candlestick patterns from recent bars."""
        patterns = []
        if len(df) < 3:
            return patterns

        c = df.iloc[-1]  # current
        p = df.iloc[-2]  # previous
        pp = df.iloc[-3]  # two bars ago

        body = c["close"] - c["open"]
        body_abs = abs(body)
        upper_wick = c["high"] - max(c["open"], c["close"])
        lower_wick = min(c["open"], c["close"]) - c["low"]
        candle_range = c["high"] - c["low"]

        p_body = p["close"] - p["open"]
        p_body_abs = abs(p_body)

        if candle_range == 0:
            return patterns

        # Doji — tiny body relative to range
        if body_abs < candle_range * 0.1:
            patterns.append({"name": "doji", "direction": "neutral", "strength": 0.3})

        # Hammer — small body at top, long lower wick (bullish reversal)
        if (lower_wick > body_abs * 2 and upper_wick < body_abs * 0.5
                and body_abs > 0 and lower_wick > candle_range * 0.6):
            patterns.append({"name": "hammer", "direction": "bullish", "strength": 0.6})

        # Shooting star — small body at bottom, long upper wick (bearish reversal)
        if (upper_wick > body_abs * 2 and lower_wick < body_abs * 0.5
                and body_abs > 0 and upper_wick > candle_range * 0.6):
            patterns.append({"name": "shooting_star", "direction": "bearish", "strength": 0.6})

        # Bullish engulfing — previous red, current green engulfs it
        if (p_body < 0 and body > 0
                and c["open"] <= p["close"] and c["close"] >= p["open"]
                and body_abs > p_body_abs):
            patterns.append({"name": "bullish_engulfing", "direction": "bullish", "strength": 0.7})

        # Bearish engulfing — previous green, current red engulfs it
        if (p_body > 0 and body < 0
                and c["open"] >= p["close"] and c["close"] <= p["open"]
                and body_abs > p_body_abs):
            patterns.append({"name": "bearish_engulfing", "direction": "bearish", "strength": 0.7})

        # Morning star (bullish 3-bar reversal)
        pp_body = pp["close"] - pp["open"]
        if (pp_body < 0 and abs(p["close"] - p["open"]) < candle_range * 0.3
                and body > 0 and c["close"] > (pp["open"] + pp["close"]) / 2):
            patterns.append({"name": "morning_star", "direction": "bullish", "strength": 0.8})

        # Evening star (bearish 3-bar reversal)
        if (pp_body > 0 and abs(p["close"] - p["open"]) < candle_range * 0.3
                and body < 0 and c["close"] < (pp["open"] + pp["close"]) / 2):
            patterns.append({"name": "evening_star", "direction": "bearish", "strength": 0.8})

        # Three white soldiers — three consecutive green candles with higher closes
        if (pp_body > 0 and p_body > 0 and body > 0
                and p["close"] > pp["close"] and c["close"] > p["close"]
                and body_abs > candle_range * 0.3):
            patterns.append({"name": "three_white_soldiers", "direction": "bullish", "strength": 0.8})

        # Three black crows — three consecutive red candles with lower closes
        if (pp_body < 0 and p_body < 0 and body < 0
                and p["close"] < pp["close"] and c["close"] < p["close"]
                and body_abs > candle_range * 0.3):
            patterns.append({"name": "three_black_crows", "direction": "bearish", "strength": 0.8})

        return patterns

    def _find_support_resistance(self, df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
        """Find support/resistance levels from pivot highs/lows."""
        highs = df["high"].values
        lows = df["low"].values
        current_price = float(df["close"].iloc[-1])

        resistance = []
        support = []

        # Find local maxima and minima
        for i in range(window, len(df) - window):
            # Local high (resistance)
            if highs[i] == max(highs[i - window:i + window + 1]):
                resistance.append(round(float(highs[i]), 2))
            # Local low (support)
            if lows[i] == min(lows[i - window:i + window + 1]):
                support.append(round(float(lows[i]), 2))

        # Deduplicate levels within 0.5% of each other
        support = _cluster_levels(sorted(set(support)), current_price)
        resistance = _cluster_levels(sorted(set(resistance), reverse=True), current_price)

        # Keep only levels near current price (within 10%)
        support = [s for s in support if s < current_price and s > current_price * 0.90]
        resistance = [r for r in resistance if r > current_price and r < current_price * 1.10]

        return support, resistance

    def _determine_trend(self, df: pd.DataFrame) -> tuple[str, float]:
        """Determine trend from EMA alignment and slope."""
        close = df["close"]

        if len(df) < 50:
            # Simple: compare first half vs second half
            mid = len(df) // 2
            if close.iloc[-1] > close.iloc[mid]:
                return "BULLISH", 0.4
            elif close.iloc[-1] < close.iloc[mid]:
                return "BEARISH", 0.4
            return "NEUTRAL", 0.2

        ema10 = close.ewm(span=10).mean()
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()

        latest_price = float(close.iloc[-1])
        e10 = float(ema10.iloc[-1])
        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])

        # Perfect bullish alignment: price > EMA10 > EMA20 > EMA50
        bullish_points = 0
        if latest_price > e10:
            bullish_points += 1
        if e10 > e20:
            bullish_points += 1
        if e20 > e50:
            bullish_points += 1
        if latest_price > e50:
            bullish_points += 1

        # EMA slope (10-bar change in EMA20)
        if len(ema20) >= 10:
            slope = (float(ema20.iloc[-1]) - float(ema20.iloc[-10])) / float(ema20.iloc[-10]) * 100
        else:
            slope = 0

        if bullish_points >= 3:
            strength = 0.5 + min(abs(slope) * 0.1, 0.5)
            return "BULLISH", min(strength, 1.0)
        elif bullish_points <= 1:
            strength = 0.5 + min(abs(slope) * 0.1, 0.5)
            return "BEARISH", min(strength, 1.0)
        else:
            return "NEUTRAL", 0.3

    def _check_volume_confirmation(self, df: pd.DataFrame, trend: str) -> bool:
        """Check if volume confirms the current trend."""
        if len(df) < 20:
            return False

        vol = df["volume"]
        vol_sma = vol.rolling(20).mean()

        recent_vol = float(vol.iloc[-5:].mean())
        avg_vol = float(vol_sma.iloc[-1])

        if avg_vol == 0:
            return False

        # Volume should be above average for trend confirmation
        vol_ratio = recent_vol / avg_vol

        if trend in ("BULLISH", "BEARISH") and vol_ratio > 1.2:
            return True
        return False

    def _assess_momentum(self, df: pd.DataFrame) -> str:
        """Assess momentum direction from price rate of change."""
        if len(df) < 14:
            return "neutral"

        close = df["close"]
        roc_5 = (float(close.iloc[-1]) / float(close.iloc[-5]) - 1) * 100
        roc_14 = (float(close.iloc[-1]) / float(close.iloc[-14]) - 1) * 100

        # Both positive and accelerating = increasing
        if roc_5 > 0 and roc_14 > 0 and roc_5 > roc_14 / 2:
            return "increasing"
        elif roc_5 < 0 and roc_14 < 0 and roc_5 < roc_14 / 2:
            return "decreasing"
        elif abs(roc_5) < 0.5:
            return "flat"
        return "mixed"

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute Average True Range."""
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])

        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)

        return float(tr.rolling(period).mean().iloc[-1])

    def _compute_score(self, trend, trend_strength, patterns, volume_confirms, momentum) -> float:
        """Compute overall score from all components."""
        score = 0.0

        # Trend component (40%)
        if trend == "BULLISH":
            score += 0.4 * trend_strength
        elif trend == "BEARISH":
            score -= 0.4 * trend_strength

        # Pattern component (30%)
        for p in patterns:
            weight = p["strength"] * 0.15  # Each pattern contributes up to 15%
            if p["direction"] == "bullish":
                score += weight
            elif p["direction"] == "bearish":
                score -= weight

        # Volume confirmation (15%)
        if volume_confirms:
            if trend == "BULLISH":
                score += 0.15
            elif trend == "BEARISH":
                score -= 0.15

        # Momentum (15%)
        if momentum == "increasing":
            score += 0.15
        elif momentum == "decreasing":
            score -= 0.15

        return max(-1.0, min(1.0, score))

    def _build_reasoning(self, trend, patterns, volume_confirms, momentum) -> str:
        """Build human-readable reasoning string."""
        parts = [f"Trend: {trend}"]
        if patterns:
            parts.append(f"Patterns: {', '.join(p['name'] for p in patterns)}")
        parts.append(f"Volume confirms: {'yes' if volume_confirms else 'no'}")
        parts.append(f"Momentum: {momentum}")
        return " | ".join(parts)


def _cluster_levels(levels: list[float], reference: float, threshold_pct: float = 0.005) -> list[float]:
    """Merge price levels within threshold_pct of each other."""
    if not levels:
        return []
    clustered = [levels[0]]
    for level in levels[1:]:
        if abs(level - clustered[-1]) / max(reference, 1) > threshold_pct:
            clustered.append(level)
    return clustered


def _score_to_recommendation(score: float) -> str:
    if score >= 0.6:
        return "STRONG_BUY"
    elif score >= 0.3:
        return "BUY"
    elif score <= -0.6:
        return "STRONG_SELL"
    elif score <= -0.3:
        return "SELL"
    return "NEUTRAL"


def _neutral(reason: str = "") -> dict:
    return {
        "recommendation": "NEUTRAL",
        "confidence": 0.0,
        "reasoning": reason or "No signal",
        "trend": "NEUTRAL",
        "trend_strength": 0.0,
        "patterns_detected": [],
        "support_levels": [],
        "resistance_levels": [],
        "long_opportunity": True,
        "long_reasoning": "No pattern data — deferring to other signals",
        "short_opportunity": True,
        "short_reasoning": "No pattern data — deferring to other signals",
    }
