import json
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("strategy")


@dataclass
class TradeSignal:
    symbol: str
    side: str  # "LONG" or "SHORT"
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    timeframe: str
    strategy: str
    signals_detail: dict
    reasoning: str


class SignalCombiner:
    """Combines signals from TradingView indicators and AI chart analysis
    to produce final trade decisions."""

    def __init__(self):
        self.config = get_config()
        self.confidence_threshold = self.config["ai"]["confidence_threshold"]
        # Weights for different signal sources
        # Multi-timeframe is most reliable; AI is least reliable
        self.weights = {
            "tradingview_summary": 0.25,
            "tradingview_indicators": 0.25,
            "ai_chart": 0.15,
            "multi_timeframe": 0.35,
        }

    def evaluate_direction(
        self,
        symbol: str,
        tv_analysis: dict,
        tv_indicator_signals: dict,
        ai_analysis: dict,
        multi_tf_analyses: dict,
    ) -> Optional[dict]:
        """Determine signal direction and confidence from all sources.

        Returns a dict with 'side', 'confidence', 'scores', 'reasoning'
        or None if no actionable signal.
        """
        # Pre-filters: reject trades in trendless or low-volume conditions
        indicators = tv_analysis.get("indicators", {})

        # ADX filter — no trend = no trade
        adx = indicators.get("adx")
        if adx is not None and adx < 25:
            logger.info("%s: ADX %.1f too low (< 25), no trend", symbol, adx)
            return None

        # Volume filter — weak volume = unreliable signal
        volume = indicators.get("volume")
        volume_sma = indicators.get("volume_sma20")
        if volume is not None and volume_sma is not None and volume_sma > 0:
            if volume < volume_sma * 1.2:
                logger.info(
                    "%s: Volume %.0f below 1.2x SMA20 (%.0f), skipping",
                    symbol, volume, volume_sma,
                )
                return None

        scores = {}

        # 1. TradingView summary score (-1 to 1)
        tv_score = self._score_tv_summary(tv_analysis)
        scores["tradingview_summary"] = tv_score

        # 2. TradingView individual indicator signals
        indicator_score = self._score_indicators(tv_indicator_signals)
        scores["tradingview_indicators"] = indicator_score

        # 3. AI chart analysis score
        ai_score = self._score_ai_analysis(ai_analysis)
        scores["ai_chart"] = ai_score

        # 4. Multi-timeframe alignment
        mtf_score = self._score_multi_timeframe(multi_tf_analyses)
        scores["multi_timeframe"] = mtf_score

        # Use adjusted weights if AI analysis is unavailable (error or empty)
        ai_available = ai_analysis and "error" not in ai_analysis and ai_analysis.get("recommendation", "NEUTRAL") != "NEUTRAL"
        if ai_available:
            weights = self.weights
        else:
            # Redistribute AI weight to other sources
            weights = {
                "tradingview_summary": 0.30,
                "tradingview_indicators": 0.30,
                "ai_chart": 0.0,
                "multi_timeframe": 0.40,
            }

        # Weighted combined score
        combined_score = sum(
            scores[key] * weights[key] for key in scores
        )

        # Determine side
        if combined_score > 0:
            side = "LONG"
        elif combined_score < 0:
            side = "SHORT"
        else:
            logger.info("%s: Combined score is 0, no signal", symbol)
            return None

        # Check if side is allowed
        allowed_sides = self.config["trading"]["allowed_sides"]
        if side not in allowed_sides:
            logger.info("%s: %s not in allowed sides", symbol, side)
            return None

        # AI veto — hard block if AI explicitly says no opportunity
        if side == "LONG" and ai_analysis.get("long_opportunity") is False:
            logger.info("%s: AI vetoed LONG opportunity, blocking trade", symbol)
            return None
        elif side == "SHORT" and ai_analysis.get("short_opportunity") is False:
            logger.info("%s: AI vetoed SHORT opportunity, blocking trade", symbol)
            return None

        confidence = min(abs(combined_score), 1.0)

        if confidence < self.confidence_threshold:
            logger.info(
                "%s: Confidence %.2f below threshold %.2f",
                symbol, confidence, self.confidence_threshold,
            )
            return None

        reasoning = self._build_reasoning(scores, side, ai_analysis)

        logger.info(
            "DIRECTION: %s %s | confidence=%.2f | scores=%s",
            side, symbol, confidence, scores,
        )
        return {
            "side": side,
            "confidence": round(confidence, 3),
            "scores": scores,
            "reasoning": reasoning,
            "timeframe": self._determine_timeframe(scores),
        }

    def build_signal(
        self,
        symbol: str,
        direction: dict,
        trade_levels: dict,
    ) -> Optional[TradeSignal]:
        """Build a TradeSignal from a validated direction and matching trade levels."""
        if not trade_levels.get("valid"):
            logger.info("%s: Trade levels invalid (RR or qty)", symbol)
            return None

        signal = TradeSignal(
            symbol=symbol,
            side=direction["side"],
            confidence=direction["confidence"],
            entry_price=trade_levels["entry_price"],
            stop_loss=trade_levels["stop_loss"],
            take_profit=trade_levels["take_profit"],
            quantity=trade_levels["quantity"],
            timeframe=direction["timeframe"],
            strategy="combined_signal",
            signals_detail=direction["scores"],
            reasoning=direction["reasoning"],
        )

        logger.info(
            "SIGNAL: %s %s | confidence=%.2f | entry=%.2f | SL=%.2f | TP=%.2f | qty=%d",
            signal.side, symbol, signal.confidence,
            signal.entry_price, signal.stop_loss, signal.take_profit, signal.quantity,
        )
        return signal

    def _score_tv_summary(self, analysis: dict) -> float:
        if "error" in analysis:
            return 0.0
        summary = analysis.get("summary", {})
        rec = summary.get("recommendation", "NEUTRAL")

        score_map = {
            "STRONG_BUY": 1.0,
            "BUY": 0.6,
            "NEUTRAL": 0.0,
            "SELL": -0.6,
            "STRONG_SELL": -1.0,
        }
        return score_map.get(rec, 0.0)

    def _score_indicators(self, indicator_signals: dict) -> float:
        overall = indicator_signals.get("overall", "NEUTRAL")
        buy_count = indicator_signals.get("buy_count", 0)
        sell_count = indicator_signals.get("sell_count", 0)
        total = buy_count + sell_count
        if total == 0:
            return 0.0
        return (buy_count - sell_count) / total

    def _score_ai_analysis(self, analysis: dict) -> float:
        rec = analysis.get("recommendation", "NEUTRAL")
        confidence = analysis.get("confidence", 0.0)

        score_map = {
            "STRONG_BUY": 1.0,
            "BUY": 0.6,
            "NEUTRAL": 0.0,
            "SELL": -0.6,
            "STRONG_SELL": -1.0,
        }
        base_score = score_map.get(rec, 0.0)
        return base_score * confidence

    def _score_multi_timeframe(self, analyses: dict) -> float:
        """Score based on multi-timeframe alignment.
        Stronger signal when all timeframes agree."""
        if not analyses:
            return 0.0

        scores = []
        for style_name, style_data in analyses.items():
            for tf_role, analysis in style_data.items():
                if isinstance(analysis, dict) and "summary" in analysis:
                    rec = analysis["summary"].get("recommendation", "NEUTRAL")
                    score_map = {
                        "STRONG_BUY": 1.0, "BUY": 0.5, "NEUTRAL": 0.0,
                        "SELL": -0.5, "STRONG_SELL": -1.0,
                    }
                    scores.append(score_map.get(rec, 0.0))

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        # Bonus for alignment (all same direction)
        all_positive = all(s > 0 for s in scores)
        all_negative = all(s < 0 for s in scores)
        if all_positive or all_negative:
            avg *= 1.3  # 30% bonus for full alignment

        return max(-1.0, min(1.0, avg))

    def _determine_timeframe(self, scores: dict) -> str:
        """Determine whether this is a day trade or swing trade signal."""
        # Higher scores on shorter timeframes -> day trade
        # For now, use a simple heuristic
        return "day_trading" if abs(scores.get("tradingview_indicators", 0)) > 0.5 else "swing_trading"

    def _build_reasoning(self, scores: dict, side: str, ai_analysis: dict) -> str:
        parts = [f"Signal: {side}"]
        for source, score in scores.items():
            direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
            parts.append(f"  {source}: {score:+.2f} ({direction})")

        ai_reasoning = ai_analysis.get("reasoning", "")
        if ai_reasoning:
            parts.append(f"  AI says: {ai_reasoning}")

        return "\n".join(parts)
