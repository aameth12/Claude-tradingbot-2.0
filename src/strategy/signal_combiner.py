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
        self.weights = {
            "tradingview_summary": 0.25,
            "tradingview_indicators": 0.25,
            "ai_chart": 0.30,
            "multi_timeframe": 0.20,
        }

    def combine_signals(
        self,
        symbol: str,
        tv_analysis: dict,
        tv_indicator_signals: dict,
        ai_analysis: dict,
        multi_tf_analyses: dict,
        current_price: float,
        atr: float,
        trade_levels: dict,
    ) -> Optional[TradeSignal]:
        """Combine all signal sources into a final trade decision."""

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

        # Weighted combined score
        combined_score = sum(
            scores[key] * self.weights[key] for key in scores
        )

        # Determine side
        if combined_score > 0:
            side = "LONG"
        elif combined_score < 0:
            side = "SHORT"
        else:
            logger.info("%s: Combined score is 0, no signal", symbol)
            return None

        # Check if SHORT is allowed
        allowed_sides = self.config["trading"]["allowed_sides"]
        if side not in allowed_sides:
            logger.info("%s: %s not in allowed sides", symbol, side)
            return None

        # Check AI short recommendation specifically
        if side == "SHORT" and ai_analysis.get("short_opportunity") is False:
            # Reduce confidence if AI doesn't see short opportunity
            combined_score *= 0.5

        confidence = min(abs(combined_score), 1.0)

        if confidence < self.confidence_threshold:
            logger.info(
                "%s: Confidence %.2f below threshold %.2f",
                symbol, confidence, self.confidence_threshold,
            )
            return None

        # Validate trade levels
        if not trade_levels.get("valid"):
            logger.info("%s: Trade levels invalid (RR or qty)", symbol)
            return None

        reasoning = self._build_reasoning(scores, side, ai_analysis)

        signal = TradeSignal(
            symbol=symbol,
            side=side,
            confidence=round(confidence, 3),
            entry_price=trade_levels["entry_price"],
            stop_loss=trade_levels["stop_loss"],
            take_profit=trade_levels["take_profit"],
            quantity=trade_levels["quantity"],
            timeframe=self._determine_timeframe(scores),
            strategy="combined_signal",
            signals_detail=scores,
            reasoning=reasoning,
        )

        logger.info(
            "SIGNAL: %s %s | confidence=%.2f | entry=%.2f | SL=%.2f | TP=%.2f | qty=%d",
            side, symbol, confidence,
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
