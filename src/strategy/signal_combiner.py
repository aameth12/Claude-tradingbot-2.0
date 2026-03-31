import json
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.utils.database import get_session, IndicatorAccuracy

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
    atr: float = 0.0  # ATR used for SL/TP calculation (for recalc after fill)


class SignalCombiner:
    """Combines signals from TradingView indicators and AI chart analysis
    to produce final trade decisions."""

    def __init__(self):
        self.config = get_config()
        self.confidence_threshold = self.config["ai"]["confidence_threshold"]
        # Weights for different signal sources (with sentiment)
        sentiment_weight = self.config.get("agents", {}).get("news_sentiment", {}).get("sentiment_weight", 0.0)
        if sentiment_weight > 0:
            remaining = 1.0 - sentiment_weight
            self.weights = {
                "tradingview_summary": round(0.25 * remaining / 0.85, 3),
                "tradingview_indicators": round(0.25 * remaining / 0.85, 3),
                "ai_chart": round(0.15 * remaining / 0.85, 3),
                "multi_timeframe": round(0.35 * remaining / 0.85, 3),
                "sentiment": sentiment_weight,
            }
        else:
            self.weights = {
                "tradingview_summary": 0.25,
                "tradingview_indicators": 0.25,
                "ai_chart": 0.15,
                "multi_timeframe": 0.35,
            }

    def _get_adaptive_weights(self) -> dict:
        """Adjust signal weights based on indicator accuracy from DB.

        Falls back to static weights if not enough accuracy data exists.
        """
        session = get_session()
        try:
            records = session.query(IndicatorAccuracy).all()
            if not records or len(records) < 5:
                return self.weights  # Not enough data yet

            accuracy_map = {r.indicator_name: r.accuracy_pct for r in records}

            # Map indicator names to signal source categories
            source_keywords = {
                "tradingview_summary": "tradingview_summary",
                "tradingview_indicators": "tradingview_indicators",
                "ai_chart": "ai_chart",
                "multi_timeframe": "multi_timeframe",
                "sentiment": "sentiment",
            }

            source_accuracy = {}
            for source in self.weights:
                keyword = source_keywords.get(source, source)
                related = [v for k, v in accuracy_map.items() if keyword in k]
                if related:
                    source_accuracy[source] = sum(related) / len(related)
                else:
                    source_accuracy[source] = 50.0  # Neutral default

            # Re-weight: higher accuracy → higher weight, normalized to sum=1.0
            total = sum(source_accuracy.values())
            if total <= 0:
                return self.weights

            adaptive = {k: round(v / total, 3) for k, v in source_accuracy.items()}
            logger.info("Adaptive weights: %s", adaptive)
            return adaptive
        except Exception as e:
            logger.warning("Failed to get adaptive weights: %s, using static", e)
            return self.weights
        finally:
            session.close()

    def _get_regime_filters(self, regime: str | None) -> tuple[float, float]:
        """Get ADX and volume filter thresholds based on market regime."""
        if regime == "TRENDING":
            return 20, 0.9   # Strict: need real trends with volume
        elif regime == "RANGING":
            return 10, 0.6   # Loose: allow mean-reversion in quiet markets
        elif regime == "VOLATILE":
            return 15, 1.0   # Need volume confirmation in volatility
        else:
            return 15, 0.8   # Default

    def evaluate_direction(
        self,
        symbol: str,
        tv_analysis: dict,
        tv_indicator_signals: dict,
        ai_analysis: dict,
        multi_tf_analyses: dict,
        sentiment_score: float = 0.0,
        confidence_threshold_override: float | None = None,
        regime: str | None = None,
    ) -> Optional[dict]:
        """Determine signal direction and confidence from all sources.

        Returns a dict with 'side', 'confidence', 'scores', 'reasoning'
        or None if no actionable signal.
        """
        # Pre-filters: regime-aware thresholds
        indicators = tv_analysis.get("indicators", {})
        adx_min, vol_min = self._get_regime_filters(regime)

        # ADX filter — no trend = no trade
        adx = indicators.get("adx")
        if adx is not None and adx < adx_min:
            logger.info("%s: ADX %.1f too low (< %.0f, regime=%s), no trend",
                        symbol, adx, adx_min, regime or "default")
            return None

        # Volume filter — weak volume = unreliable signal
        volume = indicators.get("volume")
        volume_sma = indicators.get("volume_sma20")
        if volume is not None and volume_sma is not None and volume_sma > 0:
            if volume < volume_sma * vol_min:
                logger.info(
                    "%s: Volume %.0f below %.1fx SMA20 (%.0f), skipping (regime=%s)",
                    symbol, volume, vol_min, volume_sma, regime or "default",
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

        # 5. Sentiment score (if agents enabled)
        if "sentiment" in self.weights:
            scores["sentiment"] = max(-1.0, min(1.0, sentiment_score))

        # Use adaptive weights (accuracy-driven) or fall back to static
        base_weights = self._get_adaptive_weights()

        # Adjust weights if AI analysis is unavailable
        ai_available = ai_analysis and "error" not in ai_analysis and ai_analysis.get("recommendation", "NEUTRAL") != "NEUTRAL"
        if ai_available:
            weights = base_weights
        else:
            ai_w = base_weights.get("ai_chart", 0.15)
            weights = dict(base_weights)
            weights["ai_chart"] = 0.0
            remaining_keys = [k for k in weights if k != "ai_chart" and weights[k] > 0]
            if remaining_keys:
                bonus = ai_w / len(remaining_keys)
                for k in remaining_keys:
                    weights[k] = round(weights[k] + bonus, 3)

        # Weighted combined score
        combined_score = sum(
            scores[key] * weights[key] for key in scores
        )

        # Signal convergence bonus — reward when multiple sources agree
        bullish_sources = sum(1 for s in scores.values() if s > 0.05)
        bearish_sources = sum(1 for s in scores.values() if s < -0.05)
        total_sources = len(scores)
        agreement = max(bullish_sources, bearish_sources) / total_sources if total_sources > 0 else 0

        if agreement >= 0.75:
            convergence_bonus = 1.0 + (agreement - 0.5) * 0.6  # 1.15 to 1.30
            combined_score *= convergence_bonus
            logger.info("%s: Convergence bonus %.0f%% (%d/%d sources agree)",
                        symbol, (convergence_bonus - 1) * 100,
                        max(bullish_sources, bearish_sources), total_sources)

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

        # AI veto — soft penalty instead of hard block
        ai_confidence = ai_analysis.get("confidence", 0.0)
        if side == "LONG" and ai_analysis.get("long_opportunity") is False and ai_confidence > 0.3:
            combined_score *= 0.6
            logger.info("%s: AI discourages LONG (confidence %.2f), reducing score", symbol, ai_confidence)
        elif side == "SHORT" and ai_analysis.get("short_opportunity") is False and ai_confidence > 0.3:
            combined_score *= 0.6
            logger.info("%s: AI discourages SHORT (confidence %.2f), reducing score", symbol, ai_confidence)

        confidence = min(abs(combined_score), 1.0)

        threshold = confidence_threshold_override or self.confidence_threshold
        if confidence < threshold:
            logger.info(
                "%s: Confidence %.2f below threshold %.2f",
                symbol, confidence, threshold,
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
        atr: float = 0.0,
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
            atr=atr,
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
