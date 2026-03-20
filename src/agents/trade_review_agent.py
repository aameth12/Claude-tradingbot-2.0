import json
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.utils.database import get_session, Trade, TradeReviewRecord, IndicatorAccuracy
from src.utils.logger import setup_logger

logger = setup_logger("review_agent")

# Only used for batch suggestions (every N trades, not per-trade)
BATCH_REVIEW_SYSTEM_PROMPT = """You are a trading performance analyst. Given indicator accuracy statistics from recent trades, suggest parameter adjustments to improve performance.

Respond ONLY with valid JSON, no other text."""

BATCH_REVIEW_USER_PROMPT = """Indicator accuracy over the last {n_trades} trades:
{accuracy_table}

Suggest parameter adjustments:
{{
    "suggested_adjustments": {{"param_name": "new_value"}},
    "review_text": "Brief analysis and recommendations"
}}"""


class TradeReviewAgent(BaseAgent):
    """Reviews closed trades to track indicator accuracy.

    Per-trade reviews are deterministic (no Claude API call). An optional
    batch Claude call runs every N trades for strategic suggestions.
    """

    def __init__(self):
        super().__init__(name="trade_review", default_ttl=0)  # No caching for reviews
        self.agent_config = self.config.get("agents", {}).get("trade_review", {})
        self.batch_interval = self.agent_config.get("batch_review_interval", 50)
        self._trades_since_batch = 0

    async def run(self, trade_id: int, **kwargs) -> dict:
        """Review a closed trade deterministically and store the analysis."""
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade or trade.status != "CLOSED":
                return {"error": f"Trade {trade_id} not found or not closed"}

            # Deterministic review — no Claude API call
            result = self._deterministic_review(trade)

            # Store review in DB
            review = TradeReviewRecord(
                trade_id=trade_id,
                correct_indicators=json.dumps(result.get("correct_indicators", [])),
                incorrect_indicators=json.dumps(result.get("incorrect_indicators", [])),
                suggested_adjustments=json.dumps(result.get("suggested_adjustments", {})),
                review_text=result.get("review_text", ""),
            )
            session.add(review)

            # Update indicator accuracy stats
            for ind in result.get("correct_indicators", []):
                self._update_accuracy(session, ind, correct=True)
            for ind in result.get("incorrect_indicators", []):
                self._update_accuracy(session, ind, correct=False)

            session.commit()
            logger.info("Trade #%d reviewed: %d correct, %d incorrect indicators",
                        trade_id, len(result.get("correct_indicators", [])),
                        len(result.get("incorrect_indicators", [])))

            # Check if we should run a batch Claude review for suggestions
            self._trades_since_batch += 1
            if self._trades_since_batch >= self.batch_interval:
                self._trades_since_batch = 0
                batch_result = self._run_batch_review()
                if batch_result and "error" not in batch_result:
                    result["batch_suggestions"] = batch_result

            return result

        except Exception as e:
            logger.error("Failed to review trade #%d: %s", trade_id, e)
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()

    @staticmethod
    def _deterministic_review(trade: Trade) -> dict:
        """Review a trade deterministically based on signal scores vs outcome.

        An indicator is 'correct' if its score direction matched the trade
        side AND the trade was profitable, or if its score direction opposed
        the trade side AND the trade was unprofitable (it was right to disagree).
        """
        is_winner = (trade.pnl or 0) > 0
        side = trade.side  # "LONG" or "SHORT"

        signals = {}
        try:
            signals = json.loads(trade.signals or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        correct = []
        incorrect = []

        for indicator_name, score in signals.items():
            if not isinstance(score, (int, float)):
                continue

            # Indicator agreed with trade direction?
            indicator_bullish = score > 0
            trade_is_long = side == "LONG"
            agreed_with_trade = indicator_bullish == trade_is_long

            if agreed_with_trade and is_winner:
                correct.append(indicator_name)
            elif not agreed_with_trade and not is_winner:
                correct.append(indicator_name)  # It was right to disagree
            else:
                incorrect.append(indicator_name)

        pnl = trade.pnl or 0
        pnl_pct = trade.pnl_pct or 0
        outcome = "WIN" if is_winner else "LOSS"

        return {
            "correct_indicators": correct,
            "incorrect_indicators": incorrect,
            "suggested_adjustments": {},
            "review_text": (
                f"{outcome}: {trade.side} {trade.symbol} "
                f"P&L ${pnl:+.2f} ({pnl_pct:+.1f}%) | "
                f"{len(correct)} correct, {len(incorrect)} incorrect indicators"
            ),
        }

    def _run_batch_review(self) -> dict:
        """Run a Claude API call with accumulated accuracy stats for strategic suggestions."""
        stats = self.get_accuracy_stats()
        if not stats:
            return {}

        accuracy_table = "\n".join(
            f"- {s['name']}: {s['accuracy']:.1f}% ({s['correct']}/{s['total']} correct)"
            for s in stats
        )

        total_trades = sum(s["total"] for s in stats) // max(len(stats), 1)

        result = self._call_claude(
            BATCH_REVIEW_SYSTEM_PROMPT,
            BATCH_REVIEW_USER_PROMPT.format(
                n_trades=total_trades,
                accuracy_table=accuracy_table,
            ),
            max_tokens=500,
        )

        if "error" not in result:
            logger.info("Batch review suggestions: %s", result.get("review_text", ""))
        return result

    @staticmethod
    def _update_accuracy(session, indicator_name: str, correct: bool):
        """Update rolling accuracy for an indicator."""
        record = session.query(IndicatorAccuracy).filter(
            IndicatorAccuracy.indicator_name == indicator_name
        ).first()

        if not record:
            record = IndicatorAccuracy(
                indicator_name=indicator_name,
                total_signals=0,
                correct_signals=0,
            )
            session.add(record)

        record.total_signals += 1
        if correct:
            record.correct_signals += 1
        record.accuracy_pct = round(
            (record.correct_signals / record.total_signals) * 100, 1
        ) if record.total_signals > 0 else 0.0
        record.last_updated = datetime.utcnow()

    def get_accuracy_stats(self) -> list[dict]:
        """Get indicator accuracy stats for display."""
        session = get_session()
        try:
            records = session.query(IndicatorAccuracy).order_by(
                IndicatorAccuracy.accuracy_pct.desc()
            ).all()
            return [
                {
                    "name": r.indicator_name,
                    "total": r.total_signals,
                    "correct": r.correct_signals,
                    "accuracy": r.accuracy_pct,
                }
                for r in records
            ]
        finally:
            session.close()
