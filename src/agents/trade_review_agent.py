import json
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.utils.database import get_session, Trade, TradeReviewRecord, IndicatorAccuracy
from src.utils.logger import setup_logger

logger = setup_logger("review_agent")


REVIEW_SYSTEM_PROMPT = """You are a trading performance analyst. Given a completed trade's signals and outcome, analyze which indicators were correct and which were wrong.

Respond ONLY with valid JSON, no other text."""

REVIEW_USER_PROMPT = """Completed Trade:
- Symbol: {symbol}
- Side: {side}
- Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f}
- P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)
- Exit Reason: {exit_reason}
- Signals at entry: {signals_json}

Analyze which indicators gave correct vs incorrect signals:
{{
    "correct_indicators": ["indicator1", "indicator2"],
    "incorrect_indicators": ["indicator3"],
    "suggested_adjustments": {{"param_name": new_value}},
    "review_text": "Brief analysis of what went right/wrong"
}}"""


class TradeReviewAgent(BaseAgent):
    """Reviews closed trades to track indicator accuracy."""

    def __init__(self):
        super().__init__(name="trade_review", default_ttl=0)  # No caching for reviews
        self.agent_config = self.config.get("agents", {}).get("trade_review", {})

    async def run(self, trade_id: int, **kwargs) -> dict:
        """Review a closed trade and store the analysis."""
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade or trade.status != "CLOSED":
                return {"error": f"Trade {trade_id} not found or not closed"}

            signals_json = trade.signals or "{}"

            result = self._call_claude(
                REVIEW_SYSTEM_PROMPT,
                REVIEW_USER_PROMPT.format(
                    symbol=trade.symbol,
                    side=trade.side,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price or trade.entry_price,
                    pnl=trade.pnl or 0,
                    pnl_pct=trade.pnl_pct or 0,
                    exit_reason=trade.exit_reason or "UNKNOWN",
                    signals_json=signals_json,
                ),
            )

            if "error" in result:
                return result

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
            is_winner = (trade.pnl or 0) > 0
            for ind in result.get("correct_indicators", []):
                self._update_accuracy(session, ind, correct=True)
            for ind in result.get("incorrect_indicators", []):
                self._update_accuracy(session, ind, correct=False)

            session.commit()
            logger.info("Trade #%d reviewed: %d correct, %d incorrect indicators",
                        trade_id, len(result.get("correct_indicators", [])),
                        len(result.get("incorrect_indicators", [])))
            return result

        except Exception as e:
            logger.error("Failed to review trade #%d: %s", trade_id, e)
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()

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
