import json
from datetime import datetime

from src.agents.market_regime_agent import MarketRegimeAgent, MarketRegime
from src.agents.news_sentiment_agent import NewsSentimentAgent, SentimentResult
from src.agents.trade_review_agent import TradeReviewAgent
from src.utils.config import get_config
from src.utils.database import get_session, AgentOutput
from src.utils.logger import setup_logger

logger = setup_logger("agent_manager")


class AgentManager:
    """Coordinator for all AI agents."""

    def __init__(self):
        self.config = get_config()
        self.agents_config = self.config.get("agents", {})
        self.enabled = self.agents_config.get("enabled", False)

        self.regime_agent = MarketRegimeAgent()
        self.sentiment_agent = NewsSentimentAgent()
        self.review_agent = TradeReviewAgent()

        logger.info("AgentManager initialized (enabled=%s)", self.enabled)

    async def get_market_regime(self, watchlist: list[str] | None = None) -> MarketRegime | None:
        """Get current market regime classification."""
        if not self.enabled or not self.agents_config.get("market_regime", {}).get("enabled", False):
            return None

        try:
            regime = await self.regime_agent.run(watchlist=watchlist)
            self._log_output("market_regime", None, regime.__dict__ if hasattr(regime, '__dict__') else {})
            return regime
        except Exception as e:
            logger.error("Market regime agent failed: %s", e)
            return None

    async def get_sentiment(self, symbol: str) -> SentimentResult | None:
        """Get sentiment for a symbol."""
        if not self.enabled or not self.agents_config.get("news_sentiment", {}).get("enabled", False):
            return None

        try:
            sentiment = await self.sentiment_agent.run(symbol=symbol)
            self._log_output("news_sentiment", symbol, sentiment.__dict__ if hasattr(sentiment, '__dict__') else {})
            return sentiment
        except Exception as e:
            logger.error("Sentiment agent failed for %s: %s", symbol, e)
            return None

    async def review_trade(self, trade_id: int) -> dict | None:
        """Review a closed trade."""
        if not self.enabled or not self.agents_config.get("trade_review", {}).get("enabled", False):
            return None

        try:
            result = await self.review_agent.run(trade_id=trade_id)
            self._log_output("trade_review", None, result)
            return result
        except Exception as e:
            logger.error("Trade review agent failed for #%d: %s", trade_id, e)
            return None

    async def refresh_earnings_calendar(self, watchlist: list[str]):
        """Refresh earnings dates for all watchlist symbols."""
        if not self.enabled:
            return
        try:
            await self.sentiment_agent.refresh_earnings_calendar(watchlist)
        except Exception as e:
            logger.error("Earnings calendar refresh failed: %s", e)

    def get_accuracy_stats(self) -> list[dict]:
        """Get indicator accuracy stats."""
        return self.review_agent.get_accuracy_stats()

    @staticmethod
    def _log_output(agent_name: str, symbol: str | None, output: dict):
        """Store agent output in DB for audit."""
        try:
            session = get_session()
            record = AgentOutput(
                agent_name=agent_name,
                symbol=symbol,
                output_json=json.dumps(output, default=str),
            )
            session.add(record)
            session.commit()
            session.close()
        except Exception as e:
            logger.debug("Failed to log agent output: %s", e)
