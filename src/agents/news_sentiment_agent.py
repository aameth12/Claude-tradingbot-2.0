import asyncio
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import yfinance as yf

from src.agents.base_agent import BaseAgent
from src.utils.logger import setup_logger

logger = setup_logger("sentiment_agent")


@dataclass
class SentimentResult:
    symbol: str
    earnings_date: str | None
    earnings_within_n_days: bool
    sentiment_score: float  # -1.0 to 1.0
    news_events: list[str]
    should_block_trade: bool
    timestamp: str


SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analyst. Given a stock symbol and current date, provide a brief sentiment assessment based on general market conditions and the stock's recent context.

Respond ONLY with valid JSON, no other text."""

SENTIMENT_USER_PROMPT = """Stock: {symbol}
Current Date: {current_date}
Earnings Date: {earnings_info}

Provide sentiment analysis:
{{
    "sentiment_score": -1.0 to 1.0 (negative=bearish, positive=bullish),
    "news_events": ["event1", "event2"],
    "reasoning": "brief explanation"
}}"""


class NewsSentimentAgent(BaseAgent):
    """Checks earnings calendar and provides sentiment scores."""

    def __init__(self):
        super().__init__(name="news_sentiment", default_ttl=1800)  # 30 min default
        self.agent_config = self.config.get("agents", {}).get("news_sentiment", {})
        self.earnings_block_days = self.agent_config.get("earnings_block_days", 2)
        self._earnings_cache: dict[str, str | None] = {}  # {symbol: date_str or None}
        self._earnings_cache_date: date | None = None

    async def run(self, symbol: str, **kwargs) -> SentimentResult:
        """Get sentiment for a symbol. Returns cached if fresh."""
        cache_key = f"sentiment_{symbol}"
        cached = self.get_cached(cache_key)
        if cached:
            return cached

        # Get earnings date (cached daily)
        earnings_date = await self._get_earnings_date(symbol)
        earnings_within_n = False
        if earnings_date:
            try:
                ed = datetime.strptime(earnings_date, "%Y-%m-%d").date()
                days_until = (ed - date.today()).days
                earnings_within_n = 0 <= days_until <= self.earnings_block_days
            except Exception:
                pass

        should_block = earnings_within_n

        # Get sentiment from Claude
        earnings_info = f"{earnings_date} ({(datetime.strptime(earnings_date, '%Y-%m-%d').date() - date.today()).days} days away)" if earnings_date else "Unknown"

        result = self._call_claude(
            SENTIMENT_SYSTEM_PROMPT,
            SENTIMENT_USER_PROMPT.format(
                symbol=symbol,
                current_date=date.today().isoformat(),
                earnings_info=earnings_info,
            ),
            max_tokens=500,
        )

        sentiment = SentimentResult(
            symbol=symbol,
            earnings_date=earnings_date,
            earnings_within_n_days=earnings_within_n,
            sentiment_score=max(-1.0, min(1.0, result.get("sentiment_score", 0.0))),
            news_events=result.get("news_events", []),
            should_block_trade=should_block,
            timestamp=datetime.utcnow().isoformat(),
        )

        ttl = self.agent_config.get("sentiment_refresh_minutes", 30) * 60
        self.set_cached(sentiment, cache_key, ttl)

        logger.info(
            "%s sentiment: %.2f | earnings: %s | block: %s",
            symbol, sentiment.sentiment_score, earnings_date, should_block,
        )
        return sentiment

    async def refresh_earnings_calendar(self, watchlist: list[str]):
        """Refresh earnings dates for all watchlist symbols (daily)."""
        if self._earnings_cache_date == date.today() and self._earnings_cache:
            return

        loop = asyncio.get_event_loop()
        for symbol in watchlist:
            try:
                ed = await loop.run_in_executor(None, self._fetch_earnings_date, symbol)
                self._earnings_cache[symbol] = ed
            except Exception as e:
                logger.warning("Failed to fetch earnings for %s: %s", symbol, e)
                self._earnings_cache[symbol] = None

        self._earnings_cache_date = date.today()
        logger.info("Earnings calendar refreshed for %d symbols", len(watchlist))

    async def _get_earnings_date(self, symbol: str) -> str | None:
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]
        loop = asyncio.get_event_loop()
        ed = await loop.run_in_executor(None, self._fetch_earnings_date, symbol)
        self._earnings_cache[symbol] = ed
        return ed

    @staticmethod
    def _fetch_earnings_date(symbol: str) -> str | None:
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        if isinstance(ed, list) and ed:
                            return str(ed[0].date()) if hasattr(ed[0], 'date') else str(ed[0])
                        return str(ed)
                elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    return str(val.date()) if hasattr(val, 'date') else str(val)
        except Exception as e:
            logger.debug("No earnings data for %s: %s", symbol, e)
        return None
