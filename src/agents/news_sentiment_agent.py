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


class NewsSentimentAgent(BaseAgent):
    """Checks earnings calendar and derives sentiment from price momentum.

    No Claude API calls — earnings dates come from yfinance, and sentiment
    is derived from 5-day price returns (a data-grounded signal).
    """

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

        # Derive sentiment from 5-day price momentum (no Claude API call)
        sentiment_score = await self._get_momentum_sentiment(symbol)

        sentiment = SentimentResult(
            symbol=symbol,
            earnings_date=earnings_date,
            earnings_within_n_days=earnings_within_n,
            sentiment_score=sentiment_score,
            news_events=[],
            should_block_trade=should_block,
            timestamp=datetime.utcnow().isoformat(),
        )

        ttl = self.agent_config.get("sentiment_refresh_minutes", 30) * 60
        self.set_cached(sentiment, cache_key, ttl)

        logger.info(
            "%s sentiment: %.2f (momentum) | earnings: %s | block: %s",
            symbol, sentiment.sentiment_score, earnings_date, should_block,
        )
        return sentiment

    async def _get_momentum_sentiment(self, symbol: str) -> float:
        """Derive sentiment score from 5-day price return. No API call needed."""
        try:
            loop = asyncio.get_running_loop()
            score = await loop.run_in_executor(None, self._fetch_momentum, symbol)
            return score
        except Exception as e:
            logger.debug("Could not compute momentum for %s: %s", symbol, e)
            return 0.0

    @staticmethod
    def _fetch_momentum(symbol: str) -> float:
        """Fetch 5-day return and convert to sentiment score in [-1, 1]."""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty or len(hist) < 2:
            return 0.0
        five_day_return_pct = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
        # Clamp: ±5% return maps to ±1.0 sentiment
        return max(-1.0, min(1.0, five_day_return_pct / 5.0))

    async def refresh_earnings_calendar(self, watchlist: list[str]):
        """Refresh earnings dates for all watchlist symbols (daily)."""
        if self._earnings_cache_date == date.today() and self._earnings_cache:
            return

        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
                elif hasattr(cal, 'loc') and "Earnings Date" in getattr(cal, 'index', []):
                    val = cal.loc["Earnings Date"].iloc[0]
                    return str(val.date()) if hasattr(val, 'date') else str(val)
        except Exception as e:
            logger.debug("No earnings data for %s: %s", symbol, e)
        return None
