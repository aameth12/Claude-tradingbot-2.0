import yfinance as yf
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("correlation")


class CorrelationFilter:
    """Blocks trades on highly correlated or sector-concentrated positions."""

    def __init__(self):
        self.config = get_config().get("correlation", {})
        self.enabled = self.config.get("enabled", False)
        self.max_correlation = self.config.get("max_correlation", 0.75)
        self.max_sector_positions = self.config.get("max_sector_positions", 3)
        self.sectors = self.config.get("sectors", {})
        self._returns_cache: dict[str, pd.Series] = {}

    def check(self, candidate: str, open_symbols: list[str]) -> tuple[bool, str, str | None]:
        """Check if candidate can trade given open positions.

        Returns (can_trade, reason, correlated_with).
        """
        if not self.enabled or not open_symbols:
            return True, "OK", None

        # Sector concentration check
        candidate_sector = self._get_sector(candidate)
        if candidate_sector:
            same_sector = [s for s in open_symbols if self._get_sector(s) == candidate_sector]
            if len(same_sector) >= self.max_sector_positions:
                reason = f"Sector '{candidate_sector}' full ({len(same_sector)}/{self.max_sector_positions})"
                logger.info("%s blocked: %s", candidate, reason)
                return False, reason, same_sector[0]

        # Correlation check
        try:
            candidate_returns = self._get_returns(candidate)
            if candidate_returns is None:
                return True, "OK", None

            for symbol in open_symbols:
                other_returns = self._get_returns(symbol)
                if other_returns is None:
                    continue
                # Align on common dates
                combined = pd.concat([candidate_returns, other_returns], axis=1).dropna()
                if len(combined) < 10:
                    continue
                corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                if abs(corr) > self.max_correlation:
                    reason = f"Correlation {corr:.2f} with {symbol} exceeds {self.max_correlation}"
                    logger.info("%s blocked: %s", candidate, reason)
                    return False, reason, symbol
        except Exception as e:
            logger.warning("Correlation check failed for %s: %s", candidate, e)

        return True, "OK", None

    def _get_sector(self, symbol: str) -> str | None:
        for sector_name, symbols in self.sectors.items():
            if symbol in symbols:
                return sector_name
        return None

    def _get_returns(self, symbol: str) -> pd.Series | None:
        if symbol in self._returns_cache:
            return self._returns_cache[symbol]
        try:
            df = yf.Ticker(symbol).history(period="30d")
            if df.empty or len(df) < 10:
                return None
            returns = df["Close"].pct_change(fill_method=None).dropna()
            returns.name = symbol
            self._returns_cache[symbol] = returns
            return returns
        except Exception as e:
            logger.warning("Failed to fetch returns for %s: %s", symbol, e)
            return None

    def clear_cache(self):
        self._returns_cache.clear()
