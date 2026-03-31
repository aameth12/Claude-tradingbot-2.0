import asyncio
from dataclasses import dataclass
from datetime import datetime

import yfinance as yf

from src.agents.base_agent import BaseAgent
from src.utils.logger import setup_logger

logger = setup_logger("regime_agent")


@dataclass
class MarketRegime:
    regime: str  # TRENDING, RANGING, VOLATILE
    confidence: float
    vix_level: float
    vix_trend: str  # RISING, FALLING, STABLE
    breadth_pct: float
    recommended_sl_multiplier: float
    recommended_tp_multiplier: float
    recommended_confidence_threshold: float
    recommended_position_scale: float
    timestamp: str


class MarketRegimeAgent(BaseAgent):
    """Classifies current market conditions using deterministic rules.

    No Claude API calls — regime is classified from VIX, breadth, and
    volatility data using straightforward threshold logic.
    """

    def __init__(self):
        super().__init__(name="market_regime", default_ttl=900)
        self.agent_config = self.config.get("agents", {}).get("market_regime", {})
        self.default_ttl = self.agent_config.get("refresh_interval_minutes", 15) * 60
        self.regimes = self.agent_config.get("regimes", {})

    async def run(self, watchlist: list[str] | None = None, **kwargs) -> MarketRegime:
        """Classify current market regime. Returns cached result if fresh."""
        cached = self.get_cached()
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, self._gather_market_data, watchlist or [])

        # Deterministic classification — no Claude API call needed
        result = self._classify_regime(data)

        regime_name = result["regime"]
        params = self.regimes.get(regime_name.lower(), {})

        regime = MarketRegime(
            regime=regime_name,
            confidence=result["confidence"],
            vix_level=data["vix_level"],
            vix_trend=data["vix_trend"],
            breadth_pct=data["breadth_pct"],
            recommended_sl_multiplier=params.get("sl_multiplier", 2.0),
            recommended_tp_multiplier=params.get("tp_multiplier", 4.0),
            recommended_confidence_threshold=params.get("confidence_threshold", 0.60),
            recommended_position_scale=params.get("position_scale", 1.0),
            timestamp=datetime.utcnow().isoformat(),
        )

        self.set_cached(regime)
        logger.info(
            "Market regime: %s (confidence=%.2f, VIX=%.1f, breadth=%.1f%%)",
            regime.regime, regime.confidence, regime.vix_level, regime.breadth_pct,
        )
        return regime

    @staticmethod
    def _classify_regime(data: dict) -> dict:
        """Classify market regime from data using deterministic rules."""
        vix = data["vix_level"]
        breadth = data["breadth_pct"]
        spy_rv = data["spy_rv"]
        spy_return = abs(data["spy_return"])

        # VOLATILE: high VIX or high realized volatility
        if vix > 25 or spy_rv > 2.0:
            confidence = min(1.0, (vix - 20) / 15 + spy_rv / 3.0) if vix > 20 else spy_rv / 3.0
            return {"regime": "VOLATILE", "confidence": round(min(1.0, max(0.5, confidence)), 2)}

        # TRENDING: low VIX + skewed breadth + directional move
        if vix < 20 and (breadth > 60 or breadth < 40):
            strength = abs(breadth - 50) / 50  # How far from neutral
            confidence = 0.5 + strength * 0.4 + (spy_return / 10) * 0.1
            return {"regime": "TRENDING", "confidence": round(min(1.0, confidence), 2)}

        # RANGING: everything else (moderate VIX, neutral breadth)
        confidence = 0.5 + (1.0 - abs(breadth - 50) / 50) * 0.3
        return {"regime": "RANGING", "confidence": round(min(1.0, confidence), 2)}

    def _gather_market_data(self, watchlist: list[str]) -> dict:
        """Fetch VIX, breadth, and relative performance data."""
        # VIX
        vix_level = 20.0
        vix_change = 0.0
        vix_trend = "STABLE"
        try:
            vix_df = yf.Ticker("^VIX").history(period="5d")
            if not vix_df.empty:
                vix_level = float(vix_df["Close"].iloc[-1])
                if len(vix_df) >= 2:
                    vix_change = float(vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[0])
                    if vix_change > 2:
                        vix_trend = "RISING"
                    elif vix_change < -2:
                        vix_trend = "FALLING"
        except Exception as e:
            logger.warning("Failed to fetch VIX: %s", e)

        # SPY / QQQ returns
        spy_return = 0.0
        qqq_return = 0.0
        spy_rv = 1.0
        try:
            spy_df = yf.Ticker("SPY").history(period="30d")
            if not spy_df.empty and len(spy_df) >= 5:
                spy_return = float((spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-5] - 1) * 100)
                # Realized volatility: ATR proxy
                spy_range = (spy_df["High"] - spy_df["Low"]).rolling(14).mean().iloc[-1]
                spy_rv = float(spy_range / spy_df["Close"].iloc[-1] * 100)
        except Exception as e:
            logger.warning("Failed to fetch SPY data: %s", e)

        try:
            qqq_df = yf.Ticker("QQQ").history(period="5d")
            if not qqq_df.empty and len(qqq_df) >= 2:
                qqq_return = float((qqq_df["Close"].iloc[-1] / qqq_df["Close"].iloc[0] - 1) * 100)
        except Exception as e:
            logger.warning("Failed to fetch QQQ data: %s", e)

        # Market breadth: % of watchlist above EMA50
        breadth_pct = 50.0
        if watchlist:
            above_ema50 = 0
            checked = 0
            for symbol in watchlist:
                try:
                    df = yf.Ticker(symbol).history(period="60d")
                    if not df.empty and len(df) >= 50:
                        ema50 = df["Close"].ewm(span=50).mean().iloc[-1]
                        if df["Close"].iloc[-1] > ema50:
                            above_ema50 += 1
                        checked += 1
                except Exception:
                    pass
            if checked > 0:
                breadth_pct = (above_ema50 / checked) * 100

        return {
            "vix_level": vix_level,
            "vix_change": vix_change,
            "vix_trend": vix_trend,
            "spy_return": spy_return,
            "qqq_return": qqq_return,
            "spy_rv": spy_rv,
            "breadth_pct": breadth_pct,
        }
