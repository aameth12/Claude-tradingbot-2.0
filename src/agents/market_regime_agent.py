import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime

import yfinance as yf
import pandas as pd

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


REGIME_SYSTEM_PROMPT = """You are a market regime classifier. Based on the provided market data, classify the current regime as one of: TRENDING, RANGING, or VOLATILE.

Rules:
- TRENDING: VIX < 20, breadth > 60% or < 40%, clear directional bias in SPY/QQQ
- RANGING: VIX 15-22, breadth 40-60%, SPY and QQQ moving sideways
- VOLATILE: VIX > 25, rapid breadth changes, or significant SPY/QQQ divergence

Respond ONLY with valid JSON, no other text."""

REGIME_USER_PROMPT = """Current Market Data:
- VIX: {vix_level:.2f} (5-day change: {vix_change:+.2f})
- VIX Trend: {vix_trend}
- Market Breadth: {breadth_pct:.1f}% of watchlist above EMA50
- SPY 5-day return: {spy_return:+.2f}%
- QQQ 5-day return: {qqq_return:+.2f}%
- SPY realized volatility (14-day ATR/price): {spy_rv:.2f}%

Classify the regime and recommend parameter adjustments:
{{
    "regime": "TRENDING" | "RANGING" | "VOLATILE",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""


class MarketRegimeAgent(BaseAgent):
    """Classifies current market conditions and recommends parameter adjustments."""

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

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._gather_market_data, watchlist or [])

        # Call Claude for classification
        result = self._call_claude(
            REGIME_SYSTEM_PROMPT,
            REGIME_USER_PROMPT.format(**data),
        )

        regime_name = result.get("regime", "RANGING").upper()
        if regime_name not in ("TRENDING", "RANGING", "VOLATILE"):
            regime_name = "RANGING"

        # Look up parameter adjustments from config
        params = self.regimes.get(regime_name.lower(), {})

        regime = MarketRegime(
            regime=regime_name,
            confidence=result.get("confidence", 0.5),
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
            "Market regime: %s (confidence=%.2f, VIX=%.1f)",
            regime.regime, regime.confidence, regime.vix_level,
        )
        return regime

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
