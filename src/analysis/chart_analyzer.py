import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before mplfinance import

import anthropic
import pandas as pd
import mplfinance as mpf
from PIL import Image

from src.utils.logger import setup_logger
from src.utils.config import ANTHROPIC_API_KEY, get_config, DATA_DIR

logger = setup_logger("chart_analyzer")

CHART_ANALYSIS_PROMPT = """You are an expert technical analyst. Analyze this stock chart and provide an unbiased trading recommendation. Consider BOTH long and short opportunities equally.

Stock: {symbol}
Timeframe: {timeframe}

Analyze the chart for:
1. **Trend**: Is the stock in an uptrend, downtrend, or sideways?
2. **Key Levels**: Identify support and resistance levels.
3. **Patterns**: Identify any chart patterns (head & shoulders, triangles, flags, double top/bottom, etc.)
4. **Candlestick Patterns**: Any notable candlestick patterns?
5. **Volume**: Is volume confirming the price action?
6. **Momentum**: Is momentum increasing or decreasing?

Provide your response in this exact JSON format:
{{
    "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
    "trend_strength": 0.0 to 1.0,
    "patterns_detected": ["pattern1", "pattern2"],
    "support_levels": [price1, price2],
    "resistance_levels": [price1, price2],
    "recommendation": "STRONG_BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG_SELL",
    "confidence": 0.0 to 1.0,
    "entry_zone": {{"low": price, "high": price}},
    "stop_loss_suggestion": price,
    "target_suggestion": price,
    "reasoning": "Brief explanation of analysis",
    "long_opportunity": true | false,
    "long_reasoning": "Why buying may or may not be appropriate",
    "short_opportunity": true | false,
    "short_reasoning": "Why shorting may or may not be appropriate"
}}

Respond ONLY with valid JSON, no other text.
"""


class ChartAnalyzer:
    """Uses Claude Vision to analyze stock charts for patterns and signals."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.config = get_config()["ai"]
        self.charts_dir = DATA_DIR / "charts"
        self.charts_dir.mkdir(exist_ok=True)

    def generate_chart_image(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Path:
        """Generate a candlestick chart image from OHLCV data."""
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Ensure columns are named correctly
        df.columns = [c.lower() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        chart_path = self.charts_dir / f"{symbol}_{timeframe}.png"

        # Create chart with indicators
        ema9 = df["close"].ewm(span=9).mean()
        ema21 = df["close"].ewm(span=21).mean()
        ema50 = df["close"].ewm(span=50).mean()
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        add_plots = [
            mpf.make_addplot(ema9, color="blue", width=0.8, label="EMA9"),
            mpf.make_addplot(ema21, color="orange", width=0.8, label="EMA21"),
            mpf.make_addplot(ema50, color="red", width=1.0, label="EMA50"),
            mpf.make_addplot(bb_upper, color="gray", width=0.5, linestyle="--"),
            mpf.make_addplot(bb_lower, color="gray", width=0.5, linestyle="--"),
        ]

        mpf.plot(
            df,
            type="candle",
            style="charles",
            title=f"{symbol} - {timeframe}",
            volume=True,
            addplot=add_plots,
            savefig=dict(fname=str(chart_path), dpi=150, bbox_inches="tight"),
            figsize=(14, 8),
        )

        logger.info("Chart generated: %s", chart_path)
        return chart_path

    def analyze_chart(self, chart_path: Path, symbol: str, timeframe: str) -> dict:
        """Send chart image to Claude Vision for analysis."""
        if not self.config.get("chart_analysis_enabled", True):
            return {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": "Chart analysis disabled"}

        with open(chart_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = CHART_ANALYSIS_PROMPT.format(symbol=symbol, timeframe=timeframe)

        try:
            response = self.client.messages.create(
                model=self.config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip()
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            analysis = json.loads(response_text)
            logger.info(
                "Chart analysis for %s: %s (confidence: %.2f)",
                symbol, analysis.get("recommendation"), analysis.get("confidence", 0),
            )
            return analysis

        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI response as JSON: %s", e)
            return {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": "Parse error"}
        except Exception as e:
            logger.error("Chart analysis failed for %s: %s", symbol, e)
            return {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": str(e)}

    def analyze_from_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Generate chart and analyze it in one step."""
        try:
            chart_path = self.generate_chart_image(df, symbol, timeframe)
            return self.analyze_chart(chart_path, symbol, timeframe)
        except Exception as e:
            logger.error("Full chart analysis failed for %s: %s", symbol, e)
            return {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": str(e)}
