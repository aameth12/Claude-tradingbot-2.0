import json
import time
from abc import ABC, abstractmethod

import anthropic

from src.utils.logger import setup_logger
from src.utils.config import ANTHROPIC_API_KEY, get_config

logger = setup_logger("agents")


class BaseAgent(ABC):
    """Base class for all AI agents with TTL caching and lazy Claude API access.

    The Anthropic client is only created on first _call_claude() invocation,
    so agents that don't need Claude (regime, sentiment) never instantiate it.
    """

    def __init__(self, name: str, default_ttl: int = 900):
        self.name = name
        self.default_ttl = default_ttl
        self.config = get_config()
        self._client = None  # Lazy — created on first _call_claude()
        self._cache: dict[str, dict] = {}  # {key: {"value": ..., "expires": timestamp}}

    def get_cached(self, key: str = "default"):
        """Return cached result if still valid, else None."""
        entry = self._cache.get(key)
        if entry and time.time() < entry["expires"]:
            return entry["value"]
        return None

    def set_cached(self, value, key: str = "default", ttl: int | None = None):
        """Store a value in cache with TTL."""
        self._cache[key] = {
            "value": value,
            "expires": time.time() + (ttl or self.default_ttl),
        }

    def clear_cache(self):
        self._cache.clear()

    def _call_claude(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> dict:
        """Call Claude API and parse JSON response. Creates client lazily."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY,
                timeout=30.0,
            )

        try:
            model = self.config.get("ai", {}).get("model", "claude-sonnet-4-20250514")
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            logger.info("[%s] Claude call succeeded", self.name)
            return result
        except json.JSONDecodeError as e:
            logger.error("[%s] Failed to parse Claude response: %s", self.name, e)
            return {"error": f"JSON parse error: {e}"}
        except Exception as e:
            logger.error("[%s] Claude API call failed: %s", self.name, e)
            return {"error": str(e)}

    @abstractmethod
    async def run(self, **kwargs) -> dict:
        """Execute the agent's main task. Subclasses must implement."""
        ...
