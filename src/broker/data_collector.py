"""Background IBKR data collector.

Reads ib_insync's synchronous cache accessors on a 30-second timer
so the Telegram dashboard never has to make blocking IBKR calls.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from ib_insync import IB

from src.utils.logger import setup_logger

logger = setup_logger("data_collector")

_ACCOUNT_TAGS = {
    "NetLiquidation", "TotalCashValue", "UnrealizedPnL",
    "RealizedPnL", "BuyingPower", "GrossPositionValue",
}


class IBKRDataCollector:
    """Periodically caches IBKR data for instant dashboard reads."""

    def __init__(self, ib: IB):
        self.ib = ib
        self.last_updated: Optional[datetime] = None

        # Cached data
        self.account: dict[str, float] = {}
        self.portfolio: list[dict] = []
        self.open_orders: list[dict] = []
        self.executions: list[dict] = []
        self.managed_account: str = ""

        self._subscription_attempts = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """One-time init after broker connects."""
        accounts = self.ib.managedAccounts()
        if accounts:
            self.managed_account = accounts[0]
            logger.info("Data collector started for account %s", self.managed_account)
        else:
            logger.warning("Data collector: no managed accounts found")

        # Kick off the first data subscription (non-blocking)
        await self._try_subscribe()
        # Do an immediate collect from whatever cache is available
        await self.collect()

    async def _try_subscribe(self):
        """Try to start account data subscription with a short timeout."""
        if not self.managed_account:
            return
        try:
            await asyncio.wait_for(
                self.ib.reqAccountUpdatesAsync(self.managed_account),
                timeout=5,
            )
            self._subscription_attempts = 0
            logger.info("Account data subscription active")
        except (asyncio.TimeoutError, TimeoutError):
            self._subscription_attempts += 1
            logger.warning(
                "Account subscription timed out (attempt %d/3)",
                self._subscription_attempts,
            )
        except Exception as e:
            self._subscription_attempts += 1
            logger.warning("Account subscription failed: %s", e)

    # ------------------------------------------------------------------
    # Periodic collection (called by APScheduler every 30s)
    # ------------------------------------------------------------------

    async def collect(self):
        """Read all data from ib_insync caches. Never hangs."""
        try:
            if not self.ib.isConnected():
                logger.debug("Data collector: not connected, skipping")
                return

            self._collect_account()
            self._collect_portfolio()
            self._collect_orders()
            self._collect_executions()

            # If account data still empty, retry subscription (max 3 times)
            if not self.has_data() and self._subscription_attempts < 3:
                await self._try_subscribe()
                self._collect_account()  # re-read after subscription

            self.last_updated = datetime.now(timezone.utc)
            logger.debug(
                "Data collected: account=%s, positions=%d, orders=%d, fills=%d",
                self.has_data(), len(self.portfolio),
                len(self.open_orders), len(self.executions),
            )
        except Exception as e:
            logger.warning("Data collection error: %s", e)

    def _collect_account(self):
        """Parse account values from ib_insync cache."""
        # Try accountValues first (populated by reqAccountUpdates)
        result = {}
        for item in self.ib.accountValues():
            if item.tag in _ACCOUNT_TAGS:
                if hasattr(item, "currency") and item.currency not in ("USD", ""):
                    continue
                try:
                    result[item.tag] = float(item.value)
                except (ValueError, TypeError):
                    result[item.tag] = 0.0

        # Fallback to accountSummary
        if not result or result.get("NetLiquidation", 0) <= 0:
            for item in self.ib.accountSummary():
                if item.tag in _ACCOUNT_TAGS:
                    try:
                        result[item.tag] = float(item.value)
                    except (ValueError, TypeError):
                        result[item.tag] = 0.0

        if result:
            self.account = result

    def _collect_portfolio(self):
        """Read portfolio positions from cache."""
        positions = []
        for item in self.ib.portfolio():
            if item.position == 0:
                continue
            positions.append({
                "symbol": item.contract.symbol,
                "position": item.position,
                "market_price": item.marketPrice,
                "market_value": item.marketValue,
                "average_cost": item.averageCost,
                "unrealized_pnl": item.unrealizedPNL,
                "realized_pnl": item.realizedPNL,
            })
        self.portfolio = positions

    def _collect_orders(self):
        """Read open orders from cache."""
        orders = []
        for trade in self.ib.openTrades():
            orders.append({
                "order_id": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "status": trade.orderStatus.status,
                "limit_price": trade.order.lmtPrice or 0,
                "aux_price": trade.order.auxPrice or 0,
            })
        self.open_orders = orders

    def _collect_executions(self):
        """Read today's fills from cache."""
        today = datetime.now(timezone.utc).date()
        fills = []
        for fill in self.ib.fills():
            exec_time = fill.execution.time
            # Filter for today only
            if hasattr(exec_time, "date") and exec_time.date() != today:
                continue
            fills.append({
                "symbol": fill.contract.symbol,
                "exec_id": fill.execution.execId,
                "time": exec_time,
                "side": fill.execution.side,
                "price": fill.execution.price,
                "quantity": fill.execution.shares,
                "order_id": fill.execution.orderId,
                "commission": fill.commissionReport.commission
                if fill.commissionReport else 0,
            })
        self.executions = fills

    # ------------------------------------------------------------------
    # Accessors (instant, never block)
    # ------------------------------------------------------------------

    def has_data(self) -> bool:
        """True if we have meaningful account data."""
        return bool(self.account) and self.account.get("NetLiquidation", 0) > 0

    def get_freshness(self) -> str:
        """Human-readable time since last update."""
        if not self.last_updated:
            return "never"
        delta = (datetime.now(timezone.utc) - self.last_updated).total_seconds()
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        return f"{int(delta // 3600)}h ago"
