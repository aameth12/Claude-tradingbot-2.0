import asyncio
import math
import json
from datetime import datetime, timezone
from typing import Optional

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Contract, Order, Trade as IBTrade, ExecutionFilter
from src.utils.logger import setup_logger
from src.utils.config import IB_HOST, IB_PORT, IB_CLIENT_ID, get_config

logger = setup_logger("broker")


class IBKRClient:
    """Interactive Brokers client using ib_insync for order management.

    Uses async versions of ib_insync methods (qualifyContractsAsync,
    reqHistoricalDataAsync) to avoid 'event loop already running' errors
    when combined with asyncio-based telegram bot and scheduler.
    """

    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.config = get_config()["broker"]
        self._qualified_contracts: dict[str, Stock] = {}

    async def connect(self):
        if self.connected:
            return
        try:
            await self.ib.connectAsync(
                host=IB_HOST,
                port=IB_PORT,
                clientId=IB_CLIENT_ID,
                timeout=self.config.get("timeout", 30),
                readonly=self.config.get("readonly", False),
            )
            self.connected = True
            self.ib.reqMarketDataType(3)  # 3 = delayed
            logger.info("Connected to IB Gateway at %s:%s", IB_HOST, IB_PORT)
        except Exception as e:
            logger.error("Failed to connect to IB Gateway: %s", e)
            raise

    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self._qualified_contracts.clear()
            logger.info("Disconnected from IB Gateway")

    def _ensure_connected(self):
        if not self.connected:
            raise ConnectionError("Not connected to IB Gateway. Call connect() first.")

    def create_stock_contract(self, symbol: str) -> Stock:
        return Stock(symbol, "SMART", "USD")

    async def _get_qualified_contract(self, symbol: str) -> Stock:
        """Get a qualified contract, using cache to avoid repeated async calls."""
        if symbol in self._qualified_contracts:
            return self._qualified_contracts[symbol]

        contract = self.create_stock_contract(symbol)
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            if not qualified or not qualified[0].conId:
                logger.warning("Contract qualification returned no conId for %s", symbol)
        except Exception as e:
            logger.warning("Contract qualification fallback for %s: %s", symbol, e)

        self._qualified_contracts[symbol] = contract
        return contract

    async def get_market_data(self, symbol: str) -> dict:
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)
        try:
            ticker = self.ib.reqMktData(contract, genericTickList="", snapshot=True)
            await asyncio.sleep(1)  # wait for data (reduced from 2s)
            self.ib.cancelMktData(contract)
        except Exception as e:
            logger.warning("Market data request failed for %s: %s", symbol, e)
            return {"symbol": symbol, "bid": None, "ask": None, "last": None,
                    "volume": None, "high": None, "low": None, "close": None}

        # Use last price, fallback to close, then bid/ask midpoint
        last = ticker.last
        if last is None or (isinstance(last, float) and math.isnan(last)):
            last = ticker.close
        if last is None or (isinstance(last, float) and math.isnan(last)):
            bid = ticker.bid if ticker.bid and not math.isnan(ticker.bid) else 0
            ask = ticker.ask if ticker.ask and not math.isnan(ticker.ask) else 0
            last = (bid + ask) / 2 if (bid and ask) else None

        return {
            "symbol": symbol,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "last": last,
            "volume": ticker.volume,
            "high": ticker.high,
            "low": ticker.low,
            "close": ticker.close,
        }

    async def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
    ) -> list[dict]:
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)
        try:
            bars = await asyncio.wait_for(
                self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=1,
                ),
                timeout=20,  # 20s timeout for historical data
            )
        except asyncio.TimeoutError:
            logger.warning("Historical data timeout for %s", symbol)
            return []
        return [
            {
                "date": str(bar.date),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in (bars or [])
        ]

    async def place_market_order(
        self, symbol: str, action: str, quantity: int
    ) -> IBTrade:
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        logger.info("Market order placed: %s %s %s shares", action, symbol, quantity)
        return trade

    async def place_entry_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        timeout: int = 10,
    ) -> tuple[IBTrade, float]:
        """Place a market entry order and wait for fill confirmation.

        Returns (trade, fill_price). Raises RuntimeError if not filled.
        """
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)

        # Wait for fill (poll every 0.5s)
        for _ in range(timeout * 2):
            await asyncio.sleep(0.5)
            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(
                    "Entry filled: %s %s | qty=%s | fill=%.2f",
                    action, symbol, quantity, fill_price,
                )
                return trade, fill_price

        # Not filled — cancel and raise
        self.ib.cancelOrder(trade.order)
        raise RuntimeError(f"Entry order not filled within {timeout}s for {symbol}")

    async def place_exit_orders(
        self,
        symbol: str,
        exit_action: str,
        quantity: int,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> tuple[IBTrade, IBTrade]:
        """Place SL + TP as an OCA group. Call only after entry is filled."""
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)
        oca_group = f"OCA_{symbol}_{self.ib.client.getReqId()}"

        # Take profit limit order (GTC)
        tp_order = LimitOrder(exit_action, quantity, take_profit_price)
        tp_order.ocaGroup = oca_group
        tp_order.ocaType = 1
        tp_order.tif = "GTC"
        tp_trade = self.ib.placeOrder(contract, tp_order)

        # Stop loss order (GTC)
        sl_order = StopOrder(exit_action, quantity, stop_loss_price)
        sl_order.ocaGroup = oca_group
        sl_order.ocaType = 1
        sl_order.tif = "GTC"
        sl_trade = self.ib.placeOrder(contract, sl_order)

        await asyncio.sleep(0.5)

        logger.info(
            "Exit orders placed: %s %s | qty=%s | SL=%.2f | TP=%.2f | OCA=%s",
            exit_action, symbol, quantity, stop_loss_price, take_profit_price,
            oca_group,
        )
        return tp_trade, sl_trade

    async def place_bracket_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> list[IBTrade]:
        """Place entry + exit orders using market entry + OCA exit group.

        Entry uses a market order for immediate fill, then SL/TP are placed
        only after entry is confirmed. This prevents orphan SL/TP orders.
        """
        entry_trade, fill_price = await self.place_entry_order(symbol, action, quantity)

        exit_action = "BUY" if action == "SELL" else "SELL"
        tp_trade, sl_trade = await self.place_exit_orders(
            symbol, exit_action, quantity, stop_loss_price, take_profit_price,
        )

        logger.info(
            "Bracket complete: %s %s | qty=%s | fill=%.2f | SL=%.2f | TP=%.2f",
            action, symbol, quantity, fill_price, stop_loss_price, take_profit_price,
        )
        return [entry_trade, tp_trade, sl_trade]

    async def place_trailing_stop(
        self,
        symbol: str,
        action: str,
        quantity: int,
        trail_percent: float,
    ) -> IBTrade:
        """Place a trailing stop order."""
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)

        # Reverse action for the stop (if we bought, trailing stop sells)
        stop_action = "SELL" if action == "BUY" else "BUY"
        order = Order()
        order.action = stop_action
        order.totalQuantity = quantity
        order.orderType = "TRAIL"
        order.trailingPercent = trail_percent

        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "Trailing stop placed: %s %s | qty=%s | trail=%.1f%%",
            stop_action, symbol, quantity, trail_percent,
        )
        return trade

    async def modify_stop_loss(self, order_id: int, new_stop_price: float):
        """Modify an existing stop loss order price (for manual trailing)."""
        self._ensure_connected()
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                trade.order.auxPrice = new_stop_price
                self.ib.placeOrder(trade.contract, trade.order)
                logger.info("Stop loss modified: order %s -> %.2f", order_id, new_stop_price)
                return True
        logger.warning("Order %s not found in open trades", order_id)
        return False

    async def cancel_order(self, order_id: int):
        self._ensure_connected()
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                logger.info("Order cancelled: %s", order_id)
                return True
        return False

    async def cancel_stale_orders(self, symbols: list[str] | None = None, max_age_seconds: int = 300):
        """Cancel pending orders older than max_age_seconds for given symbols.

        This prevents Error 201 (too many orders on one side) by cleaning up
        stale unfilled orders before placing new ones.
        """
        self._ensure_connected()
        now = datetime.now(timezone.utc)
        cancelled = 0

        for trade in self.ib.openTrades():
            # Only cancel for specified symbols (or all if None)
            if symbols and trade.contract.symbol not in symbols:
                continue

            status = trade.orderStatus.status
            if status in ("Cancelled", "Filled", "Inactive"):
                continue

            # Check if the order has been sitting unfilled
            if trade.log:
                order_time = trade.log[0].time
                # Handle timezone-naive timestamps from ib_insync
                if order_time.tzinfo is None:
                    order_time = order_time.replace(tzinfo=timezone.utc)
                age = (now - order_time).total_seconds()
                if age > max_age_seconds:
                    self.ib.cancelOrder(trade.order)
                    cancelled += 1
                    logger.info(
                        "Cancelled stale order: %s %s %s (age: %ds)",
                        trade.order.action, trade.contract.symbol,
                        trade.order.orderType, int(age),
                    )

        if cancelled:
            logger.info("Cancelled %d stale orders", cancelled)
            await asyncio.sleep(0.5)  # Let cancellations propagate
        return cancelled

    async def get_recent_fills(self, symbol: str) -> list[dict]:
        """Get recent execution fills for a symbol."""
        self._ensure_connected()
        fills = self.ib.fills()
        return [
            {
                "symbol": f.contract.symbol,
                "price": f.execution.avgPrice,
                "quantity": f.execution.shares,
                "side": f.execution.side,
                "time": f.execution.time,
                "order_id": f.execution.orderId,
            }
            for f in fills if f.contract.symbol == symbol
        ]

    async def get_positions(self) -> list[dict]:
        self._ensure_connected()
        positions = self.ib.positions()
        return [
            {
                "symbol": pos.contract.symbol,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost,
            }
            for pos in positions
        ]

    async def get_portfolio(self) -> list[dict]:
        """Get portfolio positions with IBKR-provided P&L and market data.

        Uses ib.portfolio() which returns richer data than ib.positions(),
        including marketPrice, marketValue, unrealizedPNL, and realizedPNL.
        """
        self._ensure_connected()
        portfolio = self.ib.portfolio()
        return [
            {
                "symbol": item.contract.symbol,
                "position": item.position,
                "market_price": item.marketPrice,
                "market_value": item.marketValue,
                "average_cost": item.averageCost,
                "unrealized_pnl": item.unrealizedPNL,
                "realized_pnl": item.realizedPNL,
            }
            for item in portfolio
        ]

    async def get_account_summary(self) -> dict:
        self._ensure_connected()
        summary = self.ib.accountSummary()
        result = {}
        for item in summary:
            result[item.tag] = item.value
        return result

    async def get_account_pnl(self) -> dict:
        """Get account-level balance and P&L from IBKR."""
        self._ensure_connected()
        tags_of_interest = {
            "NetLiquidation", "TotalCashValue", "UnrealizedPnL",
            "RealizedPnL", "BuyingPower", "GrossPositionValue",
        }
        result = {}

        # Try cached accountSummary first
        summary = self.ib.accountSummary()
        for item in summary:
            if item.tag in tags_of_interest:
                try:
                    result[item.tag] = float(item.value)
                except (ValueError, TypeError):
                    result[item.tag] = 0.0

        # Try cached accountValues
        if not result or result.get("NetLiquidation", 0) == 0:
            values = self.ib.accountValues()
            for item in values:
                if item.tag in tags_of_interest and item.currency in ("USD", ""):
                    try:
                        result[item.tag] = float(item.value)
                    except (ValueError, TypeError):
                        result[item.tag] = 0.0

        # If still empty, do an async request to populate the cache
        if not result or result.get("NetLiquidation", 0) == 0:
            try:
                await self.ib.reqAccountSummaryAsync()
                summary = self.ib.accountSummary()
                for item in summary:
                    if item.tag in tags_of_interest:
                        try:
                            result[item.tag] = float(item.value)
                        except (ValueError, TypeError):
                            result[item.tag] = 0.0
            except Exception as e:
                logger.warning("reqAccountSummaryAsync failed: %s", e)

        return result

    async def get_executions(self, symbol: str = "") -> list[dict]:
        """Get execution reports (survives reconnections unlike fills()).

        Args:
            symbol: Filter by symbol (empty = all symbols).
        """
        self._ensure_connected()
        exec_filter = ExecutionFilter()
        if symbol:
            exec_filter.symbol = symbol

        # Request fresh executions from TWS
        trades = await self.ib.reqExecutionsAsync(exec_filter)

        results = []
        for fill in self.ib.fills():
            if symbol and fill.contract.symbol != symbol:
                continue
            results.append({
                "symbol": fill.contract.symbol,
                "exec_id": fill.execution.execId,
                "time": fill.execution.time,
                "side": fill.execution.side,
                "price": fill.execution.avgPrice,
                "quantity": fill.execution.shares,
                "order_id": fill.execution.orderId,
                "commission": fill.commissionReport.commission if fill.commissionReport else 0,
            })
        return results

    async def get_open_orders(self) -> list[dict]:
        self._ensure_connected()
        trades = self.ib.openTrades()
        return [
            {
                "order_id": t.order.orderId,
                "symbol": t.contract.symbol,
                "action": t.order.action,
                "quantity": t.order.totalQuantity,
                "order_type": t.order.orderType,
                "status": t.orderStatus.status,
                "limit_price": t.order.lmtPrice,
                "aux_price": t.order.auxPrice,
            }
            for t in trades
        ]
