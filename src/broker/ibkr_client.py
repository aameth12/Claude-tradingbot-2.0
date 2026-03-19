import asyncio
import math
import json
from datetime import datetime
from typing import Optional

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Contract, Order, Trade as IBTrade
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
            # Request delayed data as fallback when live data subscription
            # is not available (avoids Error 10089)
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
            await self.ib.qualifyContractsAsync(contract)
        except RuntimeError as e:
            # "event loop already running" - proceed without qualification
            # (works for well-known US stocks on SMART exchange)
            logger.warning("Contract qualification fallback for %s: %s", symbol, e)

        self._qualified_contracts[symbol] = contract
        return contract

    async def get_market_data(self, symbol: str) -> dict:
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)
        ticker = self.ib.reqMktData(contract, genericTickList="", snapshot=True)
        await asyncio.sleep(2)  # wait for data
        self.ib.cancelMktData(contract)

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
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )
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

    async def place_bracket_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> list[IBTrade]:
        """Place entry + exit orders for a trade.

        Uses individual orders instead of ib_insync bracket orders to
        avoid Error 135 in paper trading. Entry is a limit order; SL and
        TP are placed as an OCA group so one cancels the other.
        """
        self._ensure_connected()
        contract = await self._get_qualified_contract(symbol)

        exit_action = "BUY" if action == "SELL" else "SELL"
        oca_group = f"OCA_{symbol}_{self.ib.client.getReqId()}"

        # 1. Entry limit order (DAY = valid for current session)
        entry_order = LimitOrder(action, quantity, limit_price)
        entry_order.tif = "DAY"
        entry_trade = self.ib.placeOrder(contract, entry_order)

        # 2. Take profit limit order (OCA group, GTC = good till cancelled)
        tp_order = LimitOrder(exit_action, quantity, take_profit_price)
        tp_order.ocaGroup = oca_group
        tp_order.ocaType = 1  # Cancel other orders in group
        tp_order.tif = "GTC"
        tp_trade = self.ib.placeOrder(contract, tp_order)

        # 3. Stop loss order (OCA group, GTC)
        sl_order = StopOrder(exit_action, quantity, stop_loss_price)
        sl_order.ocaGroup = oca_group
        sl_order.ocaType = 1
        sl_order.tif = "GTC"
        sl_trade = self.ib.placeOrder(contract, sl_order)

        await asyncio.sleep(0.5)

        logger.info(
            "Orders placed: %s %s | qty=%s | entry=%.2f | SL=%.2f | TP=%.2f | OCA=%s",
            action, symbol, quantity, limit_price, stop_loss_price, take_profit_price,
            oca_group,
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

    async def get_account_summary(self) -> dict:
        self._ensure_connected()
        summary = self.ib.accountSummary()
        result = {}
        for item in summary:
            result[item.tag] = item.value
        return result

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
