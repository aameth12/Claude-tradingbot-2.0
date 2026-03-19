import json
from datetime import datetime, date
from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.utils.database import get_session, Trade

logger = setup_logger("risk")


class RiskManager:
    """Manages risk for all trades: stop loss, take profit, trailing stops, position sizing."""

    def __init__(self):
        self.config = get_config()["risk"]
        self.trading_config = get_config()["trading"]

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """Calculate stop loss based on ATR."""
        multiplier = self.config["stop_loss"]["atr_multiplier"]
        if side == "LONG":
            return round(entry_price - (atr * multiplier), 2)
        else:  # SHORT
            return round(entry_price + (atr * multiplier), 2)

    def calculate_take_profit(self, entry_price: float, atr: float, side: str) -> float:
        """Calculate take profit based on ATR, ensuring minimum RR ratio."""
        multiplier = self.config["take_profit"]["atr_multiplier"]
        min_rr = self.config["risk_reward_ratio"]

        # Ensure TP distance >= RR * SL distance
        sl_distance = atr * self.config["stop_loss"]["atr_multiplier"]
        tp_distance = max(atr * multiplier, sl_distance * min_rr)

        if side == "LONG":
            return round(entry_price + tp_distance, 2)
        else:  # SHORT
            return round(entry_price - tp_distance, 2)

    def calculate_position_size(
        self, entry_price: float, stop_loss: float, portfolio_value: float
    ) -> int:
        """Calculate position size based on max risk per trade."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0

        max_risk_dollars = portfolio_value * (self.config["max_risk_per_trade_pct"] / 100)
        max_shares_by_risk = int(max_risk_dollars / risk_per_share)

        # Also respect max position size
        max_shares_by_size = int(self.trading_config["max_position_size"] / entry_price)

        quantity = min(max_shares_by_risk, max_shares_by_size)
        return max(quantity, 0)

    def validate_risk_reward(self, entry: float, stop_loss: float, take_profit: float, side: str) -> bool:
        """Validate that the trade meets minimum risk:reward ratio."""
        if side == "LONG":
            risk = entry - stop_loss
            reward = take_profit - entry
        else:
            risk = stop_loss - entry
            reward = entry - take_profit

        if risk <= 0:
            logger.warning("Invalid risk calculation: risk=%.2f", risk)
            return False

        rr = reward / risk
        min_rr = self.config["risk_reward_ratio"]

        if rr < min_rr - 0.01:  # Small tolerance for floating point rounding
            logger.warning("RR ratio %.2f below minimum %.2f", rr, min_rr)
            return False

        logger.info("RR ratio validated: %.2f (min: %.2f)", rr, min_rr)
        return True

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        highest_price: float,
        side: str,
    ) -> float:
        """Calculate new trailing stop loss level."""
        ts_config = self.config["trailing_stop"]
        if not ts_config["enabled"]:
            return current_stop

        if side == "LONG":
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            if profit_pct >= ts_config["activation_pct"]:
                # Trail from the highest price seen
                new_stop = highest_price * (1 - ts_config["trail_pct"] / 100)
                if new_stop > current_stop:
                    logger.info(
                        "Trailing stop updated: %.2f -> %.2f (highest: %.2f)",
                        current_stop, new_stop, highest_price,
                    )
                    return round(new_stop, 2)
        else:  # SHORT
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            if profit_pct >= ts_config["activation_pct"]:
                # For shorts, trail from the lowest price seen (highest_price stores lowest)
                new_stop = highest_price * (1 + ts_config["trail_pct"] / 100)
                if new_stop < current_stop:
                    logger.info(
                        "Trailing stop updated: %.2f -> %.2f (lowest: %.2f)",
                        current_stop, new_stop, highest_price,
                    )
                    return round(new_stop, 2)

        return current_stop

    def check_daily_loss_limit(self, portfolio_value: float = 100000) -> bool:
        """Check if daily loss limit has been hit (percentage of portfolio)."""
        session = get_session()
        try:
            today = date.today().isoformat()
            trades = (
                session.query(Trade)
                .filter(Trade.status == "CLOSED")
                .filter(Trade.exit_time >= today)
                .all()
            )
            daily_pnl = sum(t.pnl or 0 for t in trades)

            max_loss_pct = self.config["max_daily_loss_pct"]
            max_loss_dollars = portfolio_value * (max_loss_pct / 100)
            if daily_pnl < 0 and abs(daily_pnl) > max_loss_dollars:
                logger.warning(
                    "Daily loss limit hit: $%.2f (max: $%.2f = %.1f%% of $%.0f)",
                    daily_pnl, max_loss_dollars, max_loss_pct, portfolio_value,
                )
                return True
            return False
        finally:
            session.close()

    def can_open_new_trade(self) -> tuple[bool, str]:
        """Check if we can open a new trade based on risk rules."""
        session = get_session()
        try:
            # Check open positions count
            open_trades = session.query(Trade).filter(Trade.status == "OPEN").count()
            max_positions = self.trading_config["max_open_positions"]
            if open_trades >= max_positions:
                return False, f"Max open positions reached ({open_trades}/{max_positions})"

            # Check daily trade count (0 = unlimited)
            max_daily = self.trading_config["max_daily_trades"]
            if max_daily > 0:
                today = date.today().isoformat()
                daily_trades = (
                    session.query(Trade)
                    .filter(Trade.entry_time >= today)
                    .count()
                )
                if daily_trades >= max_daily:
                    return False, f"Max daily trades reached ({daily_trades}/{max_daily})"

            # Check daily loss limit
            if self.check_daily_loss_limit():
                return False, "Daily loss limit reached"

            return True, "OK"
        finally:
            session.close()

    def get_trade_levels(self, entry_price: float, atr: float, side: str, portfolio_value: float) -> dict:
        """Get all trade levels: SL, TP, position size, validated."""
        stop_loss = self.calculate_stop_loss(entry_price, atr, side)
        take_profit = self.calculate_take_profit(entry_price, atr, side)
        quantity = self.calculate_position_size(entry_price, stop_loss, portfolio_value)
        valid_rr = self.validate_risk_reward(entry_price, stop_loss, take_profit, side)

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quantity": quantity,
            "side": side,
            "risk_per_share": round(risk, 2),
            "reward_per_share": round(reward, 2),
            "risk_reward_ratio": round(rr_ratio, 2),
            "total_risk": round(risk * quantity, 2),
            "total_reward": round(reward * quantity, 2),
            "valid": valid_rr and quantity > 0,
        }
