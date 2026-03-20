"""Daily performance target tracking with auto-ratcheting goals.

Compares each day's win rate and P&L % against targets. When the bot
beats its targets consistently, the targets automatically increase.
"""
from datetime import date, timedelta

from src.utils.database import get_session, DailySummary, PerformanceTarget, Trade
from src.utils.config import get_config
from src.utils.logger import setup_logger

logger = setup_logger("performance_tracker")

# How many recent days to look at for ratcheting
LOOKBACK_DAYS = 5


class PerformanceTracker:
    """Tracks daily targets and ratchets them up based on recent performance."""

    def __init__(self):
        config = get_config().get("performance_targets", {})
        self.initial_win_rate = config.get("initial_win_rate", 50.0)
        self.initial_pnl_pct = config.get("initial_pnl_pct", 2.0)
        self.win_rate_step = config.get("win_rate_step", 2.0)  # increase by 2% when beating target
        self.pnl_pct_step = config.get("pnl_pct_step", 0.5)  # increase by 0.5% when beating target
        self.lookback_days = config.get("lookback_days", LOOKBACK_DAYS)

    def get_today_targets(self) -> dict:
        """Get today's targets. Creates them if they don't exist."""
        session = get_session()
        try:
            today_str = date.today().isoformat()
            record = session.query(PerformanceTarget).filter(
                PerformanceTarget.date == today_str
            ).first()

            if record:
                return {
                    "date": record.date,
                    "win_rate_target": record.win_rate_target,
                    "pnl_pct_target": record.pnl_pct_target,
                    "win_rate_actual": record.win_rate_actual,
                    "pnl_pct_actual": record.pnl_pct_actual,
                    "win_rate_hit": record.win_rate_hit,
                    "pnl_pct_hit": record.pnl_pct_hit,
                    "streak": record.streak,
                }

            # Calculate today's targets from recent performance
            targets = self._calculate_next_targets(session)
            record = PerformanceTarget(
                date=today_str,
                win_rate_target=targets["win_rate_target"],
                pnl_pct_target=targets["pnl_pct_target"],
                streak=targets["streak"],
            )
            session.add(record)
            session.commit()

            logger.info(
                "Today's targets set: win_rate=%.1f%%, pnl=%.2f%%, streak=%d",
                targets["win_rate_target"], targets["pnl_pct_target"], targets["streak"],
            )

            return {
                "date": today_str,
                "win_rate_target": targets["win_rate_target"],
                "pnl_pct_target": targets["pnl_pct_target"],
                "win_rate_actual": None,
                "pnl_pct_actual": None,
                "win_rate_hit": None,
                "pnl_pct_hit": None,
                "streak": targets["streak"],
            }
        finally:
            session.close()

    def evaluate_day(self, portfolio_value: float = 100000) -> dict:
        """Evaluate today's performance against targets. Called at end of day.

        Returns a dict with results and the comparison message.
        """
        session = get_session()
        try:
            today_str = date.today().isoformat()

            # Get today's closed trades
            trades = session.query(Trade).filter(
                Trade.status == "CLOSED",
                Trade.exit_time >= today_str,
            ).all()

            if not trades:
                return {"message": "No trades today — no comparison to make.", "trades": 0}

            winners = [t for t in trades if (t.pnl or 0) > 0]
            total_pnl = sum(t.pnl or 0 for t in trades)
            win_rate = round(len(winners) / len(trades) * 100, 1)
            pnl_pct = round(total_pnl / portfolio_value * 100, 2)

            # Get or create today's target record
            record = session.query(PerformanceTarget).filter(
                PerformanceTarget.date == today_str
            ).first()

            if not record:
                targets = self._calculate_next_targets(session)
                record = PerformanceTarget(
                    date=today_str,
                    win_rate_target=targets["win_rate_target"],
                    pnl_pct_target=targets["pnl_pct_target"],
                    streak=targets["streak"],
                )
                session.add(record)

            # Record actuals
            record.win_rate_actual = win_rate
            record.pnl_pct_actual = pnl_pct
            record.win_rate_hit = win_rate >= record.win_rate_target
            record.pnl_pct_hit = pnl_pct >= record.pnl_pct_target

            # Update streak
            both_hit = record.win_rate_hit and record.pnl_pct_hit
            if both_hit:
                record.streak = (record.streak or 0) + 1
            else:
                record.streak = 0

            session.commit()

            # Build comparison message
            msg = self._build_report(record, trades, total_pnl)

            logger.info(
                "Day evaluated: win_rate=%.1f%% (target %.1f%%), pnl=%.2f%% (target %.2f%%)",
                win_rate, record.win_rate_target, pnl_pct, record.pnl_pct_target,
            )

            return {
                "message": msg,
                "trades": len(trades),
                "win_rate": win_rate,
                "pnl_pct": pnl_pct,
                "win_rate_target": record.win_rate_target,
                "pnl_pct_target": record.pnl_pct_target,
                "win_rate_hit": record.win_rate_hit,
                "pnl_pct_hit": record.pnl_pct_hit,
                "streak": record.streak,
            }
        finally:
            session.close()

    def get_recent_history(self, days: int = 7) -> list[dict]:
        """Get recent daily target results for display."""
        session = get_session()
        try:
            records = session.query(PerformanceTarget).order_by(
                PerformanceTarget.date.desc()
            ).limit(days).all()

            return [
                {
                    "date": r.date,
                    "win_rate_target": r.win_rate_target,
                    "pnl_pct_target": r.pnl_pct_target,
                    "win_rate_actual": r.win_rate_actual,
                    "pnl_pct_actual": r.pnl_pct_actual,
                    "win_rate_hit": r.win_rate_hit,
                    "pnl_pct_hit": r.pnl_pct_hit,
                    "streak": r.streak,
                }
                for r in records
            ]
        finally:
            session.close()

    def _calculate_next_targets(self, session) -> dict:
        """Calculate targets for today based on recent performance."""
        # Get recent target records
        recent = session.query(PerformanceTarget).filter(
            PerformanceTarget.win_rate_actual.isnot(None),
        ).order_by(PerformanceTarget.date.desc()).limit(self.lookback_days).all()

        if not recent:
            # First day — use initial targets
            return {
                "win_rate_target": self.initial_win_rate,
                "pnl_pct_target": self.initial_pnl_pct,
                "streak": 0,
            }

        # Get the previous day's targets and streak
        last = recent[0]
        prev_wr_target = last.win_rate_target
        prev_pnl_target = last.pnl_pct_target
        prev_streak = last.streak or 0

        # Calculate averages from recent days
        actual_win_rates = [r.win_rate_actual for r in recent if r.win_rate_actual is not None]
        actual_pnl_pcts = [r.pnl_pct_actual for r in recent if r.pnl_pct_actual is not None]

        avg_win_rate = sum(actual_win_rates) / len(actual_win_rates) if actual_win_rates else 0
        avg_pnl_pct = sum(actual_pnl_pcts) / len(actual_pnl_pcts) if actual_pnl_pcts else 0

        # Ratchet up: if we beat the target recently, raise it
        # Rule: if avg of last N days beats current target, step up
        new_wr_target = prev_wr_target
        new_pnl_target = prev_pnl_target

        if avg_win_rate > prev_wr_target and len(actual_win_rates) >= 3:
            new_wr_target = prev_wr_target + self.win_rate_step
            logger.info(
                "Win rate target raised: %.1f%% -> %.1f%% (avg was %.1f%%)",
                prev_wr_target, new_wr_target, avg_win_rate,
            )

        if avg_pnl_pct > prev_pnl_target and len(actual_pnl_pcts) >= 3:
            new_pnl_target = prev_pnl_target + self.pnl_pct_step
            logger.info(
                "P&L target raised: %.2f%% -> %.2f%% (avg was %.2f%%)",
                prev_pnl_target, new_pnl_target, avg_pnl_pct,
            )

        # If we missed target badly (avg < 80% of target over last N days), ease off
        if actual_win_rates and avg_win_rate < prev_wr_target * 0.8 and prev_wr_target > self.initial_win_rate:
            new_wr_target = max(self.initial_win_rate, prev_wr_target - self.win_rate_step)
            logger.info("Win rate target eased: %.1f%% -> %.1f%%", prev_wr_target, new_wr_target)

        if actual_pnl_pcts and avg_pnl_pct < prev_pnl_target * 0.5 and prev_pnl_target > self.initial_pnl_pct:
            new_pnl_target = max(self.initial_pnl_pct, prev_pnl_target - self.pnl_pct_step)
            logger.info("P&L target eased: %.2f%% -> %.2f%%", prev_pnl_target, new_pnl_target)

        # Cap at reasonable maximums
        new_wr_target = min(new_wr_target, 90.0)
        new_pnl_target = min(new_pnl_target, 20.0)

        return {
            "win_rate_target": round(new_wr_target, 1),
            "pnl_pct_target": round(new_pnl_target, 2),
            "streak": prev_streak if (last.win_rate_hit and last.pnl_pct_hit) else 0,
        }

    @staticmethod
    def _build_report(record: PerformanceTarget, trades: list, total_pnl: float) -> str:
        """Build the daily performance report message."""
        wr_icon = "HIT" if record.win_rate_hit else "MISS"
        pnl_icon = "HIT" if record.pnl_pct_hit else "MISS"

        winners = len([t for t in trades if (t.pnl or 0) > 0])

        lines = [
            f"DAILY PERFORMANCE REPORT",
            f"{'='*30}",
            f"",
            f"Win Rate:  {record.win_rate_actual:.1f}%  vs  {record.win_rate_target:.1f}% target  [{wr_icon}]",
            f"  ({winners}/{len(trades)} trades won)",
            f"",
            f"P&L:  {record.pnl_pct_actual:+.2f}%  vs  {record.pnl_pct_target:+.2f}% target  [{pnl_icon}]",
            f"  (${total_pnl:+,.2f})",
            f"",
        ]

        both_hit = record.win_rate_hit and record.pnl_pct_hit
        if both_hit:
            streak = record.streak or 1
            lines.append(f"BOTH TARGETS HIT! Streak: {streak} day(s)")
            lines.append(f"Targets will increase tomorrow.")
        elif record.win_rate_hit or record.pnl_pct_hit:
            lines.append(f"Partial hit — keep pushing.")
        else:
            lines.append(f"Missed both targets. Reviewing strategy...")

        return "\n".join(lines)
