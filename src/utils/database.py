from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Boolean, Text, Enum as SqlEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker
from src.utils.config import DATA_DIR

DB_PATH = DATA_DIR / "trading_bot.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(String(5), nullable=False)  # LONG or SHORT
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    status = Column(String(10), nullable=False, default="OPEN")  # OPEN, CLOSED, CANCELLED
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    strategy = Column(String(50), nullable=True)
    timeframe = Column(String(10), nullable=True)
    signals = Column(Text, nullable=True)  # JSON of signals that triggered the trade
    order_id = Column(Integer, nullable=True)  # IBKR order ID
    exit_reason = Column(String(20), nullable=True)  # SL_HIT, TP_HIT, TRAILING_STOP, MANUAL
    notes = Column(Text, nullable=True)


class DailySummary(Base):
    __tablename__ = "daily_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    best_trade_pnl = Column(Float, default=0.0)
    worst_trade_pnl = Column(Float, default=0.0)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    strategy = Column(String(50), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(String(10), nullable=False)
    end_date = Column(String(10), nullable=False)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    total_return_pct = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    details = Column(Text, nullable=True)  # JSON with full trade list


class AgentOutput(Base):
    __tablename__ = "agent_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=True)
    output_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class TradeReviewRecord(Base):
    __tablename__ = "trade_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=False, index=True)
    correct_indicators = Column(Text, nullable=True)  # JSON list
    incorrect_indicators = Column(Text, nullable=True)  # JSON list
    suggested_adjustments = Column(Text, nullable=True)  # JSON dict
    review_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class IndicatorAccuracy(Base):
    __tablename__ = "indicator_accuracy"

    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_name = Column(String(30), nullable=False, index=True)
    total_signals = Column(Integer, default=0)
    correct_signals = Column(Integer, default=0)
    accuracy_pct = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
    # Migrate existing tables: add exit_reason column if missing
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "trades" in insp.get_table_names():
        columns = [c["name"] for c in insp.get_columns("trades")]
        if "exit_reason" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE trades ADD COLUMN exit_reason VARCHAR(20)"))
                conn.commit()


def get_session():
    return SessionLocal()
