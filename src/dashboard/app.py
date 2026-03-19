import json
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.database import get_session, Trade, DailySummary, BacktestResult, init_db
from src.utils.config import get_config


def get_all_trades():
    session = get_session()
    try:
        trades = session.query(Trade).order_by(Trade.entry_time.desc()).all()
        return [
            {
                "id": t.id, "symbol": t.symbol, "side": t.side,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "quantity": t.quantity, "stop_loss": t.stop_loss,
                "take_profit": t.take_profit, "status": t.status,
                "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                "entry_time": str(t.entry_time), "exit_time": str(t.exit_time),
                "strategy": t.strategy, "timeframe": t.timeframe,
            }
            for t in trades
        ]
    finally:
        session.close()


def get_open_positions():
    session = get_session()
    try:
        return session.query(Trade).filter(Trade.status == "OPEN").all()
    finally:
        session.close()


def main():
    st.set_page_config(
        page_title="AI Trading Bot Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("AI Trading Bot Dashboard")

    init_db()
    config = get_config()
    trades_data = get_all_trades()
    df_trades = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

    # --- Top Metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)

    closed_trades = [t for t in trades_data if t["status"] == "CLOSED"]
    open_positions = [t for t in trades_data if t["status"] == "OPEN"]
    total_pnl = sum(t.get("pnl") or 0 for t in closed_trades)
    winners = [t for t in closed_trades if (t.get("pnl") or 0) > 0]
    win_rate = (len(winners) / len(closed_trades) * 100) if closed_trades else 0

    with col1:
        st.metric("Total P&L", f"${total_pnl:+,.2f}")
    with col2:
        st.metric("Total Trades", len(closed_trades))
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric("Open Positions", len(open_positions))
    with col5:
        today_pnl = sum(
            t.get("pnl") or 0
            for t in closed_trades
            if t.get("exit_time", "").startswith(date.today().isoformat())
        )
        st.metric("Today's P&L", f"${today_pnl:+,.2f}")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "P&L Chart", "Trade History", "Open Positions", "Watchlist", "Backtest Results"
    ])

    with tab1:
        st.subheader("Cumulative P&L")
        if closed_trades:
            pnl_df = pd.DataFrame(closed_trades)
            pnl_df["exit_time"] = pd.to_datetime(pnl_df["exit_time"])
            pnl_df = pnl_df.sort_values("exit_time")
            pnl_df["cumulative_pnl"] = pnl_df["pnl"].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pnl_df["exit_time"],
                y=pnl_df["cumulative_pnl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="green" if total_pnl >= 0 else "red", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,0,0.1)" if total_pnl >= 0 else "rgba(255,0,0,0.1)",
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # P&L by symbol
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("P&L by Symbol")
                symbol_pnl = pnl_df.groupby("symbol")["pnl"].sum().sort_values(ascending=True)
                fig2 = go.Figure(go.Bar(
                    x=symbol_pnl.values,
                    y=symbol_pnl.index,
                    orientation="h",
                    marker_color=["green" if v > 0 else "red" for v in symbol_pnl.values],
                ))
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                st.subheader("Win/Loss Distribution")
                fig3 = go.Figure(go.Histogram(
                    x=[t["pnl"] for t in closed_trades],
                    nbinsx=20,
                    marker_color="steelblue",
                ))
                fig3.update_layout(
                    xaxis_title="P&L ($)",
                    yaxis_title="Count",
                    height=300,
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No closed trades yet. Start trading to see P&L data.")

    with tab2:
        st.subheader("Trade History")
        if not df_trades.empty:
            display_cols = [
                "symbol", "side", "entry_price", "exit_price", "quantity",
                "stop_loss", "take_profit", "pnl", "pnl_pct", "status",
                "entry_time", "exit_time", "strategy",
            ]
            available_cols = [c for c in display_cols if c in df_trades.columns]
            st.dataframe(
                df_trades[available_cols],
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No trades recorded yet.")

    with tab3:
        st.subheader("Open Positions")
        open_data = [t for t in trades_data if t["status"] == "OPEN"]
        if open_data:
            open_df = pd.DataFrame(open_data)
            st.dataframe(open_df[["symbol", "side", "entry_price", "quantity", "stop_loss", "take_profit", "entry_time"]])
        else:
            st.info("No open positions.")

    with tab4:
        st.subheader("Watchlist")
        watchlist = config["watchlist"]
        st.write("Currently monitoring:", ", ".join(watchlist))
        st.markdown("---")
        st.subheader("Risk Settings")
        risk = config["risk"]
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Risk:Reward Ratio", f"1:{risk['risk_reward_ratio']}")
        with col_r2:
            st.metric("Max Risk/Trade", f"{risk['max_risk_per_trade_pct']}%")
        with col_r3:
            st.metric("Max Daily Loss", f"{risk['max_daily_loss_pct']}%")

    with tab5:
        st.subheader("Backtest Results")
        session = get_session()
        try:
            results = session.query(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(20).all()
            if results:
                bt_data = [
                    {
                        "Symbol": r.symbol, "Strategy": r.strategy,
                        "Timeframe": r.timeframe,
                        "Trades": r.total_trades, "Win Rate": f"{r.win_rate:.1f}%",
                        "Return": f"{r.total_return_pct:+.2f}%",
                        "Max DD": f"{r.max_drawdown_pct:.2f}%",
                        "Sharpe": f"{r.sharpe_ratio:.2f}",
                        "Profit Factor": f"{r.profit_factor:.2f}",
                        "Period": f"{r.start_date} to {r.end_date}",
                    }
                    for r in results
                ]
                st.dataframe(pd.DataFrame(bt_data), use_container_width=True)
            else:
                st.info("No backtest results yet. Use /backtest in Telegram to run one.")
        finally:
            session.close()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Bot Configuration")
        st.write(f"Mode: **{config['trading']['mode']}**")
        st.write(f"Max Positions: **{config['trading']['max_open_positions']}**")
        st.write(f"Max Daily Trades: **{config['trading']['max_daily_trades']}**")
        st.write(f"Max Position Size: **${config['trading']['max_position_size']}**")
        st.markdown("---")
        st.write(f"Trailing Stop: **{'ON' if risk['trailing_stop']['enabled'] else 'OFF'}**")
        st.write(f"Trail %: **{risk['trailing_stop']['trail_pct']}%**")
        st.markdown("---")
        if st.button("Refresh Data"):
            st.rerun()


if __name__ == "__main__":
    main()
