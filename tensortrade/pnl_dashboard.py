"""
pnl_dashboard.py
===============
Interactive dashboard for TensorTrade P&L analysis and performance tracking.

This dashboard provides comprehensive views of trading performance including:
- Real-time P&L tracking
- Trade-level analytics
- Performance attribution
- Risk metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from db_utils import get_engine
from trade_pnl_tracker import TradePnLTracker
from sqlalchemy import text


def load_trade_data(engine, episode_ids=None):
    """Load trade data from database"""
    query = """
    SELECT 
        trade_id, episode_id, symbol, entry_timestamp, exit_timestamp,
        entry_price, exit_price, quantity, realized_pnl, unrealized_pnl,
        holding_period_minutes, trade_type, commission_cost, slippage_cost, net_pnl
    FROM tt_trades 
    """
    
    if episode_ids:
        placeholders = ','.join(['%s'] * len(episode_ids))
        query += f" WHERE episode_id IN ({placeholders})"
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params=episode_ids)
    else:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)


def load_portfolio_snapshots(engine, episode_ids=None):
    """Load portfolio snapshot data"""
    query = """
    SELECT 
        episode_id, timestamp, symbol, total_quantity, average_cost,
        market_value, unrealized_pnl, cost_basis
    FROM tt_portfolio_snapshots 
    """
    
    if episode_ids:
        placeholders = ','.join(['%s'] * len(episode_ids))
        query += f" WHERE episode_id IN ({placeholders})"
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params=episode_ids)
    else:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)


def load_episode_data(engine):
    """Load episode summary data"""
    query = """
    SELECT 
        episode_id, created_at, finalized_at, stop_reason, 
        max_drawdown, sharpe_ratio, turnover
    FROM tt_episodes 
    ORDER BY created_at DESC
    """
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def create_pnl_timeline(trades_df):
    """Create P&L timeline chart"""
    if trades_df.empty:
        return go.Figure()
    
    # Calculate cumulative P&L
    closed_trades = trades_df[trades_df['realized_pnl'].notna()].copy()
    if closed_trades.empty:
        return go.Figure()
    
    closed_trades = closed_trades.sort_values('exit_timestamp')
    closed_trades['cumulative_pnl'] = closed_trades['net_pnl'].cumsum()
    
    fig = go.Figure()
    
    # Add cumulative P&L line
    fig.add_trace(go.Scatter(
        x=closed_trades['exit_timestamp'],
        y=closed_trades['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='blue'),
        hovertemplate='%{x}Cumulative P&L: $%{y:,.2f}'
    ))
    
    # Add individual trade markers
    colors = ['green' if pnl > 0 else 'red' for pnl in closed_trades['net_pnl']]
    fig.add_trace(go.Scatter(
        x=closed_trades['exit_timestamp'],
        y=closed_trades['cumulative_pnl'],
        mode='markers',
        marker=dict(
            color=colors,
            size=8,
            symbol='circle'
        ),
        name='Individual Trades',
        text=closed_trades['symbol'],
        hovertemplate='%{text}Trade P&L: $%{customdata:,.2f}Cumulative: $%{y:,.2f}',
        customdata=closed_trades['net_pnl']
    ))
    
    fig.update_layout(
        title='P&L Timeline',
        xaxis_title='Date',
        yaxis_title='Cumulative P&L ($)',
        hovermode='x unified'
    )
    
    return fig


def create_trade_distribution(trades_df):
    """Create trade P&L distribution histogram"""
    if trades_df.empty or trades_df['net_pnl'].isna().all():
        return go.Figure()
    
    closed_trades = trades_df[trades_df['net_pnl'].notna()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=closed_trades['net_pnl'],
        nbinsx=30,
        name='Trade P&L Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add mean line
    mean_pnl = closed_trades['net_pnl'].mean()
    fig.add_vline(x=mean_pnl, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: ${mean_pnl:.2f}")
    
    fig.update_layout(
        title='Trade P&L Distribution',
        xaxis_title='Trade P&L ($)',
        yaxis_title='Frequency'
    )
    
    return fig


def create_symbol_performance(trades_df):
    """Create symbol performance breakdown"""
    if trades_df.empty:
        return go.Figure()
    
    closed_trades = trades_df[trades_df['net_pnl'].notna()]
    symbol_pnl = closed_trades.groupby('symbol')['net_pnl'].agg(['sum', 'count', 'mean']).reset_index()
    symbol_pnl.columns = ['symbol', 'total_pnl', 'trade_count', 'avg_pnl']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=symbol_pnl['symbol'],
        y=symbol_pnl['total_pnl'],
        name='Total P&L by Symbol',
        marker_color=['green' if x > 0 else 'red' for x in symbol_pnl['total_pnl']],
        text=[f"${x:,.0f}" for x in symbol_pnl['total_pnl']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='P&L by Symbol',
        xaxis_title='Symbol',
        yaxis_title='Total P&L ($)'
    )
    
    return fig


def create_holding_period_analysis(trades_df):
    """Create holding period vs P&L analysis"""
    if trades_df.empty:
        return go.Figure()
    
    closed_trades = trades_df[
        (trades_df['net_pnl'].notna()) & 
        (trades_df['holding_period_minutes'].notna())
    ]
    
    if closed_trades.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    colors = ['green' if pnl > 0 else 'red' for pnl in closed_trades['net_pnl']]
    
    fig.add_trace(go.Scatter(
        x=closed_trades['holding_period_minutes'],
        y=closed_trades['net_pnl'],
        mode='markers',
        marker=dict(
            color=colors,
            size=8,
            opacity=0.7
        ),
        text=closed_trades['symbol'],
        hovertemplate='%{text}Holding Period: %{x} minP&L: $%{y:,.2f}'
    ))
    
    fig.update_layout(
        title='Holding Period vs P&L',
        xaxis_title='Holding Period (minutes)',
        yaxis_title='Trade P&L ($)'
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="TensorTrade P&L Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 TensorTrade P&L Dashboard")
    st.markdown("Real-time trading performance analytics and P&L tracking")
    
    # Database connection
    db_url = st.sidebar.text_input(
        "Database URL",
        value=os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/tensortrade'),
        type="password"
    )
    
    try:
        engine = get_engine(db_url)
        st.sidebar.success("✅ Database connected")
    except Exception as e:
        st.sidebar.error(f"❌ Database connection failed: {e}")
        return
    
    # Load episode data
    episodes_df = load_episode_data(engine)
    
    if episodes_df.empty:
        st.warning("No episodes found in database. Run some training first!")
        return
    
    # Episode selection
    st.sidebar.subheader("Episode Selection")
    
    if st.sidebar.checkbox("Show all episodes"):
        selected_episodes = episodes_df['episode_id'].tolist()
    else:
        # Single episode selection
        episode_options = {
            f"Episode {row['episode_id']} ({row['created_at'].strftime('%Y-%m-%d %H:%M')})": row['episode_id']
            for _, row in episodes_df.head(20).iterrows()
        }
        
        if episode_options:
            selected_episode_key = st.sidebar.selectbox("Select Episode", list(episode_options.keys()))
            selected_episodes = [episode_options[selected_episode_key]]
        else:
            selected_episodes = []
    
    if not selected_episodes:
        st.warning("Please select at least one episode.")
        return
    
    # Load trade and portfolio data
    trades_df = load_trade_data(engine, selected_episodes)
    portfolio_df = load_portfolio_snapshots(engine, selected_episodes)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(trades_df[trades_df['net_pnl'].notna()])
        st.metric("Total Trades", total_trades)
    
    with col2:
        total_pnl = trades_df['net_pnl'].sum() if not trades_df.empty else 0
        st.metric("Total P&L", f"${total_pnl:,.2f}")
    
    with col3:
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0]) if not trades_df.empty else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        avg_pnl = trades_df['net_pnl'].mean() if not trades_df.empty else 0
        st.metric("Avg Trade P&L", f"${avg_pnl:,.2f}")
    
    # Charts
    st.subheader("📈 P&L Timeline")
    pnl_timeline = create_pnl_timeline(trades_df)
    st.plotly_chart(pnl_timeline, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Trade Distribution")
        trade_dist = create_trade_distribution(trades_df)
        st.plotly_chart(trade_dist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Performance by Symbol")
        symbol_perf = create_symbol_performance(trades_df)
        st.plotly_chart(symbol_perf, use_container_width=True)
    
    st.subheader("⏱️ Holding Period Analysis")
    holding_analysis = create_holding_period_analysis(trades_df)
    st.plotly_chart(holding_analysis, use_container_width=True)
    
    # Detailed tables
    st.subheader("📋 Detailed Trade Data")
    
    if not trades_df.empty:
        # Format the dataframe for display
        display_df = trades_df.copy()
        
        # Format timestamps
        if 'entry_timestamp' in display_df.columns:
            display_df['entry_timestamp'] = pd.to_datetime(display_df['entry_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        if 'exit_timestamp' in display_df.columns:
            display_df['exit_timestamp'] = pd.to_datetime(display_df['exit_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Format currency columns
        currency_cols = ['entry_price', 'exit_price', 'realized_pnl', 'unrealized_pnl', 'net_pnl', 'commission_cost', 'slippage_cost']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No trade data available for selected episodes.")
    
    # Portfolio snapshots
    if not portfolio_df.empty:
        st.subheader("💼 Portfolio Snapshots")
        
        # Show latest snapshots
        latest_snapshots = portfolio_df.sort_values('timestamp').groupby(['episode_id', 'symbol']).tail(1)
        
        display_portfolio = latest_snapshots.copy()
        display_portfolio['timestamp'] = pd.to_datetime(display_portfolio['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        currency_cols = ['average_cost', 'market_value', 'unrealized_pnl', 'cost_basis']
        for col in currency_cols:
            if col in display_portfolio.columns:
                display_portfolio[col] = display_portfolio[col].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_portfolio, use_container_width=True)


if __name__ == "__main__":
    main()
