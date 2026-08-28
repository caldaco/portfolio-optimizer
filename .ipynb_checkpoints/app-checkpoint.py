"""
Portfolio Optimizer - Interactive Dashboard
============================================

A real-time interactive tool to optimize multi-asset portfolios.
Enter your investment amount and desired tickers, get instant allocations.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy.optimize import minimize
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Portfolio Optimizer Dashboard")
st.markdown("---")


# ============================================================================
# HELPER FUNCTIONS (from main file)
# ============================================================================

def calculate_portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility."""
    cov_times = np.dot(weights, cov_matrix)
    variance = np.dot(weights, cov_times)
    volatility = np.sqrt(variance)
    return volatility


def min_variance_portfolio(mean_returns, cov_matrix):
    """Minimum Variance Portfolio (Markowitz)."""
    n_assets = len(mean_returns)
    
    def objective(weights):
        return calculate_portfolio_volatility(weights, cov_matrix)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate):
    """Maximum Sharpe Ratio Portfolio."""
    n_assets = len(mean_returns)
    
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        cov_times = np.dot(weights, cov_matrix)
        variance = np.dot(weights, cov_times)
        volatility = np.sqrt(variance)
        sharpe_ratio = (portfolio_return - risk_free_rate) / volatility
        return -(sharpe_ratio)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0.15, 0.85) for _ in range(n_assets))
    x0 = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def risk_parity_portfolio(mean_returns, cov_matrix):
    """Risk Parity Portfolio."""
    n_assets = len(mean_returns)
    
    def objective(weights):
        cov_times = np.dot(weights, cov_matrix)
        port_vol = np.sqrt(np.dot(weights, cov_times))
        risk_contribution = weights * cov_times / port_vol
        target_rc = np.sum(risk_contribution) / n_assets
        return np.sum((risk_contribution - target_rc) ** 2)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x


def calculate_backtest_metrics(backtest_df, risk_free_rate=0.03):
    """Calculate performance metrics from backtest."""
    returns = backtest_df['portfolio_value'].pct_change().dropna()
    
    total_return = backtest_df['portfolio_value'].iloc[-1] / backtest_df['portfolio_value'].iloc[0] - 1
    years = len(backtest_df) / 252
    cagr = (total_return + 1) ** (1/years) - 1
    
    volatility = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility
    
    cummax = backtest_df['portfolio_value'].cummax()
    drawdown = (backtest_df['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min()
    
    return {
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Total Return': total_return
    }


def backtest_strategy(weights, prices, returns, rebalance_freq='Q'):
    """Backtest a strategy with periodic rebalancing."""
    portfolio_values = [1.0]
    dates = [prices.index[0]]
    current_weights = weights.copy()
    
    if rebalance_freq == 'Q':
        rebalance_period = 63
    elif rebalance_freq == 'M':
        rebalance_period = 21
    else:
        rebalance_period = 252
    
    for i in range(1, min(len(prices), len(returns))):
        day_returns = returns.iloc[i]
        pct_change = np.sum(current_weights * day_returns)
        new_value = portfolio_values[-1] * (1 + pct_change)
        portfolio_values.append(new_value)
        dates.append(prices.index[i])
        
        if i % rebalance_period == 0:
            current_weights = weights.copy()
    
    backtest_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values
    }).set_index('date')
    
    return backtest_df


# ============================================================================
# SIDEBAR INPUTS
# ============================================================================

st.sidebar.header("📋 Portfolio Setup")

investment_amount = st.sidebar.number_input(
    "💰 Investment Amount ($)",
    min_value=100,
    value=100000,
    step=5000,
    help="Total amount you want to invest"
)

st.sidebar.markdown("---")

# Ticker input
st.sidebar.subheader("📊 Select Tickers")

# Predefined portfolios for quick selection
portfolio_presets = {
    "Custom": [],
    "Conservative (Bonds & Stable)": ['BND', 'VGIT', 'SHV'],
    "Balanced": ['SPY', 'BND', 'VEA', 'GLD'],
    "Aggressive Growth": ['QQQ', 'VUG', 'VGRO'],
    "Global Diversified": ['SPY', 'EFA', 'BND', 'GLD', 'DBC', 'VNQ'],
}

preset_choice = st.sidebar.selectbox(
    "Quick Preset",
    options=list(portfolio_presets.keys()),
    help="Select a preset or choose Custom to add your own"
)

if preset_choice == "Custom":
    tickers_input = st.sidebar.text_input(
        "Enter Tickers (comma-separated)",
        value="SPY,BND,GLD",
        help="e.g., SPY,BND,GLD,QQQ"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
else:
    tickers = portfolio_presets[preset_choice]
    st.sidebar.info(f"Using preset: {', '.join(tickers)}")

# Validate tickers
if not tickers or tickers == ['']:
    st.sidebar.error("❌ Please enter at least one ticker")
    st.stop()

st.sidebar.markdown("---")

# Risk-free rate and lookback
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.1,
    help="Current risk-free rate for Sharpe calculations"
) / 100

lookback_years = st.sidebar.slider(
    "Historical Data Lookback (years)",
    min_value=1,
    max_value=10,
    value=5,
    help="Years of historical data to analyze"
)

rebalance_freq = st.sidebar.selectbox(
    "Rebalancing Frequency",
    options=['Q', 'M', 'A'],
    format_func=lambda x: {'Q': 'Quarterly', 'M': 'Monthly', 'A': 'Annual'}[x]
)

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

st.sidebar.markdown("---")

with st.spinner("📥 Loading market data..."):
    try:
        end_date = date.today().strftime('%Y-%m-%d')
        start_date = (date.today() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        
        prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
        
        if len(prices.columns) == 1:
            prices.columns = tickers
            
        prices = prices[tickers]
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        std_devs = returns.std() * np.sqrt(252)
        
        st.sidebar.success(f"✓ Data loaded: {len(prices)} trading days")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        st.stop()


# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Optimization", "💹 Risk Analysis", "📈 Backtest", "📊 Analytics", "📊 DCA Tracker", "🔍 Overfitting"])


# ============================================================================
# TAB 1: OPTIMIZATION
# ============================================================================

with tab1:
    st.header("Portfolio Optimization Results")
    st.markdown("Optimal allocations for your investment amount")
    
    # Optimization choice
    col1, col2 = st.columns([1, 2])
    with col1:
        strategy = st.radio(
            "Select Strategy",
            options=["Max Sharpe", "Min Variance", "Risk Parity", "Equal Weight"],
            help="Choose optimization strategy"
        )
    
    # Run optimization
    if strategy == "Max Sharpe":
        weights = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
        strategy_name = "Maximum Sharpe Ratio"
    elif strategy == "Min Variance":
        weights = min_variance_portfolio(mean_returns, cov_matrix)
        strategy_name = "Minimum Variance"
    elif strategy == "Risk Parity":
        weights = risk_parity_portfolio(mean_returns, cov_matrix)
        strategy_name = "Risk Parity"
    else:
        n_assets = len(tickers)
        weights = np.array([1/n_assets] * n_assets)
        strategy_name = "Equal Weight"
    
    # Calculate metrics
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Display metrics
    st.subheader(f"📊 {strategy_name} Portfolio")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Expected Return", f"{portfolio_return:.2%}")
    with metric_cols[1]:
        st.metric("Volatility", f"{portfolio_volatility:.2%}")
    with metric_cols[2]:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    with metric_cols[3]:
        st.metric("Investment Amount", f"${investment_amount:,.0f}")
    
    st.markdown("---")
    
    # Allocation table
    st.subheader("💰 Dollar Allocation")
    
    allocation_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight (%)': weights * 100,
        'Dollar Amount': weights * investment_amount,
    })
    allocation_df = allocation_df.sort_values('Dollar Amount', ascending=False).reset_index(drop=True)
    
    # Display with formatting
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            allocation_df.style.format({
                'Weight (%)': '{:.2f}',
                'Dollar Amount': '${:,.2f}'
            }),
            use_container_width=True
        )
    
    # Pie chart
    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=allocation_df['Ticker'],
            values=allocation_df['Dollar Amount'],
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        fig.update_layout(
            height=400,
            showlegend=True,
            title_text="Portfolio Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Asset statistics
    st.subheader("📈 Individual Asset Statistics")
    
    asset_stats = pd.DataFrame({
        'Ticker': tickers,
        'Annual Return': mean_returns.values * 100,
        'Annual Volatility': std_devs.values * 100,
        'Allocation %': weights * 100,
    }).sort_values('Allocation %', ascending=False)
    
    st.dataframe(
        asset_stats.style.format({
            'Annual Return': '{:.2f}%',
            'Annual Volatility': '{:.2f}%',
            'Allocation %': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    # Download allocation
    csv = allocation_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Allocation (CSV)",
        data=csv,
        file_name=f"portfolio_allocation_{strategy.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )


# ============================================================================
# TAB 2: RISK ANALYSIS
# ============================================================================

with tab2:
    st.header("Risk Analysis")
    st.markdown("Comprehensive risk metrics for your portfolio")
    
    # Backtest to get returns
    prices_subset = prices[tickers] if len(tickers) > 1 else prices
    returns_subset = returns[tickers] if len(tickers) > 1 else returns
    
    backtest_df = backtest_strategy(weights, prices_subset, returns_subset, rebalance_freq)
    backtest_returns = backtest_df['portfolio_value'].pct_change().dropna()
    
    # VaR calculations
    st.subheader("Value-at-Risk (VaR) Analysis")
    
    hist_var_95 = backtest_returns.quantile(0.05) * 100
    hist_var_99 = backtest_returns.quantile(0.01) * 100
    
    mean_ret = backtest_returns.mean()
    std_ret = backtest_returns.std()
    
    param_var_95 = (mean_ret + std_ret * stats.norm.ppf(0.05)) * 100
    param_var_99 = (mean_ret + std_ret * stats.norm.ppf(0.01)) * 100
    
    # CVaR calculations
    cvar_95 = backtest_returns[backtest_returns <= backtest_returns.quantile(0.05)].mean() * 100
    cvar_99 = backtest_returns[backtest_returns <= backtest_returns.quantile(0.01)].mean() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("VaR (95% confidence)", f"{hist_var_95:.2f}%", 
                 help="95% of days, losses won't exceed this")
        st.metric("VaR (99% confidence)", f"{hist_var_99:.2f}%",
                 help="99% of days, losses won't exceed this")
    
    with col2:
        st.metric("CVaR (95% confidence)", f"{cvar_95:.2f}%",
                 help="Average loss on worst 5% of days")
        st.metric("CVaR (99% confidence)", f"{cvar_99:.2f}%",
                 help="Average loss on worst 1% of days")
    
    st.markdown("---")
    
    # Risk comparison table
    st.subheader("📊 VaR Methods Comparison")
    
    var_comparison = pd.DataFrame({
        'Confidence Level': ['95%', '95%', '99%', '99%'],
        'Method': ['Historical', 'Parametric', 'Historical', 'Parametric'],
        'VaR': [hist_var_95, param_var_95, hist_var_99, param_var_99]
    })
    
    st.dataframe(
        var_comparison.style.format({'VaR': '{:.2f}%'}),
        use_container_width=True
    )
    
    st.info(
        "**Historical VaR** uses real past data (captures tail risk). "
        "**Parametric VaR** assumes normal distribution (faster, may underestimate extreme risk)."
    )
    
    st.markdown("---")
    
    # Stress testing
    st.subheader("🚨 Stress Test Scenarios")
    
    scenarios = {
        '2008 Financial Crisis': -0.0916,
        'COVID Crash (March 2020)': -0.1203,
        'Black Monday (1987)': -0.2205,
        'Moderate Correction': -0.05,
    }
    
    stress_results = []
    for scenario_name, shock in scenarios.items():
        loss_dollar = investment_amount * shock
        stress_results.append({
            'Scenario': scenario_name,
            'Daily Loss %': shock * 100,
            'Dollar Loss': loss_dollar
        })
    
    stress_df = pd.DataFrame(stress_results)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            stress_df.style.format({
                'Daily Loss %': '{:.2f}%',
                'Dollar Loss': '${:,.2f}'
            }),
            use_container_width=True
        )
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(
                x=stress_df['Scenario'],
                y=stress_df['Daily Loss %'],
                marker_color=['darkred' if x < -10 else 'red' if x < -5 else 'orange' 
                             for x in stress_df['Daily Loss %']],
                text=stress_df['Dollar Loss'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Loss: %{y:.2f}%<extra></extra>'
            )
        ])
        fig.update_layout(
            title='Stress Test: Portfolio Loss Scenarios',
            xaxis_title='Scenario',
            yaxis_title='Daily Loss %',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 3: BACKTEST
# ============================================================================

with tab3:
    st.header("📈 Historical Backtest")
    rebalance_freq_name = {'Q': 'quarterly', 'M': 'monthly', 'A': 'annual'}[rebalance_freq]
    st.markdown(f"Performance simulation over {lookback_years} years with {rebalance_freq_name} rebalancing")
    
    # Calculate backtest metrics
    metrics = calculate_backtest_metrics(backtest_df, risk_free_rate)
    
    # Display metrics
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("CAGR", f"{metrics['CAGR']:.2%}")
    with metric_cols[1]:
        st.metric("Volatility", f"{metrics['Volatility']:.2%}")
    with metric_cols[2]:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe']:.3f}")
    with metric_cols[3]:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
    with metric_cols[4]:
        st.metric("Total Return", f"{metrics['Total Return']:.2%}")
    
    st.markdown("---")
    
    # Cumulative returns chart
    st.subheader("Cumulative Returns Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_df.index,
        y=backtest_df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Portfolio Growth: ${backtest_df["portfolio_value"].iloc[0]:,.0f} → ${backtest_df["portfolio_value"].iloc[-1]:,.0f}',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Drawdown chart
    st.subheader("Drawdown Analysis")
    
    cummax = backtest_df['portfolio_value'].cummax()
    drawdown = (backtest_df['portfolio_value'] - cummax) / cummax * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)',
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Drawdown</b>: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=350,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly returns heatmap
    st.markdown("---")
    st.subheader("Monthly Returns Heatmap")
    
    monthly_returns = backtest_df['portfolio_value'].pct_change().resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_returns_pivot = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot_table = monthly_returns_pivot.pivot(index='Year', columns='Month', values='Return')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:pivot_table.shape[1]],
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot_table.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Return %"),
        hovertemplate='<b>%{y}</b> - %{x}<br>Return: %{text:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Returns (%)',
        height=300,
        xaxis_title='Month',
        yaxis_title='Year'
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================

with tab4:
    st.header("📊 Correlation & Analytics")
    
    # Correlation matrix
    st.subheader("Asset Correlation Matrix")
    
    corr_matrix = returns[tickers].corr() if len(tickers) > 1 else pd.DataFrame([[1.0]], index=tickers, columns=tickers)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{text:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=400,
        title='Asset Correlation'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Return distribution
    st.subheader("Return Distributions")
    
    fig = go.Figure()
    
    for ticker in tickers[:6]:  # Limit to 6 for clarity
        ticker_returns = returns[ticker] * 100
        fig.add_trace(go.Histogram(
            x=ticker_returns,
            name=ticker,
            opacity=0.7,
            nbinsx=50,
            hovertemplate='<b>%{fullData.name}</b><br>Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Daily Return Distributions',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        height=400,
        barmode='overlay',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    summary_stats = pd.DataFrame({
        'Ticker': tickers,
        'Mean Daily Return': (returns[tickers].mean() * 100).values,
        'Std Dev': (returns[tickers].std() * 100).values,
        'Skewness': [returns[t].skew() for t in tickers],
        'Kurtosis': [returns[t].kurtosis() for t in tickers],
    })
    
    st.dataframe(
        summary_stats.style.format({
            'Mean Daily Return': '{:.3f}%',
            'Std Dev': '{:.2f}%',
            'Skewness': '{:.3f}',
            'Kurtosis': '{:.3f}'
        }),
        use_container_width=True
    )
    
    st.info(
        "**Skewness**: Negative = left tail risk (bigger downside). "
        "**Kurtosis**: Higher = fatter tails (more extreme events)."
    )



# ============================================================================
# FLEXIBLE DCA TRACKER - SUPPORTS ANY TICKERS
# ============================================================================

import json
import os
from datetime import datetime

DCA_FILE = "dca_log.json"
TICKERS_CONFIG_FILE = "dca_tickers.json"

def load_dca_data():
    """Load DCA history from file."""
    if os.path.exists(DCA_FILE):
        try:
            with open(DCA_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_dca_data(data):
    """Save DCA history to file."""
    with open(DCA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_dca_tickers():
    """Load selected tickers from config."""
    if os.path.exists(TICKERS_CONFIG_FILE):
        try:
            with open(TICKERS_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('tickers', ['SPY', 'BND', 'GLD'])
        except:
            return ['SPY', 'BND', 'GLD']
    return ['SPY', 'BND', 'GLD']

def save_dca_tickers(tickers):
    """Save selected tickers to config."""
    with open(TICKERS_CONFIG_FILE, 'w') as f:
        json.dump({'tickers': tickers}, f, indent=2)

def add_dca_entry(ticker_amounts, note=""):
    """Add a new DCA entry with flexible tickers."""
    dca_data = load_dca_data()
    total = sum(ticker_amounts.values())
    
    entry = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'amounts': ticker_amounts,
        'total': float(total),
        'note': note
    }
    dca_data.append(entry)
    save_dca_data(dca_data)
    return entry

def update_dca_entry(index, ticker_amounts, note=""):
    """Update an existing DCA entry."""
    dca_data = load_dca_data()
    if 0 <= index < len(dca_data):
        total = sum(ticker_amounts.values())
        dca_data[index] = {
            'date': dca_data[index]['date'],
            'amounts': ticker_amounts,
            'total': float(total),
            'note': note
        }
        save_dca_data(dca_data)
        return True
    return False

def delete_dca_entry(index):
    """Delete a DCA entry."""
    dca_data = load_dca_data()
    if 0 <= index < len(dca_data):
        dca_data.pop(index)
        save_dca_data(dca_data)
        return True
    return False

# ============================================================================
# TAB 5: FLEXIBLE DCA TRACKER
# ============================================================================

with tab5:
    st.header("📊 Dollar-Cost Averaging Tracker")
    st.markdown("Track your investments with flexible tickers")
    
    # ===== Ticker Configuration =====
    with st.expander("⚙️ Configure Tickers", expanded=False):
        st.markdown("Set which tickers to track in your DCA")
        
        current_tickers = load_dca_tickers()
        
        tickers_input = st.text_input(
            "Enter Tickers (comma-separated)",
            value=",".join(current_tickers),
            help="e.g., SPY,BND,GLD or QQQ,VTI,AGG",
            placeholder="SPY,BND,GLD"
        )
        
        if st.button("💾 Save Ticker Configuration"):
            new_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            if len(new_tickers) > 0:
                save_dca_tickers(new_tickers)
                st.success(f"✅ Tickers updated to: {', '.join(new_tickers)}")
                st.rerun()
            else:
                st.error("❌ Please enter at least one ticker")
    
    st.markdown("---")
    
    # ===== Get current tickers =====
    tracked_tickers = load_dca_tickers()
    
    # ===== Input Section =====
    st.subheader("💰 Log New Investment")
    
    st.markdown(f"**Tracking:** {', '.join(tracked_tickers)}")
    
    # Create input columns for each ticker
    ticker_cols = st.columns(len(tracked_tickers))
    ticker_amounts = {}
    
    for i, ticker in enumerate(tracked_tickers):
        with ticker_cols[i]:
            amount = st.number_input(
                f"{ticker} Amount ($)",
                min_value=0.0,
                value=100.0 / len(tracked_tickers),  # Default: equal split
                step=5.0,
                key=f"input_{ticker}"
            )
            ticker_amounts[ticker] = amount
    
    total_amount = sum(ticker_amounts.values())
    
    # Show allocation breakdown
    if total_amount > 0:
        pct_cols = st.columns(len(tracked_tickers))
        for i, ticker in enumerate(tracked_tickers):
            with pct_cols[i]:
                pct = (ticker_amounts[ticker] / total_amount) * 100
                st.metric(f"{ticker} %", f"{pct:.1f}%")
    
    st.metric("Total Investment", f"${total_amount:.2f}")
    
    note = st.text_input("Note (optional)", placeholder="e.g., 'Good tips' or 'Weekly savings'")
    
    if st.button("✅ Log Investment", key="log_investment"):
        if total_amount > 0:
            entry = add_dca_entry(ticker_amounts, note)
            st.success(f"✅ Logged ${total_amount:.2f} investment!")
            st.balloons()
            st.rerun()
        else:
            st.error("❌ Investment amount must be greater than $0")
    
    st.markdown("---")
    
    # ===== Load and Display Data =====
    dca_data = load_dca_data()
    
    if len(dca_data) == 0:
        st.info("📝 No investments logged yet. Log your first investment above!")
    else:
        # ===== Summary Metrics =====
        total_invested = sum([entry['total'] for entry in dca_data])
        num_weeks = len(dca_data)
        avg_weekly = total_invested / num_weeks
        
        # Calculate totals per ticker
        ticker_totals = {ticker: 0 for ticker in tracked_tickers}
        for entry in dca_data:
            for ticker, amount in entry['amounts'].items():
                if ticker in ticker_totals:
                    ticker_totals[ticker] += amount
        
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Total Invested", f"${total_invested:.2f}")
        with summary_cols[1]:
            st.metric("Number of Weeks", num_weeks)
        with summary_cols[2]:
            st.metric("Average/Week", f"${avg_weekly:.2f}")
        with summary_cols[3]:
            st.metric("Target 1-Year", f"${avg_weekly * 52:.2f}")
        
        st.markdown("---")
        
        # ===== Investment History Table with Edit/Delete =====
        st.subheader("📋 Investment History")
        
        # Build history table
        history_records = []
        for entry in dca_data:
            record = {'Date': entry['date']}
            for ticker in tracked_tickers:
                amount = entry['amounts'].get(ticker, 0)
                record[ticker] = f"${amount:.2f}"
            record['Total'] = f"${entry['total']:.2f}"
            record['Note'] = entry['note']
            history_records.append(record)
        
        history_df = pd.DataFrame(history_records)
        st.dataframe(history_df, use_container_width=True)
        
        st.markdown("---")
        
        # ===== Edit/Delete Section =====
        st.subheader("✏️ Edit or Delete Entries")
        
        entry_to_edit = st.selectbox(
            "Select entry to edit",
            options=range(len(dca_data)),
            format_func=lambda i: f"{dca_data[i]['date']} - ${dca_data[i]['total']:.2f}"
        )
        
        if entry_to_edit is not None:
            selected_entry = dca_data[entry_to_edit]
            
            st.markdown(f"**Editing:** {selected_entry['date']}")
            
            # Create edit columns for each ticker
            edit_cols = st.columns(len(tracked_tickers))
            edit_amounts = {}
            
            for i, ticker in enumerate(tracked_tickers):
                with edit_cols[i]:
                    current_amount = selected_entry['amounts'].get(ticker, 0)
                    amount = st.number_input(
                        f"Edit {ticker} ($)",
                        min_value=0.0,
                        value=current_amount,
                        step=5.0,
                        key=f"edit_{ticker}_{entry_to_edit}"
                    )
                    edit_amounts[ticker] = amount
            
            edit_note = st.text_input(
                "Edit Note",
                value=selected_entry['note'],
                key=f"edit_note_{entry_to_edit}"
            )
            
            edit_total = sum(edit_amounts.values())
            st.metric("New Total", f"${edit_total:.2f}")
            
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                if st.button("💾 Save Changes", key=f"save_{entry_to_edit}"):
                    if edit_total > 0:
                        update_dca_entry(entry_to_edit, edit_amounts, edit_note)
                        st.success("✅ Entry updated!")
                        st.rerun()
                    else:
                        st.error("❌ Total must be greater than $0")
            
            with button_col2:
                if st.button("🗑️ Delete Entry", key=f"delete_{entry_to_edit}"):
                    delete_dca_entry(entry_to_edit)
                    st.success("✅ Entry deleted!")
                    st.rerun()
            
            with button_col3:
                if st.button("❌ Cancel", key=f"cancel_{entry_to_edit}"):
                    st.info("Edit cancelled")
        
        st.markdown("---")
        
        # ===== Cumulative Investment Chart =====
        st.subheader("📈 Cumulative Investments by Asset")
        
        cumulative_data = []
        cumulative_amounts = {ticker: 0 for ticker in tracked_tickers}
        
        for entry in dca_data:
            for ticker in tracked_tickers:
                cumulative_amounts[ticker] += entry['amounts'].get(ticker, 0)
            
            row = {'Date': entry['date']}
            row.update(cumulative_amounts.copy())
            cumulative_data.append(row)
        
        cumulative_df = pd.DataFrame(cumulative_data)
        
        # Create stacked area chart
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b']
        
        for i, ticker in enumerate(tracked_tickers):
            fig.add_trace(go.Scatter(
                x=cumulative_df['Date'],
                y=cumulative_df[ticker],
                mode='lines',
                name=ticker,
                line=dict(color=colors[i % len(colors)], width=2),
                stackgroup='one',
                hovertemplate=f'<b>{{%x}}</b><br>{ticker}: ${{%y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Cumulative Investment by Asset',
            xaxis_title='Date',
            yaxis_title='Cumulative Amount ($)',
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== Asset Allocation Pie Chart =====
        st.subheader("💹 Total Asset Allocation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=list(ticker_totals.keys()),
                values=list(ticker_totals.values()),
                textposition='inside',
                textinfo='label+percent',
                marker=dict(colors=colors[:len(tracked_tickers)]),
                hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<extra></extra>'
            )])
            fig.update_layout(height=400, title='Investment Allocation')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            allocation_records = []
            for ticker in tracked_tickers:
                amount = ticker_totals[ticker]
                pct = (amount / total_invested) * 100 if total_invested > 0 else 0
                allocation_records.append({
                    'Ticker': ticker,
                    'Total': f'${amount:.2f}',
                    '%': f'{pct:.1f}%'
                })
            
            allocation_table = pd.DataFrame(allocation_records)
            st.dataframe(allocation_table, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ===== Weekly Amount Chart =====
        st.subheader("📊 Weekly Investment by Asset")
        
        weekly_records = []
        for entry in dca_data:
            record = {'Date': entry['date']}
            for ticker in tracked_tickers:
                record[ticker] = entry['amounts'].get(ticker, 0)
            weekly_records.append(record)
        
        weekly_df = pd.DataFrame(weekly_records)
        
        fig = go.Figure()
        
        for i, ticker in enumerate(tracked_tickers):
            fig.add_trace(go.Bar(
                x=weekly_df['Date'],
                y=weekly_df[ticker],
                name=ticker,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Weekly Investment by Asset',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            height=350,
            barmode='stack',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== Future Projections =====
        st.subheader("🔮 12-Month Projection")
        
        col1, col2, col3 = st.columns(3)
        
        projected_52_weeks = avg_weekly * 52
        growth_5pct = projected_52_weeks * 1.05
        growth_10pct = projected_52_weeks * 1.10
        
        with col1:
            st.metric(
                "Projected 1-Year Investment",
                f"${projected_52_weeks:.2f}",
                f"+${projected_52_weeks - total_invested:.2f}"
            )
        
        with col2:
            st.metric(
                "With 5% Growth",
                f"${growth_5pct:.2f}",
                f"+${growth_5pct - projected_52_weeks:.2f}"
            )
        
        with col3:
            st.metric(
                "With 10% Growth",
                f"${growth_10pct:.2f}",
                f"+${growth_10pct - projected_52_weeks:.2f}"
            )
        
        st.info(
            f"💡 At your current pace of **${avg_weekly:.2f}/week**, "
            f"you'll invest **${projected_52_weeks:.2f}** in the next 12 months. "
            f"With historical market returns of 7-10%, your portfolio could be worth "
            f"**${growth_10pct:.2f}** by next year!"
        )



# ============================================================================
# TAB 6: OVERFITTING DETECTION
# ============================================================================
# Add this to your app.py
# Change: tab1, tab2, tab3, tab4, tab5 = st.tabs(...)
# To:     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(...)
# And add this code after tab5 section
# ============================================================================

with tab6:
    st.header("🔍 Overfitting Detection")
    st.markdown("Test if your optimizer generalizes or just fits the past")
    
    # ===== Test 1: Out-of-Sample =====
    st.subheader("Test 1: Out-of-Sample Testing")
    st.markdown("Optimize on first half of data, test on second half (data optimizer never saw)")
    
    # Split data
    split_point = len(returns) // 2
    
    train_returns = returns.iloc[:split_point]
    test_returns = returns.iloc[split_point:]
    
    # Optimize on TRAIN data only
    train_mean = train_returns.mean() * 252
    train_cov = train_returns.cov() * 252
    train_weights = max_sharpe_portfolio(train_mean, train_cov, risk_free_rate)
    
    # Calculate Sharpe on TRAIN
    train_sharpe = (np.dot(train_weights, train_mean) - risk_free_rate) / np.sqrt(np.dot(train_weights, np.dot(train_cov, train_weights)))
    
    # NOW test on TEST data (optimizer never saw this)
    test_mean = test_returns.mean() * 252
    test_cov = test_returns.cov() * 252
    test_sharpe = (np.dot(train_weights, test_mean) - risk_free_rate) / np.sqrt(np.dot(train_weights, np.dot(test_cov, train_weights)))
    
    degradation = train_sharpe - test_sharpe
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Train Sharpe (Optimized On)", f"{train_sharpe:.3f}")
    with col2:
        st.metric("Test Sharpe (Never Saw)", f"{test_sharpe:.3f}")
    with col3:
        st.metric("Degradation", f"{degradation:.3f}")
    
    # Interpretation
    if degradation < 0.1:
        st.success("✅ **Excellent!** Minimal overfitting. Your model generalizes well!")
    elif degradation < 0.3:
        st.warning("⚠️ **Mild overfitting** detected. Model is okay but could be better.")
    else:
        st.error("❌ **Severe overfitting!** Your weights may not work in the future.")
    
    st.info(
        "💡 **What this means:** If you optimized on 2016-2021 data and test on 2021-2026 data, "
        "do the weights still work? Small drop = generalizes. Large drop = overfitting."
    )
    
    st.markdown("---")
    
    # ===== Test 2: Stability Testing =====
    st.subheader("Test 2: Stability Testing")
    st.markdown("Remove one month at a time. Do weights change drastically?")
    
    with st.spinner("Testing stability across different time periods..."):
        perturbation_results = []
        
        # Try removing different months
        for i in range(len(returns) - 21, 0, -21):  # 21 trading days = ~1 month
            perturbed_returns = pd.concat([returns.iloc[:i], returns.iloc[i+21:]])
            
            if len(perturbed_returns) > 50:  # Need enough data
                perturbed_mean = perturbed_returns.mean() * 252
                perturbed_cov = perturbed_returns.cov() * 252
                perturbed_weights = max_sharpe_portfolio(perturbed_mean, perturbed_cov, risk_free_rate)
                
                weight_change = np.sum(np.abs(weights - perturbed_weights))
                perturbation_results.append(weight_change)
        
        avg_stability = np.mean(perturbation_results)
        max_stability = np.max(perturbation_results)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Weight Change", f"{avg_stability:.4f}")
    with col2:
        st.metric("Max Weight Change", f"{max_stability:.4f}")
    with col3:
        st.metric("# Periods Tested", len(perturbation_results))
    
    # Interpretation
    if avg_stability < 0.05:
        st.success("✅ **Stable!** Weights don't change much when removing time periods.")
    elif avg_stability < 0.15:
        st.warning("⚠️ **Somewhat unstable.** Weights shift when data changes.")
    else:
        st.error("❌ **Very unstable!** Weights change drastically. Overfitting detected.")
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=perturbation_results,
        mode='lines+markers',
        name='Weight Change',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        hovertemplate='Period %{x}<br>Weight Change: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="Good", annotation_position="right")
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange", annotation_text="Warning", annotation_position="right")
    
    fig.update_layout(
        title='Weight Stability Across Time Periods',
        xaxis_title='Time Period',
        yaxis_title='Weight Change (L1 Norm)',
        height=350,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ===== Test 3: Compare to Benchmarks =====
    st.subheader("Test 3: Comparison to Simple Benchmarks")
    st.markdown("Does your optimizer beat simple strategies? If barely, you're probably overfitting.")
    
    # Your optimizer backtest
    backtest_df = backtest_strategy(weights, prices[tickers] if len(tickers) > 1 else prices, returns[tickers] if len(tickers) > 1 else returns, rebalance_freq)
    optimized_metrics = calculate_backtest_metrics(backtest_df, risk_free_rate)
    
    # Benchmark 1: Equal Weight
    equal_weights = np.array([1/len(tickers)] * len(tickers))
    equal_backtest = backtest_strategy(equal_weights, prices[tickers] if len(tickers) > 1 else prices, returns[tickers] if len(tickers) > 1 else returns, rebalance_freq)
    equal_metrics = calculate_backtest_metrics(equal_backtest, risk_free_rate)
    
    # Benchmark 2: Min Variance
    min_var_weights = min_variance_portfolio(mean_returns, cov_matrix)
    min_var_backtest = backtest_strategy(min_var_weights, prices[tickers] if len(tickers) > 1 else prices, returns[tickers] if len(tickers) > 1 else returns, rebalance_freq)
    min_var_metrics = calculate_backtest_metrics(min_var_backtest, risk_free_rate)
    
    # Compare
    comparison_df = pd.DataFrame({
        'Strategy': ['Your Optimizer', 'Equal Weight', 'Min Variance'],
        'Sharpe': [optimized_metrics['Sharpe'], equal_metrics['Sharpe'], min_var_metrics['Sharpe']],
        'CAGR': [optimized_metrics['CAGR'], equal_metrics['CAGR'], min_var_metrics['CAGR']],
        'Volatility': [optimized_metrics['Volatility'], equal_metrics['Volatility'], min_var_metrics['Volatility']],
        'Max Drawdown': [optimized_metrics['Max Drawdown'], equal_metrics['Max Drawdown'], min_var_metrics['Max Drawdown']]
    })
    
    st.dataframe(
        comparison_df.style.format({
            'Sharpe': '{:.3f}',
            'CAGR': '{:.2%}',
            'Volatility': '{:.2%}',
            'Max Drawdown': '{:.2%}'
        }),
        use_container_width=True
    )
    
    # Interpretation
    optimizer_advantage = optimized_metrics['Sharpe'] - equal_metrics['Sharpe']
    
    if optimizer_advantage > 0.3:
        st.success("✅ **Strong improvement!** Your optimizer clearly beats simple strategies.")
    elif optimizer_advantage > 0.1:
        st.warning("⚠️ **Marginal improvement.** Could be overfitting rather than true skill.")
    else:
        st.error("❌ **No real improvement.** Optimizer probably overfits. Use simple strategy instead.")
    
    # Chart
    fig = go.Figure(data=[
        go.Bar(name='Sharpe', x=comparison_df['Strategy'], y=comparison_df['Sharpe']),
        go.Bar(name='CAGR', x=comparison_df['Strategy'], y=comparison_df['CAGR']),
        go.Bar(name='Volatility', x=comparison_df['Strategy'], y=comparison_df['Volatility'])
    ])
    
    fig.update_layout(
        title='Strategy Comparison',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ===== Test 4: Forward Testing =====
    st.subheader("Test 4: Forward Testing (Most Recent Period)")
    st.markdown("Train on past 4 years, test on most recent year (like predicting the future)")
    
    with st.spinner("Running forward test..."):
        # Split: past 4 years for training, most recent 1 year for testing
        forward_split = len(returns) - 252  # 1 year = 252 trading days
        
        forward_train = returns.iloc[:forward_split]
        forward_test = returns.iloc[forward_split:]
        
        # Optimize on training period
        forward_train_mean = forward_train.mean() * 252
        forward_train_cov = forward_train.cov() * 252
        forward_weights = max_sharpe_portfolio(forward_train_mean, forward_train_cov, risk_free_rate)
        
        # Calculate Sharpe on both
        forward_train_sharpe = (np.dot(forward_weights, forward_train_mean) - risk_free_rate) / np.sqrt(np.dot(forward_weights, np.dot(forward_train_cov, forward_weights)))
        
        forward_test_mean = forward_test.mean() * 252
        forward_test_cov = forward_test.cov() * 252
        forward_test_sharpe = (np.dot(forward_weights, forward_test_mean) - risk_free_rate) / np.sqrt(np.dot(forward_weights, np.dot(forward_test_cov, forward_weights)))
        
        forward_degradation = forward_train_sharpe - forward_test_sharpe
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sharpe (Past 4 Years)", f"{forward_train_sharpe:.3f}")
    with col2:
        st.metric("Sharpe (Most Recent Year)", f"{forward_test_sharpe:.3f}")
    with col3:
        st.metric("Degradation", f"{forward_degradation:.3f}")
    
    if forward_degradation < 0.2:
        st.success("✅ **Good forward performance!** Recent period shows continued strength.")
    elif forward_degradation < 0.4:
        st.warning("⚠️ **Weaker recently.** Strategy may be adapting to new market conditions.")
    else:
        st.error("❌ **Much worse recently.** Weights optimized for past conditions, not current.")
    
    st.markdown("---")
    
    # ===== Overall Assessment =====
    st.subheader("📊 Overall Overfitting Assessment")
    
    # Score each test
    scores = []
    
    # Test 1 score
    if degradation < 0.1:
        scores.append(3)
    elif degradation < 0.3:
        scores.append(2)
    else:
        scores.append(1)
    
    # Test 2 score
    if avg_stability < 0.05:
        scores.append(3)
    elif avg_stability < 0.15:
        scores.append(2)
    else:
        scores.append(1)
    
    # Test 3 score
    if optimizer_advantage > 0.3:
        scores.append(3)
    elif optimizer_advantage > 0.1:
        scores.append(2)
    else:
        scores.append(1)
    
    # Test 4 score
    if forward_degradation < 0.2:
        scores.append(3)
    elif forward_degradation < 0.4:
        scores.append(2)
    else:
        scores.append(1)
    
    overall_score = np.mean(scores)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if overall_score > 2.5:
            st.success(f"**Score: {overall_score:.1f}/3.0**\n\n✅ **LOW OVERFITTING**\n\nYour model generalizes well!")
        elif overall_score > 1.5:
            st.warning(f"**Score: {overall_score:.1f}/3.0**\n\n⚠️ **MODERATE OVERFITTING**\n\nUse with caution.")
        else:
            st.error(f"**Score: {overall_score:.1f}/3.0**\n\n❌ **HIGH OVERFITTING**\n\nDon't use this model!")
    
    with col2:
        st.write("")  # Spacing
    
    with col3:
        st.write("")  # Spacing
    
    st.markdown("---")
    
    # ===== Recommendations =====
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if degradation > 0.3:
        recommendations.append("🔴 **Use longer lookback period** (try 10+ years instead of 5)")
    
    if avg_stability > 0.1:
        recommendations.append("🔴 **Add stability constraints** (prevent extreme allocations)")
    
    if optimizer_advantage < 0.15:
        recommendations.append("🔴 **Consider simple 60/30/10** instead of optimizer")
    
    if forward_degradation > 0.3:
        recommendations.append("🔴 **Optimizer not working recently** — market conditions changed")
    
    if not recommendations:
        recommendations.append("✅ Your optimizer looks solid! No major overfitting detected.")
    
    for rec in recommendations:
        st.write(rec)
# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <small>💡 Pro tip: Use this dashboard to explore different allocations and understand your portfolio's risk/return tradeoff.</small>
    </div>
    """,
    unsafe_allow_html=True
)
