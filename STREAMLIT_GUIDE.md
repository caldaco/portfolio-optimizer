# 🎯 Portfolio Optimizer Dashboard - Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_dashboard.txt
```

Or individually:
```bash
pip install streamlit plotly yfinance pandas numpy scipy matplotlib
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

This will:
- Open your browser automatically to `http://localhost:8501`
- Display a live, interactive dashboard
- Update in real-time as you change parameters

---

## 🎨 Features

### Left Sidebar - Controls
- **💰 Investment Amount**: Enter how much money you want to invest
- **📊 Select Tickers**: Choose from presets or enter custom tickers
- **Risk-Free Rate**: Adjust for Sharpe calculations
- **Historical Lookback**: Change years of data (1-10 years)
- **Rebalancing Frequency**: Quarterly, Monthly, or Annual

### Tab 1: 🎯 Optimization
**INPUT YOUR MONEY & GET ALLOCATIONS**

- Select optimization strategy:
  - **Max Sharpe** (best risk-adjusted returns) ⭐ Recommended
  - **Min Variance** (lowest risk)
  - **Risk Parity** (equal risk contribution)
  - **Equal Weight** (baseline)

**OUTPUT:**
- Expected return & volatility for your strategy
- **Dollar allocation** for each ticker
- **Percentage allocation** pie chart
- Individual asset statistics

**Download:** Export allocation to CSV

---

### Tab 2: 💹 Risk Analysis
**UNDERSTAND YOUR PORTFOLIO'S RISK**

**Value-at-Risk (VaR):**
- Daily loss thresholds at 95% and 99% confidence
- Historical vs Parametric comparison
- What it means: "95% of days, I won't lose more than X%"

**Conditional VaR (CVaR):**
- Average loss on worst days
- More conservative than VaR

**Stress Testing:**
- Portfolio loss in 2008 financial crisis
- COVID crash scenario
- Black Monday 1987
- Shows dollar losses too

---

### Tab 3: 📈 Backtest
**SEE HISTORICAL PERFORMANCE**

**Charts:**
1. **Cumulative Returns**: How $100k would have grown over 5 years
2. **Drawdown Analysis**: Worst losses over time
3. **Monthly Returns Heatmap**: See which months were winners/losers

**Metrics:**
- CAGR (Compound Annual Growth Rate)
- Volatility
- Sharpe Ratio
- Max Drawdown (worst peak-to-trough loss)

---

### Tab 4: 📊 Analytics
**DEEP DIVE INTO CORRELATIONS**

**Correlation Matrix:**
- How your assets move together
- Red = negative correlation (diversification benefit)
- Blue = positive correlation (move together)

**Return Distributions:**
- Visualize daily return patterns
- Spot outliers and tail risk

**Summary Statistics:**
- Skewness: Measures downside tail risk
- Kurtosis: Measures extreme events

---

## 🚀 Common Use Cases

### Use Case 1: "I have $50k. How should I allocate it?"

1. Enter: **$50,000**
2. Choose tickers: **SPY, BND, GLD** (or use preset)
3. Select: **Max Sharpe** strategy
4. See: Recommended allocations with expected returns
5. Download: CSV file for your records

### Use Case 2: "What if I only had bonds?"

1. Select preset: **Conservative (Bonds & Stable)**
2. Set: **Min Variance** strategy
3. Review: Lower returns, but much lower risk
4. Compare: Metrics with other strategies

### Use Case 3: "How much could I lose in a market crash?"

1. Go to: **Risk Analysis** tab
2. Check: **Stress Test** scenarios
3. See: Dollar loss for Black Monday scenario
4. Understand: Your portfolio's crash risk

### Use Case 4: "How did this allocation perform historically?"

1. Go to: **Backtest** tab
2. Review: 5-year cumulative returns chart
3. Check: Maximum drawdown period
4. Understand: What you could have earned or lost

---

## 💡 Tips

### Best Practices

1. **Start with Max Sharpe**: Best for most investors (optimizes risk-adjusted returns)
2. **Compare strategies**: Use the same investment amount across all strategies
3. **Check correlation**: Tab 4 shows if your assets move together (bad for diversification)
4. **Understand your risk**: VaR tells you typical losses; CVaR tells you worst-case losses
5. **Look at drawdown**: Max Drawdown is what you'd experience in a crash

### Interpretation Guide

| Metric | Good | Okay | Risky |
|--------|------|------|-------|
| Sharpe | >1.0 | 0.5-1.0 | <0.5 |
| Volatility | <12% | 12-18% | >18% |
| Max Drawdown | >-20% | -20% to -30% | <-30% |
| VaR 95% | >-2% | -2% to -5% | <-5% |

### Correlation Interpretation

- **+1.0** = Perfect positive (move exactly together)
- **0.0** = Uncorrelated (independent)
- **-1.0** = Perfect negative (move opposite)

**Best portfolios have correlations near 0.0** (assets move independently, diversification benefit)

---

## 🎬 Example Walk-Through

Let's say you have $100,000 and want to invest in a balanced portfolio:

1. **Set Investment Amount**: $100,000

2. **Choose Tickers**: Use preset "Global Diversified"
   - SPY (US Stocks) - growth
   - EFA (International) - diversification
   - BND (Bonds) - stability
   - GLD (Gold) - crisis protection
   - DBC (Commodities) - inflation hedge
   - VNQ (REITs) - real estate exposure

3. **Select Max Sharpe Strategy** → Get allocations like:
   - SPY: $38,200
   - BND: $25,400
   - GLD: $18,700
   - Others: $17,700

4. **Check Risk Analysis**:
   - VaR 95%: -1.28% (95% of days, don't lose more)
   - CVaR 95%: -1.92% (worst days average -1.92%)
   - Black Monday stress: -22% (could lose $22k)

5. **Review Backtest**:
   - 5-year CAGR: 16.5% (would have $265k)
   - Max Drawdown: -17% (worst period)
   - Sharpe: 1.04 (excellent)

6. **Analyze Correlations**:
   - Bonds & Gold: negative (diversify US stock crashes)
   - Commodities & Stocks: low (independent)
   - REITs & Bonds: moderate (similar rate sensitivity)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"

```bash
pip install streamlit
```

### "No module named 'plotly'"

```bash
pip install plotly
```

### "Error loading data for ticker XYZ"

- Ticker might be invalid or delisted
- Try a different ticker
- Check Yahoo Finance for valid tickers

### Dashboard runs slow

- Too many years of lookback data
- Reduce "Historical Data Lookback" to 3-5 years
- Use fewer tickers

---

## 📱 Mobile Access

To access dashboard from phone on same network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then visit: `http://<your-computer-ip>:8501` from phone

---

## 🎓 Learning More

### Key Concepts Explained

**Sharpe Ratio**: Measures how much return you get per unit of risk
- Formula: (Return - Risk-Free Rate) / Volatility
- Higher is better
- >1.0 is excellent, <0.5 is poor

**Value at Risk (VaR)**: Threshold loss at given confidence
- "I'm 95% confident I won't lose more than X% on any given day"
- Used by banks and hedge funds for risk management

**Conditional VaR (CVaR)**: Average loss when VaR threshold is breached
- "If I'm in the worst 5% of days, average loss is Y%"
- More realistic measure of tail risk

**Sharpe Ratio vs Volatility**:
- Volatility = how much it swings (up or down)
- Sharpe = return per unit of swing (quality of returns)
- High Sharpe with lower volatility = better

---

## 🚀 Next Steps

1. **Play with different inputs** to see how allocations change
2. **Compare strategies** side-by-side for the same investment
3. **Stress test** your allocation in different scenarios
4. **Share results** with others using export feature
5. **Integrate with your broker** for real implementation

---

**Questions?** Check the inline help (hover over metric names) or review the portfolio_risk_optimizer.py source code for methodology details.

Good luck with your portfolio! 💰
