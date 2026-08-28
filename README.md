# 🎯 Portfolio Optimizer Dashboard

A professional Streamlit application for portfolio optimization, risk analysis, and Dollar-Cost Averaging tracking. Built with data science best practices including rigorous overfitting detection.

**[Quick Start](#quick-start)** | **[Features](#features)** | **[Performance](#-performance-results)**

---

## 🎬 Overview

This project combines **quantitative portfolio optimization** with **interactive data visualization** to help investors:
- Optimize asset allocations using Modern Portfolio Theory (Markowitz)
- Track weekly DCA investments with flexible ticker support
- Analyze risk metrics (VaR, CVaR, stress testing)
- Validate strategies against overfitting through rigorous testing
- Backtest historical performance over 5+ years

**Key Result:** Portfolio optimizer achieved **Sharpe ratio 0.998** with **15.4% CAGR** over 5 years, validated with 2.8/3.0 overfitting score (low overfitting).

---

## ✨ Features

### 🎯 Tab 1: Portfolio Optimization
- Multiple strategies: Max Sharpe, Min Variance, Risk Parity, Equal Weight
- Custom ticker selection and manual allocation input
- Real-time risk metrics: Sharpe, volatility, expected return
- Pie chart visualization and CSV export

### 💹 Tab 2: Risk Analysis
- Value-at-Risk (VaR) at 95% and 99% confidence
- Conditional VaR for tail risk analysis
- Stress testing: 2008 crisis, COVID crash, Black Monday scenarios

### 📈 Tab 3: Backtesting
- 5-year historical simulation with quarterly rebalancing
- Performance metrics: CAGR, Sharpe, Max Drawdown
- Cumulative returns and drawdown charts
- Monthly returns heatmap

### 📊 Tab 4: Analytics
- Correlation matrix for asset relationships
- Return distributions and tail risk analysis
- Summary statistics: skewness, kurtosis

### 💰 Tab 5: DCA Tracker
- Configure any set of tickers to track
- Log weekly investments with flexible allocations
- Edit/delete past entries
- Persistent JSON storage
- Cumulative growth charts and 12-month projections

### 🔍 Tab 6: Overfitting Detection
- Out-of-sample testing
- Weight stability analysis
- Benchmark comparison
- Forward testing on recent data
- Overall overfitting score (0-3 scale)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

Open browser to `http://localhost:8501`

---

## 📈 Example: Optimize $100K Portfolio

1. Go to **Tab 1 (Optimization)**
2. Enter: **$100,000** investment amount
3. Select: **SPY, BND, GLD** tickers
4. Choose: **Max Sharpe** strategy
5. See recommended allocation:
   - SPY: $31,200 (31%)
   - BND: $15,000 (15%)
   - GLD: $53,800 (54%)
   - Expected return: **15.4%** annually
   - Sharpe: **0.998** (excellent)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit 1.28+ |
| Analysis | Pandas, NumPy |
| Optimization | SciPy (SLSQP) |
| Visualization | Plotly |
| Data | yfinance |
| Storage | JSON |

---

## 📊 Performance Results

### Backtest: SPY/BND/GLD (5 years)

| Strategy | CAGR | Volatility | Sharpe | Max DD |
|----------|------|-----------|--------|--------|
| **Max Sharpe** | **15.51%** | 12.98% | **1.04** | -17.04% |
| Min Variance | 1.48% | 5.62% | -0.27 | -14.24% |
| Equal Weight | 9.93% | 10.72% | 0.65 | -17.57% |

### Validation Results

| Test | Score | Status |
|------|-------|--------|
| Out-of-Sample | 0.12 | ✅ Low degradation |
| Stability | 0.053 avg | ✅ Stable weights |
| vs Benchmarks | +0.16 Sharpe | ✅ Real edge |
| Forward Test | -0.423 | ✅✅ **Improved on new data** |
| **Overall** | **2.8/3.0** | ✅ **Low overfitting** |

---

## 📁 Project Structure

```
portfolio-optimizer/
├── app.py                              # Main dashboard
├── portfolio_risk_optimizer_final.py    # Optimizer functions
├── requirements.txt                    # Dependencies
├── .gitignore
└── README.md
```

---

## 💡 Key Insights

- **Constrained optimization** (0.15-0.85 bounds) prevents extreme allocations
- **Out-of-sample validation** proves model generalizes
- **Forward testing shows improvement** on recent data (not overfitting!)
- **Bonds are essential** for downside protection, even if recent years underperformed
- **Weekly DCA beats** trying to time the market

---

## 📝 License

MIT License - Free to use for personal or commercial projects

---

## 👤 Author

**Christopher Aldaco** - Mathematics & Business @ University of Washington

Pursuing finance and data science internships in banking, investment management, and quantitative finance.

---

**Status:** ✅ Production Ready | Last Updated: August 2026
