# Portfolio Performance Analysis

This project analyzes portfolio construction, diversification, and the relationship between risk and return for NSE stocks.

## Features

- Data collection from Yahoo Finance for NSE stocks
- Return analysis: mean returns, volatility, correlations
- Portfolio optimization: Efficient frontier calculation
- Performance evaluation: Sharpe ratio and comparison with individual assets
- Interactive Dashboard using Streamlit

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, matplotlib, yfinance, scipy, streamlit

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the analysis script:
```
python portfolio_analysis.py
```

Run the interactive dashboard:
```
python -m streamlit run dashboard.py
```

This will open a web browser with the dashboard showing historical prices, asset statistics, efficient frontier, and portfolio metrics.

## Output

- Console: Efficient Frontier plot, portfolio metrics
- Dashboard: Interactive charts and metrics

## Troubleshooting

- Ensure internet connection for data fetching
- If yfinance fails, check ticker symbols
- For optimization issues, verify scipy installation
- Dashboard requires streamlit to be installed