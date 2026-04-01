import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.title("Portfolio Performance Analysis Dashboard")

st.sidebar.header("Settings")

# Default tickers
default_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=', '.join(default_tickers))
tickers = [t.strip() for t in tickers_input.split(',')]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-12-31'))

@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

data = fetch_data(tickers, start_date, end_date)

if data.empty:
    st.error("No data fetched. Check tickers and dates.")
else:
    st.subheader("Historical Prices")
    st.line_chart(data)

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(tickers)

    st.subheader("Asset Statistics")
    stats = pd.DataFrame({
        'Mean Return': mean_returns * 252,
        'Volatility': returns.std() * np.sqrt(252)
    })
    st.dataframe(stats)

    # Portfolio Performance Function
    def portfolio_performance(weights, mean_returns, cov_matrix):
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return port_return, port_vol

    # Efficient Frontier
    target_returns = np.linspace(stats['Mean Return'].min(), stats['Mean Return'].max(), 50)
    efficient_volatilities = []

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0, 1) for _ in range(num_assets))

    for target in target_returns:
        cons = constraints + ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target},)
        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1], 
                          num_assets*[1./num_assets], 
                          method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            efficient_volatilities.append(result.fun)
        else:
            efficient_volatilities.append(np.nan)

    st.subheader("Efficient Frontier")
    fig, ax = plt.subplots()
    ax.plot(efficient_volatilities, target_returns, 'b--')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    st.pyplot(fig)

    # Equal Weight Portfolio
    weights = np.array([1/num_assets] * num_assets)
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = port_return / port_vol  # Assuming rf=0

    st.subheader("Equal Weight Portfolio")
    st.metric("Expected Return", f"{port_return:.2%}")
    st.metric("Volatility", f"{port_vol:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.subheader("Portfolio Weights")
    weight_df = pd.DataFrame({'Asset': tickers, 'Weight': weights})
    st.bar_chart(weight_df.set_index('Asset'))