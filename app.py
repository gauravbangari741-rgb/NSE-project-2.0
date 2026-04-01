import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page config for dark theme
st.set_page_config(
    page_title="Portfolio Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #0e1117;
    }
    .css-1lcbmhc {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Portfolio Performance Analysis Dashboard")

# Sidebar for settings
st.sidebar.header("Settings")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
start_date = pd.to_datetime('today') - pd.DateOffset(years=3)
end_date = pd.to_datetime('today')

# Fetch data
@st.cache_data
def fetch_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        if data.empty:
            st.error("No data fetched. Please check tickers.")
            return pd.DataFrame()
        # Handle missing data
        data = data.fillna(method='ffill').dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = fetch_data(tickers, start_date, end_date)

if data.empty:
    st.stop()

# Calculate returns
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(tickers)

# Annualize
annual_returns = mean_returns * 252
annual_vols = returns.std() * np.sqrt(252)

# Section 1: Historical Prices
st.header("📈 Historical Prices")
fig_prices = go.Figure()
for ticker in tickers:
    fig_prices.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))
fig_prices.update_layout(
    title="Historical Adjusted Closing Prices",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark"
)
st.plotly_chart(fig_prices, use_container_width=True)

# Section 2: Asset Statistics
st.header("📊 Asset Statistics")
stats_df = pd.DataFrame({
    'Asset': tickers,
    'Mean Return (%)': (annual_returns * 100).round(2),
    'Volatility (%)': (annual_vols * 100).round(2)
})
st.dataframe(stats_df, use_container_width=True)

# Portfolio calculations
np.random.seed(42)
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    port_return = np.sum(annual_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - 0.02) / port_vol
    
    results[0,i] = port_return
    results[1,i] = port_vol
    results[2,i] = sharpe

# Section 3: Efficient Frontier
st.header("🎯 Efficient Frontier")
fig_frontier = go.Figure()
fig_frontier.add_trace(go.Scatter(
    x=results[1,:],
    y=results[0,:],
    mode='markers',
    marker=dict(
        size=5,
        color=results[2,:],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Sharpe Ratio")
    ),
    name="Random Portfolios"
))
fig_frontier.update_layout(
    title="Efficient Frontier (5000 Random Portfolios)",
    xaxis_title="Volatility",
    yaxis_title="Expected Return",
    template="plotly_dark"
)
st.plotly_chart(fig_frontier, use_container_width=True)

# Equal Weight Portfolio
equal_weights = np.array([1/num_assets] * num_assets)
equal_return = np.sum(annual_returns * equal_weights)
equal_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix * 252, equal_weights)))
equal_sharpe = (equal_return - 0.02) / equal_vol

# Section 4: Equal Weight Portfolio
st.header("⚖️ Equal Weight Portfolio")
col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{equal_return:.2%}")
col2.metric("Volatility", f"{equal_vol:.2%}")
col3.metric("Sharpe Ratio", f"{equal_sharpe:.2f}")

# Portfolio Weights Bar Chart
weights_df = pd.DataFrame({
    'Asset': tickers,
    'Weight': equal_weights
})
fig_weights = px.bar(weights_df, x='Asset', y='Weight', title="Equal Weight Portfolio Allocation")
fig_weights.update_layout(template="plotly_dark")
st.plotly_chart(fig_weights, use_container_width=True)