import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# List of NSE stock tickers
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']

# Time period
start_date = '2020-01-01'
end_date = '2023-12-31'

# Fetch historical data
print("Fetching data...")
data = yf.download(tickers, start=start_date, end=end_date)['Close']
print("Data fetched successfully")

# Calculate daily returns
returns = data.pct_change().dropna()
print("Returns calculated")

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Number of assets
num_assets = len(tickers)

# Function to calculate portfolio return and volatility
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252  # Annualized
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized
    return returns, std

# Function to minimize volatility for a given return
def minimize_volatility(weights, mean_returns, cov_matrix, target_return):
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return port_vol

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)  # Weights sum to 1
bounds = tuple((0, 1) for asset in range(num_assets))  # No short selling

# Target returns for efficient frontier
target_returns = np.linspace(0.05, 0.25, 50)
efficient_volatilities = []

for target in target_returns:
    constraints_with_return = constraints + ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target},)
    result = minimize(minimize_volatility, num_assets*[1./num_assets], args=(mean_returns, cov_matrix, target),
                      method='SLSQP', bounds=bounds, constraints=constraints_with_return)
    efficient_volatilities.append(result.fun)

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(efficient_volatilities, target_returns, 'b--', linewidth=2)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.grid(True)
plt.show()

# Calculate Sharpe ratio for a sample portfolio (equal weight)
weights = np.array([1/num_assets] * num_assets)
port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
sharpe_ratio = port_return / port_vol  # Assuming risk-free rate = 0

print(f"Equal Weight Portfolio:")
print(f"Expected Annual Return: {port_return:.2%}")
print(f"Annual Volatility: {port_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Performance evaluation: Compare with individual assets
individual_returns = mean_returns * 252
individual_vols = returns.std() * np.sqrt(252)

print("\nIndividual Assets:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: Return {individual_returns[i]:.2%}, Vol {individual_vols[i]:.2%}")

if __name__ == "__main__":
    pass  # Code runs above