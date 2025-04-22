# SmartPortfolio: Portfolio Optimization and Risk Analysis Tool

# Required installations:
# pip install yfinance dash pandas numpy matplotlib scipy plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from scipy.optimize import minimize
import yfinance as yf
import webbrowser

# === 1. Configuration ===
tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'V', 'XOM', 'JNJ', 'PFE', 'TGT', 'WMT']
benchmark_ticker = 'SPY'
risk_free_rate = 0.02  # Annualized risk-free rate (e.g., 2%)

# === 2. Fetch Data ===
def fetch_data(tickers):
    """Fetch daily stock/ETF data from Yahoo Finance."""
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            df = yf.download(ticker, start="2000-01-01", end="2025-01-01")
            df = df[['Close']].rename(columns={'Close': ticker})
            if data.empty:
                data = df
            else:
                data = data.join(df, how='outer')
            print(f"Data for {ticker} fetched successfully.")
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
    if data.empty:
        raise ValueError("No data retrieved for any tickers. Exiting.")
    return data.sort_index()

price_data = fetch_data(tickers + [benchmark_ticker])

# === 3. Return & Risk Calculations ===
def calculate_returns(price_data):
    returns = price_data.pct_change().dropna()
    return returns

returns = calculate_returns(price_data)
mean_returns = returns[tickers].mean()
cov_matrix = returns[tickers].cov()
benchmark_returns = returns[[benchmark_ticker]]

# === 4. Portfolio Optimization ===
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return ret, vol

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate the negative Sharpe ratio for optimization."""
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

def optimize_portfolio(mean_returns, cov_matrix):
    """Optimize portfolio to maximize the Sharpe ratio."""
    num_assets = len(mean_returns)
    if num_assets == 0:
        raise ValueError("No assets available for optimization. Check your data.")
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    result = minimize(negative_sharpe, initial_guess,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

opt_weights = optimize_portfolio(mean_returns, cov_matrix)
opt_ret, opt_vol = portfolio_performance(opt_weights, mean_returns, cov_matrix)
opt_sharpe = (opt_ret - risk_free_rate) / opt_vol

# === 5. Efficient Frontier ===
def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    """Generate the efficient frontier."""
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        if i % 1000 == 0:
            print(f"Processing portfolio {i}/{num_portfolios}...")
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std
    print("Efficient frontier calculation completed.")
    return results, weights_record

print("Generating efficient frontier...")
results, weights_record = generate_efficient_frontier(mean_returns, cov_matrix)

# === 6. Benchmark Comparison ===
benchmark_annual_return = benchmark_returns.mean().iloc[0] * 252
benchmark_volatility = benchmark_returns.std().iloc[0] * np.sqrt(252)
benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_volatility

# === 7. Dash App ===
app = Dash(__name__)

app.layout = html.Div([
    html.H1("SmartPortfolio Optimizer"),
    dcc.Graph(id='efficient-frontier'),
    html.Div([
        html.H4("Optimal Portfolio Weights:"),
        html.Ul([html.Li(f"{ticker}: {weight:.2%}") for ticker, weight in zip(tickers, opt_weights)])
    ]),
    html.Div([
        html.H4("Performance Comparison vs. S&P 500 (SPY):"),
        html.Ul([
            html.Li(f"Optimized Portfolio - Return: {opt_ret:.2%}, Volatility: {opt_vol:.2%}, Sharpe Ratio: {opt_sharpe:.2f}"),
            html.Li(f"S&P 500 (SPY) - Return: {benchmark_annual_return:.2%}, Volatility: {benchmark_volatility:.2%}, Sharpe Ratio: {benchmark_sharpe:.2f}")
        ])
    ])
])

@app.callback(
    Output('efficient-frontier', 'figure'),
    Input('efficient-frontier', 'id')
)
def update_graph(_):
    trace = go.Scatter(
        x=results[0],
        y=results[1],
        mode='markers',
        marker=dict(color=results[2], colorscale='Viridis', showscale=True),
        text=[f"Sharpe Ratio: {sr:.2f}" for sr in results[2]]
    )
    return {
        'data': [trace],
        'layout': go.Layout(
            title='Efficient Frontier',
            xaxis={'title': 'Volatility'},
            yaxis={'title': 'Expected Return'},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8050/")
    app.run(debug=True)

