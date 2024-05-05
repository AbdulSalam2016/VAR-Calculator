# Value at Risk (VaR) Calculator

# Purpose:
# This tool calculates the Value at Risk (VaR) for a portfolio using both Monte Carlo simulations and Parametric methods. It provides some results and visualizations tailored to the user's inputs.
#
# Instructions:
# From the sidebar, choose your desired stocks. This selection will determine the composition of the portfolio for which VaR is calculated. Select at least two stocks.
# Specify the historical period for adjusted close daily stock price data retrieval, which will be sourced directly from the Yahoo Finance live API.
# Ensure the total sum of the portfolio weights for the selected stocks equals 1. Adjust the weights accordingly to meet this requirement.

# Credit:
# This tool was developed by Dr. Yakubu Abdul-Salam, Associate Professor of Energy Economics at the University of Aberdeen, UK.

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap visualization

def fetch_data(tickers, end_date, years_back):
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years=years_back)
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.pct_change().dropna(), data

def monte_carlo_var(returns, weights, portfolio_value, num_simulations, confidence_interval, num_days):
    weights = np.array(weights)
    daily_returns = returns.dot(weights)
    daily_portfolio_return = daily_returns.mean() * num_days
    daily_port_std_dev = np.sqrt(daily_returns.var() * num_days)
    sim_returns = np.random.normal(daily_portfolio_return, daily_port_std_dev, num_simulations)
    sim_portfolio_values = portfolio_value * (1 + sim_returns)
    VaR = portfolio_value - np.percentile(sim_portfolio_values, 100 - confidence_interval)
    return VaR, sim_portfolio_values

def parametric_var(returns, weights, portfolio_value, confidence_interval):
    weights = np.array(weights)
    daily_returns = returns.dot(weights)
    mean = daily_returns.mean()
    std_dev = np.sqrt(daily_returns.var())
    z_score = norm.ppf(1 - confidence_interval / 100)
    VaR = -z_score * std_dev * portfolio_value
    return VaR


# Sidebar for input parameters
st.sidebar.title('Input Parameters')
tickers_options = ['AMZN', 'GOOG', 'NG.L', 'BP.L', 'CNA.L', 'SSE.L', 'WMT', 'TSLA', 'UKW.L', 'XOM', 'CVX', 'PBR', 'EOAN.SW', 'META', '^GSPC', 'VBMFX', 'AAPL']
selected_tickers = st.sidebar.multiselect('Select Tickers for Portfolio', options=tickers_options, default=['AMZN', 'GOOG'])
max_end_date = pd.to_datetime('today') - pd.DateOffset(days=1)
end_date = st.sidebar.date_input('Select End Date for historical price series', max_end_date, max_value=max_end_date)
years_back = st.sidebar.number_input("Select number of years for the historical price series", min_value=1, value=5)
portfolio_value = st.sidebar.number_input('Portfolio Value', min_value=1000, value=1000000)
num_simulations = st.sidebar.number_input('Number of Monte Carlo Simulations', min_value=1000, value=10000)
confidence_interval = st.sidebar.slider('Confidence Interval (%)', min_value=90, max_value=99, value=95)
num_days = st.sidebar.number_input('Number of Days for VaR', min_value=1, value=1)

weights = {}
total_weight = 0
if selected_tickers:
    for ticker in selected_tickers:
        weights[ticker] = st.sidebar.slider(f'Portfolio Weight for {ticker} (%)', min_value=0, max_value=100, value=100 // len(selected_tickers), step=1)
        total_weight += weights[ticker]

# Main area for outputs
st.title('Value at Risk (VaR) Calculator')
st.subheader('Purpose:')
st.markdown("This tool calculates the Value at Risk (VaR) for a portfolio using both the :blue[Monte Carlo simulations] and :blue[Parametric] methods. It provides some results and visualizations tailored to the user's inputs. This tool was developed by :blue[Dr. Yakubu Abdul-Salam], Associate Professor of Energy Economics at the University of Aberdeen, UK.")

st.subheader('Instructions:')
st.markdown(
    """
    1. From the sidebar, choose your desired tickers/stocks. This selection will determine the composition of the portfolio for which VaR is calculated. Select at least two tickers/stocks.
    2. Specify the historical period for :blue[adjusted close daily ticker/stock price data] retrieval, which will be sourced directly from the :blue[Yahoo Finance live API and servers].
    3. Ensure the total sum of the portfolio weights for the selected tickers/stocks equals 1. Adjust the weights in the sidebar accordingly to meet this requirement.
    """
)

# st.subheader('Credit:')
# st.markdown("This tool was developed by :blue[Dr. Yakubu Abdul-Salam], Associate Professor of Energy Economics at the University of Aberdeen, UK.")

if st.button(':white[Calculate VaR]', key='calculate_button', type='primary', use_container_width=1, help='Click to calculate VaR'):
    weight_list = [weights[ticker] / 100 for ticker in selected_tickers]
    if np.isclose(sum(weight_list), 1.0):
        try:
            _, prices = fetch_data(selected_tickers, end_date, years_back)
            returns = prices.pct_change().dropna()
            mc_VaR, mc_values = monte_carlo_var(returns, weight_list, portfolio_value, num_simulations, confidence_interval, num_days)
            param_VaR = parametric_var(returns, weight_list, portfolio_value, confidence_interval)
            st.success(f'Monte Carlo simulation method: VaR over {num_days} day(s) period at {confidence_interval}% confidence interval: ${mc_VaR:,.2f}')
            st.success(f'Parametric method: VaR over 1 day period only at {confidence_interval}% confidence interval: ${param_VaR:,.2f}')

            # Visualization split into two rows
            fig, axs = plt.subplots(3, 2, figsize=(15, 12))
            plt.subplots_adjust(hspace=0.5)

            # Histogram of portfolio values
            axs[0, 0].hist(mc_values, bins=30, color='blue', edgecolor='black')
            #axs[0, 0].axvline(x=portfolio_value - mc_VaR, color='red', linestyle='dashed', label='Value at Risk'
            #axs[0, 0].axvline(x=mc_VaR, color='red', linestyle='dashed', label='Value at Risk')
            axs[0, 0].set_title('Distribution of Simulated Portfolio Values')
            axs[0, 0].legend()

            # Histogram of Value at Risk (VaR)
            axs[0, 1].hist(mc_values - portfolio_value + mc_VaR, bins=30, color='green', edgecolor='black')
            axs[0, 1].axvline(x=mc_VaR, color='purple', linestyle='dashed', label='Expected value at risk')
            axs[0, 1].set_title('Distribution of Portfolio Value at Risk')
            axs[0, 1].legend()

            # Correlation heatmap
            corr = prices.corr()
            sns.heatmap(corr, ax=axs[1, 0], annot=True, cmap='coolwarm')
            axs[1, 0].set_title('Price Correlation Matrix for Selected Tickers')

            # Adjusted Close Prices
            prices.plot(ax=axs[1, 1], title='Adjusted Close Prices')
            returns.plot(ax=axs[2, 0], title='Daily Returns')

            # Histogram of returns for each selected ticker
            for ticker in selected_tickers:
                axs[2, 1].hist(returns[ticker], bins=30, alpha=0.5, label=f'{ticker} returns')
            axs[2, 1].set_title('Histogram of Returns for Selected Tickers')
            axs[2, 1].legend()

            st.pyplot(fig)
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.error('Total weights must sum to 100%.')
