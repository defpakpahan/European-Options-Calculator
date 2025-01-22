import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import exp, sqrt
import yfinance as yf
import pandas as pd
from io import BytesIO

# Models

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Real-Time Price Comparison
def fetch_real_time_price(ticker):
    try:
        price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        return price
    except Exception as e:
        return None

# Value at Risk (VaR) and Expected Shortfall (ES)
def calculate_var_es(returns, confidence_level):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    es = sorted_returns[:index].mean()
    return var, es

# Streamlit App
st.title("Options Pricing Calculator")

# Inputs
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Stock Ticker", "AAPL")
    real_time_price = fetch_real_time_price(ticker)
    if real_time_price:
        S = real_time_price
        st.success(f"Current Stock Price: ${S:.2f}")
    else:
        S = st.number_input("Stock Price", min_value=1.0, value=100.0)

    option_type = st.selectbox("Option Type", ["Call", "Put"])

    K = st.number_input("Strike Price", min_value=1.0, value=100.0)
    sigma = st.number_input("Volatility (%)", min_value=0.0, value=20.0) / 100
    r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0) / 100

    T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0)

    contracts = st.number_input("Number of Contracts", min_value=1, value=1, step=1)
    fee = st.number_input("Associated Fee per Contract ($)", min_value=0.0, value=0.0)

    confidence_level = st.slider("Confidence Level for VaR and ES", min_value=0.8, max_value=0.99, value=0.95)

# Black-Scholes Calculation
option_price = black_scholes(S, K, T, r, sigma, option_type=option_type.lower())
total_cost = (option_price + fee) * contracts * 100

# Outputs
st.header("Results")
st.write(f"**Option Price:** ${option_price:.2f}")
st.write(f"**Total Cost for {contracts} Contracts (including fees):** ${total_cost:.2f}")

# Risk Metrics
st.subheader("Risk Metrics")
simulated_returns = np.random.normal(loc=r - 0.5 * sigma ** 2, scale=sigma, size=1000) * contracts * 100
var, es = calculate_var_es(simulated_returns, confidence_level)
st.write(f"Value at Risk (VaR) at {confidence_level*100:.0f}% confidence: ${var:.2f}")
st.write(f"Expected Shortfall (ES) at {confidence_level*100:.0f}% confidence: ${es:.2f}")

# Greeks Calculation
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -((S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - (r * K * exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = (K * T * exp(-r * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))) / 100
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

greeks = calculate_greeks(S, K, T, r, sigma, option_type=option_type.lower())
st.subheader("Greeks")
columns = st.columns(3)
columns[0].write(f"Delta: {greeks['Delta']:.4f}")
columns[1].write(f"Gamma: {greeks['Gamma']:.4f}")
columns[2].write(f"Theta: {greeks['Theta']:.4f}")
columns[0].write(f"Vega: {greeks['Vega']:.4f}")
columns[1].write(f"Rho: {greeks['Rho']:.4f}")

# Payoff Diagram
st.subheader("Payoff Diagram")
S_range = np.linspace(0.5 * S, 1.5 * S, 100)
if option_type.lower() == "call":
    payoff = np.maximum(S_range - K, 0) - option_price - fee
else:
    payoff = np.maximum(K - S_range, 0) - option_price - fee

break_even = K + option_price if option_type.lower() == "call" else K - option_price

fig, ax = plt.subplots()
ax.plot(S_range, payoff, label="Payoff")
ax.axhline(0, color="black", linestyle="--")
ax.axvline(break_even, color="blue", linestyle="--", label=f"Break Even: ${break_even:.2f}")
ax.set_title("Payoff Diagram")
ax.set_xlabel("Stock Price (S)")
ax.set_ylabel("Payoff ($)")
ax.legend()
st.pyplot(fig)
st.write(f"Break Even Point: ${break_even:.2f}")

# Heatmap of Option Prices (Spot Price vs Volatility)
st.subheader("Heatmap: Option Price vs Volatility")
volatility_range = np.linspace(0.1, 0.9, 10)
spot_prices = np.linspace(0.8 * S, 1.2 * S, 10)
heatmap_data = np.array([
    [black_scholes(spot, K, T, r, vol, option_type=option_type.lower()) for spot in spot_prices]
    for vol in volatility_range
])

fig, ax = plt.subplots()
c = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", origin="lower",
              extent=[spot_prices.min(), spot_prices.max(), volatility_range.min(), volatility_range.max()])
fig.colorbar(c, ax=ax, label="Option Price ($)")
ax.set_title(option_type.upper())
ax.set_xlabel("Spot Price ($)")
ax.set_ylabel("Volatility")
st.pyplot(fig)

# PnL Matrix
st.subheader("PnL Matrix")
pnl_matrix = np.zeros_like(heatmap_data)
for i, vol in enumerate(volatility_range):
    for j, spot in enumerate(spot_prices):
        pnl_matrix[i, j] = (heatmap_data[i, j] - option_price - fee) * contracts * 100

pnl_df = pd.DataFrame(pnl_matrix, index=[f"{vol:.2f}" for vol in volatility_range], columns=[f"{spot:.2f}" for spot in spot_prices])
st.dataframe(pnl_df.style.applymap(lambda x: f"color: {'#39e75f' if x > 0 else '#ee2400'}").format("${:.2f}"))
st.write("X-axis: Spot Prices ($), Y-axis: Volatility")

# Export Functionality
def export_results():
    output = BytesIO()
    data = []
    for i, vol in enumerate(volatility_range):
        for j, spot in enumerate(spot_prices):
            pnl = (heatmap_data[i, j] - option_price - fee) * contracts * 100
            data.append({"Spot Price ($)": spot, "Volatility": vol, "Option Price ($)": heatmap_data[i, j], "PnL ($)": pnl})

    df = pd.DataFrame(data)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Heatmap Data", index=False)
    return output.getvalue()

if st.button("Export Heatmap Data"):
    st.download_button(label="Download Heatmap Data as Excel", data=export_results(), file_name="heatmap_data.xlsx")