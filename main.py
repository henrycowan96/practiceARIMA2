import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

st.title("Chocolate Sales Forecast with ARIMA")

try:
    data = pd.read_csv("chocolate_sales.csv", parse_dates=["date"], index_col="date")
    if "sales" not in data.columns:
        st.error("Missing 'sales' column.")
        st.stop()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Train/test split
train = data.iloc[:104]
test = data.iloc[104:]

# Fit auto_arima
try:
    model = auto_arima(train["sales"], seasonal=True, m=52, trace=True, suppress_warnings=True)
    forecast = model.predict(n_periods=len(test))
    rmse = np.sqrt(mean_squared_error(test["sales"], forecast))
    st.write(f"Test RMSE: {rmse:.2f}")

    # Refit on full data and forecast next 10 weeks
    model.update(data["sales"])
    future = model.predict(n_periods=10)
    future_index = pd.date_range(start=data.index[-1], periods=11, freq='W')[1:]
    forecast_df = pd.Series(future, index=future_index)

    # Plot
    st.line_chart(pd.concat([data["sales"], forecast_df]))

except Exception as e:
    st.error(f"Model error: {e}")

