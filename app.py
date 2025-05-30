
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pickle
from datetime import timedelta

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast using ARIMA")

uploaded_file = st.file_uploader("Upload your gold price CSV file (with 'date' and 'price' columns)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.dropna()

    st.subheader("📊 Raw Data Preview")
    st.write(df.tail())

    # Train ARIMA model on full data
    model_arima = ARIMA(df['price'], order=(2, 1, 0))
    model_arima_fit = model_arima.fit()
    with open("arima_model.pkl", "wb") as f:
        pickle.dump(model_arima_fit, f)
    st.success("✅ ARIMA model trained and saved.")

    # Forecast next 30 days from a selected date
    st.subheader("📅 Forecast for Next 30 Days")
    selected_date = st.date_input("Select a date to start forecasting from (must be after the last date)", df.index[-1].date())

    if selected_date > df.index[-1].date():
        steps = 30
        model = pickle.load(open("arima_model.pkl", "rb"))
        forecast = model.forecast(steps=steps)
        forecast_dates = pd.date_range(start=pd.to_datetime(selected_date), periods=steps)
        forecast_series = pd.Series(forecast, index=forecast_dates)

        fig, ax = plt.subplots()
        df['price'].plot(ax=ax, label="Historical")
        forecast_series.plot(ax=ax, label="30-Day Forecast", color='red')
        ax.legend()
        ax.set_title("Gold Price Forecast for 30 Days")
        st.pyplot(fig)
    else:
        st.warning("📅 Please select a date after the last date in your data.")

else:
    st.info("📤 Please upload a CSV file to continue.")
