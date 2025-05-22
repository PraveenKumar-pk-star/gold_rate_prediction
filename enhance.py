import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast using ARIMA")

uploaded_file = st.file_uploader("Upload your gold price CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    st.sidebar.header("🔧 ARIMA Model Settings")
    p = st.sidebar.number_input("AR (p)", min_value=0, max_value=5, value=1)
    d = st.sidebar.number_input("I (d)", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("MA (q)", min_value=0, max_value=5, value=1)
    forecast_days = st.sidebar.slider("Forecast Period (days)", min_value=7, max_value=60, value=30)

    st.sidebar.header("📅 Filter Date Range")
    min_date, max_date = df.index.min(), df.index.max()
    start_date = st.sidebar.date_input("Start Date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    end_date = st.sidebar.date_input("End Date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    st.subheader("📊 Raw Data")
    st.write(df.head())

    if df.isnull().sum().sum() > 0:
        st.warning("Missing values found!")
        df = df.dropna()

    st.subheader("🟡 Gold Price Over Time")
    st.line_chart(df['price'])

    st.subheader("📌 Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, color='orange', ax=ax)
    st.pyplot(fig)

    df['price_change'] = df['price'].diff()
    df['price_return'] = df['price'].pct_change() * 100
    df['MA7'] = df['price'].rolling(window=7).mean()
    df['MA14'] = df['price'].rolling(window=14).mean()
    df['volatility'] = df['price_return'].rolling(window=7).std()

    st.subheader("📉 Daily Return (%)")
    st.line_chart(df['price_return'])

    st.subheader("📈 Moving Averages")
    fig, ax = plt.subplots()
    df[['price', 'MA7', 'MA14']].plot(ax=ax)
    st.pyplot(fig)

    st.subheader("🔴 Rolling Volatility (7 Days)")
    st.line_chart(df['volatility'])

    monthly_change = df['price'].resample('M').last().diff()
    st.subheader("📅 Monthly Price Change")
    fig, ax = plt.subplots()
    monthly_change.plot(kind='bar', color='brown', ax=ax)
    st.pyplot(fig)

    test_ratio = 0.2
    split_index = int(len(df) * (1 - test_ratio))
    train = df['price'][:split_index]
    test = df['price'][split_index:]

    st.subheader("🔮 ARIMA Forecasting")
    model_arima = ARIMA(train, order=(p, d, q))
    model_arima_fit = model_arima.fit()
    forecast = model_arima_fit.forecast(steps=forecast_days)

    last_date = train.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast_price': forecast}).set_index('date')

    fig, ax = plt.subplots()
    train.plot(label='Historical Price', ax=ax)
    forecast_df.plot(label='Forecast', ax=ax, color='green')
    ax.set_title("Gold Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    if len(test) >= forecast_days:
        mse = mean_squared_error(test[:forecast_days], forecast)
        rmse = np.sqrt(mse)
        error_percentage = (rmse / df['price'].mean()) * 100
        accuracy = 100 - error_percentage

        st.markdown(f"""
            ### 🧪 Evaluation Metrics:
            - **Mean Squared Error (MSE):** `{mse:.2f}`
            - **Root MSE:** `{rmse:.2f}`
            - **Error %:** `{error_percentage:.2f}%`
            - **Model Accuracy:** `{accuracy:.2f}%`
        """)

        st.subheader("📉 Final Forecast vs Actual")
        fig, ax = plt.subplots()
        test[:forecast_days].plot(label='Actual Price', ax=ax, color='gold')
        forecast.plot(label='Predicted Price', ax=ax, color='green')
        ax.legend()
        ax.set_title("ARIMA Model: Actual vs Predicted")
        st.pyplot(fig)

    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')
else:
    st.info("Please upload a CSV file with 'date' and 'price' columns to begin.")
