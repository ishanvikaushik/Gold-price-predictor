import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np

# Page config
st.set_page_config(page_title="Gold Price Predictor", layout="centered")
st.title("📈 Gold Price Predictor (India)")
st.markdown("Predict gold price in ₹ for any future date (up to 2 years) based on historical trends using Meta’s Prophet model.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/gold_price_india_cleaned.csv")
    df.rename(columns={"Date": "ds", "Price": "y"}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # Filter out any accidental future dates
    df = df[df['ds'] <= pd.Timestamp.today()]
    return df

df = load_data()

# Data Validation 
st.subheader("Data Quality Check")
missing_values = df.isnull().sum()
invalid_dates = df['ds'].isna().sum()
invalid_prices = df['y'].isna().sum()

if missing_values.sum() == 0 and invalid_dates == 0 and invalid_prices == 0:
    st.success(" Data is clean — no missing values or invalid entries.")
else:
    st.warning("Warning: Data contains missing values. Please review the source file.")
    st.write("Missing values per column:")
    st.write(missing_values)

#  Train Prophet Model 
model = Prophet(daily_seasonality=True)
model.add_country_holidays(country_name='IN')
model.fit(df)

#  Forecast Future Prices
last_date = df['ds'].max()
future = model.make_future_dataframe(periods=730)
forecast = model.predict(future)

# Filter forecast starting from today
today = pd.Timestamp.today().normalize()
forecast_future = forecast[forecast['ds'] >= today]

# Predict Specific Date 
st.subheader("🔍 Select a Date to Predict")
selected_date = st.date_input("Choose a future date (within next 2 years):", min_value=today.date())

if st.button("Predict Gold Price"):
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    match = forecast[forecast['ds'].dt.strftime('%Y-%m-%d') == selected_date_str]

    if not match.empty:
        yhat = match.iloc[0]['yhat']
        lower = match.iloc[0]['yhat_lower']
        upper = match.iloc[0]['yhat_upper']
        st.success(f"Predicted price on {selected_date_str}: ₹{yhat:.2f}")
        st.info(f"Confidence Interval: ₹{lower:.2f} – ₹{upper:.2f}")
    else:
        st.warning("Selected date is beyond forecast range. Try a closer date.")

#  Forecast Chart (Past + Future) 
st.subheader("Forecasted Trend (Past + Future)")
plot = plot_plotly(model, forecast)
st.plotly_chart(plot, use_container_width=True)

# Evaluate Model 
st.markdown("## Model Evaluation")

def evaluate_model(df):
    split_idx = int(len(df) * 0.9)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    model_eval = Prophet()
    model_eval.fit(train)

    future_eval = model_eval.make_future_dataframe(periods=len(test))
    forecast_eval = model_eval.predict(future_eval)

    predicted = forecast_eval[['ds', 'yhat']].tail(len(test)).reset_index(drop=True)
    actual = test[['ds', 'y']].reset_index(drop=True)

    rmse = np.sqrt(mean_squared_error(actual['y'], predicted['yhat']))
    return rmse, predicted, actual

rmse, predicted, actual = evaluate_model(df)
st.success(f" RMSE (Root Mean Squared Error): ₹{rmse:.2f}")
st.caption("Lower RMSE means better prediction accuracy.")

#  Plot Predicted vs Actual 
st.subheader(" Predicted vs Actual Prices (Test Set)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], mode='lines+markers', name='Actual Price'))
fig.add_trace(go.Scatter(x=predicted['ds'], y=predicted['yhat'], mode='lines+markers', name='Predicted Price'))
fig.update_layout(xaxis_title='Date', yaxis_title='Gold Price (₹/gram)', legend_title='Legend', template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# Display Next 7 Days Prediction 
st.subheader(" Recent Predictions (Next 7 Days)")
next_7_days = forecast_future.head(7)[['ds', 'yhat']]
next_7_days = next_7_days.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price (₹)'})
st.dataframe(next_7_days)
