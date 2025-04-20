import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# App title
st.title("📈 Stock Price Predictor App")

# User input for stock ticker
stock = st.text_input("Enter the Stock ID (Ticker)", "GOOG")

# Fetch historical stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

# Display stock data
st.subheader("📊 Stock Data")
st.write(google_data)

# Check for MultiIndex issue
if isinstance(google_data.columns, pd.MultiIndex):
    close_col = ('Close', stock)  # MultiIndex case
else:
    close_col = 'Close'  # SingleIndex case

# Moving Averages
google_data['MA_for_250_days'] = google_data[close_col].rolling(250).mean()
google_data['MA_for_200_days'] = google_data[close_col].rolling(200).mean()
google_data['MA_for_100_days'] = google_data[close_col].rolling(100).mean()

# Plotting function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label='Moving Average')
    plt.plot(full_data[close_col], 'b', label='Actual Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'g', label='Extra Data')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Trends")
    return fig

# Plot Moving Averages
st.subheader("📉 Close Price vs MA (250 days)")
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

st.subheader("📉 Close Price vs MA (200 days)")
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

st.subheader("📉 Close Price vs MA (100 days)")
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader("📉 MA (100 days) vs MA (250 days)")
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Splitting data for model
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data[close_col][splitting_len:])

# Set model path
model_path = r"C:\Users\ROG\OneDrive\Desktop\PROJECT FOLDERS\stock_price_prediction-main\project\Latest_stock_price_model.keras"

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
    st.write("🔹 Please ensure the file is in the correct location or retrain the model.")
    st.stop()

# Load trained LSTM model
try:
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test.values.reshape(-1, 1))

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse transform the predictions and actual values
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Calculate error metrics
mae = mean_absolute_error(inv_y_test, inv_predictions)
mse = mean_squared_error(inv_y_test, inv_predictions)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((inv_y_test - inv_predictions) / inv_y_test)) * 100

# Display metrics
st.subheader("📌 Error Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("MAPE", f"{mape:.2f}%")

# Prepare DataFrame for results
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# Add per-row absolute and percentage error
ploting_data['absolute_error'] = np.abs(ploting_data['original_test_data'] - ploting_data['predictions'])
ploting_data['squared_error'] = ploting_data['absolute_error'] ** 2
ploting_data['percentage_error'] = (ploting_data['absolute_error'] / ploting_data['original_test_data']) * 100

# Display Table
st.subheader("📊 Original vs Predicted Values with Errors")
st.write(ploting_data)

# Plot predictions vs actual data
st.subheader("📉 Actual Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data[close_col][:splitting_len + 100], ploting_data['original_test_data']], axis=0), label="Original")
plt.plot(pd.concat([pd.Series([np.nan] * (splitting_len + 100), index=google_data.index[:splitting_len + 100]), ploting_data['predictions']], axis=0), label="Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Stock Price Prediction for {stock}")
plt.legend()
st.pyplot(fig)

# Plot error metrics over time
st.subheader("📈 Error Metrics Over Time")

def plot_error_series(y, label, color):
    fig = plt.figure(figsize=(15, 4))
    plt.plot(ploting_data.index, y, color=color)
    plt.xlabel("Date")
    plt.ylabel(label)
    plt.title(f"{label} Over Time")
    return fig

st.pyplot(plot_error_series(ploting_data['absolute_error'], "Absolute Error", "orange"))
st.pyplot(plot_error_series(ploting_data['squared_error'], "Squared Error", "red"))
st.pyplot(plot_error_series(ploting_data['percentage_error'], "Percentage Error (%)", "green"))

# **Future Predictions**
N_future_days = 10  # Change the number of days to predict

# Use the last 100 days of data for future prediction
last_100_scaled = scaled_data[-100:]
future_predictions_scaled = []
input_seq = last_100_scaled.copy()

# Predict future stock prices
for _ in range(N_future_days):
    input_reshaped = np.reshape(input_seq, (1, input_seq.shape[0], 1))
    pred_scaled = model.predict(input_reshaped, verbose=0)
    future_predictions_scaled.append(pred_scaled[0])
    input_seq = np.append(input_seq[1:], pred_scaled, axis=0)

# Inverse transform to get original prices
future_predictions = scaler.inverse_transform(future_predictions_scaled)

# Generate future dates
last_date = google_data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=N_future_days, freq='B')

# Create DataFrame for future predictions
future_df = pd.DataFrame({
    'Predicted Price': future_predictions.flatten()
}, index=future_dates)

# Display future predictions
st.subheader("📅 Future Stock Price Prediction")
st.write(future_df)

# Plot future predictions
st.subheader("📉 Future Stock Price Prediction Plot")
fig_future = plt.figure(figsize=(10, 5))
plt.plot(future_df, marker='o', color='purple')
plt.title(f"Future Stock Price Prediction for {stock}")
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.grid(True)
st.pyplot(fig_future)
