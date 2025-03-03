
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

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

# Handle MultiIndex issue for accessing 'Close' prices
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

# Load trained LSTM model
model_path = r"C:\Users\ROG\OneDrive\Desktop\stock_price_prediction-main\project\Latest_stock_price_model.keras"

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

# Prepare DataFrame for results
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_predictions.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# Display Results
st.subheader("📊 Original vs Predicted Values")
st.write(ploting_data)

# Plot predictions vs actual data
st.subheader("📉 Actual Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data[close_col][:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data - Not Used for Training", "Original Test Data", "Predicted Test Data"])
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Stock Price Prediction for {stock}")
st.pyplot(fig)
