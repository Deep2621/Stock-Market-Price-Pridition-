import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# Streamlit title
st.title("Stock Market Price Prediction using LSTM")

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol")

# Function to fetch stock data
def fetch_stock_data(stock_symbol, start_date="2020-01-01"):
    stock_data = yf.download(stock_symbol, start=start_date, interval="1d")
    return stock_data

# LSTM Sequence Creation
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Predict Function
def predict_stock_price(stock_symbol):
    # Fetch data
    df = fetch_stock_data(stock_symbol)
    
    # Preprocess Data
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Create sequences
    seq_length = 50
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)

    # Make Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot Graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_actual, color='blue', label='Actual Prices')
    ax.plot(predictions, color='red', label='Predicted Prices')
    ax.set_title(f'{stock_symbol} Stock Price Prediction')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Price')
    ax.legend()

    return fig

# Button to trigger prediction
if st.button("Predict Stock Prices"):
    st.write(f"Fetching Data for {stock_symbol}...")
    
    # Plot and display graph
    fig = predict_stock_price(stock_symbol)
    st.pyplot(fig)
