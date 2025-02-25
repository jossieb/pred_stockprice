"""
Predict stock prices using a trained LSTM model.
This script loads a pre-trained model and scaler, preprocesses new data,
and generates predictions for stock prices.

Author: Jos van der Have aka jossieb
Date: 2025 Q1
Version: 4.0 25-02-2025
License: MIT
Example: python predict.py DGTL.MI local 1
"""

import warnings

warnings.filterwarnings("ignore")

import os

# Suppress warnings and TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import argparse
import numpy as np
import pandas as pd
import talib
import joblib
import yfinance as yf
from datetime import date, timedelta
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import local
import json


def parse_arguments():
    """Parse command-line arguments for stock symbol, data type, and run number."""
    parser = argparse.ArgumentParser(
        description="Predict stock prices using a trained model."
    )
    parser.add_argument("symbol", type=str, help="Stock symbol to predict prices for")
    parser.add_argument(
        "data_type", type=str, help="Location for data: 'yahoo' or 'local'"
    )
    parser.add_argument("run_nr", type=str, help="Run number (whole number)")
    return parser.parse_args()


def load_data(symbol, data_type):
    """Load stock data from either local storage or Yahoo Finance."""
    if data_type == "local":
        data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
        data = pd.read_csv(data_path)
        print(f"Data retrieved from: {data_path}")
    else:
        stock = yf.Ticker(symbol)
        end_date = date.today()
        start_date = end_date - timedelta(days=1000)
        data = stock.history(start=start_date, end=end_date)
        data.columns = data.columns.str.lower()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        print("Data retrieved from Yahoo Finance")
    return data


def add_technical_indicators(data):
    """Add technical indicators to the DataFrame."""
    if len(data) < 52:
        raise ValueError("Not enough data to calculate technical indicators")

    indicators = {
        "obv": talib.OBV(data["close"], data["volume"]),
        "rsi": talib.RSI(data["close"], timeperiod=14),
        "macd": talib.MACD(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)[
            0
        ],
        "adx": talib.ADX(data["high"], data["low"], data["close"], timeperiod=14),
        "sma_14": talib.SMA(data["close"], timeperiod=14),
        "ema_14": talib.EMA(data["close"], timeperiod=14),
        "bollinger_high": talib.BBANDS(
            data["close"], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0
        )[0],
        "bollinger_low": talib.BBANDS(
            data["close"], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0
        )[2],
        "stoch_oscillator": talib.STOCH(
            data["high"],
            data["low"],
            data["close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )[0],
    }

    data = data.assign(**indicators).bfill().ffill()
    # data = data.assign(**indicators).fillna(method="bfill").fillna(method="ffill") #deprecated

    return data


def preprocess_data(data, scaler, features, seq_length):
    """Preprocess data for prediction using the same scaler as during training."""
    data_subset = data[features].dropna()
    data_scaled = scaler.fit_transform(data_subset)

    x, y = [], []
    for i in range(len(data_scaled) - seq_length):
        x.append(data_scaled[i : i + seq_length])
        y.append(data_scaled[i + seq_length, features.index("close")])
    return np.array(x), np.array(y)


# Haal de waarde van "weight" op
def get_value_by_key(run_data, key):
    """Haal de waarde op voor een gegeven key uit de run-data."""
    for item in run_data:
        if item["key"] == key:
            return item["value\r"].strip()  # Verwijder \r en spaties
    return None


@register_keras_serializable()
def weighted_mse_exp(y_true, y_pred):
    global run_nr  # Declare run_nr as global to access it inside this function

    # Parse de JSON-data
    with open("model_vars.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Navigeer naar de run-data
    run_key = f"run{run_nr}"
    run_data = data["model_vars"][run_key]

    # Haal de waarde van "weight" op
    weight_value = get_value_by_key(run_data, "weight")

    # Converteer naar float
    if weight_value is not None:
        weight_value = float(weight_value)
        print(f"Value of 'weight' is: {weight_value}")
    else:
        print("Key 'weight' not found in JSON file.")

    """Custom loss function with exponential weighting for recent data."""
    seq_length = tf.shape(y_true)[0]
    weights = tf.exp(weight_value * tf.range(seq_length, dtype=tf.float32))
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


def load_model_and_scaler(symbol, run_nr):
    """Load the trained model and scaler from disk."""
    model_path = os.path.join(local.lfolder, f"{symbol}_{local.lmodel}")
    scaler_path = os.path.join(local.lfolder, f"{symbol}_{local.lscaler}")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler not found for symbol {symbol} and run {run_nr}.")
        sys.exit(1)

    custom_objects = {"weighted_mse_exp": weighted_mse_exp}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    return model, scaler


def generate_predictions(model, scaler, x, y, seq_length):
    global symbol
    global features

    """Make predictions"""

    predictions = model.predict(x)

    # Create an array with the same shape as the original data used for scaling
    predictions_full = np.zeros((predictions.shape[0], len(features)))
    predictions_full[:, features.index("close")] = predictions[:, 0].flatten()

    # Inverse scale the predictions
    predictions_rescaled = scaler.inverse_transform(predictions_full)[
        :, features.index("close")
    ]

    # Get the last date from the data
    today = date.today()

    # Calculate the next working day
    next_working_day = today + timedelta(days=1)
    while next_working_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_working_day += timedelta(days=1)

    # Save predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": [next_working_day],
            "predicted_close": [predictions_rescaled[-1]],
        }
    )

    print(
        "Predicted price for the next working day: ",
        [next_working_day],
        predictions_rescaled[-1],
    )

    if not os.path.exists(os.path.join(local.dfolder, local.predfile)):
        predictions_df.to_csv(os.path.join(local.dfolder, local.predfile), index=False)

    else:
        predictions_df.to_csv(
            os.path.join(local.dfolder, local.predfile),
            mode="a",
            header=False,
            index=False,
        )
    print(f"Prediction saved to: {os.path.join(local.dfolder, local.predfile)}")

    return predictions_rescaled[-1]


def main():
    # Declare run_nr, symbol and features as global to access it inside other functions
    global run_nr
    global symbol
    global features

    args = parse_arguments()
    symbol, data_type, run_nr = args.symbol, args.data_type, args.run_nr

    if not all([symbol, data_type, run_nr]):
        print("Please provide all required arguments: symbol, data_type, and run_nr.")
        sys.exit(1)

    print(
        f"Execute prediction for -> Symbol: {symbol}, Type: {data_type}, Run: {run_nr}"
    )

    data = load_data(symbol, data_type)
    data = add_technical_indicators(data)
    data["daily_return"] = data["close"].pct_change()

    features = [
        "close",
        "daily_return",
        "obv",
        "rsi",
        "macd",
        "adx",
        "sma_14",
        "ema_14",
        "bollinger_high",
        "bollinger_low",
        "stoch_oscillator",
    ]

    model, scaler = load_model_and_scaler(symbol, run_nr)
    x, y = preprocess_data(data, scaler, features, seq_length=40)
    predicted_close = generate_predictions(model, scaler, x, y, seq_length=40)


if __name__ == "__main__":
    main()
