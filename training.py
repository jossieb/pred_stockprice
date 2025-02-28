"""
Training script for a stock price prediction model using an LSTM neural network.
This script retrieves historical stock price data, adds technical indicators,
preprocesses the data, and trains the model using Keras Tuner for hyperparameter tuning.
The best model and scaler are saved for future use.

Author: Jos van der Have aka jossieb
Date: 2025 Q1
Version: 4.0 / 25-02-2025
License: MIT
Example: python training.py DGTL.MI local 1
"""

import sys
import warnings
import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import yfinance as yf
from datetime import date, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable
from kerastuner.tuners import RandomSearch
import tensorflow as tf
import local

# Suppress warnings and TensorFlow logging
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Parse command-line arguments
def parse_arguments():
    """Parse command-line arguments for stock symbol, data type, and run number."""
    parser = argparse.ArgumentParser(
        description="Train a stock price prediction model."
    )
    parser.add_argument(
        "symbol", type=str, help="Stock symbol to retrieve historical data for"
    )
    parser.add_argument(
        "data_type", type=str, help="Location for data: 'yahoo' or 'local'"
    )
    parser.add_argument("run_nr", type=str, help="Run number (whole number)")
    return parser.parse_args()


# Load parameters from JSON
def load_parameters(run_nr):
    """Load parameters for the specified run from a JSON file."""
    with open("model_vars.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    run_key = f"run{run_nr}"
    try:
        run_data = data["model_vars"][run_key]
        params = {}
        for item in run_data:
            key = item["key"]
            value = item["value\r"].strip()
            try:
                if "." in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value  # Keep as string if it cannot be converted
        return params
    except KeyError:
        print(f"Run {run_key} not defined in JSON file.")
        sys.exit(1)


# Custom loss function with exponential weighting
@register_keras_serializable()
def weighted_mse_exp(y_true, y_pred):
    """Custom loss function with exponential weighting for recent data."""
    seq_length = tf.shape(y_true)[0]
    weights = tf.exp(myweight * tf.range(seq_length, dtype=tf.float32))
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


# Load stock data
def load_data(symbol, data_type):
    """Load stock data from either local storage or Yahoo Finance."""
    if data_type == "local":
        data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
        data = pd.read_csv(data_path)
        print(f"Data retrieved from: {data_path}")
    else:
        stock = yf.Ticker(symbol)
        data = stock.history(period="max")
        data.columns = data.columns.str.lower()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        print("Data retrieved from Yahoo Finance")
        # save the data to a CSV
        
        myfile = symbol + "_" + local.csvfile
        data.to_csv(os.path.join(local.dfolder, myfile), index=True)
        print(f"Data saved to: {os.path.join(local.dfolder, myfile)}")

    return data.tail(myrows)


# Add technical indicators to the data
def add_technical_indicators(data):
    """Add technical indicators to the DataFrame."""
    if len(data) < 52:
        raise ValueError("Not enough data to calculate technical indicators")
    else:
        data["obv"] = talib.OBV(data["close"], data["volume"])
        data["rsi"] = talib.RSI(data["close"], timeperiod=14)
        data["macd"] = talib.MACD(
            data["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )[0]
        data["adx"] = talib.ADX(data["high"], data["low"], data["close"], timeperiod=14)
        data["sma_14"] = talib.SMA(data["close"], timeperiod=14)
        data["ema_14"] = talib.EMA(data["close"], timeperiod=14)
        data["bollinger_high"] = talib.BBANDS(
            data["close"], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0
        )[0]
        data["bollinger_low"] = talib.BBANDS(
            data["close"], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0
        )[2]
        data["stoch_oscillator"] = talib.STOCH(
            data["high"],
            data["low"],
            data["close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )[0]

    return data


# Preprocess data for training
def preprocess_data(data, seq_length):
    """Preprocess data by scaling, adding noise, and creating sequences."""
    data_subset = data[features].dropna()
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data_subset)
    noise = np.random.normal(0, mynoise, data_scaled.shape)
    data_scaled_noisy = data_scaled + noise

    x, y = [], []
    for i in range(len(data_scaled_noisy) - seq_length):
        x.append(data_scaled_noisy[i : i + seq_length])
        y.append(data_scaled_noisy[i + seq_length, features.index("close")])

    return np.array(x), np.array(y), scaler


# Build model for hyperparameter tuning
def build_model(hp):
    """Build an LSTM model with hyperparameters for tuning."""
    inputs = Input(shape=(40, len(features)))
    x = LSTM(hp.Int("units1", 50, 256, step=64), return_sequences=True)(inputs)
    x = Attention()([x, x])
    x = Dropout(mydropout)(x)
    x = LSTM(hp.Int("units2", 32, 128, step=32))(x)
    outputs = Dense(1, kernel_regularizer=l2(0.01))(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("lr", mylearning_rate, 1e-3)),
        loss=weighted_mse_exp,
        metrics=["mse"],
    )
    return model


# Plot training history
def plot_training_history(history, run_nr):
    """Plot training and validation loss."""
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(f"accuracy_{run_nr}.png", dpi=300)
    plt.close()


# Main function
def main():
    args = parse_arguments()
    symbol, data_type, run_nr = args.symbol, args.data_type, args.run_nr

    if not all([symbol, data_type, run_nr]):
        print("Please provide all required arguments: symbol, data_type, and run_nr.")
        sys.exit(1)

    print(f"Symbol: {symbol}, Type: {data_type}, Run: {run_nr}")

    global myrun, myweight, myrows, mytresh, myseq_length, mynoise, mydropout, mylearning_rate, mytest_size, mymax_trials, myexecutions_per_trial, mypatience, myfactor, mymin_lr, myepochs, mybatch_size
    params = load_parameters(run_nr)

    # Debugging: Print the keys and values of the params dictionary
    print("Loaded parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    (
        myrun,
        myweight,
        myrows,
        mytresh,
        myseq_length,
        mynoise,
        mydropout,
        mylearning_rate,
        mytest_size,
        mymax_trials,
        myexecutions_per_trial,
        mypatience,
        myfactor,
        mymin_lr,
        myepochs,
        mybatch_size,
    ) = params.values()

    data = load_data(symbol, data_type, myrows)
    data = add_technical_indicators(data)
    data["daily_return"] = data["close"].pct_change()

    global features
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

    x, y, scaler = preprocess_data(data, seq_length=40)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=mytest_size, shuffle=False
    )

    if os.path.exists("keras_tuner"):
        import shutil

        shutil.rmtree("keras_tuner")

    tuner = RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=mymax_trials,
        executions_per_trial=myexecutions_per_trial,
        directory="keras_tuner",
        project_name="stock_prediction",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=mypatience, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=myfactor, patience=mypatience, min_lr=mymin_lr
    )

    try:
        tuner.search(
            x_train,
            y_train,
            epochs=myepochs,
            batch_size=mybatch_size,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, reduce_lr],
        )
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        sys.exit(1)

    best_model = tuner.get_best_models(num_models=1)[0]
    history = best_model.fit(
        x_train,
        y_train,
        epochs=myepochs,
        batch_size=mybatch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr],
    )

    model_path = os.path.join(local.lfolder, f"{symbol}_{local.lmodel}")
    scaler_path = os.path.join(local.lfolder, f"{symbol}_{local.lscaler}")
    best_model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Best model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    plot_training_history(history, run_nr)


if __name__ == "__main__":
    main()
