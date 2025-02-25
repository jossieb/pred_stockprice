import warnings

warnings.filterwarnings("ignore")

import os

# Suppress warnings and TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import talib
from tensorflow.keras.saving import register_keras_serializable
import json
import local
import os
from sklearn.model_selection import train_test_split
import joblib


def load_data(symbol):
    """Load stock data from local."""

    data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
    data = pd.read_csv(data_path)
    data.tail(300)
    print(f"Data retrieved from: {data_path}")
    return data


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

    # Parse de JSON-data
    with open("model_vars.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Navigeer naar de run-data
    run_key = "run1"
    run_data = data["model_vars"][run_key]

    # Haal de waarde van "weight" op
    weight_value = get_value_by_key(run_data, "weight")

    # Converteer naar float
    if weight_value is not None:
        weight_value = float(weight_value)
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


# SHAP-explainer instellen
def shap_explanation(model, x_sample, x_train):
    print("Vergelijk de verschillende indicatoren op effectiviteit.")

    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

    x_train_2d = x_train.reshape(x_train.shape[0], -1)
    x_sample_2d = x_sample.reshape(x_sample.shape[0], -1)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x_sample)

    return shap_values


# ===========================================
# Laad het getrainde model
symbol = "DGTL.MI"
run_nr = "run1"

data = load_data(symbol)
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=5,
    validation_data=(x_test, y_test),
    verbose=0,
)

# Neem een subset van de testdata voor SHAP
x_train = x_train.astype(np.float32)
sample_size = 20  # Aantal samples voor SHAP (pas aan indien nodig)
x_sample = x_test[:sample_size]
x_sample = x_sample.astype(np.float32)

# Bereken SHAP-waarden
shap_values = shap_explanation(model, x_sample, x_train)
shap_values = np.squeeze(shap_values, axis=-1)

shap_values_mean = np.mean(shap_values, axis=1)

# SHAP summary plot
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_mean, x_sample[:, 0, :], feature_names=features)
save_path = "shap_summary_plot.png"
fig.savefig(save_path, dpi=300)
plt.show(fig)
plt.close(fig)  # Sluit de plot om geheugen vrij te maken
