"""
Predict stock prices using a trained LSTM model.
This script loads a pre-trained model and scaler, preprocesses new data,
and generates predictions for stock prices.

Author: Jos van der Have aka jossieb
Date: 2025 Q1
Version: 6.0
License: MIT
Example: python predict.py DGTL.MI local 1
"""

import warnings

warnings.filterwarnings("ignore")

import os

# Suppress warnings and TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime, date, timedelta
import sys
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta

# import talib
import joblib
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import local
import json
import get_news
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        sys.exit(3)


def load_data(symbol, data_type, myrows):
    """Load stock data from either local storage or Yahoo Finance."""
    data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
    sent2_path = os.path.join(local.dfolder, f"{symbol}_avg_{local.sentfile}")
    # add latest news sentiment
    items = 1000
    get_news.yf_news(symbol, items)
    avg_sent = pd.read_csv(sent2_path)

    if data_type == "local":
        data = pd.read_csv(data_path)
        # Werk de bestaande DataFrame bij met waarden uit news_sent
        data.update(avg_sent)
        data["sentiment"] = data["sentiment"].fillna(0)
        data.to_csv(data_path, index=False)
        print(f"Data retrieved from: {data_path}")
    else:
        stock = yf.Ticker(symbol)
        data = stock.history(period="max")
        data = data.reset_index()
        data.columns = data.columns.str.lower()
        data["date"] = data["date"].dt.date
        # save the data to a CSV
        data.to_csv(data_path)
        print("Data retrieved from Yahoo Finance")

        # Voeg de DataFrames samen op basis van de datum
        avg_sent["pub_date"] = pd.to_datetime(avg_sent["pub_date"])
        data["date"] = pd.to_datetime(data["date"])
        # Verwijder tijdzone-informatie (als deze aanwezig is)
        avg_sent["pub_date"] = avg_sent["pub_date"].dt.tz_localize(None)
        data["date"] = data["date"].dt.tz_localize(None)
        data = pd.merge(data, avg_sent, left_on="date", right_on="pub_date", how="left")
        # Verwijder de overbodige kolom 'pub_date' (deze is nu gelijk aan 'date')
        data = data.drop(columns=["pub_date"])
        data["sentiment"] = data["sentiment"].fillna(0)
        # sla de nieuwe file op
        data.to_csv(data_path, index=False)

    return data.tail(myrows)


def add_technical_indicators(data):
    """Add technical indicators to the DataFrame."""
    if len(data) < 52:
        raise ValueError("Not enough data to calculate technical indicators")
    else:
        # On-Balance Volume (OBV)
        data["obv"] = ta.obv(data["close"], data["volume"])
        # Relative Strength Index (RSI)
        data["rsi"] = ta.rsi(data["close"], length=14)
        # Moving Average Convergence Divergence (MACD)
        macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
        data["macd"] = macd["MACD_12_26_9"]
        # Average Directional Index (ADX)
        adx = ta.adx(data["high"], data["low"], data["close"], length=14)
        data["adx"] = adx["ADX_14"]
        # Simple Moving Average (SMA)
        data["sma_14"] = ta.sma(data["close"], length=14)
        # Exponential Moving Average (EMA)
        data["ema_14"] = ta.ema(data["close"], length=14)
        # Bollinger Bands
        bbands = ta.bbands(data["close"], length=14, std=2)
        data["bollinger_high"] = bbands["BBU_14_2.0"]
        data["bollinger_low"] = bbands["BBL_14_2.0"]
        # Stochastic Oscillator
        stoch = ta.stoch(data["high"], data["low"], data["close"], k=14, d=3)
        data["stoch_oscillator"] = stoch["STOCHk_14_3_3"]

    return data


def preprocess_data(data, scaler, features, seq_length):
    """Preprocess data for prediction using the same scaler as during training."""
    data_subset = data[features].dropna()
    data_scaled = scaler.fit_transform(data_subset)

    x, y_close, y_trend = [], [], []
    for i in range(len(data_scaled) - seq_length):
        x.append(data_scaled[i : i + seq_length])
        y_close.append(data_scaled[i + seq_length, features.index("close")])
        y_trend.append(data_scaled[i + seq_length, features.index("daily_trend")])
    return np.array(x), np.array(y_close), np.array(y_trend)


@register_keras_serializable()
def weighted_mse_exp(y_true, y_pred):
    global run_nr, myweight

    # Converteer naar float
    if myweight is not None:
        myweight = float(myweight)
    else:
        print("Key 'weight' not found in JSON file.")

    """Custom loss function with exponential weighting for recent data."""
    seq_length = tf.shape(y_true)[0]
    weights = tf.exp(myweight * tf.range(seq_length, dtype=tf.float32))
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


def load_model_and_scaler(symbol, run_nr):
    """Load the trained model and scaler from disk."""
    model_path = os.path.join(local.lfolder, f"{symbol}_{local.lmodel}")
    scaler_path = os.path.join(local.lfolder, f"{symbol}_{local.lscaler}")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler not found for symbol {symbol} and run {run_nr}.")
        sys.exit(4)

    custom_objects = {"weighted_mse_exp": weighted_mse_exp}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    return model, scaler


def generate_predictions(
    data, model, scaler, x, y_close, y_trend, next_working_day, seq_length
):
    global symbol, features

    """Make predictions"""

    predictions = model.predict(x)
    predictions_close = predictions[0]
    predictions_trend = predictions[1]

    # Create an array with the same shape as the original data used for scaling
    predictions_full = np.zeros((predictions_close.shape[0], len(features)))
    predictions_full[:, features.index("close")] = predictions_close.flatten()
    predictions_full[:, features.index("daily_trend")] = predictions_trend.flatten()

    # Inverse scale the predictions
    predictions_rescaled = scaler.inverse_transform(predictions_full)
    # Get the last prediction
    predicted_close = predictions_rescaled[-1, features.index("close")]
    predicted_trend = predictions_rescaled[-1, features.index("daily_trend")]

    # bepaal verschil laatste met voorspelde
    last_close = data["close"].values[-1]
    verschil = predicted_close - last_close
    verandering = (verschil / predicted_close) * 100  # % verandering

    # Converteer next_working_day naar een string in het formaat y-m-d
    next_working_day_str = next_working_day.strftime("%Y-%m-%d")

    # Save predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": [next_working_day_str],
            "predicted_close": [predicted_close],
            "verandering": [verandering],
        }
    )

    print(
        "Predicted price for the next working day:",
        next_working_day_str,
        "Eur:",
        predicted_close,
    )
    print(
        "Difference in percentage for the next working day:",
        next_working_day_str,
        "%:",
        verandering,
    )

    pred_path = os.path.join(local.dfolder, f"{symbol}_{local.predfile}")

    if not os.path.exists(pred_path):
        predictions_df.to_csv(pred_path, index=False)

    else:
        predictions_df.to_csv(
            os.path.join(pred_path),
            mode="a",
            header=False,
            index=False,
        )
    print(f"Prediction saved to: {pred_path}")

    # beeld de stijging/daling af in een kleurenspectrum
    # Definieer het kleurenspectrum
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["red", "green"])

    # Verkrijg de kleur uit het kleurenspectrum
    color = cmap(verandering)

    # Maak een figuur en assen
    fig, ax = plt.subplots(figsize=(10, 3))
    # Voeg titel toe
    ax.set_title(
        "Voorspelde Koersverandering in %", fontsize=16, fontweight="bold", pad=20
    )

    # Toon het constante kleurenspectrum als achtergrond
    gradient = np.linspace(0, 1, 256)
    ax.imshow(np.array([gradient]), aspect="auto", cmap=cmap, extent=[-10, 10, 0, 1])
    # Voeg labels toe
    ax.text(-6, 1.05, "DALEN", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(
        0,
        1.05,
        "NEUTRAAL",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        6, 1.05, "STIJGEN", ha="center", va="center", fontsize=12, fontweight="bold"
    )
    # Toon de trendwaarde als een verticale lijn
    ax.axvline(x=verandering, color="black", linestyle="--", linewidth=2)

    # Voeg trendwaarde, datum en koers toe als tekst
    ax.text(
        verandering,
        0.5,
        f"Koers: EUR {predicted_close:.2f}\nVerandering: {verandering:.2f}%\nDatum: {next_working_day_str}",
        ha="center",
        va="center",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Verwijder assen en ticks
    ax.set_yticks([])
    ax.set_ylim(0, 1)

    # Pas x-as aan
    ax.set_xlim(-10, 10)
    ax.set_xticks([-10, 0, 10])  # Toon x-as markeringen

    # Verwijder de randen van de plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()

    plot_path = os.path.join(local.dfolder, f"{symbol}_nextday_trend.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return predicted_close, predicted_trend


def plot_predict_reality(symbol, next_working_day):
    data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
    pred_path = os.path.join(local.dfolder, f"{symbol}_{local.predfile}")

    # Converteer next_working_day naar een string in het formaat y-m-d
    next_working_day_str = next_working_day.strftime("%Y-%m-%d")

    # Lees de gegevens uit de tekstbestanden
    df_predicted = pd.read_csv(pred_path)
    df_actual = pd.read_csv(data_path)
    # Zet de datumkolommen om in datetime-objecten
    df_predicted["date"] = pd.to_datetime(df_predicted["date"])
    df_actual["date"] = pd.to_datetime(df_actual["date"])
    # Filter de gegevens vanaf 2025-01-01
    df_predicted = df_predicted[df_predicted["date"] >= "2025-03-01"]
    df_actual = df_actual[df_actual["date"] >= "2025-03-01"]
    # Maak de plot
    plt.figure(figsize=(12, 6))

    plt.plot(
        df_predicted["date"],
        df_predicted["predicted_close"],
        label="Voorspelde slotkoers",
    )
    plt.plot(df_actual["date"], df_actual["close"], label="Werkelijke slotkoers")
    # Voeg de voorspelde waarde voor next working day handmatig toe
    plt.scatter(
        pd.to_datetime(next_working_day_str),
        df_predicted[df_predicted["date"] == pd.to_datetime(next_working_day_str)][
            "predicted_close"
        ].values[0],
        color="blue",
    )

    # Voeg labels en een titel toe
    plt.xlabel("Datum")
    plt.ylabel("Slotkoers")
    plt.title("Voorspelde vs. Werkelijke slotkoers")

    # Voeg een legenda toe
    plt.legend()
    # Stel de x-as expliciet in

    plt.xlim(pd.to_datetime("2025-03-03"), df_predicted["date"].max())
    plt.grid(True)
    plot_path = os.path.join(local.dfolder, f"{symbol}_predict_reality.png")

    plt.savefig(plot_path, dpi=300)
    plt.close()


def main():
    # Declare run_nr, symbol and features as global to access it inside other functions
    global run_nr, symbol, features, myweight

    args = parse_arguments()
    symbol, data_type, run_nr = args.symbol, args.data_type, args.run_nr

    if not all([symbol, data_type, run_nr]):
        print("Please provide all required arguments: symbol, data_type, and run_nr.")
        sys.exit(1)

    if data_type == "local":
        data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
        if not os.path.exists(data_path):
            print(f"Local file {data_path} not found.")
            sys.exit(2)

    print(
        f"Execute prediction for -> Symbol: {symbol}, Type: {data_type}, Run: {run_nr}"
    )

    params = load_parameters(run_nr)

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
    # add the daily trend (up/down)
    max_abs_change = data["daily_return"].abs().max()
    data["daily_trend"] = data["daily_return"] / max_abs_change
    data["daily_trend"] = data["daily_trend"].fillna(0)

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
        "sentiment",
        "daily_trend",
    ]

    model, scaler = load_model_and_scaler(symbol, run_nr)
    x, y_close, y_trend = preprocess_data(data, scaler, features, seq_length=40)

    # Get the last date from the data
    data = data.reset_index()  # use date as regular column to get the last one in
    most_recent_date = data["date"].max()
    print(f"Most recent stock date in file:", most_recent_date)
    # Converteer de string naar een datetime-object
    # Verwijder de tijdcomponent
    most_recent_date_str = str(most_recent_date).split()[0]
    most_recent_date = datetime.strptime(str(most_recent_date_str), "%Y-%m-%d")

    # Calculate the next working day
    next_working_day = most_recent_date + timedelta(days=1)
    while next_working_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_working_day += timedelta(days=1)

    generate_predictions(
        data,
        model,
        scaler,
        x,
        y_close,
        y_trend,
        next_working_day,
        seq_length=40,
    )

    plot_predict_reality(symbol, next_working_day)


if __name__ == "__main__":
    main()
