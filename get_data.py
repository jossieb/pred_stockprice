# =======================================================
# get_data.py - Get data from Yahoo to local
# =======================================================
#
# Author: Jos van der Have
# Date: 2025 Q1
# Version: 1.0
# license: MIT
# Example: python get_data.py DGTL.MI
# =======================================================
import warnings

warnings.filterwarnings("ignore")

import local
import argparse
import os
import pandas as pd
import yfinance as yf
from datetime import date, timedelta


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Get stock prices from Yahoo.")
parser.add_argument(
    "symbol", type=str, help="Stock symbol to retrieve historical data for"
)

args = parser.parse_args()

symbol = args.symbol  # Use the stock symbol provided as a command-line argument


if not symbol:
    print("Please provide a stock symbol as a command-line argument.")
    exit(1)

# Load data from Yahoo
stock = yf.Ticker(symbol)  # Use yfinance to get more data
data = stock.history(period="max")
data = pd.DataFrame(data)
data.columns = data.columns.str.lower()  # Ensure all column names are lowercase
data.index = pd.to_datetime(data.index)
data = data.sort_index()
print("Data retrieved from Yahoo Finance")

myfile = symbol + "_" + local.csvfile
data.to_csv(os.path.join(local.dfolder, myfile), index=True)
print(f"Data saved to: {os.path.join(local.dfolder, myfile)}")
