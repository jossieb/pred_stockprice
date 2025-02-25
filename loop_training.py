# The script loops through all the rundata in model_vars.json
#
# Author: Jos van der Have
# Date: 2025 Q1
# Version: 1.0  /  22-02-2025
# license: MIT
# Example: python loop_training.py DGTL.MI local
# When Yahoo is in error try: pip install --upgrade yfinance
# =======================================================
import json
import argparse
import subprocess
import pandas as pd
import sys


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a stock price prediction model.")
parser.add_argument(
    "symbol", type=str, help="Stock symbol to retrieve historical data for"
)
parser.add_argument(
    "data_type", type=str, help="location for data = yahoo or local(default)"
)

args = parser.parse_args()

symbol = args.symbol  # Use the stock symbol provided as a command-line argument
data_type = args.data_type  # Use the data type

if not symbol:
    print("Please provide a stock symbol as a command-line argument.")
    exit(1)
if not data_type:
    print("Please provide a data type (local or yahoo) as a command-line argument.")
    exit(1)

# Stap 1: JSON inlezen
with open("model_vars.json", "r") as f:  # Pas de bestandsnaam aan indien nodig
    data = json.load(f)

# Stap 2: Runs ophalen
runs = data.get("model_vars", {})  # Dictionary van runs


# Stap 3: Voor elke run training.py starten
for run_name, run_data in runs.items():

    run_value = next(
        (item["value\r"].strip() for item in run_data if item["key"] == "run"), None
    )
    print(
        f"🔄 Start training voor {run_name} en waarde ... {symbol} {data_type} {run_value}"
    )

    # Stap 4: Start training.py als subprocess
    # venv_python = r"venv\Scripts\python.exe"  # Voor Windows
    result = subprocess.run(
        [sys.executable, "training.py", symbol, data_type, run_value],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Fout bij run {run_name}: {result.returncode}")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
    else:
        # Stap 5: Toon output (optioneel, handig voor debugging)
        print(f"✅ run {run_name} voltooid.")
        print(f"Output: {result.stdout}")

print("🎉 Alle runs zijn uitgevoerd!")

# Unieke variabelen ophalen in de originele volgorde uit de eerste run
first_run_key = next(iter(runs))  # Eerste key pakken (bijv. 'run1')
ordered_keys = [item["key"] for item in runs[first_run_key]]  # Volgorde behouden

# Data opslaan met de juiste volgorde
data_dict = {"Name": ordered_keys}

for run_name, run_data in runs.items():
    values = {
        item["key"]: item["value\r"].strip() for item in run_data
    }  # "\r" en witruimte verwijderen
    data_dict[run_name] = [
        values.get(key, "N/A") for key in ordered_keys
    ]  # "N/A" als key ontbreekt

# Data omzetten naar Pandas DataFrame
df = pd.DataFrame(data_dict)

# Print tabel
print(df)
df.to_csv("mv_output.csv", index=False)
