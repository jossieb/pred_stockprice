import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Maak een DataFrame met de gegevens uit de tabel
data = {
    "Datum": [
        "250224",
        "250225",
        "250226",
        "250227",
        "250303",
        "250304",
        "250305",
        "250306",
    ],
    "Voorspeld": [
        "EUR 9,54",
        "EUR 9,97",
        "EUR 9,81",
        "EUR 10,03",
        "EUR 9,99",
        "EUR 9,95",
        "EUR 9,87",
        "EUR 9,64",
    ],
    "Werkelijk": [
        "EUR 10,16",
        "EUR 10,38",
        "EUR 10,40",
        "EUR 10,21",
        "EUR 10,28",
        "EUR 9,79",
        "EUR 9,73",
        "EUR ",
    ],
}

# Converteer naar een DataFrame
df = pd.DataFrame(data)

# Converteer de 'Datum'-kolom naar datetime
df["Datum"] = pd.to_datetime(df["Datum"], format="%y%m%d")

# Verwijder 'EUR ' en vervang komma's door punten in de valuta-kolommen
for col in ["Voorspeld", "Werkelijk"]:
    df[col] = df[col].str.replace("EUR ", "").str.replace(",", ".").astype(float)
# Converteer de 'Datum'-kolom naar strings (voor categorische x-as)
df["Datum_str"] = df["Datum"].dt.strftime("%Y-%m-%d")

# Plot de gegevens
plt.figure(figsize=(10, 6))
plt.plot(
    df["Datum_str"],
    df["Voorspeld"],
    label="Voorspeld",
    marker="x",
    linestyle="--",
)
plt.plot(
    df["Datum_str"], df["Werkelijk"], label="Werkelijk", marker="s", linestyle="-."
)

# Configureer de x-as als categorisch
plt.gca().set_xticks(df["Datum_str"])  # Gebruik alleen de datums uit de dataset
plt.gca().set_xticklabels(
    df["Datum_str"], rotation=45
)  # Draai datums voor betere leesbaarheid

# Voeg titel en labels toe
plt.title("Voorspelde vs Werkelijke Slotkoersen (zonder weekenden)")
plt.xlabel("Datum")
plt.ylabel("Koers (EUR)")
plt.legend()
plt.grid(True)

# Toon de plot
plt.tight_layout()  # Zorgt voor betere opmaak
plt.show()
