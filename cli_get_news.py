"""
Get financial news items and use the daily sentiment als input for the model

Author: Jos van der Have aka jossieb
Date: 2025 Q1
Version: 5.0 07-03-2025
License: MIT
Example: python get_news.py DGTL.MI 100
"""

import yfinance as yf
import pandas as pd
from textblob import TextBlob
import local
import os
from datetime import datetime, timezone
import argparse


def parse_arguments():
    """Parse command-line arguments for stock symbol, nr of news items, and run number."""
    parser = argparse.ArgumentParser(
        description="Predict stock prices using a trained model."
    )
    parser.add_argument("symbol", type=str, help="Stock symbol to predict prices for")
    parser.add_argument(
        "items", type=str, help="Max. number of newsitems (whole number)"
    )
    return parser.parse_args()


def get_news_sentiment(symbol, items):
    # get list of news
    news = yf.Search(symbol, items).news
    news = pd.DataFrame(news)
    # drop niet gebruikte kolommen
    news = news.drop(["uuid", "publisher", "thumbnail", "relatedTickers"], axis=1)

    # Bepaal het titel sentiment (polarity loopt van -1 tot 1)
    for index, row in news.iterrows():
        title = row["title"]
        blob_title = TextBlob(title)
        news.at[index, "sentiment"] = blob_title.sentiment.polarity

    # Controleer of 'providerPublishTime' aanwezig is in de DataFrame
    if "providerPublishTime" in news.columns:
        # Voeg een kolom toe met de publicatiedatum
        news["pub_date"] = news["providerPublishTime"].apply(
            lambda x: datetime.fromtimestamp(x, timezone.utc).strftime(
                "%Y-%m-%d 00:00:00+01:00"
            )
        )
    else:
        print("'providerPublishTime' niet gevonden in nieuwsgegevens.")

    return news


def store_news_data(news_sent, data_path):

    try:
        # Lees de bestaande CSV in een DataFrame
        existing_df = pd.read_csv(data_path)
        # Voeg de nieuwe DataFrame toe aan de bestaande DataFrame en sla op als csv
        combi_df = pd.concat([existing_df, news_sent], ignore_index=True)
        combi_df = combi_df.fillna(0)
        combi_df.to_csv(data_path, index=False)
        print(f"Nieuws toegevoegd aan:", data_path)
    except:
        # maak een nieuwe csv
        news_sent = news_sent.fillna(0)
        news_sent.to_csv(data_path, index=False)
        print(f"Nieuws file gemaakt:", data_path)

    return


def store_avg_data(symbol, sent_path):
    # Groepeer DataFrame op de datumkolom en bereken het gemiddelde sentiment
    news_date = pd.read_csv(sent_path)
    avg_sent = news_date.groupby("pub_date")["sentiment"].mean().reset_index()
    # sla de het gemiddelde sentiment op
    sent2_path = os.path.join(local.dfolder, f"{symbol}_avg_{local.sentfile}")
    avg_sent["pub_date"] = pd.to_datetime(avg_sent["pub_date"])
    avg_sent["pub_date"] = avg_sent["pub_date"].dt.tz_localize(None)
    avg_sent.to_csv(sent2_path, index=False)
    print(f"Avg sentiment data saved to: {sent2_path}")


def main():
    args = parse_arguments()
    symbol, items = args.symbol, args.items

    if not all([symbol, items]):
        print("Please provide all required arguments: symbol, items")
        sys.exit(1)

    print(f"Get news for -> Symbol: {symbol}, nr of items: {items}")

    # bepaal sentiment op nieuws
    news_sent = get_news_sentiment(symbol, items)
    sent_path = os.path.join(local.dfolder, f"{symbol}_{local.sentfile}")
    # sla nieuws data op in een csv
    store_news_data(news_sent, sent_path)
    # vul stock data aan met nieuws sentiment
    store_avg_data(symbol, sent_path)


if __name__ == "__main__":
    main()
