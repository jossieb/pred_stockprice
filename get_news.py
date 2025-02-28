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
    parser.add_argument("items", type=str, help="Number of newsitems (whole number)")
    return parser.parse_args()


def get_news_sentiment(symbol, items):
    # Haal de ticker informatie op
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
        combi_df.to_csv(data_path, index=False)
        print(f"Nieuws toegevoegd aan:", data_path)
    except:
        # maak een nieuwe csv
        news_sent.to_csv(data_path, index=False)
        print(f"Nieuws file gemaakt:", data_path)

    return


def add_stock_data(symbol, data_path):
    # Groepeer DataFrame op de datumkolom en bereken het gemiddelde sentiment
    new_date = pd.read_csv(data_path)
    avg_sent = new_date.groupby("pub_date")["sentiment"].mean()
    print(avg_sent)

    # lees de lokale stockfile
    data_path = os.path.join(local.dfolder, f"{symbol}_{local.csvfile}")
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.lower()
    # Voeg het gemiddelde sentiment toe aan de lokale stockfile
    print()
    data = pd.merge(data, avg_sent, left_on="date", right_on="pub_date", how="left")

    # sla de uitgebreide stockfile op
    data.to_csv(data_path, index=False)
    print(f"Stock data saved to: {data_path}")


def main():
    args = parse_arguments()
    symbol, items = args.symbol, args.items

    if not all([symbol, items]):
        print("Please provide all required arguments: symbol, items")
        sys.exit(1)

    print(f"Get news for -> Symbol: {symbol}, nr of items: {items}")

    data_path = os.path.join(local.dfolder, f"{symbol}_{local.sentfile}")

    # bepaal sentiment op nieuws
    news_sent = get_news_sentiment(symbol, items)

    # sla nieuws data op in een csv
    store_news_data(news_sent, data_path)

    # vul stock data aan met nieuws sentiment
    add_stock_data(symbol, data_path)


if __name__ == "__main__":
    main()
