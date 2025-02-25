
import yfinance as yf
from textblob import TextBlob
import pandas as pd
import requests
import pandas as pd
from datetime import datetime, timedelta

# Definieer het aandeel
symbol = 'ADRNY'

# Haal het aandeelobject op
#stock = yf.Ticker(symbol)

# Haal de nieuwsartikelen op
#news = stock.news

# Functie om sentimentanalyse uit te voeren
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def get_yahoo_news_rss(symbol, years=5):
    """
    Haal nieuwsberichten op van de laatste X jaar voor een aandeel via de Yahoo Finance RSS-feed.
    """
    # Bereken de startdatum (vandaag - X jaar)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # URL voor de Yahoo Finance RSS-feed
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

    # Haal de RSS-feed op
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Kon de RSS-feed niet ophalen. Statuscode: {response.status_code}")

    # Parse de RSS-feed (eenvoudige XML-parsing)
    from xml.etree import ElementTree as ET
    root = ET.fromstring(response.content)

    # Verzamel nieuwsberichten
    news_items = []
    for item in root.findall(".//item"):
        title = item.find("title").text
        link = item.find("link").text
        pub_date = item.find("pubDate").text
        pub_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")

        # Filter berichten op basis van de startdatum
        if pub_date >= start_date:
            news_items.append({
                "title": title,
                "link": link,
                "pub_date": pub_date.strftime("%Y-%m-%d %H:%M:%S")
            })

    return pd.DataFrame(news_items)

# Voeg sentimentanalyse toe aan de nieuwsartikelen
#for article in news:
    # Haal de 'summary' op uit de 'content' dictionary
    #summary = article['content'].get('summary', '')
    #article['sentiment'] = analyze_sentiment(summary)
    # Haal de datum uit de 'pubDate'
    #pdate = article['content'].get('pubDate', '')
    #article['date'] = pdate[0:10]

# Converteer de nieuwsartikelen naar een DataFrame
#news_df = pd.DataFrame(news)

# Groepeer de artikelen per datum en bereken het gemiddelde sentiment
#average_sentiment_per_date = news_df.groupby('date')['sentiment'].mean()

# Toon het gemiddelde sentiment per datum
#print(average_sentiment_per_date)

# Voorbeeld: Haal nieuwsberichten op voor AHOLD
news_df = get_yahoo_news_rss(symbol, years=5)
print(news_df)