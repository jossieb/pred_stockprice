
# ML Aandelen koers voorspeller

**Het AI model als kristallen bol?**<br>
In het kader van mijn zoektocht naar antwoord op de vraag **Is AI magie
of zijn het gewoon wat regels Python code?** Dacht ik waarom wagen we
niet een poging om de heilige graal te ontdekken/ontwikkelen in de
aandelenhandel. Met al die beschikbare AI kennis die voor het oprapen
ligt moet het toch te doen zijn om een *aandelenkoers voorspeller* te
ontwikkelen?!

**Het idee**<br>
Met behulp van Python, AI-modellen, gratis beschikbare data en wat
coderen de koers van morgen voorspellen voor een specifiek aandeel (of
ETF).<br>
<br>
In bijgaande blog houd ik de ontwikkeling van het model bij.<br>
[Link naar blog](https://www.stillhaveit.nl/blog_stock)

---

![Kristallen bol](https://www.stillhaveit.nl/static/ball2.jpg "AI koersvoorspeller")

---

**Requirements**<br>
pip install -r requirements.txt

**Huidige versie 4**<br>
De DeepSeek beoordeling van het model. Het model toont de training en validatieverlies (loss) over verschillende epochs. Hier zijn enkele observaties en beoordelingen:<br>
- Trend van de Loss: Zowel de training als de validatieverlies nemen af naarmate het aantal epochs toeneemt. Dit is een goed teken, omdat het aangeeft dat het model leert en beter presteert op zowel de trainings- als de validatiedata.<br>
- Overfitting: Er lijkt geen duidelijke overfitting te zijn, omdat de validatieverlies niet significant begint te stijgen terwijl de trainingverlies blijft dalen. Overfitting zou zich manifesteren als een toenemende validatieverlies terwijl de trainingverlies blijft dalen.<br>
- Convergentie: De verlieswaarden lijken te convergeren naar een bepaald punt, wat aangeeft dat het model een stabiel niveau van prestaties bereikt. Dit is positief, omdat het suggereert dat het model niet verder hoeft te worden getraind om betere resultaten te behalen.<br>
<br>
**De onderdelen**<br>
- training.py<br>
    Traint het model obv Yahoo data of eerder opgehaalde data lokaal<br>
    Aanroep:    python training.py <br>
    parameters: symbool / AAPL (voorbeeld) <br>
                data / local of yahoo <br>
                run / Xe run nummer in het JSON bestand model_vars.json<br>
<br>
- loop_training.py<br>
    Runt de training op basis van het aantal items dat is beschreven in het JSON bestand model_vars.json<br>
    Aanroep:    python loop_training.py <br>
    parameters: symbool / AAPL (voorbeeld) <br>
                data / local of yahoo <br>
<br>
- predict.py<br>
    Gebruikt het model uit de training om een voorspelling te geven van de koers van de volgende werkdag.<br>
    Aanroep:    python predict.py <br>
    parameters: symbool / AAPL (voorbeeld) <br>
                data / local of yahoo <br>
                run / Xe run nummer in het JSON bestand model_vars.json<br> 
<br>
- test_shap.py<br>
    Genereert inzicht in de mate van belangrijkheid van de verschillende technische indicatoren.<br>
    Aanroep:    python predict.py <br>
    parameters: symbool / AAPL (voorbeeld) <br>
                data / local of yahoo <br>
                run / Xe run nummer in het JSON bestand model_vars.json<br> 
<br>
- get_data.py<br>
    Haalt de meest recente Yahoo data op voor een specifiek symbool (bv AAPL).<br>
    Handig bij het testen zodat je niet meerdere keren op een dag de yahoo data hoeft op te halen<br>
    Na de yahoo data opgehaald te hebben kan je vervolgens bij het trainen de parameter local gebruiken<br>
    Aanroep:    python predict.py <br>
    parameter(s): symbool / AAPL (voorbeeld) <br>
<br>
- get_news.py<br>
    Haalt de titels van nieuws items op bij yahoo met als zoekgegeven een specifiek symbool (bv AAPL). Op basis van
    de totel wordt vervolgens het sentiment (-1 tot 1) bepaald. Daarna wordt het gemiddelde sentiment van een dag
    bepaald over de opgehaalde nieuwsitems. Het gemiddelde sentiment wordt als indicator toegevoegd aan de stock_data.csv en gebruikt in het model. De opgehaalde items worden toegevoegd aan een lokale csv file (news_sentiment.csv) Deze methode is niet ideaal maar de inhoud van de nieuwsitems is niet gratis op te halen helaas.<br>
    <b>Deze nieuws sentiment functionaliteit is nog steeds 'work in progress'. In het huidige model heeft het nog geen effect.</b><br>
    Aanroep:    python get_news.py <br>
    parameter(s):   symbool / <br>
                    aantal nieuwsitems / hoeveelheid nieuws dat wordt opgehaald<br>

<br>
