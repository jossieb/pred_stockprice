
# ML Geld Machine

**Het AI model als kristallen bol?**

In het kader van mijn zoektocht naar antwoord op de vraag **Is AI magie
of zijn het gewoon wat regels Python code?** Dacht ik waarom wagen we
niet een poging om de heilige graal te ontdekken/ontwikkelen in de
aandelenhandel. Met al die beschikbare AI kennis die voor het oprapen
ligt moet het toch te doen zijn om een *aandelenkoers voorspeller* te
ontwikkelen?!

In dit blogitem wil ik wekelijks bijhouden welke vorderingen het model
maakt. Het doel is vergelijkbaar met de zoektocht naar de \"steen der
wijzen\" die gewone metalen in goud zou kunnen veranderen, een doel dat
uiteindelijk onbereikbaar bleek. Maar\... veel mensen zoeken naar de
\"heilige graal\" van koersvoorspelling, een model dat perfecte
voorspellingen kan doen, wat ook onbereikbaar lijkt gezien de inherente
onvoorspelbaarheid van de financiële markten. Bedenk echter dat hoewel
perfecte voorspellingen onbereikbaar zijn, AI-modellen nuttige inzichten
kunnen bieden en helpen bij besluitvorming, zelfs als ze niet altijd
perfect zijn. We zullen zien hoe ver we komen!

Als je zo dit blogitem doorleest dan krijg je misschien het idee dat ik
een *Quant* ben, een quantitative analist. Ik moet echter bekennen dat
een flink aandeel van mijn kennis op gebied van Machine learning te
danken is aan mijn gebruik van AI hulpmiddelen zoals *Github co-pilot*.
Dus bij het bouwen van de kristallen bol is AI al een geweldig
hulpmiddel!. Ik denk dat je wel 50% aan kennis en efficiency verdient
door er handig gebruik van te maken. De stelling dat je beter niet meer
kunt leren programmeren lijkt mij steeds meer juist als je bedenkt dat
we nog maar in het prille begin van het AI tijdperk zitten.

**Het idee**\
Met behulp van Python, AI-modellen, gratis beschikbare data en wat
coderen de koers van morgen voorspellen voor een specifiek aandeel (of
ETF).

**De Ingrediënten**

Wat Python code

Gratis beschikbaarheid van koers data

Gratis beschikbaarheid van aanvullende economische data

Open source AI-modellen

Aanvullende AI-hulp (Co-pilot, Deepseek, ChatGPT, Claude 3.5 Sonnet, Le
Chat)

Wat kennis van bovenstaande

Als je het chronologisch wilt volgen kan je best bij week 7 beginnen met
lezen.

![](https://www.stillhaveit.nl/static/ball.jpg)

------------------------------------------------------------------------

**Versie 3, week 8**

Allereerst is de dataleverancier veranderd. De gegevens worden nu
opgehaald vanuit Yahoo Finance. De belangrijkste reden is dat Yahoo niet
alleen US data levert maar ook data van Euronext, ook handig is dat de
gewenste data in Euro\'s is in plaats van Dollars. Nog een voordeel is
dat ook koersen van ETF\'s beschikbaar zijn waar ik meer gebruik van
maak dan van aandelenkoersen.

Als nieuwe bron van AI kennis heb ik ook [Le
Chat](https://chat.mistral.ai/){target="_blank" alt="Le Chat"} ingezet.
Het Franse model dat, volgens mijn eerste indruk, niet onder doet voor
ChatGPT of Deepseek.

De volgende verbeteringen hebben te maken met het zwaarder laten wegen
van de meest recente gegevens ten opzichte van gegevens die ouder zijn.
Om dit te realiseren zijn de volgende verbeteringen doorgevoerd:

Er zijn een aantal technische indicatoren toegevoegd om maar zoveel
mogelijk, relevante, gegevens te vinden. Toegevoegd zijn bijvoorbeeld:
SMA (Simple Moving Average) en EMA (Exponential Moving Average).\
De gebruite features zijn nu: close, daily_return, obv, williams_r, roc,
rsi, macd, adx, aroon_up, aroon_down, sma_7, ema_7, sma_14, ema_14,
bollinger_high, bollinger_low, atr, keltner_high, keltner_low, cmf, cci,
stoch_oscillator, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
chikou_span, vwap, fib_236, fib_382, fib_500 en fib_618.


Er is een Gewogen Verliesfunctie ingezet: Deze berekent een gewogen Mean
Squared Error (MSE), waarbij recente fouten zwaarder worden gewogen dan
oudere fouten.

Een andere grote wijziging is het inzetten van een **tuner**, deze Keras
Tuner wordt gebruikt om de beste hyperparameters te vinden. Er worden
callbacks toegevoegd voor early stopping en het verlagen van de learning
rate. Het beste model dat mbv de tuner is gevonden wordt geëvalueerd op
de testset en vervolgens opgeslagen. Ook de scaler wordt opgeslagen voor
toekomstig gebruik. Model en scaler worden vervolgens gebruikt door de
voorspel functie. In de afbeelding hieronder zie je de tuner op zoek
naar het beste model.

![](https://www.stillhaveit.nl/static/tuner.jpg)

Om inzicht te krijgen in het resultaat van training en validatie om
daarmee overfitting inzichtelijk te maken zijn deze waarden weergegeven
in een grafiek. Als training daalt maar validatie stijgt is dat een
indicatie voor overfitting. Overfitting is een ongewenst gedrag van
machine learning wanneer het model nauwkeurige voorspellingen geeft voor
trainingsdata, maar niet voor nieuwe data. Hieronder een voorbeeld van
de weergave.

![](/static/overfit.jpg)

De koersen van de ETF / DGTL.MI zijn:
- SLOTKOERS: van dinsdag 18 februari was € 10,87
- VOORSPELLING: voor woensdag 19 februari is **Eur 10,36**

![](/static/predict_v3.jpg)

**De Realiteit (woensdag 19 feb)**
????? morgen verder

------------------------------------------------------------------------

**Versie 1, week 7**

**De Data**
Om het leuk te houden willen we alle gebruikte data gratis beschikbaar
hebben. Mocht, in een later stadium, het model het gewenste effect
hebben en de euro's binnen stromen dan is betalen voor de data
natuurlijk geen probleem!\
De historische aandelen data halen we op vanaf financialmodeling. Deze
levert 5 jaar historische koersdata. Daarnaast willen we zoveel als
mogelijk aanvullende economische data toevoegen om het model met zoveel
mogelijk relevante gegevens te 'voeden'. Dit kan met de Python Technical
Analysis Library.\
Een aantal voorbeelden die we inzetten in ons model zijn:\
• On-Balance Volume (OBV) → Een indicator die het volume relateert aan
prijsveranderingen.\
• Williams %R → Een momentumindicator die overkochte en oversold
condities meet.\
• Rate of Change (ROC) → Meet de procentuele verandering in prijs over
een bepaalde periode.\
• Relative Strength Index (RSI) → Een momentumindicator die de snelheid
en verandering van prijsbewegingen meet.\
• Moving Average Convergence Divergence (MACD) → Een trendvolgende
momentumindicator.\
• Bollinger Bands → Een volatiliteitsindicator die een boven- en
onderband rond een voortschrijdend gemiddelde plaatst.\
• Average Directional Index (ADX) → Meet de sterkte van een trend.\
• Chaikin Money Flow (CMF) → Combineert prijs en volume om de geldstroom
te meten.\
• Commodity Channel Index (CCI) → Meet de afwijking van de prijs ten
opzichte van het statistisch gemiddelde.\
• Stochastic Oscillator → Een momentumindicator die de sluitingsprijs
vergelijkt met het prijsbereik over een bepaalde periode.\
• Average True Range (ATR) → Meet de volatiliteit door het bereik van
prijsbewegingen te analyseren.\
• Aroon Indicator → Identificeert trendveranderingen en de sterkte van
een trend.\
• Keltner Channel High → Een volatiliteitsindicator die een bovenband
rond een voortschrijdend gemiddelde plaatst, gebaseerd op de gemiddelde
true range (ATR).\
• Keltner Channel Low → Een volatiliteitsindicator die een onderband
rond een voortschrijdend gemiddelde plaatst, gebaseerd op de gemiddelde
true range (ATR).\
• Average True Range (ATR) → Een volatiliteitsindicator die het
gemiddelde van de true range over een bepaalde periode meet.

![](https://www.stillhaveit.nl/static/predict.jpg)
<br>
**Het Proces**
1. Als we de data binnen hebben kan het echte werk beginnen! Voor het
gemak en latere visuele validatie slaan we de data op in een CSV. Als
blijkt dat het model goed bruikbaar is dan volgt in een volgende versie
het vastleggen van de data in een database zodat we zelf nog langere
historie kunnen gaan vormen.

1. Voor het gebruik binnen de applicatie plaatsen we de data in een
Pandas Dataframe vergelijkbaar met een tabel met rijen en kolommen.

1. Dan splitsen we de data in een deel voor training en een deel, van
100 dagen, voor testen.

1. We schalen de data zodat het bruikbaar is voor de computer. Oftewel
we normaliseren de waarden binnen een bepaald bereik (standaard tussen 0
en 1).
Waarom schalen inzetten?
• Verbetering van machine learning prestaties
• Vermijden van dominantie door grote getallen
• Nuttig voor neurale

1. Nu gaan we het AI-model definiëren om onze voorspelling vorm te
geven. In ons geval is dat een neuronaal netwerk met behulp van Keras en
TensorFlow, specifiek een Long Short Term Memory (LSTM)-gebaseerd model
voor tijdreeksvoorspellingen, zoals het voorspellen van
aandelenkoersen.
Wat doet het?
• Het neemt tijdreeksen als input (bijvoorbeeld historische
aandelenprijzen).
• Het gebruikt een LSTM-laag om patronen en trends in de tijdreeks te
leren.
• Het geeft een voorspelling als output, bijvoorbeeld de aandelenprijs
van morgen.

1. Het bovenstaande model zetten we in om het te trainen met de
train-dataset. Doel is de patronen in de gegevens te onderkennen en
ervan te leren.
• Het doet dit 50 keer (aantal epochs).
• In elke epoch verwerkt het model een bepaald aantal batches
(steps_per_epoch).
• Na elke epoch wordt de validatiedataset (test_dataset) gebruikt om te
zien hoe goed het model presteert.

7 Nu gaan we met het test-deel van de data bekijken hoe goed ons model
is in het voorspellen van de koers.

1. De uitkomst van deze voorspelling test wordt weergegeven in 2 waarden
als mate van afwijking:
Mean Squared Error (MSE) is een maat voor de gemiddelde kwadratische
afwijking tussen de voorspellingen van je model en de werkelijke
waarden; 0.4970483243207762<br>
Mean Absolute Error (MAE) berekent de gemiddelde absolute afwijking
tussen de voorspellingen en de werkelijke waarden, zonder dat grotere
fouten zwaarder worden gewogen zoals bij de MSE; 0.5616767144227245<br>

1. In ons geval is de koers van AHOLD genomen als training en testdata.
We weten dat de koers momenteel zo rond de $35,00 schommelt. Als we
deze koers gebruiken om onze afwijking te berekenen dan zien we het
volgende:

De MSE van 0.497 betekent dat de gemiddelde kwadratische afwijking
tussen de voorspelde waarden en de werkelijke waarden $0.50 is. Dit is
een kleine afwijking ten opzichte van de aandelenprijs van $35,00.
Vergelijking: Als de aandelenprijs gemiddeld rond de $35,00 ligt, dan
is een MSE van 0.497 relatief laag, wat betekent dat het model goed
presteert. Bijvoorbeeld: een fout van $0.50 betekent dat het model
gemiddeld een fout heeft van ongeveer 1.4% van de voorspelde prijs.
De MAE van 0.56 betekent dat de gemiddelde absolute afwijking tussen de
voorspelde en werkelijke waarden $0.56 EUR is. Vergelijking: Een MAE
van $0.56 is ook relatief klein als we kijken naar een aandelenkoers
van $35,00, wat betekent dat de voorspellingen meestal binnen ongeveer
1.6% van de werkelijke waarde liggen.

1.  We slaan het model op zodat we het kunnen gebruiken bij onze
dagelijkse voorspelling

**De Output**
De output van het idee is dus de heilige graal voor ons als 'newbee
day-trader'. DE manier om snel rijk te worden met een minimale
inspanning. Oftewel vandaag de beurskoers van morgen voorspellen met ons
AI-model en kopen die stocks. (donderdag 13 feb) Het model geeft aan dat
de slotprijs voor morgen \$32,94 is. Best een flinke afwijking van de
koers van vandaag \$35,02. Het goede nieuws is dat als het klopt we snel
veel geld (hadden) kunnen verdienen.De output van het idee is dus de
heilige graal voor ons als 'newbee day-trader'. DE manier om snel rijk
te worden met een minimale inspanning.

![](https://www.stillhaveit.nl/static/60_days.png)
Als extraatje kunnen we de voorspelde prijs ook in een grafiek (Matplot)
bekijken tov de prijs over laatste 60 dagen.
![](https://www.stillhaveit.nl/static/stock_price.png)

**De Realiteit (vrijdag 14 feb)**
Het model behoeft nog wat aanscherping en verdere training. De
voorspelling zat er nu nog flink naast want de werkelijke slotkoers was
$35,90. Het model had een voorspelling gedaan van $32,94. Dat is een
afwijking van $2,96.

Versie 2 heeft een aantal verbeteringen ondergaan. Of deze verbeteringen
het gewenste resultaat leveren moet natuurlijk nog blijken.
Hier de voorspelling van het model voor de slotkoers van maandag 17
februari: $36,18.
![](https://www.stillhaveit.nl/static/predict2.jpg)

</div>
:::

