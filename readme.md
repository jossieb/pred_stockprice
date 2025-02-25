
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
ETF).

**Requirements**<br>
pip install -r requirements.txt

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