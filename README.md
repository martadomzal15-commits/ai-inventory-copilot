# AI Navigator – aplikacja Streamlit

Pilotaż: jeden magazyn (Gądki), 30 SKU, dane syntetyczne. Zakładki: prognoza popytu, rekomendacje zamówień, symulacja AS-IS vs AI, model ROI.

## Uruchomienie lokalnie
    pip install -r requirements.txt
    streamlit run app.py

## Wdrożenie (Streamlit Community Cloud, bezpłatnie)
1. Załóż konto na github.com i nowe repozytorium (może zostać dotychczasowa nazwa, np. `ai-inventory-copilot`; zmiana nazwy repozytorium zmienia adres aplikacji).
2. Wgraj do niego zawartość tego folderu: `app.py`, `copilot.py`, `requirements.txt` oraz folder `dane/` z dwoma plikami CSV.
3. Wejdź na share.streamlit.io, zaloguj się kontem GitHub, wybierz **Create app**, wskaż repozytorium, gałąź `main` i plik `app.py`, kliknij **Deploy**.
4. Po kilku minutach dostaniesz stały adres aplikacji.

## Własne dane
Wybierz w panelu bocznym „Własne pliki CSV". Wymagane kolumny opisane są pod polami wgrywania.


## SECOND LIFE (nowy moduł)
Zakładka ♻️ SECOND LIFE wskazuje partie zagrożone przeterminowaniem (FEFO, bieżący popyt) i proponuje plan: transfer między magazynami, promocję, zwrot do dostawcy lub odpis – z kosztami. Dane: `dane/partie_siec.csv` (syntetyczne, generator `neuca_partie_generator.py`); własny plik w panelu bocznym (kolumny: magazyn, sku_id, sku_name, category, partia_id, ilosc, dni_do_waznosci, unit_cost_pln, price_pln, temperature, popyt_dzienny). Wszystkie parametry kosztowe to założenia.
Pliki do wgrania na GitHub: `app.py`, `second_life.py`, `dane/partie_siec.csv`.

## Asystent planisty (SECOND LIFE)
Czat pod planem działań. Dwa tryby: **Demo** (reguły, bez klucza) i **LLM (Claude API)**. LLM nie liczy – wywołuje narzędzia z `second_life.py` (podsumowanie, lista partii, plan, co-jeśli), a aplikacja ostrzega o liczbach spoza wyników narzędzi. Klucz API: w Streamlit Cloud dodaj sekret `ANTHROPIC_API_KEY` (Manage app → Settings → Secrets) albo wpisz go w polu na czas sesji. Nie umieszczaj klucza w repozytorium.
Do wgrania na GitHub: `app.py`, `second_life.py`, `asystent.py`, `requirements.txt` (dodano `anthropic`), `dane/partie_siec.csv`.
