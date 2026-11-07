# AI Inventory Copilot – aplikacja Streamlit

Pilotaż: jeden magazyn (Gądki), 30 SKU, dane syntetyczne. Zakładki: prognoza popytu, rekomendacje zamówień, symulacja AS-IS vs AI, model ROI.

## Uruchomienie lokalnie
    pip install -r requirements.txt
    streamlit run app.py

## Wdrożenie (Streamlit Community Cloud, bezpłatnie)
1. Załóż konto na github.com i nowe repozytorium (np. `ai-inventory-copilot`).
2. Wgraj do niego zawartość tego folderu: `app.py`, `copilot.py`, `requirements.txt` oraz folder `dane/` z dwoma plikami CSV.
3. Wejdź na share.streamlit.io, zaloguj się kontem GitHub, wybierz **Create app**, wskaż repozytorium, gałąź `main` i plik `app.py`, kliknij **Deploy**.
4. Po kilku minutach dostaniesz stały adres aplikacji.

## Własne dane
Wybierz w panelu bocznym „Własne pliki CSV". Wymagane kolumny opisane są pod polami wgrywania.
