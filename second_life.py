"""SECOND LIFE - ratowanie wartości leków zagrożonych przeterminowaniem (moduł AI Inventory Copilot).
Wszystkie parametry kosztowe i popytowe to ZAŁOŻENIA (dane syntetyczne), nie dane NEUCA."""
import numpy as np, pandas as pd

# opcjonalna kolumna: refundowany (True = brak promocji, cena urzędowa)
KOLUMNY = ['magazyn', 'sku_id', 'sku_name', 'category', 'partia_id', 'ilosc', 'dni_do_waznosci', 'unit_cost_pln', 'price_pln', 'temperature', 'popyt_dzienny']
DOMYSLNE = dict(
    prog_obserwacji=0.7,        # partia 'obserwuj', gdy zapas (FEFO) pokrywa >= 70% popytu do końca ważności
    transport_dni=2,            # czas transferu między magazynami
    min_dni_po_transferze=14,   # minimalny pozostały termin w magazynie docelowym
    transfer_stały_zl=150.0, transfer_chlod_zl=150.0, transfer_proc_wartosci=0.015,
    rabat=0.20, elastycznosc=1.5, min_dni_promocji=10,
    zwrot_odzysk=0.60, zwrot_min_dni=60, zwrot_koszt_zl=100.0)

def sprawdz(p):
    miss = [k for k in KOLUMNY if k not in p.columns]
    if miss: raise ValueError('W pliku partii brakuje kolumn: ' + ', '.join(miss) + '.')
    if len(p) == 0: raise ValueError('Plik partii jest pusty.')
    if (p['ilosc'] < 0).any() or (p['dni_do_waznosci'] < 0).any(): raise ValueError('Ilość i dni do ważności nie mogą być ujemne.')
    return p

def ryzyko(p, par=None):
    """FEFO: dla każdej partii liczy ile sztuk nie zdąży się sprzedać przed końcem ważności przy bieżącym tempie popytu."""
    par = {**DOMYSLNE, **(par or {})}; p = sprawdz(p.copy())
    p = p.sort_values(['magazyn', 'sku_id', 'dni_do_waznosci']).reset_index(drop=True)
    p['skumulowane'] = p.groupby(['magazyn', 'sku_id'])['ilosc'].cumsum()
    p['sprzedaz_do_terminu'] = p['popyt_dzienny'] * p['dni_do_waznosci']
    p['szt_zagrozone'] = (p['skumulowane'] - p['sprzedaz_do_terminu']).clip(lower=0).clip(upper=p['ilosc']).round(0)
    p['wartosc_zagrozona_zl'] = p['szt_zagrozone'] * p['unit_cost_pln']
    pokrycie = p['skumulowane'] / p['sprzedaz_do_terminu'].replace(0, np.nan)
    p['status'] = np.where(p['szt_zagrozone'] > 0, 'ZAGROŻONA', np.where(pokrycie >= par['prog_obserwacji'], 'OBSERWUJ', 'OK'))
    return p

def _pojemnosc(p, mag, sku, T, par):
    """Ile sztuk magazyn docelowy sprzeda przed terminem T (przy FEFO) ponad własne partie o terminie <= T."""
    m = p[(p.magazyn == mag) & (p.sku_id == sku)]
    if len(m) == 0: return 0.0, 0.0
    mu = float(m['popyt_dzienny'].iloc[0]); okno = T - par['transport_dni']
    if okno < par['min_dni_po_transferze'] or mu <= 0: return 0.0, mu
    wlasne = float(m.loc[m.dni_do_waznosci <= T, 'ilosc'].sum())
    return max(0.0, mu * okno - wlasne), mu

def plan(p, par=None):
    """Dla każdej zagrożonej partii dobiera działania (transfer -> promocja -> zwrot -> odpis) wg wartości netto na sztukę."""
    par = {**DOMYSLNE, **(par or {})}; r = ryzyko(p, par)
    cap_zuzyte = {}; wiersze = []
    for _, b in r[r.status == 'ZAGROŻONA'].sort_values('wartosc_zagrozona_zl', ascending=False).iterrows():
        U = int(b.szt_zagrozone); c, cena, T = float(b.unit_cost_pln), float(b.price_pln), int(b.dni_do_waznosci)
        zostalo = U; akcje = []
        # 1. transfer (do magazynu z wolnym popytem): korzyść = uniknięty zakup c; koszt = stały + chłodnia + % wartości
        opcje = []
        for dest in r.magazyn.unique():
            if dest == b.magazyn: continue
            cap, mu = _pojemnosc(r, dest, b.sku_id, T, par); cap -= cap_zuzyte.get((dest, b.sku_id), 0.0)
            if cap >= 1: opcje.append((dest, cap))
        for dest, cap in sorted(opcje, key=lambda x: -x[1]):
            n = int(min(zostalo, np.floor(cap)))
            if n < 1: continue
            koszt = par['transfer_stały_zl'] + (par['transfer_chlod_zl'] if str(b.temperature).startswith('cold') else 0) + par['transfer_proc_wartosci'] * n * c
            netto = n * c - koszt
            if netto <= 0: continue
            cap_zuzyte[(dest, b.sku_id)] = cap_zuzyte.get((dest, b.sku_id), 0.0) + n
            akcje.append(dict(akcja=f'TRANSFER → {dest}', szt=n, odzysk_zl=n * c, koszt_zl=koszt, netto_zl=netto,
                              termin_dzialania_dni=max(T - par['transport_dni'] - 2, 0), zalozenie='uniknięty zakup w magazynie docelowym; wymaga akceptacji QA (GDP)'))
            zostalo -= n
            if zostalo < 1: break
        # 2. promocja: dodatkowa sprzedaż = popyt * elastyczność * rabat * dni; korzyść = przychód po rabacie
        refund = bool(b.get('refundowany', False)) if 'refundowany' in r.columns else False  # leki refundowane: cena urzędowa, brak promocji
        if zostalo >= 1 and not refund and T >= par['min_dni_promocji'] and cena * (1 - par['rabat']) > 0 and b.popyt_dzienny > 0:
            dod = par['elastycznosc'] * par['rabat'] * b.popyt_dzienny * T
            n = int(min(zostalo, np.floor(dod)))
            if n >= 1:
                akcje.append(dict(akcja=f'PROMOCJA −{par["rabat"]*100:.0f}%', szt=n, odzysk_zl=n * cena * (1 - par['rabat']), koszt_zl=0.0,
                                  netto_zl=n * cena * (1 - par['rabat']), termin_dzialania_dni=max(T - 10, 0), zalozenie='dodatkowy popyt = elastyczność × rabat; kanibalizacja pominięta'))
                zostalo -= n
        # 3. zwrot do dostawcy
        if zostalo >= 1 and T >= par['zwrot_min_dni']:
            netto = zostalo * c * par['zwrot_odzysk'] - par['zwrot_koszt_zl']
            if netto > 0:
                akcje.append(dict(akcja='ZWROT do dostawcy', szt=int(zostalo), odzysk_zl=zostalo * c * par['zwrot_odzysk'], koszt_zl=par['zwrot_koszt_zl'], netto_zl=netto,
                                  termin_dzialania_dni=max(T - par['zwrot_min_dni'], 0), zalozenie='założony odzysk % kosztu i minimalny termin zwrotu; do potwierdzenia z dostawcą'))
                zostalo = 0
        if zostalo >= 1:
            akcje.append(dict(akcja='ODPIS (brak opłacalnej alternatywy)', szt=int(zostalo), odzysk_zl=0.0, koszt_zl=0.0, netto_zl=0.0, termin_dzialania_dni=T, zalozenie='strata = koszt zakupu'))
        for a in akcje:
            wiersze.append(dict(magazyn=b.magazyn, sku_id=b.sku_id, sku_name=b.sku_name, partia_id=b.partia_id, dni_do_waznosci=T,
                                szt_zagrozone=U, wartosc_zagrozona_zl=round(b.wartosc_zagrozona_zl, 0), **{k: (round(v, 0) if isinstance(v, float) else v) for k, v in a.items()}))
    return r, pd.DataFrame(wiersze)

def podsumowanie(r, pl):
    zagr = float(r.wartosc_zagrozona_zl.sum()); netto = float(pl.netto_zl.sum()) if len(pl) else 0.0
    return dict(partie=len(r), zagrozone=int((r.status == 'ZAGROŻONA').sum()), obserwuj=int((r.status == 'OBSERWUJ').sum()),
                wartosc_zagrozona_zl=zagr, wartosc_uratowana_zl=netto, udzial_uratowany=(netto / zagr if zagr else 0.0))
