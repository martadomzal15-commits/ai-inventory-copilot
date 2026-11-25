"""Asystent planisty dla modułu SECOND LIFE.
Zasada: LLM NIE liczy. Wszystkie liczby pochodzą z narzędzi (second_life.py); asystent je wyjaśnia i porównuje.
Dwa tryby: 'demo' (reguły, bez klucza i bez sieci) oraz 'llm' (Claude API z wywoływaniem narzędzi)."""
import json, re
import pandas as pd
import second_life as sl

DOZWOLONE = {'rabat': (0, .6), 'elastycznosc': (0, 5), 'transport_dni': (0, 14), 'transfer_stały_zl': (0, 5000), 'transfer_chlod_zl': (0, 5000),
             'transfer_proc_wartosci': (0, .2), 'zwrot_odzysk': (0, 1), 'zwrot_min_dni': (0, 365), 'min_dni_po_transferze': (0, 90)}

SYSTEM = ("Jesteś asystentem planisty zapasów w hurtowni farmaceutycznej (projekt AI Navigator, moduł SECOND LIFE). Dane są syntetyczne. "
 "Zasady: (1) Wszystkie liczby bierz WYŁĄCZNIE z wyników narzędzi; nie licz sam i nie zgaduj. Brakującą informację nazwij brakiem. "
 "(2) Parametry kosztowe (rabat, elastyczność, odzysk ze zwrotu, koszty transferu) to założenia – mów o tym. "
 "(3) Plan jest rekomendacją; decyduje planista, a transfer i wstrzymanie partii wymagają akceptacji QA (GDP). Nie twierdź, że wykonałeś jakiekolwiek działanie. "
 "(4) Odpowiadaj po polsku, krótko i konkretnie; porównując działania podawaj netto w zł i termin działania. Nie udzielaj porad medycznych.")

NARZEDZIA = [
 dict(name='podsumowanie_sieci', description='Podsumowanie sieci: liczba partii zagrożonych/obserwowanych, wartość zagrożona i uratowana planem.', input_schema=dict(type='object', properties={})),
 dict(name='lista_partii', description='Lista partii wg statusu (ZAGROŻONA, OBSERWUJ, OK), opcjonalnie filtr magazyn i fragment nazwy produktu.',
      input_schema=dict(type='object', properties=dict(status=dict(type='string'), magazyn=dict(type='string'), produkt=dict(type='string')))),
 dict(name='plan_dzialan', description='Zaproponowany plan działań (transfer, promocja, zwrot, odpis) z kosztami; filtr magazyn i produkt.',
      input_schema=dict(type='object', properties=dict(magazyn=dict(type='string'), produkt=dict(type='string')))),
 dict(name='co_jesli', description='Symulacja zmiany założeń (nie zmienia bieżących ustawień). Dozwolone klucze: ' + ', '.join(DOZWOLONE) + '. Wartości procentowe jako ułamki (rabat 0.3 = 30%). Zwraca różnicę względem planu bazowego.',
      input_schema=dict(type='object', properties=dict(zmiany=dict(type='object')), required=['zmiany'])),
]

def _filtr(df, magazyn=None, produkt=None):
    if magazyn: df = df[df['magazyn'].str.lower().str.contains(str(magazyn).lower(), na=False)]
    if produkt: df = df[df['sku_name'].str.lower().str.contains(str(produkt).lower(), na=False)]
    return df

def _zaokr(d):
    return json.loads(d.round(1).to_json(orient='records', force_ascii=False))

def wykonaj_narzedzie(nazwa, arg, partie, par):
    arg = arg or {}
    r, pl = sl.plan(partie, par)
    if nazwa == 'podsumowanie_sieci':
        return {k: (round(v, 0) if isinstance(v, float) and k != 'udzial_uratowany' else (round(v, 3) if isinstance(v, float) else v)) for k, v in sl.podsumowanie(r, pl).items()}
    if nazwa == 'lista_partii':
        d = _filtr(r, arg.get('magazyn'), arg.get('produkt'))
        if arg.get('status'): d = d[d.status == str(arg['status']).upper().replace('ZAGROZONA', 'ZAGROŻONA')]
        k = ['magazyn', 'sku_name', 'partia_id', 'ilosc', 'dni_do_waznosci', 'popyt_dzienny', 'sprzedaz_do_terminu', 'szt_zagrozone', 'wartosc_zagrozona_zl', 'status']
        return {'liczba': len(d), 'partie': _zaokr(d.sort_values('wartosc_zagrozona_zl', ascending=False)[k].head(20))}
    if nazwa == 'plan_dzialan':
        d = _filtr(pl, arg.get('magazyn'), arg.get('produkt')) if len(pl) else pl
        return {'liczba': len(d), 'dzialania': _zaokr(d.drop(columns=['sku_id'], errors='ignore')) if len(d) else []}
    if nazwa == 'co_jesli':
        zm = arg.get('zmiany') or {}; bad = [k for k in zm if k not in DOZWOLONE]
        if bad: return {'blad': 'Niedozwolone klucze: ' + ', '.join(bad), 'dozwolone': list(DOZWOLONE)}
        for k, v in zm.items():
            lo, hi = DOZWOLONE[k]
            if not isinstance(v, (int, float)) or not lo <= v <= hi: return {'blad': f'Wartość {k} poza zakresem {lo}-{hi}.'}
        r2, pl2 = sl.plan(partie, {**par, **zm}); a, b = sl.podsumowanie(r, pl), sl.podsumowanie(r2, pl2)
        return {'zmiany': zm, 'bazowy_uratowane_zl': round(a['wartosc_uratowana_zl']), 'po_zmianie_uratowane_zl': round(b['wartosc_uratowana_zl']),
                'roznica_zl': round(b['wartosc_uratowana_zl'] - a['wartosc_uratowana_zl']), 'wartosc_zagrozona_zl': round(b['wartosc_zagrozona_zl'])}
    return {'blad': f'Nieznane narzędzie: {nazwa}'}

# ---------- kontrola liczb ----------
def _liczby(t):
    t = re.sub(r'(?<=\d)[\s ](?=\d{3}\b)', '', t)
    return [float(x.replace(',', '.')) for x in re.findall(r'\d+(?:[.,]\d+)?', t)]

def liczby_spoza_narzedzi(odp, wyniki, prog=20):
    """Zwraca liczby >= prog z odpowiedzi, których nie ma w wynikach narzędzi (tolerancja zaokrągleń)."""
    zrodlo = [x for w in wyniki for x in _liczby(json.dumps(w, ensure_ascii=False))]
    zrodlo += [x * 100 for x in zrodlo if 0 < x <= 1]  # ułamki podawane jako procenty
    return sorted({x for x in _liczby(odp) if x >= prog and not any(abs(x - z) <= max(0.5, 0.005 * abs(z)) for z in zrodlo)})

# ---------- tryb LLM ----------
def odpowiedz_llm(historia, pytanie, partie, par, klient, model='claude-sonnet-5-5', max_krokow=6):
    """historia: lista {'role','content'} (tekst). klient: anthropic.Anthropic lub obiekt zgodny. Zwraca (tekst, wyniki_narzedzi)."""
    msgs = [*historia, {'role': 'user', 'content': pytanie}]; wyniki = []
    for _ in range(max_krokow):
        r = klient.messages.create(model=model, max_tokens=1200, system=SYSTEM, tools=NARZEDZIA, messages=msgs)
        if r.stop_reason != 'tool_use':
            return ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text'), wyniki
        msgs.append({'role': 'assistant', 'content': r.content}); res = []
        for b in r.content:
            if getattr(b, 'type', '') == 'tool_use':
                w = wykonaj_narzedzie(b.name, b.input, partie, par); wyniki.append(w)
                res.append({'type': 'tool_result', 'tool_use_id': b.id, 'content': json.dumps(w, ensure_ascii=False)})
        msgs.append({'role': 'user', 'content': res})
    return 'Przerwano: zbyt wiele kroków narzędzi. Zadaj pytanie węziej.', wyniki

# ---------- tryb demo (bez LLM) ----------
def odpowiedz_demo(pytanie, partie, par):
    q = pytanie.lower(); r, pl = sl.plan(partie, par); s = sl.podsumowanie(r, pl); wyn = []
    fm = lambda x: f'{x:,.0f}'.replace(',', ' ')
    norm = lambda x: x.lower().replace('ł', 'l').replace('ó', 'o')
    mag = next((m for m in r.magazyn.unique() if norm(m) in norm(q)), None)
    sku = next((n for n in r.sku_name.unique() if n.split('_')[0].lower() in q), None)
    m = re.search(r'rabat\D{0,6}(\d{1,2})', q); m2 = re.search(r'(?:koszt|kosztuje)\D{0,15}transfer\D{0,6}(\d+)|transfer\D{0,15}(\d+)\s*(?:zł|zl)', q)
    if m or m2:
        zm = {}
        if m: zm['rabat'] = int(m.group(1)) / 100
        if m2: zm['transfer_stały_zl'] = float(m2.group(1) or m2.group(2))
        w = wykonaj_narzedzie('co_jesli', {'zmiany': zm}, partie, par); wyn.append(w)
        if 'blad' in w: return w['blad'], wyn
        return (f"Przy zmianie założeń {zm}: wartość uratowana planem {fm(w['po_zmianie_uratowane_zl'])} zł wobec {fm(w['bazowy_uratowane_zl'])} zł w planie bazowym "
                f"(różnica {fm(w['roznica_zl'])} zł). To symulacja – bieżące ustawienia się nie zmieniły."), wyn
    if sku and re.search(r'dlaczego|wyja|czemu|co z|plan|zrobi', q):
        w = wykonaj_narzedzie('plan_dzialan', {'produkt': sku.split('_')[0], 'magazyn': mag}, partie, par); wyn.append(w)
        b = wykonaj_narzedzie('lista_partii', {'status': 'ZAGROŻONA', 'produkt': sku.split('_')[0], 'magazyn': mag}, partie, par); wyn.append(b)
        if not b['partie']: return f'Dla {sku} nie ma partii zagrożonych przeterminowaniem przy bieżącym tempie popytu.', wyn
        t = [f"**{sku}** – partie zagrożone:"]
        for x in b['partie']:
            t.append(f"- {x['magazyn']}, partia {x['partia_id']}: {x['ilosc']:.0f} szt., ważność za {x['dni_do_waznosci']:.0f} dni, popyt ok. {x['popyt_dzienny']:.1f} szt./dzień, "
                     f"do terminu sprzeda się ok. {x['sprzedaz_do_terminu']:.0f} szt. (FEFO), więc zagrożone ok. {x['szt_zagrozone']:.0f} szt. ({fm(x['wartosc_zagrozona_zl'])} zł).")
        t.append('Proponowane działania:')
        for x in w['dzialania']:
            t.append(f"- {x['partia_id']}: {x['akcja']}, {x['szt']:.0f} szt., netto {fm(x['netto_zl'])} zł, działać w ciągu {x['termin_dzialania_dni']:.0f} dni.")
        t.append('To rekomendacja oparta na założeniach kosztowych; decyduje planista, a transfer wymaga akceptacji QA.')
        return '\n'.join(t), wyn
    if 'obserw' in q:
        w = wykonaj_narzedzie('lista_partii', {'status': 'OBSERWUJ', 'magazyn': mag}, partie, par); wyn.append(w)
        if not w['partie']: return 'Brak partii do obserwacji.', wyn
        return 'Do obserwacji:\n' + '\n'.join(f"- {x['magazyn']}, {x['sku_name']}, partia {x['partia_id']}: {x['ilosc']:.0f} szt., ważność za {x['dni_do_waznosci']:.0f} dni, popyt {x['popyt_dzienny']:.1f} szt./dzień" for x in w['partie']), wyn
    if re.search(r'podsum|ile|ogółem|wartość|wartosc|sytuac|stan', q):
        w = wykonaj_narzedzie('podsumowanie_sieci', {}, partie, par); wyn.append(w)
        return (f"W sieci jest {w['zagrozone']} partii zagrożonych i {w['obserwuj']} do obserwacji (na {w['partie']} partii). Wartość zagrożona: {fm(w['wartosc_zagrozona_zl'])} zł; "
                f"plan ratuje ok. {fm(w['wartosc_uratowana_zl'])} zł ({w['udzial_uratowany']*100:.0f}%)."), wyn
    return ('Mogę: podsumować sytuację („podsumuj sieć”), wyjaśnić produkt („dlaczego trastuzumab w Ołtarzewie”), pokazać partie do obserwacji, '
            'albo przeliczyć scenariusz („co jeśli rabat 30%”, „co jeśli transfer kosztuje 500 zł”). Tryb LLM rozumie dowolne pytania.'), wyn
