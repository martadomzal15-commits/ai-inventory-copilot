import io
import numpy as np, pandas as pd, streamlit as st, altair as alt
import second_life as sl, asystent as asy
import datetime
from copilot import (brakujace_kolumny_symulacji, wczytaj_dane, do_tygodni, trenuj_i_prognozuj, metryki, mape, symuluj, podsumowanie, rekomendacje, roi)

st.set_page_config(page_title='AI Navigator', page_icon='📦', layout='wide')
st.title('📦 AI Navigator')
st.caption('Inteligentne rekomendacje zapasów i ich wpływ na wynik finansowy – pilotaż: jeden magazyn, 30 SKU. Dane syntetyczne.')

# ---------- dane wejściowe ----------
with st.sidebar:
    st.header('1. Dane')
    zrodlo = st.radio('Źródło danych', ['Przykładowe (syntetyczne)', 'Własne pliki CSV'])
    if zrodlo == 'Własne pliki CSV':
        f_h = st.file_uploader('Historia popytu (min. ok. 1 rok)', type='csv')
        f_t = st.file_uploader('Okres bieżący / testowy (z kolumnami zapasów)', type='csv')
    else:
        f_h, f_t = 'dane/neuca_gadki_30sku_historia_2lata.csv', 'dane/neuca_gadki_30sku_180d.csv'
    st.caption('Wymagane kolumny: date, sku_id, sku_name, category, demand_units. Plik bieżący dodatkowo: stock_start, stock_end, on_order_units, delivered_units, days_to_expiry_oldest, lead_time_days, lead_time_sigma, fulfillment_rate, shelf_life_days, is_deficit, price_pln, unit_cost_pln.')
    st.header('2. SECOND LIFE (partie)')
    f_p = st.file_uploader('Partie z terminami ważności (CSV, sieć magazynów) – opcjonalnie', type='csv')
    st.caption('Bez pliku używane są przykładowe, syntetyczne partie dla 3 magazynów (stan na 27.12.2025).')

if f_h is None or f_t is None:
    st.info('Wgraj oba pliki CSV w panelu bocznym albo wybierz dane przykładowe.'); st.stop()

@st.cache_data(show_spinner='Wczytuję dane i trenuję model…')
def policz(h_bytes, t_bytes):
    h, t = wczytaj_dane(io.BytesIO(h_bytes), io.BytesIO(t_bytes))
    wh, wt = do_tygodni(h), do_tygodni(t)
    te, sez, _ = trenuj_i_prognozuj(wh, wt)
    return h, t, te, sez

def bajty(f): return open(f, 'rb').read() if isinstance(f, str) else f.getvalue()
try:
    hist, test, te, sez = policz(bajty(f_h), bajty(f_t))
except Exception as e:
    st.error(f'Nie udało się przetworzyć danych: {e}'); st.stop()

mag = ', '.join(sorted(test['magazine'].dropna().astype(str).unique())) if 'magazine' in test.columns else 'nie podano (brak kolumny magazine)'
st.markdown(f"**Wczytane dane:** magazyn **{mag}** · {test['sku_id'].nunique()} SKU · historia {hist['date'].min():%d.%m.%Y}–{hist['date'].max():%d.%m.%Y} · okres bieżący {test['date'].min():%d.%m.%Y}–{test['date'].max():%d.%m.%Y}")

@st.cache_data(show_spinner='Symulacja AS-IS vs AI (30 przebiegów)…')
def sym(h_bytes, t_bytes):
    h, t, te_, _ = policz(h_bytes, t_bytes)
    return symuluj(h, t, te_, n_sim=30)
brak = brakujace_kolumny_symulacji(test); wynik = None
if brak:
    st.info('Prognoza popytu działa, ale rekomendacje, symulacja i ROI wymagają w pliku bieżącym dodatkowych kolumn: ' + ', '.join(brak) + '.')
else:
    try:
        wynik = sym(bajty(f_h), bajty(f_t)); pod = podsumowanie(wynik)
    except Exception as e:
        st.warning(f'Symulacja niedostępna dla tych danych: {e}'); wynik = None


def opis_wyniku(m):
    d = m['LightGBM'] - m['MA4']  # >0: MA4 lepsza
    if abs(d) < 1: return 'Na tych danych MA4 i model ML są porównywalne – różnica poniżej 1 pp mieści się w szumie jednego podziału train/test.'
    if d >= 1: return f'Na tych danych prosta średnia MA4 jest lepsza od LightGBM o {d:.1f} pp.'
    return f'Na tych danych LightGBM jest lepszy od MA4 o {-d:.1f} pp (jeden podział train/test – wynik wymaga potwierdzenia).'

tab1, tab2, tab3, tab4, tab5 = st.tabs(['📈 Prognoza popytu', '✅ Rekomendacje', '⚖️ AS-IS vs AI', '💰 ROI', '♻️ SECOND LIFE'])

# ---------- 1. prognoza ----------
with tab1:
    m = metryki(te); c = st.columns(3)
    for col, (k, v) in zip(c, m.items()): col.metric(f'MAPE – {k}', f'{v}%')
    st.caption(f'Prognoza na tydzień do przodu, {te.week.nunique()} tygodni testowych, {te.sku_id.nunique()} SKU. SKU sezonowych (reguła z danych uczących): {len(sez)}. ' + opis_wyniku(m))
    sku = st.selectbox('SKU', te.sort_values('sku_name').sku_name.unique())
    d = te[te.sku_name == sku].set_index('week')[['demand', 'ma4', 'lgb']].rename(columns={'demand': 'Rzeczywisty popyt', 'ma4': 'MA4', 'lgb': 'LightGBM'})
    import altair as alt
    dl = d.reset_index().melt('week', var_name='Seria', value_name='popyt')
    st.altair_chart(alt.Chart(dl).mark_line().encode(x=alt.X('week:T', title='Tydzień'), y=alt.Y('popyt:Q', title='Popyt (szt. na tydzień)'),
        color=alt.Color('Seria:N', scale=alt.Scale(domain=['Rzeczywisty popyt', 'MA4', 'LightGBM'], range=['#d62728', '#1f77b4', '#2ca02c']), legend=alt.Legend(orient='top', title=None)),
        strokeDash=alt.StrokeDash('Seria:N', scale=alt.Scale(domain=['Rzeczywisty popyt', 'MA4', 'LightGBM'], range=[[1, 0], [5, 3], [5, 3]]), legend=None)), width='stretch')
    kat = te.groupby('category').apply(lambda x: pd.Series({'MA4': mape(x.demand, x.ma4), 'LightGBM': mape(x.demand, x.lgb)}), include_groups=False).round(1)
    st.subheader('MAPE wg kategorii (%)'); 
    kl = kat[['MA4', 'LightGBM']].reset_index().melt('category', var_name='Model', value_name='mape')
    st.altair_chart(alt.Chart(kl).mark_bar().encode(x=alt.X('category:N', title=None, axis=alt.Axis(labelAngle=-90)), xOffset='Model:N',
        y=alt.Y('mape:Q', title='MAPE (%)', scale=alt.Scale(domain=[0, 40], nice=False)),
        color=alt.Color('Model:N', scale=alt.Scale(domain=['MA4', 'LightGBM'], range=['#1f77b4', '#2ca02c']), legend=alt.Legend(orient='top', title=None))), width='stretch')

# ---------- 2. rekomendacje ----------
with tab2:
    try:
        if brak: raise ValueError('wymagane są dodatkowe kolumny w pliku bieżącym (lista w komunikacie nad zakładkami)')
        rek = rekomendacje(test, te)
        c = st.columns(4)
        c[0].metric('Do zamówienia', int((rek.rekomendacja == 'ZAMÓW').sum())); c[1].metric('Do wstrzymania / zmniejszenia', int(rek.rekomendacja.str.startswith('WSTRZ').sum()))
        c[2].metric('SKU z ryzykiem', int((rek.ryzyko != '-').sum())); c[3].metric('Wartość zapasu (zł)', f"{rek.wartosc_zapasu_zl.sum():,.0f}".replace(',', ' '))
        filtr = st.multiselect('Pokaż rekomendacje', sorted(rek.rekomendacja.unique()), default=sorted(rek.rekomendacja.unique()))
        st.dataframe(rek[rek.rekomendacja.isin(filtr)], width="stretch", hide_index=True)
        st.download_button('Pobierz rekomendacje (CSV)', rek.to_csv(index=False).encode('utf-8-sig'), 'rekomendacje.csv', 'text/csv')
        st.caption('Reguła: punkt zamówienia = prognoza × czas dostawy (z poprawką na zawodność dostawcy) + zapas bezpieczeństwa (95% obsługi). Ryzyko przeterminowania: zapas nie rozejdzie się przed końcem ważności najstarszej partii (<90 dni).')
    except Exception as e:
        st.warning(f'Rekomendacje niedostępne: {e}')

# ---------- 3. AS-IS vs AI ----------
with tab3:
    if wynik is None: st.info('Brak wyników symulacji.')
    else:
        a, b = pod.loc['ASIS'], pod.loc['AI']; c = st.columns(3)
        c[0].metric('Średni zapas (zł)', f'{b.zapas_sr_zl:,.0f}'.replace(',', ' '), f'{(b.zapas_sr_zl/a.zapas_sr_zl-1)*100:.0f}% vs AS-IS', delta_color='inverse')
        c[1].metric('Fill rate', f'{b.fill_rate*100:.1f}%', f'{(b.fill_rate-a.fill_rate)*100:+.1f} pp vs AS-IS')
        c[2].metric('Utracona marża (zł)', f'{b.utracona_marza_zl:,.0f}'.replace(',', ' '), f'{b.utracona_marza_zl-a.utracona_marza_zl:,.0f}'.replace(',', ' '), delta_color='inverse')
        tp = pod.rename(index={'ASIS': 'AS-IS', 'AI': 'AI'}).loc[['AS-IS', 'AI']].copy()
        tp['fill_rate'] = (tp['fill_rate'] * 100).round(1).astype(str) + '%'
        tp = tp.round({'zapas_sr_zl': 0, 'odpisy_zl': 0, 'utracona_marza_zl': 0}).astype({'zapas_sr_zl': int, 'odpisy_zl': int, 'utracona_marza_zl': int})
        tp.columns = ['Średni zapas (zł)', 'Odpisy (zł)', 'Utracona marża (zł)', 'Fill rate']
        st.dataframe(tp, width="stretch")
        kz = wynik.pivot_table(index='category', columns='scenariusz', values='zapas_sr_zl', aggfunc='sum').rename(columns={'ASIS': 'AS-IS'}).round(0)[['AS-IS', 'AI']]
        st.subheader('Średnia wartość zapasu wg kategorii (zł)'); 
        kzl = kz.reset_index().melt('category', var_name='Scenariusz', value_name='zapas')
        st.altair_chart(alt.Chart(kzl).mark_bar().encode(x=alt.X('category:N', title=None, axis=alt.Axis(labelAngle=-90)), xOffset='Scenariusz:N',
            y=alt.Y('zapas:Q', title='Zapas (zł)', scale=alt.Scale(zero=True)),
            color=alt.Color('Scenariusz:N', scale=alt.Scale(domain=['AS-IS', 'AI'], range=['#7f7f7f', '#ff7f0e']), legend=alt.Legend(orient='top', title=None))), width='stretch')
        st.caption('AS-IS to założony proces (stała średnia popytu, duże partie, bez uwzględnienia dostawcy), a nie dane NEUCA. Ten sam popyt i te same losowania dostaw w obu scenariuszach; wynik = średnia z 30 przebiegów. '
                   'Odpisy w oknie 180 dni nie różnią się (krótki okres wobec terminów ważności).')

# ---------- 4. ROI ----------
with tab4:
    if wynik is None: st.info('Brak wyników symulacji.')
    else:
        st.subheader('Założenia (możesz zmieniać)')
        sc = {'Ostrożny': (.06, .04, .70, .50, 0.0, 100000, 25000), 'Bazowy': (.08, .07, .85, .75, 0.0, 70000, 18000), 'Optymistyczny': (.10, .10, 1.0, 1.0, .20, 50000, 12000)}
        wyb = st.radio('Scenariusz startowy', list(sc), index=1, horizontal=True); p = sc[wyb]
        c1, c2, c3, c4 = st.columns(4)
        kk = c1.slider('Koszt kapitału (%/rok)', 0.0, 15.0, p[0]*100, 0.5) / 100; ku = c2.slider('Pozostały koszt utrzymania zapasu (%)', 0.0, 20.0, p[1]*100, 0.5) / 100
        re = c3.slider('Realizacja redukcji zapasu (%)', 0, 100, int(p[2]*100), 5) / 100; om = c4.slider('Odzyskana marża z braków (%)', 0, 100, int(p[3]*100), 5) / 100
        c5, c6, c7, c8 = st.columns(4)
        uo = c5.slider('Unikane odpisy (% odpisów z danych)', 0, 100, int(p[4]*100), 5) / 100; wd = c6.number_input('Koszt wdrożenia (zł)', 0, 1000000, p[5], 5000)
        ut = c7.number_input('Koszt utrzymania (zł/rok)', 0, 500000, p[6], 1000); dy = c8.slider('Stopa dyskontowa (%)', 1.0, 15.0, 8.0, 0.5) / 100
        odp = float((test.expired_writeoff_units * test.unit_cost_pln).sum()) if 'expired_writeoff_units' in test else 0.0
        dni = test['date'].nunique()
        r = roi(a.zapas_sr_zl - b.zapas_sr_zl, a.utracona_marza_zl - b.utracona_marza_zl, odp,
                dict(koszt_kapitalu=kk, koszt_utrzymania=ku, realizacja=re, odzysk_marzy=om, unikane_odpisy=uo, wdrozenie=wd, utrzymanie=ut, dyskonto=dy), dni)
        c = st.columns(4); fm = lambda x: f'{x:,.0f}'.replace(',', ' ')
        c[0].metric('Korzyść roczna (zł)', fm(r['rocznie'])); c[1].metric('NPV 3 lata (zł)', fm(r['npv'])); c[2].metric('ROI 3 lata', f"{r['roi_3l']*100:.0f}%")
        c[3].metric('Okres zwrotu', f"{r['zwrot_lata']:.1f} lat" if np.isfinite(r['zwrot_lata']) else 'brak zwrotu')
        kb = pd.Series({'Koszt kapitału': r['korzysci']['kapital'], 'Koszt utrzymania zapasu': r['korzysci']['utrzymanie_zapasu'], 'Odzyskana marża': r['korzysci']['marza'], 'Unikane odpisy': r['korzysci']['odpisy']})
        st.subheader('Struktura korzyści rocznych (zł)'); st.bar_chart(kb)
        st.caption(f"Maksymalny koszt wdrożenia przy NPV = 0: {fm(r['maks_wdrozenie'])} zł. Marża brutto NEUCA ok. 11,5% (raport III kw. 2025); koszty wdrożenia i pozostałe parametry to założenia własne. Wynik dotyczy pilotażu jednego magazynu.")

# ---------- 5. SECOND LIFE ----------
with tab5:
    st.subheader('SECOND LIFE: ratowanie wartości leków zagrożonych przeterminowaniem')
    st.caption('Które partie nie zdążą się sprzedać przed końcem ważności (reguła FEFO, bieżące tempo popytu), kiedy działać i gdzie przenieść zapas w sieci – po uwzględnieniu kosztów. Planista zatwierdza plan; transfer i wstrzymanie partii wymagają akceptacji QA (GDP).')
    try:
        partie = pd.read_csv(f_p) if f_p is not None else pd.read_csv('dane/partie_siec.csv')
        with st.expander('Założenia kosztowe i popytowe (własne – można zmieniać)'):
            c1, c2, c3, c4 = st.columns(4)
            par = dict(
                transport_dni=c1.number_input('Czas transferu (dni)', 0, 14, 2), transfer_stały_zl=c1.number_input('Koszt stały transferu (zł)', 0, 5000, 150, 10),
                transfer_chlod_zl=c2.number_input('Dopłata za łańcuch 2–8°C (zł)', 0, 5000, 150, 10), transfer_proc_wartosci=c2.slider('Koszt transferu (% wartości)', 0.0, 10.0, 1.5, 0.5) / 100,
                rabat=c3.slider('Rabat w promocji (%)', 0, 50, 20, 5) / 100, elastycznosc=c3.slider('Elastyczność popytu na rabat', 0.0, 4.0, 1.5, 0.1),
                zwrot_odzysk=c4.slider('Odzysk przy zwrocie (% kosztu)', 0, 100, 60, 5) / 100, zwrot_min_dni=c4.number_input('Min. dni do ważności dla zwrotu', 0, 365, 60, 5),
                min_dni_po_transferze=c1.number_input('Min. termin po transferze (dni)', 0, 90, 14))
        r, pl = sl.plan(partie, par); s = sl.podsumowanie(r, pl)
        c = st.columns(4); fm = lambda x: f'{x:,.0f}'.replace(',', ' ')
        c[0].metric('Partie zagrożone / obserwowane', f"{s['zagrozone']} / {s['obserwuj']}"); c[1].metric('Wartość zagrożona (zł)', fm(s['wartosc_zagrozona_zl']))
        c[2].metric('Wartość uratowana planem (zł)', fm(s['wartosc_uratowana_zl'])); c[3].metric('Udział uratowany', f"{s['udzial_uratowany']*100:.0f}%")
        if len(pl) == 0:
            st.success('Brak partii zagrożonych przeterminowaniem przy bieżącym tempie popytu.')
        else:
            st.markdown('**Plan działań do zatwierdzenia** (od najwyższej wartości zagrożonej)')
            v = pl.rename(columns={'magazyn': 'Magazyn', 'sku_name': 'Produkt', 'partia_id': 'Partia', 'dni_do_waznosci': 'Dni do ważności', 'szt_zagrozone': 'Szt. zagrożone',
                'wartosc_zagrozona_zl': 'Wartość zagrożona (zł)', 'akcja': 'Działanie', 'szt': 'Szt.', 'odzysk_zl': 'Odzysk (zł)', 'koszt_zl': 'Koszt (zł)', 'netto_zl': 'Netto (zł)',
                'termin_dzialania_dni': 'Działać w ciągu (dni)', 'zalozenie': 'Założenie'}).drop(columns=['sku_id'])
            st.dataframe(v, width='stretch', hide_index=True)
            st.download_button('Pobierz plan (CSV)', pl.to_csv(index=False).encode('utf-8-sig'), 'plan_second_life.csv', 'text/csv')
            g = r[r.status == 'ZAGROŻONA'].groupby('magazyn').wartosc_zagrozona_zl.sum().rename('Wartość zagrożona').to_frame().join(pl.groupby('magazyn').netto_zl.sum().rename('Uratowane planem')).fillna(0).reset_index().melt('magazyn', var_name='Miara', value_name='zl')
            st.altair_chart(alt.Chart(g).mark_bar().encode(x=alt.X('magazyn:N', title=None), xOffset='Miara:N', y=alt.Y('zl:Q', title='zł', scale=alt.Scale(zero=True)),
                color=alt.Color('Miara:N', scale=alt.Scale(domain=['Wartość zagrożona', 'Uratowane planem'], range=['#d62728', '#2ca02c']), legend=alt.Legend(orient='top', title=None))), width='stretch')
        ob = r[r.status == 'OBSERWUJ']
        if len(ob):
            st.markdown('**Do obserwacji** (zapas pokrywa ≥70% popytu do końca ważności – jeszcze bez straty, ale blisko granicy)')
            st.dataframe(ob[['magazyn', 'sku_name', 'partia_id', 'ilosc', 'dni_do_waznosci', 'popyt_dzienny']].rename(columns={'magazyn': 'Magazyn', 'sku_name': 'Produkt', 'partia_id': 'Partia', 'ilosc': 'Ilość', 'dni_do_waznosci': 'Dni do ważności', 'popyt_dzienny': 'Popyt dzienny (szt.)'}), width='stretch', hide_index=True)
        st.caption('Dane partii i wszystkie parametry kosztowe są syntetyczne/założone, nie pochodzą z NEUCA. Plan to rekomendacja: człowiek decyduje, a transfer i wstrzymanie partii wymagają akceptacji osoby odpowiedzialnej (QA).')

        # ---------- zatwierdzenie planu (człowiek decyduje) ----------
        st.divider(); st.subheader('Zatwierdzenie planu')
        if len(pl):
            kz1, kz2 = st.columns([2, 1]); kto = kz1.text_input('Planista (imię i nazwisko)', key='sl_kto'); uw = kz1.text_input('Uwagi / warunki (np. po akceptacji QA)', key='sl_uw')
            if kz2.button('Zatwierdzam plan', disabled=not kto.strip()):
                zat = pl.copy(); zat['zatwierdzil'] = kto.strip(); zat['uwagi'] = uw; zat['data_zatwierdzenia'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                st.session_state['sl_zat'] = zat
            if 'sl_zat' in st.session_state:
                st.success('Plan zatwierdzony w tej sesji (aplikacja nie składa zamówień ani transferów). Pobierz zapis decyzji:')
                st.download_button('Pobierz zatwierdzony plan (CSV)', st.session_state['sl_zat'].to_csv(index=False).encode('utf-8-sig'), 'plan_zatwierdzony.csv', 'text/csv')

        # ---------- asystent planisty ----------
        st.divider(); st.subheader('Asystent planisty')
        tryb = st.radio('Tryb asystenta', ['Demo (reguły, bez klucza)', 'LLM (Claude API)'], horizontal=True, key='sl_tryb')
        klucz = None
        if tryb.startswith('LLM'):
            try: klucz = st.secrets['ANTHROPIC_API_KEY']
            except Exception: klucz = None
            if not klucz: klucz = st.text_input('Klucz API (tylko na czas sesji, nie jest zapisywany)', type='password', key='sl_klucz') or None
            st.caption('W trybie LLM pytania i wyniki narzędzi są wysyłane do zewnętrznego dostawcy modelu. Tu dane są syntetyczne; przy danych firmowych wymagana jest zgoda zgodna z polityką spółki. Liczby w odpowiedzi pochodzą z narzędzi, a asystent je kontroluje.')
        else:
            st.caption('Tryb demo odpowiada regułami na podstawie tych samych narzędzi i tabel, co tryb LLM. Przykłady: „podsumuj sieć”, „dlaczego trastuzumab w Ołtarzewie”, „co jeśli rabat 30%”, „partie do obserwacji”.')
        hist_ch = st.session_state.setdefault('sl_chat', [])
        for m in hist_ch:
            with st.chat_message(m['role']): st.markdown(m['content'])
        pyt = st.chat_input('Zapytaj asystenta o partie, działania lub scenariusze…', key='sl_pyt')
        if pyt:
            hist_ch.append({'role': 'user', 'content': pyt})
            try:
                if tryb.startswith('LLM'):
                    if not klucz: raise ValueError('podaj klucz API albo przełącz na tryb demo.')
                    import anthropic
                    odp, wyn = asy.odpowiedz_llm([dict(role=m['role'], content=m['content']) for m in hist_ch[:-1]], pyt, partie, par, anthropic.Anthropic(api_key=klucz))
                else:
                    odp, wyn = asy.odpowiedz_demo(pyt, partie, par)
                sp = asy.liczby_spoza_narzedzi(odp, wyn)
                if sp: odp += '\n\n⚠️ Uwaga: liczby ' + ', '.join(f'{x:g}' for x in sp) + ' nie pochodzą z wyników narzędzi – sprawdź je w tabeli planu.'
            except Exception as e:
                odp = f'Asystent niedostępny: {e}'
            hist_ch.append({'role': 'assistant', 'content': odp}); st.rerun()
    except Exception as e:
        st.warning(f'SECOND LIFE niedostępny: {e}')
