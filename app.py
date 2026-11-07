import io
import numpy as np, pandas as pd, streamlit as st, altair as alt
from copilot import (brakujace_kolumny_symulacji, wczytaj_dane, do_tygodni, trenuj_i_prognozuj, metryki, mape, symuluj, podsumowanie, rekomendacje, roi)

st.set_page_config(page_title='AI Inventory Copilot', page_icon='📦', layout='wide')
st.title('📦 AI Inventory Copilot')
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

tab1, tab2, tab3, tab4 = st.tabs(['📈 Prognoza popytu', '✅ Rekomendacje', '⚖️ AS-IS vs AI', '💰 ROI'])

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
