"""AI Inventory Copilot - moduł prognozy popytu (używany w Colab i w aplikacji Streamlit)."""
import pandas as pd, numpy as np, lightgbm as lgb

FEATURES = ['lag_1','lag_2','lag_4','lag_52','ma4','std4','woy','month','sku_num']

def wczytaj_dane(hist, test):
    """hist/test: ścieżka albo obiekt pliku. Wymagane kolumny: date, sku_id, sku_name, category, demand_units."""
    h = pd.read_csv(hist, parse_dates=['date']); t = pd.read_csv(test, parse_dates=['date'])
    for d in (h, t):
        miss = {'date','sku_id','sku_name','category','demand_units'} - set(d.columns)
        if miss: raise ValueError(f'Brak kolumn: {miss}')
    return h, t

def do_tygodni(df):
    """Tygodnie wtorek-poniedziałek; niepełne tygodnie odrzucone."""
    d = df.copy(); d['week'] = d['date'].dt.to_period('W-MON').dt.start_time
    w = d.groupby(['week','sku_id','sku_name','category'], as_index=False).agg(demand=('demand_units','sum'), days=('date','nunique'))
    return w[w['days']==7].drop(columns='days').reset_index(drop=True)

def cechy(w):
    """Cechy z przeszłości (shift) - bez wycieku z przyszłości i bez kolumny seasonality_index."""
    w = w.sort_values(['sku_id','week']).reset_index(drop=True); g = w.groupby('sku_id')['demand']
    for k in (1,2,4,52): w[f'lag_{k}'] = g.shift(k)
    w['ma4'] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    w['std4'] = g.transform(lambda s: s.shift(1).rolling(4).std())
    w['woy'] = w['week'].dt.isocalendar().week.astype(int); w['month'] = w['week'].dt.month
    w['sku_num'] = w['sku_id'].str[-3:].astype(int)
    return w

def sezonowe_sku(train, prog=1.25):
    """SKU sezonowe: stosunek najwyższej do najniższej średniej miesięcznej w danych UCZĄCYCH > prog."""
    m = train.groupby(['sku_id', train['week'].dt.month])['demand'].mean().unstack()
    return list(m.index[(m.max(axis=1) / m.min(axis=1)) > prog])

def trenuj_i_prognozuj(w_hist, w_test):
    """Uczy na historii, zwraca prognozę 1 tydzień do przodu dla okresu testowego: ma4, lgb, hyb."""
    all_ = cechy(pd.concat([w_hist, w_test], ignore_index=True))
    split = w_test['week'].min()
    tr = all_[(all_['week'] < split)].dropna(subset=FEATURES); te = all_[all_['week'] >= split].copy()
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1).fit(tr[FEATURES], tr['demand'])
    te['lgb'] = np.maximum(m.predict(te[FEATURES]), 0)
    sez = sezonowe_sku(all_[all_['week'] < split])
    te['hyb'] = np.where(te['sku_id'].isin(sez), te['lgb'], te['ma4'])
    return te, sez, m

def mape(a, p):
    k = a > 0; return float(np.mean(np.abs(a[k]-p[k]) / a[k]) * 100)

def metryki(te):
    return {n: round(mape(te['demand'], te[c]), 1) for n, c in [('MA4','ma4'),('LightGBM','lgb'),('Hybryda','hyb')]}


# ================= KROK 2-3: rekomendacje i symulacja AS-IS vs AI =================
Z = 1.65  # poziom obsługi ok. 95%
SHARE_BULK = {'SKU_044':5,'SKU_045':5,'SKU_034':4,'SKU_007':3,'SKU_016':3,'SKU_046':3,'SKU_020':2.5,'SKU_029':3,'SKU_033':3,'SKU_040':3}

def _losowe(sku_id, n, seed=0):
    r = np.random.default_rng(int(sku_id[-3:]) + 1000 + 7919*seed)
    return dict(z_lt=r.normal(size=n), z_fill=r.normal(size=n), u_def=r.random(n), extra=r.integers(10,35,n), u_shelf=r.random(n))

def sym_sku(dem, mu_t, sigma, p, init, polityka, bulk=1.0, review=7, seed=0, z_ai=Z):
    """Symulacja dzienna jednego SKU. p: dict(price,cost,L,Lsig,fill,deficit,shelf). polityka: 'asis' (stały popyt + duże partie) lub 'ai' (prognoza + ograniczenie pokrycia)."""
    n = len(dem); rnd = _losowe(p['sku_id'], n, seed)
    L, shelf = int(p['L']), p['shelf']
    ss_const = Z * sigma * np.sqrt(L)
    batches = [[init['stock']/3, max(init['days'],5)], [init['stock']/3, max(init['days'],5)+0.3*shelf], [init['stock']/3, max(init['days'],5)+0.6*shelf]]
    pipe = []; out = []
    for i in range(n):
        for a in [x for x in pipe if x[0]==i]: batches.append([a[1], a[2]])
        pipe = [x for x in pipe if x[0]>i]
        batches.sort(key=lambda b: b[1]); need = dem[i]; sold = 0
        for b in batches:
            t = min(b[0], need); b[0] -= t; need -= t; sold += t
        for b in batches: b[1] -= 1
        wo = sum(b[0] for b in batches if b[1] <= 0)
        batches = [b for b in batches if b[1] > 0 and b[0] > 1e-9]
        end = sum(b[0] for b in batches); onord = sum(x[1] for x in pipe)
        mu = mu_t[i]
        if polityka == 'asis':
            rop = mu*L + ss_const; S = (mu*(L+14) + ss_const) * bulk
        else:
            # AI: uwzględnia zmienność popytu ORAZ czasu dostawy i zawodność dostawcy (fulfillment_rate)
            L_eff = L + (1 - p['fill']) * 22 * (1 if p['deficit'] else 0)
            ss = z_ai * np.sqrt(L_eff * sigma**2 + (mu * p['Lsig'])**2)
            rop = mu*L_eff + ss; S = min(mu*(L_eff+review) + ss, mu * 0.4 * shelf)
        if end + onord <= rop and S - end - onord > 0:
            q = S - end - onord
            lt = max(1, int(round(L + p['Lsig']*rnd['z_lt'][i])))
            if p['deficit'] and rnd['u_def'][i] > p['fill']: lt += int(rnd['extra'][i])
            q *= min(1, max(0.2, max(p['fill'],0.5) + 0.1*rnd['z_fill'][i]))
            pipe.append((i+lt, q, shelf*(0.35 + 0.55*rnd['u_shelf'][i])))
        out.append((end, dem[i]-sold, wo))
    return np.array(out)

def symuluj(hist, test, wt_forecast, n_sim=30, z_ai=Z):
    """Porównanie AS-IS vs AI na tym samym popycie. wt_forecast: tabela te z kolumną ma4 (prognoza tygodniowa)."""
    cat = test.groupby('sku_id').first()
    dni = sorted(test['date'].unique()); n = len(dni)
    tyg = pd.Series(pd.to_datetime(dni)).dt.to_period('W-MON').dt.start_time.values
    fc = wt_forecast.pivot(index='week', columns='sku_id', values='ma4')
    res = []
    for sku, g in test.sort_values('date').groupby('sku_id'):
        c = cat.loc[sku]; dem = g['demand_units'].values.astype(float)
        sig = hist[hist.sku_id==sku]['demand_units'].std(); mu_const = hist[hist.sku_id==sku]['demand_units'].mean()
        mu_ai = pd.Series(tyg).map(fc[sku]/7).ffill().bfill().values
        p = dict(sku_id=sku, price=c.price_pln, cost=c.unit_cost_pln, L=c.lead_time_days, Lsig=c.lead_time_sigma,
                 fill=c.fulfillment_rate, deficit=bool(c.is_deficit), shelf=c.shelf_life_days)
        init = dict(stock=g['stock_start'].iloc[0], days=g['days_to_expiry_oldest'].iloc[0]+1)
        for pol, mu, bulk in [('asis', np.full(n, mu_const), SHARE_BULK.get(sku,1.0)), ('ai', mu_ai, 1.0)]:
            os_ = [sym_sku(dem, mu, sig, p, init, pol, bulk, seed=k, z_ai=z_ai) for k in range(n_sim)]
            o = np.mean(os_, axis=0)
            res.append(dict(sku_id=sku, sku_name=c.sku_name, category=c.category, scenariusz=pol.upper(),
                zapas_sr_zl=float((o[:,0]*p['cost']).mean()), odpisy_zl=float((o[:,2]*p['cost']).sum()),
                utracona_marza_zl=float((o[:,1]*(p['price']-p['cost'])).sum()), utracona_sprzedaz_zl=float((o[:,1]*p['price']).sum()),
                popyt_szt=float(dem.sum()), braki_szt=float(o[:,1].sum())))
    return pd.DataFrame(res)

def podsumowanie(r):
    s = r.groupby('scenariusz').agg(zapas_sr_zl=('zapas_sr_zl','sum'), odpisy_zl=('odpisy_zl','sum'),
        utracona_marza_zl=('utracona_marza_zl','sum'), braki=('braki_szt','sum'), popyt=('popyt_szt','sum'))
    s['fill_rate'] = 1 - s['braki']/s['popyt']
    return s.drop(columns=['braki','popyt']).round(3)


# ================= REKOMENDACJE I ROI (używane w aplikacji) =================
def rekomendacje(test, te, z=Z, review=7):
    """Rekomendacja zamówień na ostatni dzień danych testowych. Prognoza = średnia z 4 ostatnich pełnych tygodni (MA4)."""
    t = test.sort_values('date'); ost = t.groupby('sku_id').tail(1).set_index('sku_id')
    wk = do_tygodni(t); mu = (wk.sort_values('week').groupby('sku_id').tail(4).groupby('sku_id')['demand'].mean() / 7)
    sig = t.groupby('sku_id')['demand_units'].std(); rows = []
    for sku, c in ost.iterrows():
        m = mu[sku]; s = sig[sku]; L = c.lead_time_days
        L_eff = L + ((1 - c.fulfillment_rate) * 22 if c.is_deficit else 0)
        ss = z * np.sqrt(L_eff * s**2 + (m * c.lead_time_sigma)**2)
        rop = m * L_eff + ss; S = min(m * (L_eff + review) + ss, m * 0.4 * c.shelf_life_days)
        pos = c.stock_end + c.on_order_units; pokrycie = c.stock_end / m if m > 0 else np.inf
        braki = pos <= rop; nadmiar = pos > 1.5 * S
        przeterm = bool(c.stock_end > 0 and pokrycie > c.days_to_expiry_oldest and c.days_to_expiry_oldest < 90)
        if braki: rek, q = 'ZAMÓW', int(np.ceil(S - pos))
        elif nadmiar: rek, q = 'WSTRZYMAJ / ZMNIEJSZ ZAPAS', 0
        else: rek, q = 'OK', 0
        ryzyko = [x for x, f in [('Brak towaru', braki), ('Nadmiar zapasu', nadmiar), ('Przeterminowanie', przeterm)] if f]
        rows.append(dict(sku_id=sku, sku_name=c.sku_name, category=c.category, prognoza_dzienna=round(m, 1), stan_magazynowy=round(c.stock_end),
            w_drodze=round(c.on_order_units), punkt_zamowienia=round(rop), poziom_docelowy=round(S), pokrycie_dni=round(min(pokrycie, 999)),
            dni_do_waznosci_najstarszej=int(c.days_to_expiry_oldest), rekomendacja=rek, ilosc_do_zamowienia=q,
            ryzyko=', '.join(ryzyko) if ryzyko else '-', wartosc_zapasu_zl=round(c.stock_end * c.unit_cost_pln)))
    df = pd.DataFrame(rows)
    kol = {'ZAMÓW': 0, 'WSTRZYMAJ / ZMNIEJSZ ZAPAS': 1, 'OK': 2}
    df['_k'] = df['rekomendacja'].map(kol)
    return df.sort_values(['_k', 'wartosc_zapasu_zl'], ascending=[True, False]).drop(columns='_k').reset_index(drop=True)

def roi(delta_zapas, delta_marza, odpisy_plik, zal, dni=180):
    """zal: koszt_kapitalu, koszt_utrzymania, realizacja, odzysk_marzy, unikane_odpisy, wdrozenie, utrzymanie, dyskonto. Zwraca dict wyników."""
    a = 365 / dni; d = delta_zapas * zal['realizacja']
    k = dict(kapital=d * zal['koszt_kapitalu'], utrzymanie_zapasu=d * zal['koszt_utrzymania'],
             marza=delta_marza * a * zal['odzysk_marzy'], odpisy=odpisy_plik * a * zal['unikane_odpisy'])
    rocznie = sum(k.values()); netto = rocznie - zal['utrzymanie']; r = zal['dyskonto']
    npv = -zal['wdrozenie'] + sum(netto / (1 + r)**t for t in (1, 2, 3))
    return dict(korzysci=k, rocznie=rocznie, netto=netto, npv=npv,
                roi_3l=(3 * netto - zal['wdrozenie']) / zal['wdrozenie'] if zal['wdrozenie'] else np.nan,
                zwrot_lata=zal['wdrozenie'] / netto if netto > 0 else np.inf,
                maks_wdrozenie=netto * (1 - (1 + r)**-3) / r if netto > 0 else 0)
