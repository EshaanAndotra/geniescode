"""Leakage-safe direct benchmark-price forecasting.

Every historical feature is computed from rows at or before its origin.
Known demand is the only permitted future covariate. Training targets are
restricted to target_week <= cutoff, including overlapping forecast horizons.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

SELECTED_MODEL = 'revert_0.35'


def _base_features(data):
    prices = data['prices'].sort_values(['material','week','country'])
    o = data['outcomes'].sort_values(['material','week'])
    p = prices.groupby(['material','week']).agg(spot=('spot_price','mean'),
        futures=('futures_4w','mean'), spot_min=('spot_price','min'),
        spot_max=('spot_price','max'))
    p['spread'] = np.log(p.futures / p.spot)
    p['dispersion'] = np.log(p.spot_max / p.spot_min)
    p = p.join(o.set_index(['material','week'])[['disrupted']])
    sh = data['supply_history'].merge(data['suppliers'],on=['material','country'],validate='many_to_one')
    sh['capacity_ratio'] = sh.capacity_units / sh.normal_capacity_units
    sh['lead_extra'] = sh.lead_time_weeks - sh.base_lead_weeks
    ag = sh.groupby(['material','week']).agg(cap_min=('capacity_ratio','min'),
        cap_mean=('capacity_ratio','mean'),lead_extra=('lead_extra','max'),
        landed=('landed_cost','mean'))
    p = p.join(ag)
    tf = data['tariffs'].copy()
    tf['announced_change'] = (tf.announced_rate - tf.tariff_rate).fillna(0)
    p = p.join(tf.groupby(['material','week']).agg(tariff=('tariff_rate','mean'),announcement=('announced_change','max')))
    # Supplier-linked macro and shipping series, with each supplier counted once.
    links = data['suppliers'][['material','country','lane']]
    ma = data['macro'].merge(links,on='country')
    p = p.join(ma.groupby(['material','week'])[['pmi','fx_rate','cpi_yoy','policy_rate','tension_index']].mean())
    ship = data['shipping'].merge(links,on='lane')
    p = p.join(ship.groupby(['material','week'])[['container_rate','transit_days','congestion_index']].mean())
    p = p.join(data['demand'].set_index(['material','week'])[['demand_units']])
    group = p.groupby(level='material')
    for lag in [1,2,4,8,13,26,52]:
        p[f'ret{lag}'] = np.log(p.spot / group.spot.shift(lag))
    for width in [3,8,26]:
        avg = group.spot.transform(lambda s: s.ewm(span=width,adjust=False).mean())
        p[f'ema{width}'] = np.log(avg / p.spot)
    for col in ['spread','dispersion','disrupted','cap_min','lead_extra','pmi','tension_index','transit_days','congestion_index']:
        p[col+'_4mean'] = p.groupby(level='material')[col].transform(lambda s:s.rolling(4,min_periods=1).mean())
    for col in ['pmi','fx_rate','cpi_yoy','tension_index','container_rate','transit_days','congestion_index']:
        p[col+'_change4'] = p[col]-p.groupby(level='material')[col].shift(4)
    p['disrupted_rate'] = p.groupby(level='material').disrupted.transform(lambda s:s.expanding().mean())
    p['landed_ratio'] = p.landed / p.spot
    p['global_ret4'] = p.groupby(level='week').ret4.transform('mean')
    p['global_ret13'] = p.groupby(level='week').ret13.transform('mean')
    p = p.reset_index()
    p['season_sin'] = np.sin(2*np.pi*p.week/52)
    p['season_cos'] = np.cos(2*np.pi*p.week/52)
    return p


def _examples(data, origin, requested_weeks=None):
    # Truncate first, not just after joining: no future feature preprocessing.
    past = {k:(v.loc[v.week<=origin].copy() if 'week' in v and k!='demand' else v.copy()) for k,v in data.items()}
    b = _base_features(past)
    if requested_weeks is None:
        base = b.loc[b.week>=27]
        hs = range(1,27)
    else:
        base = b.loc[b.week==origin]
        hs = [int(w)-origin for w in requested_weeks]
    examples = []
    for h in hs:
        t = base.copy()
        t['horizon'] = h
        t['hroot'] = np.sqrt(h)
        t['hlog'] = np.log1p(h)
        t['target_week'] = t.week+h
        for col in ['ret1','ret4','ret13','ret26','ema3','ema8','ema26','spread','disrupted','cap_min','dispersion','lead_extra','tension_index']:
            t[col+'_h'] = t[col] * np.sqrt(h)
        examples.append(t)
    ex = pd.concat(examples,ignore_index=True)
    dem = data['demand'][['week','material','demand_units']].rename(columns={'week':'target_week','demand_units':'future_demand'})
    ex = ex.merge(dem,on=['material','target_week'],how='left',validate='many_to_one')
    ex['demand_ratio'] = np.log(ex.future_demand/ex.demand_units)
    if requested_weeks is None:
        target = past['outcomes'][['week','material','benchmark_price']].rename(columns={'week':'target_week','benchmark_price':'target_price'})
        ex = ex.merge(target,on=['material','target_week'],how='inner',validate='many_to_one')
        ex = ex.loc[ex.target_week<=origin].copy()
        ex['target'] = np.log(ex.target_price/ex.spot)
    return ex


def _matrix(ex):
    omit = ['material','week','target_week','target_price','target','spot','futures','spot_min','spot_max','landed','demand_units','future_demand']
    x = ex.drop(columns=[c for c in omit if c in ex]).replace([np.inf,-np.inf],np.nan).fillna(0)
    for mat in sorted(ex.material.unique()):
        x['material_'+mat] = (ex.material==mat).astype(float)
    return x


def candidate_forecasts(data, origin, weeks, fit_complex=True):
    test = _examples(data,origin,weeks)
    result = test[['material','target_week','spot']].rename(columns={'target_week':'week','spot':'carry'})
    # Parsimonious causal references, compared before selecting a model.
    result['futures'] = test.futures.values
    result['futures_half'] = np.sqrt(test.spot.values * test.futures.values)
    for strength in [0.15,0.35,0.6,1.0]:
        result[f'revert_{strength}'] = test.spot.values * np.exp(strength*test.ema26.values*(1-np.exp(-test.horizon.values/8)))
    for strength in [0.1,0.25,0.5]:
        result[f'trend_{strength}'] = test.spot.values*np.exp(np.clip(strength*test.ret13.values*test.horizon.values/13,-0.2,0.2))
    if fit_complex:
        train = _examples(data,origin)
        x = _matrix(train)
        xt = _matrix(test).reindex(columns=x.columns,fill_value=0)
        y = train.target
        for alpha in [100,1000,10000]:
            model = make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(x,y)
            yp = np.clip(model.predict(xt),-0.6,0.6)
            result[f'ridge_{alpha}'] = test.spot.values * np.exp(yp)
        model = HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=7,
            learning_rate=.05,min_samples_leaf=100,l2_regularization=20,random_state=2026)
        model.fit(x,y)
        result['hist_gb'] = test.spot.values*np.exp(np.clip(model.predict(xt),-.6,.6))
        result['ridge_blend'] = .5*result.carry+.5*result.ridge_10000
        result['hist_blend'] = .5*result.carry+.5*result.hist_gb
    return result


def predict_prices(data, origin, weeks):
    """Return frozen-origin forecasts for requested weeks (no recursive updates)."""
    f = candidate_forecasts(data,int(origin),list(weeks),fit_complex=SELECTED_MODEL not in ['carry','futures','futures_half'] and not SELECTED_MODEL.startswith(('revert_','trend_')))
    return f[['week','material',SELECTED_MODEL]].rename(columns={SELECTED_MODEL:'price_forecast'})


def validate_price_models(data, origins=(104,130,156), test_origin=182):
    rows=[]
    predictions=[]
    actual=data['outcomes'][['week','material','benchmark_price']]
    for origin in [*origins,test_origin]:
        pred = candidate_forecasts(data,origin,range(origin+1,origin+27)).merge(actual,on=['week','material'],validate='one_to_one')
        pred['origin']=origin
        predictions.append(pred)
        for col in pred.columns:
            if col in ['week','material','benchmark_price','origin']:
                continue
            for region, mask in [('all',np.ones(len(pred),dtype=bool)),('h1_13',pred.week<=origin+13),('h14_26',pred.week>origin+13)]:
                err=np.abs(pred.loc[mask,col]/pred.loc[mask,'benchmark_price']-1)
                rows.append(dict(origin=origin,split='test' if origin==test_origin else 'tuning',window=region,model=col,mape=float(err.mean())))
        print(f'price origin {origin} complete',flush=True)
    return pd.DataFrame(rows),pd.concat(predictions,ignore_index=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='C:/Users/eshaa/Downloads')
    ap.add_argument('--validate',action='store_true')
    args=ap.parse_args()
    names=['prices','outcomes','demand','supply_history','tariffs','shipping','macro','suppliers']
    data={n:pd.read_csv(Path(args.data_dir)/(n+'.csv')) for n in names}
    if args.validate:
        metrics,preds=validate_price_models(data)
        metrics.to_csv('price_validation.csv',index=False)
        preds.to_csv('price_backtest_predictions.csv',index=False)
        print(metrics.query("window=='all'").pivot(index='model',columns='origin',values='mape').to_string())
    else:
        predict_prices(data,208,range(209,235)).to_csv('price_forecasts.csv',index=False)
