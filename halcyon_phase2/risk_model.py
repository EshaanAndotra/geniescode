"""Direct multi-horizon disruption forecasts with purged rolling-origin validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from news_features import get_news_features

def build_features(data):
    """One feature row per material/origin; no input newer than that origin."""
    sp=data['suppliers']; max_week=int(data['outcomes'].week.max())
    p=data['prices'].merge(data['supply_history'],on=['week','date','material','country'],validate='one_to_one')
    p=p.merge(data['tariffs'],on=['week','date','material','country'],validate='one_to_one')
    p=p.merge(sp,on=['material','country'],validate='many_to_one')
    p=p.merge(data['shipping'],on=['week','date','lane'],validate='many_to_one')
    p=p.merge(data['macro'],on=['week','date','country'],validate='many_to_one')
    p['capacity_ratio']=p.capacity_units/p.normal_capacity_units
    p['lead_excess']=p.lead_time_weeks-p.base_lead_weeks
    p['futures_premium']=p.futures_4w/p.spot_price-1
    p['tariff_excess']=p.tariff_rate-p.base_tariff
    p['announced_change']=(p.announced_rate-p.tariff_rate).fillna(0)
    p['announce_wait']=(p.announced_effective_week-p.week).fillna(30)
    fields=['capacity_ratio','lead_excess','futures_premium','tariff_excess','announced_change',
            'announce_wait','tension_index','pmi','fx_rate','cpi_yoy','policy_rate',
            'congestion_index','transit_days','container_rate']
    agg=p.groupby(['material','week'])[fields].agg(['mean','min','max'])
    agg.columns=['_'.join(c) for c in agg.columns]
    outcome=data['outcomes'].set_index(['material','week']).sort_index()
    agg=agg.join(outcome[['disrupted','benchmark_price']]).sort_index()
    rows=[]
    for material in sorted(sp.material.unique()):
        g=agg.loc[material].sort_index()
        suppliers=sp[sp.material==material]
        for origin in range(26,max_week+1):
            hist=g.loc[:origin]; last=hist.iloc[-1]
            f={'origin':origin,'material':material}
            for col in agg.columns:
                if col=='benchmark_price': continue
                f[col]=float(last[col])
                if col.endswith('_mean') or col=='disrupted':
                    for window in [4,13]:
                        f[f'{col}_avg{window}']=float(hist[col].tail(window).mean())
                    f[f'{col}_delta4']=float(last[col]-hist[col].iloc[-5])
            y=hist.disrupted.to_numpy()
            run=1
            for v in y[-2::-1]:
                if v!=y[-1]: break
                run+=1
            f['state_age']=run
            f['historical_rate']=float(y.mean())
            f['recent_rate26']=float(hist.disrupted.tail(26).mean())
            prices=hist.benchmark_price
            for w in [1,4,13,26]:
                f[f'price_return{w}']=float(prices.iloc[-1]/prices.iloc[max(0,len(prices)-1-w)]-1)
            f['price_vol13']=float(prices.pct_change().tail(13).std())
            for c in ['KES','VAL','SUN','ORI','HAL','AUR']:
                f['supplies_'+c]=float(c in suppliers.country.values)
            for m in sorted(sp.material.unique()): f['is_'+m]=float(m==material)
            f.update(get_news_features(data['news'],sp,origin,material))
            rows.append(f)
    return pd.DataFrame(rows).fillna(0)

def horizon_rows(features,data,origins=None):
    fs=(features if origins is None else features[features.origin.isin(origins)]).copy()
    rows=pd.concat([fs.assign(horizon=h,week=fs.origin+h) for h in range(1,27)],ignore_index=True)
    demand=data['demand'].rename(columns={'demand_units':'known_demand'})
    rows=rows.merge(demand[['week','material','known_demand']],on=['week','material'],how='left',validate='many_to_one')
    # Known forward production schedule is supplied for all 234 weeks.
    # Do not normalize using future data: capacity is static and already known.
    capacity=data['suppliers'].groupby('material').normal_capacity_units.sum()
    rows['known_demand_ratio']=rows.known_demand/rows.material.map(capacity)
    rows=rows.drop(columns='known_demand')
    rows['horizon_log']=np.log1p(rows.horizon)
    rows['target_sin']=np.sin(2*np.pi*rows.week/52)
    rows['target_cos']=np.cos(2*np.pi*rows.week/52)
    targets=data['outcomes'][['week','material','disrupted']].rename(columns={'disrupted':'target'})
    return rows.merge(targets,on=['week','material'],how='left',validate='many_to_one')

def markov_forecast(data,origin,frame):
    """Shrunk two-state transition model with geometric episode duration."""
    hist=data['outcomes'].query('week<=@origin').sort_values(['material','week'])
    transitions=[]
    for _,g in hist.groupby('material'):
        y=g.disrupted.to_numpy(); transitions.extend(zip(y[:-1],y[1:]))
    a=np.asarray(transitions)
    global01=float(a[a[:,0]==0,1].mean()); global10=float((1-a[a[:,0]==1,1]).mean())
    result=[]
    for row in frame.itertuples():
        y=hist[hist.material==row.material].disrupted.to_numpy()
        n0=max(0,(y[:-1]==0).sum()); n1=max(0,(y[:-1]==1).sum())
        p01=(np.sum((y[:-1]==0)&(y[1:]==1))+global01*45)/(n0+45)
        p10=(np.sum((y[:-1]==1)&(y[1:]==0))+global10*30)/(n1+30)
        eq=p01/(p01+p10)
        result.append(eq+(float(y[-1])-eq)*(1-p01-p10)**row.horizon)
    return np.clip(result,0.01,0.99)

def fit_predict_candidates(data,table,origin):
    train=table[(table.week<=origin)&table.target.notna()]
    test=table[table.origin==origin].copy()
    columns=[c for c in table if c not in ['origin','material','week','target']]
    X=train[columns].fillna(0); Xt=test[columns].fillna(0)
    y=train.target.astype(int)
    hist=data['outcomes'].query('week<=@origin').groupby('material').disrupted.mean()
    output=test[['origin','week','material','horizon','target']].copy()
    output['historical']=test.material.map(hist).to_numpy()
    output['markov']=markov_forecast(data,origin,test)
    models={
       'hgb_small':HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=7,min_samples_leaf=100,l2_regularization=30,learning_rate=.05,early_stopping=False,random_state=2026),
       'hgb_deep':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=120,l2_regularization=40,learning_rate=.04,early_stopping=False,random_state=2026),
       'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.015,max_iter=700))}
    for name,model in models.items():
        model.fit(X,y)
        output[name]=np.clip(model.predict_proba(Xt)[:,1],.005,.995)
    output['blend']=.5*output.hgb_small+.25*output.logistic+.25*output.markov
    output['conservative']=.5*output.blend+.5*output.historical
    from risk_challenger import onset_forecast
    for shrinkage in [10,30]:
        for decay in [3,6]:
            output[f'onset_s{shrinkage}_d{decay}']=onset_forecast(data,table,origin,shrinkage,decay)
    return output

def select_and_forecast(data, output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(exist_ok=True,parents=True)
    features=build_features(data)
    features.to_csv(output_dir/'risk_features.csv',index=False)
    table=horizon_rows(features,data)
    cuts=[104,130,156,182]
    frames=[]
    for origin in cuts:
        print(f'Risk rolling-origin fold {origin}',flush=True)
        frames.append(fit_predict_candidates(data,table,origin))
    validation=pd.concat(frames,ignore_index=True)
    candidates=['historical','markov','hgb_small','hgb_deep','logistic','blend','conservative',
                'onset_s10_d3','onset_s10_d6','onset_s30_d3','onset_s30_d6']
    tuning=validation[validation.origin<182]
    metrics={c:float(np.mean((tuning[c]-tuning.target)**2)) for c in candidates}
    winner=min(metrics,key=metrics.get)
    hold=validation[validation.origin==182]
    hold_metrics={c:float(np.mean((hold[c]-hold.target)**2)) for c in candidates}
    final=fit_predict_candidates(data,table,int(data['outcomes'].week.max()))
    final['disruption_prob']=final[winner]
    validation['disruption_prob']=validation[winner]
    validation.to_csv(output_dir/'risk_backtest_predictions.csv',index=False)
    final.to_csv(output_dir/'risk_forecasts.csv',index=False)
    report={'selected_model':winner,'tuning_origins':cuts[:-1],'holdout_origin':182,
            'tuning_brier':metrics,'holdout_brier':hold_metrics,
            'holdout_brier_skill':1-hold_metrics[winner]/hold_metrics['historical'],
            'feature_count':len(features.columns)-2,'max_training_target_week':int(data['outcomes'].week.max()),
            'validation':'Fixed origin, horizons 1-26; all training labels end at or before each origin. Model chosen only on origins104/130/156.'}
    (output_dir/'risk_validation.json').write_text(json.dumps(report,indent=2))
    return final[['week','material','disruption_prob']],report,validation
