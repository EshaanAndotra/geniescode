"""Supplier-aware, fixed-origin forecasts for the final 13-week horizon.

Every feature uses contemporaneous/past records; every fitted target ends at
or before the forecast origin. Known future demand and announced tariff dates
are the only future covariates. Supplier shock proxies are audited to outcomes.
"""
from pathlib import Path
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from news_features import _prepare,TYPES


def supplier_news_panel(data):
    """Track each supplier's warning stages and age without reading future news."""
    records=_prepare(data['news']); byweek={}
    for r in records:byweek.setdefault(r['week'],[]).append(r)
    maxweek=int(data['outcomes'].week.max());result=[]
    for s in data['suppliers'].itertuples():
        states={}
        for w in range(1,maxweek+1):
            for r in byweek.get(w,[]):
                if r['materials'] and s.material not in r['materials']:continue
                channel='lane' if r['lanes'] else 'supplier'
                match=s.lane in r['lanes'] if r['lanes'] else s.country in r['countries']
                if not match:continue
                if r['stage']=='resolved':
                    states={k:v for k,v in states.items() if k[0]!=channel}
                    continue
                key=(channel,r['type']);old=states.get(key)
                first=old['first'] if old and w-old['last']<=13 else w
                stage_first=old['stage_first'] if old and old['stage']==r['stage'] and w-old['last']<=13 else w
                states[key]={'stage':r['stage'],'last':w,'first':first,'stage_first':stage_first}
            row={'week':w,'material':s.material,'country':s.country}
            for kind in TYPES:
                st=[v for (channel,k),v in states.items() if k==kind and w-v['last']<= (20 if v['stage']=='active' else 10)]
                for stage in ['early','late','active']:
                    found=[v for v in st if v['stage']==stage]
                    row[f'n_{kind}_{stage}']=float(bool(found))
                row[f'n_{kind}_age']=min([w-v['first'] for v in st]+[26])
                row[f'n_{kind}_last']=min([w-v['last'] for v in st]+[26])
            result.append(row)
    return pd.DataFrame(result)


def make_panel(data):
    keys=['week','material','country']
    p=data['prices'].drop(columns='date').merge(data['supply_history'].drop(columns='date'),on=keys,validate='one_to_one')
    p=p.merge(data['tariffs'].drop(columns='date'),on=keys,validate='one_to_one').merge(data['suppliers'],on=['material','country'],validate='many_to_one')
    p=p.merge(data['shipping'].drop(columns='date'),on=['week','lane'],validate='many_to_one').merge(data['macro'].drop(columns='date'),on=['week','country'],validate='many_to_one')
    p=p.merge(data['demand'].drop(columns='date'),on=['week','material'],validate='many_to_one')
    p=p.merge(supplier_news_panel(data),on=keys,validate='one_to_one').sort_values(['material','country','week']).reset_index(drop=True)
    p['capacity_ratio']=p.capacity_units/p.normal_capacity_units
    p['lead_extra']=p.lead_time_weeks-p.base_lead_weeks
    p['tariff_extra']=p.tariff_rate-p.base_tariff
    p['shock']=((p.capacity_ratio<.85)|(p.lead_extra>0)|(p.tariff_extra>.01)).astype(int)
    proxy=p.groupby(['week','material']).shock.max()
    truth=data['outcomes'].set_index(['week','material']).disrupted
    if not np.array_equal(proxy.reindex(truth.index).values,truth.values):
        raise ValueError('Supplier shock proxy no longer matches material outcomes; inspect before retraining.')
    p['premium']=np.log(p.futures_4w/p.spot_price)
    p['landed_ratio']=p.landed_cost/p.spot_price
    p['demand_capacity']=p.demand_units/p.normal_capacity_units
    p['announce_delta']=(p.announced_rate-p.tariff_rate).fillna(0)
    p['announce_wait']=(p.announced_effective_week-p.week).fillna(50)
    g=p.groupby(['material','country'],sort=False)
    for lag in [1,4,13,26]:p[f'return{lag}']=np.log(p.spot_price/g.spot_price.shift(lag))
    for col in ['premium','shock','capacity_ratio','lead_extra','tension_index','congestion_index']:
        p[col+'_mean4']=g[col].transform(lambda x:x.rolling(4,min_periods=1).mean())
    for col in ['pmi','fx_rate','tension_index','container_rate','transit_days','congestion_index']:
        p[col+'_delta4']=p[col]-g[col].shift(4)
    p['shock_prior']=g.shock.transform(lambda x:x.expanding().mean())
    p['ema26']=np.log(g.spot_price.transform(lambda x:x.ewm(span=26,adjust=False).mean())/p.spot_price)
    p['price_vol']=g.return1.transform(lambda x:x.rolling(13,min_periods=2).std())
    p['state_age']=g.shock.transform(lambda x:x.groupby(x.ne(x.shift()).cumsum()).cumcount()+1)
    p['tariff_age']=g.tariff_extra.transform(lambda x:x.groupby(x.ne(x.shift()).cumsum()).cumcount()+1)
    for col in ['country','lane']:
        for value in sorted(p[col].unique()):p[col+'_'+value]=(p[col]==value).astype(float)
    return p


def examples(panel,data):
    base=panel[panel.week>=2].rename(columns={'week':'origin'})
    ex=pd.concat([base.assign(horizon=h,week=base.origin+h) for h in range(1,14)],ignore_index=True)
    future=panel[['week','material','country','shock','spot_price','capacity_ratio','lead_extra']].rename(columns={c:'target_'+c for c in ['shock','spot_price','capacity_ratio','lead_extra']})
    ex=ex.merge(future,on=['week','material','country'],how='left',validate='many_to_one')
    ex['target_return']=np.log(ex.target_spot_price/ex.spot_price)
    ex['announced_active']=((ex.announce_delta>0)&(ex.horizon>=ex.announce_wait)).astype(float)
    ex['after_announcement']=np.maximum(0,ex.horizon-ex.announce_wait)
    ex['horizon_log']=np.log1p(ex.horizon)
    ex['target_sin']=np.sin(2*np.pi*ex.week/52);ex['target_cos']=np.cos(2*np.pi*ex.week/52)
    dem=data['demand'][['week','material','demand_units']].rename(columns={'demand_units':'future_demand'})
    ex=ex.merge(dem,on=['week','material'],how='left',validate='many_to_one')
    ex['demand_ratio']=ex.future_demand/ex.demand_units
    for col in ['return4','return13','premium','ema26']:
        ex[col+'_h']=ex[col]*np.sqrt(ex.horizon)
    return ex


def feature_columns(ex):
    omit={'week','origin','material','country','country_name','lane','spot_price','futures_4w','landed_cost',
          'capacity_units','normal_capacity_units','demand_units','future_demand','announced_rate','announced_effective_week'}
    return [c for c in ex.select_dtypes(include='number') if c not in omit and not c.startswith('target_') or c in ['target_sin','target_cos']]


def country_transition(panel,origin,test):
    hist=panel[panel.week<=origin];seqs={k:g.sort_values('week').shock.to_numpy() for k,g in hist.groupby(['material','country'])}
    transitions=np.array([a for y in seqs.values() for a in zip(y[:-1],y[1:])])
    g01=(transitions[(transitions[:,0]==0),1].sum()+1)/((transitions[:,0]==0).sum()+100)
    g10=((1-transitions[(transitions[:,0]==1),1]).sum()+2)/((transitions[:,0]==1).sum()+20)
    probs=[]
    for r in test.itertuples():
        y=seqs[(r.material,r.country)];p01=(((y[:-1]==0)&(y[1:]==1)).sum()+g01*50)/((y[:-1]==0).sum()+50)
        p10=(((y[:-1]==1)&(y[1:]==0)).sum()+g10*30)/((y[:-1]==1).sum()+30)
        eq=p01/(p01+p10);probs.append(eq+(y[-1]-eq)*(1-p01-p10)**r.horizon)
    return np.asarray(probs)


def predict_candidates(data,panel,table,origin):
    train=table[(table.week<=origin)&table.target_shock.notna()]
    test=table[table.origin==origin].copy()
    cols=feature_columns(table);X=train[cols].replace([np.inf,-np.inf],np.nan).fillna(0);Xt=test[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    f=test[['origin','week','horizon','material','country','lane','spot_price','premium','ema26','capacity_ratio','lead_extra','base_lead_weeks',
            'announced_active','announce_wait','announce_delta','tariff_rate','base_tariff','normal_capacity_units']].copy()
    f['transition']=country_transition(panel,origin,test)
    with threadpool_limits(limits=4):
        for name,leaves,l2 in [('tree_small',7,12),('tree_large',15,25)]:
            model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=leaves,min_samples_leaf=60,
                l2_regularization=l2,learning_rate=.06,early_stopping=False,random_state=2026)
            if train.target_shock.nunique()>1:
                model.fit(X,train.target_shock)
                f[name]=np.clip(model.predict_proba(Xt)[:,1],.005,.995)
            else:f[name]=f.transition
        # Confirmed scheduled duties are observable forward commitments. Their
        # effect is explicit rather than relying on trees to discover equality.
        for name in ['tree_small','tree_large']:
            active=f.announced_active.eq(1)&(f.horizon-f.announce_wait<=7)
            f.loc[active,name]=np.maximum(f.loc[active,name],.98)
        f['tree_blend']=.45*f.tree_small+.35*f.tree_large+.20*f.transition
        for name,target,lower,upper in [('capacity_mean','target_capacity_ratio',0,1.0),('delay_mean','target_lead_extra',0,10),('price_tree_return','target_return',-.4,.5)]:
            reg=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=70,
                l2_regularization=12,learning_rate=.05,early_stopping=False,random_state=2026)
            reg.fit(X,train[target]);f[name]=np.clip(reg.predict(Xt),lower,upper)
        pricecols=[c for c in cols if c in ['horizon','horizon_log','premium','premium_mean4','return1','return4','return13','return26','ema26','price_vol','capacity_ratio','lead_extra','tariff_extra','announce_delta','announced_active','tension_index_delta4','fx_rate_delta4','demand_ratio','premium_h','return4_h','return13_h','ema26_h']]
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=1000))
        ridge.fit(X[pricecols],train.target_return);f['price_ridge_return']=np.clip(ridge.predict(Xt[pricecols]),-.4,.5)
    f['price_carry']=f.spot_price
    f['price_original']=f.spot_price*np.exp(.35*f.ema26*(1-np.exp(-f.horizon/8)))
    f['price_futures']=f.spot_price*np.exp(np.clip(f.premium,-.2,.2))
    f['price_tree']=f.spot_price*np.exp(f.price_tree_return)
    f['price_ridge']=f.spot_price*np.exp(f.price_ridge_return)
    f['price_blend']=.5*f.price_tree+.25*f.price_ridge+.25*f.price_carry
    f['price_conservative']=.5*f.price_tree+.5*f.price_carry
    return f.reset_index(drop=True)


def aggregate_predictions(supplier):
    result=[]
    probcols=['transition','tree_small','tree_large','tree_blend']
    pricecols=[c for c in supplier if c.startswith('price_') and not c.endswith('_return')]
    for (origin,week,mat),g in supplier.groupby(['origin','week','material']):
        row={'origin':origin,'week':week,'material':mat,'horizon':week-origin}
        for c in probcols:
            row[c+'_union']=1-np.prod(1-g[c])
            row[c+'_lane']=1-np.prod(1-g.groupby('lane')[c].max())
            row[c+'_max']=g[c].max()
        for c in pricecols:row[c]=g[c].mean()
        result.append(row)
    return pd.DataFrame(result)


def tariff_survival(panel,origin,elapsed):
    """Empirical survival of announced tariff episodes, including censoring."""
    episodes=[]
    for (mat,country),g in panel[panel.week<=origin].groupby(['material','country']):
        g=g.sort_values('week');mask=g.tariff_extra.gt(.01).to_numpy();weeks=g.week.to_numpy()
        start=None
        for i,hit in enumerate(mask):
            if hit and start is None:start=int(weeks[i])
            if start is not None and (not hit or i==len(mask)-1):
                end=int(weeks[i] if hit else weeks[i-1]);complete=not hit
                episodes.append((country,start,end-start+1,complete));start=None
    episodes=list(set(episodes)) # a country-wide tariff is not three independent events
    if not episodes:return float(np.exp(-max(0,elapsed-7)/12))
    survival=1.0
    for t in sorted({length for _,_,length,complete in episodes if complete and length<=elapsed}):
        at_risk=sum(length>=t for _,_,length,_ in episodes)
        deaths=sum(length==t and complete for _,_,length,complete in episodes)
        survival*=1-deaths/(at_risk+1.0)
    return float(survival)


def add_robust_candidates(frame,panel,data):
    """Preserve active-event persistence; use signal models for new warnings.

    The gate depends only on the origin's state, never on realized future
    shocks. Explicit announced duties use the empirically observed duration.
    """
    a=frame.copy();origin=int(a.origin.iloc[0]);now=panel[panel.week.eq(origin)]
    active=data['outcomes'].query('week==@origin').set_index('material').disrupted
    warning_cols=[c for c in now if c.startswith('n_') and c.endswith(('_early','_late'))]
    signals=(now[warning_cols].sum(axis=1)>0)|(now.premium>.03)
    warned=now.assign(warned=signals).groupby('material').warned.max()
    # The original Markov path is available directly from origin-local labels
    # for final forecasts, when no historical reference candidate was stored.
    if 'original_risk' not in a or a.original_risk.isna().all():
        from risk_model import markov_forecast
        a['original_risk']=markov_forecast(data,origin,a)
    if 'original_price' not in a or a.original_price.isna().all():
        from price_model import predict_prices
        a=a.drop(columns='original_price',errors='ignore').merge(predict_prices(data,origin,range(origin+1,origin+14)).rename(columns={'price_forecast':'original_price'}),on=['week','material'])
    # Expert gate: trust the direct supplier model when markets/transport
    # corroborate stress; otherwise retain the less brittle transition path.
    # These fixed thresholds are expressed in source units, not future labels.
    stress=now.assign(strong=(now.premium>.035)|(now.tension_index_delta4>12)|
        (now.congestion_index_delta4>8)|(now.announce_delta>0)|
        (now.tension_index>40)|(now.congestion_index>45)).groupby('material').strong.max()
    a['gated_risk']=np.where(a.material.map(stress),.8*a.tree_blend_union+.2*a.original_risk,a.original_risk)
    for source in ['tree_blend_union','tree_small_lane']:
        weight=np.where(a.material.map(warned),.8,.35)
        weight=np.where(a.material.map(active).eq(1),0,weight)
        a['robust_'+source]=weight*a[source]+(1-weight)*a.original_risk
    a['price_mix']=.5*a.price_futures+.5*a.price_tree
    a['price_stable_mix']=.5*a.original_price+.5*a.price_blend
    a['known_tariff_probability']=0.0
    for r in now.itertuples():
        if r.announce_delta>0 and r.announce_wait>0:
            for i in a.index[a.material.eq(r.material)&a.horizon.ge(r.announce_wait)]:
                elapsed=int(a.loc[i,'horizon']-r.announce_wait)
                a.loc[i,'known_tariff_probability']=max(a.loc[i,'known_tariff_probability'],.99*tariff_survival(panel,origin,elapsed))
        elif r.tariff_extra>.01:
            elapsed0=int(r.tariff_age)-1;surv0=max(tariff_survival(panel,origin,elapsed0),1e-3)
            for i in a.index[a.material.eq(r.material)]:
                conditional=tariff_survival(panel,origin,elapsed0+int(a.loc[i,'horizon']))/surv0
                a.loc[i,'known_tariff_probability']=max(a.loc[i,'known_tariff_probability'],.99*conditional)
    for col in ['robust_tree_blend_union','robust_tree_small_lane']:
        a[col]=np.maximum(a[col],a.known_tariff_probability).clip(.005,.995)
    for col in ['tree_blend_union','tree_small_lane']:
        a['scheduled_'+col]=np.maximum(a[col],a.known_tariff_probability).clip(.005,.995)
    # A future announcement is different from guessing the remaining duration
    # of an already-active tariff. Preserve the exact known start date and
    # only impose high confidence for a conservative historical duration.
    completed=[]
    for _,g in panel[panel.week<=origin].groupby(['material','country']):
        hit=g.sort_values('week').tariff_extra.gt(.01).to_numpy();run=0
        for state in hit:
            if state:run+=1
            elif run:completed.append(run);run=0
    guaranteed_span=min(completed) if completed else 6
    a['gated_announced']=a.gated_risk
    for r in now.itertuples():
        if r.announce_delta>0 and r.announce_wait>0:
            mask=a.material.eq(r.material)&a.horizon.ge(r.announce_wait)&a.horizon.lt(r.announce_wait+guaranteed_span)
            a.loc[mask,'gated_announced']=np.maximum(a.loc[mask,'gated_announced'],.98)
    return a
