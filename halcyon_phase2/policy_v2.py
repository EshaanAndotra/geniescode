"""Forecast-conditioned procurement scenarios and score-aware validation."""
import numpy as np
import pandas as pd
from policy import _optimize_orders,_initial_stock,_scenarios,COUNTRIES,plan_policy

CONFIGS={
 'historical_lean':dict(min_share=.15,max_share=.70,safety_weeks=.5,max_order_weeks=2.0,conditional=False),
 'conditional_balanced':dict(min_share=.20,max_share=.60,safety_weeks=.5,max_order_weeks=2.0,conditional=True),
 'conditional_lean':dict(min_share=.15,max_share=.70,safety_weeks=.35,max_order_weeks=2.0,conditional=True),
 'conditional_diverse':dict(min_share=.25,max_share=.50,safety_weeks=.35,max_order_weeks=2.0,conditional=True),
 'conditional_resilient':dict(min_share=.15,max_share=.70,safety_weeks=1.0,max_order_weeks=2.0,conditional=True),
 'conditional_flexible':dict(min_share=.08,max_share=.84,safety_weeks=.6,max_order_weeks=2.2,conditional=True),
 'hedged25':dict(anchor_weight=.25),
 'hedged50':dict(anchor_weight=.50),
}

def blend_policies(anchor,conditional,anchor_weight):
    a=anchor.set_index(['week','material']);b=conditional.set_index(['week','material']).reindex(a.index)
    cols=['share_'+c for c in COUNTRIES]
    quantity=anchor_weight*a[cols].multiply(a.order_units,axis=0)+(1-anchor_weight)*b[cols].multiply(b.order_units,axis=0)
    totals=quantity.sum(axis=1);shares=quantity.div(totals.replace(0,np.nan),axis=0).fillna(a[cols])
    result=shares.copy();result['order_units']=totals
    return result.reset_index()

def conditioned_scenarios(data,history,suppliers,origin,start,end,forecast,risk_column,price_column):
    countries=suppliers.country.tolist();T=end-start+1;C=len(countries)
    normal=suppliers.normal_capacity_units.to_numpy(float);baselead=suppliers.base_lead_weeks.to_numpy(int)
    f=forecast.set_index(['week','country']).reindex(pd.MultiIndex.from_product([range(start,end+1),countries],names=['week','country']))
    if f[risk_column].isna().any():raise ValueError('Incomplete supplier forecasts')
    prob=f[risk_column].to_numpy().reshape(T,C).clip(.005,.99)
    mean_cap=f.capacity_mean.to_numpy().reshape(T,C).clip(.05,.985)
    mean_delay=f.delay_mean.to_numpy().reshape(T,C).clip(0,8)
    hist=history.merge(suppliers[['country','normal_capacity_units','base_lead_weeks']],on='country')
    hist['ratio']=hist.capacity_units/hist.normal_capacity_units;hist['extra']=hist.lead_time_weeks-hist.base_lead_weeks
    shocked=hist[(hist.ratio<.85)|(hist.extra>0)]
    default_bad=np.array([shocked.loc[shocked.country.eq(c),'ratio'].median() if shocked.country.eq(c).any() else .5 for c in countries])
    implied=(mean_cap-(1-prob)*.985)/np.maximum(prob,.05)
    badcap=np.clip(.7*implied+.3*default_bad[None,:],0,.95)
    bad_delay=np.clip(np.rint(mean_delay/np.maximum(prob,.08)),0,9).astype(int)
    # Continuous paths share a shock quantile across weeks and across a lane.
    # This preserves correlated Merrow supplier failures rather than treating
    # two suppliers on the same closed route as independent protection.
    cap=[];lead=[]
    count=16;lanes=list(sorted(suppliers.lane.unique()))
    rng=np.random.default_rng(2026+origin)
    uniforms={lane:(rng.permutation(count)+.5)/count for lane in lanes}
    for s in range(count):
        hit=np.stack([uniforms[lane][s]<prob[:,c] for c,lane in enumerate(suppliers.lane)],axis=1)
        cap.append(normal[None,:]*np.where(hit,badcap,.985))
        lead.append(baselead[None,:]+np.where(hit,bad_delay,0))
    cap=np.array(cap);lead=np.array(lead)
    # Unannounced disruptions remain possible even in a quiet feature state.
    hc,hl,_,hw,cur=_scenarios(history,suppliers,origin,start,T,8)
    cap=np.concatenate([cap,hc]);lead=np.concatenate([lead,hl]);weights=np.r_[np.repeat(.85/count,count),.15*hw]
    latest=history.sort_values('week').groupby('country').tail(1).set_index('country').reindex(countries)
    p0=data['prices'].query('week==@origin').set_index(['material','country'])
    tr=data['tariffs'].query('week==@origin').set_index(['material','country'])
    mat=str(suppliers.iloc[0].material);cost=np.zeros((T,C))
    for c,country in enumerate(countries):
        prior_spot=float(p0.loc[(mat,country),'spot_price']);tariff=tr.loc[(mat,country)]
        shipping=max(0,float(latest.loc[country,'landed_cost'])-prior_spot*(1+tariff.tariff_rate))
        for t,w in enumerate(range(start,end+1)):
            rate=tariff.tariff_rate
            if np.isfinite(tariff.announced_effective_week) and tariff.announced_effective_week<=w:
                rate=tariff.announced_rate
            cost[t,c]=float(f.loc[(w,country),price_column])*(1+rate)+shipping
    return cap,lead,np.tile(cost[None,:,:],(len(cap),1,1)),weights,float(latest.landed_cost.mean())

def plan_v2(data,origin,start,end,supplier_forecast,material_forecast,risk_model,price_model,config='conditional_balanced'):
    if origin>=start:raise ValueError('Forecast origin must precede decision window')
    if not supplier_forecast.origin.eq(origin).all():raise ValueError('Wrong supplier forecast origin')
    params=CONFIGS[config];rows=[]
    if 'anchor_weight' in params:
        reference=material_forecast.copy()
        reference['disruption_prob']=reference.original_risk
        reference['price_forecast']=reference.original_price
        anchor=plan_policy(data,origin,start,end,reference)
        conditional=plan_v2(data,origin,start,end,supplier_forecast,material_forecast,risk_model,price_model,'conditional_flexible')
        return blend_policies(anchor,conditional,params['anchor_weight'])
    clean_risk=risk_model.removeprefix('scheduled_').removeprefix('robust_')
    riskcol=clean_risk.rsplit('_',1)[0] if clean_risk.startswith(('tree_','transition_')) else 'transition'
    if risk_model in ['gated_risk','gated_announced']:riskcol='tree_blend'
    pricecol=price_model if price_model not in ['original_price'] else 'price_original'
    supplier_forecast=supplier_forecast.copy()
    supplier_forecast['price_mix']=.5*supplier_forecast.price_futures+.5*supplier_forecast.price_tree
    supplier_forecast['price_stable_mix']=.5*supplier_forecast.price_original+.5*supplier_forecast.price_blend
    hist=data['supply_history'].query('week<=@origin')
    for mat,sp in data['suppliers'].groupby('material',sort=True):
        sp=sp.reset_index(drop=True);h=hist[hist.material.eq(mat)]
        sf=supplier_forecast[supplier_forecast.material.eq(mat)]
        mf=material_forecast[material_forecast.material.eq(mat)&material_forecast.week.between(start,end)].sort_values('week')
        demand=data['demand'].query('material==@mat and @start<=week<=@end').sort_values('week').demand_units.to_numpy(float)
        if params['conditional']:
            cap,lead,cost,weights,reference=conditioned_scenarios(data,h,sp,origin,start,end,sf,riskcol,pricecol)
        else:
            cap,lead,cost,weights,cur=_scenarios(h,sp,origin,start,len(demand),12);reference=float(cur.mean())
        initial=_initial_stock(data['demand'],mat,start,end,origin)
        q=_optimize_orders(demand,initial,cap,lead,cost,weights,reference,sp.normal_capacity_units.to_numpy(float),
             params['min_share'],params['max_share'],params['safety_weeks'],mf.disruption_prob.to_numpy(float),params['max_order_weeks'])
        for t,w in enumerate(range(start,end+1)):
            total=float(q[t].sum());shares=q[t]/total if total>1e-8 else np.ones(3)/3
            r={'week':w,'material':mat,'order_units':total,**{f'share_{c}':0.0 for c in COUNTRIES}}
            r.update({f'share_{c}':float(v) for c,v in zip(sp.country,shares)});rows.append(r)
    return pd.DataFrame(rows)

def score_components(prediction,actual,naive,oracle):
    brier=float(((prediction.disruption_prob-prediction.disrupted)**2).mean())
    base_brier=float(((prediction.historical_rate-prediction.disrupted)**2).mean())
    mape=float(abs(prediction.price_forecast/prediction.benchmark_price-1).mean())
    base_mape=float(abs(prediction.price_carry/prediction.benchmark_price-1).mean())
    bskill=1-brier/base_brier if base_brier>1e-12 else 0
    pskill=1-mape/base_mape if base_mape>1e-12 else 0
    P=float(np.clip(.6*bskill+.4*pskill,0,1))
    denom=naive['cost']-oracle['cost'];F=float(np.clip((naive['cost']-actual['cost'])/denom,0,1)) if denom>1e-8 else 0
    nf=naive['shock_fill_rate'];af=actual['shock_fill_rate']
    fill=float(np.clip((af-nf)/(1-nf),0,1)) if nf is not None and nf<1-1e-10 else 0
    R=.7*fill+.3*actual['diversification']
    return dict(brier=brier,baseline_brier=base_brier,mape=mape,baseline_mape=base_mape,P=P,F=F,R=R,
        score_proxy=30*P+40*F+30*R,cost=actual['cost'],cost_saving=1-actual['cost']/naive['cost'],
        fill_rate=actual['fill_rate'],shock_fill=af,naive_shock_fill=nf,diversification=actual['diversification'])
