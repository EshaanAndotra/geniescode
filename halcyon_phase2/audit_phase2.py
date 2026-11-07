"""Audit the actually submitted Phase1 CSV against the newly revealed public weeks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from data import load_data, FILES
from policy import simulate_policy

ROOT=Path(__file__).resolve().parent

def hindsight_policy(data,start,end):
    """Perfect-information procurement LP; comparison only, never a forecast."""
    rows=[]
    for mat,sp in data['suppliers'].groupby('material'):
        countries=sp.country.tolist(); T=end-start+1; C=len(countries)
        d=data['demand'].query('material==@mat and @start<=week<=@end').sort_values('week').demand_units.to_numpy(float)
        supply=data['supply_history'].query('material==@mat and @start<=week<=@end').set_index(['week','country'])
        ref=float(supply.landed_cost.mean()); n=T*C+2*T
        obj=np.zeros(n); A=np.zeros((T,n)); b=-d.copy();b[0]+=5*d.mean()
        bounds=[]
        for t,w in enumerate(range(start,end+1)):
            for c,country in enumerate(countries):
                s=supply.loc[(w,country)];i=t*C+c;arrival=t+int(s.lead_time_weeks)
                obj[i]=s.landed_cost-(.7*ref if arrival>=T else 0)
                if arrival<T:A[arrival,i]-=1
                bounds.append((0,float(s.capacity_units)))
            A[t,T*C+t]=1
            if t:A[t,T*C+t-1]=-1
            A[t,T*C+T+t]=-1
            obj[T*C+t]=.02*ref-(.7*ref if t==T-1 else 0)
            obj[T*C+T+t]=5*ref
        bounds.extend([(0,None)]*T);bounds.extend([(0,float(x)) for x in d])
        res=linprog(obj,A_eq=A,b_eq=b,bounds=bounds,method='highs')
        if not res.success:raise RuntimeError(res.message)
        q=res.x[:T*C].reshape(T,C)
        for t,w in enumerate(range(start,end+1)):
            total=q[t].sum();shares=q[t]/total if total>1e-9 else np.ones(C)/C
            row={'week':w,'material':mat,'order_units':total,**{f'share_{c}':0.0 for c in ['KES','VAL','SUN','ORI','HAL','AUR']}}
            row.update({f'share_{c}':float(v) for c,v in zip(countries,shares)});rows.append(row)
    return pd.DataFrame(rows)

def main():
    data=load_data(ROOT/'inputs_phase2');old=load_data(ROOT/'inputs');out=ROOT/'outputs_phase2';out.mkdir(exist_ok=True)
    prefix={}
    for name in FILES:
        a=old[name];b=data[name]
        if 'week' in b and name!='demand':b=b[b.week<=208].reset_index(drop=True)
        prefix[name]=a.equals(b)
    sub=pd.read_csv(ROOT/'outputs/submission.csv');truth=data['outcomes'].query('209<=week<=221').copy()
    truth['id']='w'+truth.week.astype(str)+'_'+truth.material
    v=truth.merge(sub,on='id',validate='one_to_one')
    prior=old['outcomes'].groupby('material').disrupted.mean();last=old['outcomes'].query('week==208').set_index('material').benchmark_price
    v['base_prob']=v.material.map(prior);v['base_price']=v.material.map(last)
    for name,pred,target in [('brier','disruption_prob','disrupted'),('base_brier','base_prob','disrupted')]:v[name]=(v[pred]-v[target])**2
    v['ape']=(v.price_forecast/v.benchmark_price-1).abs();v['base_ape']=(v.base_price/v.benchmark_price-1).abs()
    actual=simulate_policy(data,sub,209,221,208)
    naive=simulate_policy(data,old['sample_submission'],209,221,208)
    oracle=simulate_policy(data,hindsight_policy(data,209,221),209,221,208)
    skill=1-v.brier.mean()/v.base_brier.mean();pskill=1-v.ape.mean()/v.base_ape.mean()
    P=np.clip(.6*skill+.4*pskill,0,1)
    F=np.clip((naive['cost']-actual['cost'])/(naive['cost']-oracle['cost']),0,1)
    fill=np.clip((actual['shock_fill_rate']-naive['shock_fill_rate'])/(1-naive['shock_fill_rate']),0,1) if naive['shock_fill_rate']<1 else 0
    R=.7*fill+.3*actual['diversification']
    metrics={'prefix_identical':prefix,'reported_kaggle_score':55,'public_disrupted_rows':int(v.disrupted.sum()),'public_rows':len(v),
        'historical_disruption_rate':float(old['outcomes'].disrupted.mean()),'public_disruption_rate':float(v.disrupted.mean()),
        'predicted_disruption_rate':float(v.disruption_prob.mean()),'brier':float(v.brier.mean()),'baseline_brier':float(v.base_brier.mean()),
        'brier_skill':float(skill),'mape':float(v.ape.mean()),'baseline_mape':float(v.base_ape.mean()),'price_skill':float(pskill),
        'prediction_component':float(P),'prediction_points':float(30*P),
        'local_financial_component':float(F),'local_resilience_component':float(R),'local_score_proxy':float(100*(.3*P+.4*F+.3*R)),
        'local_policy':{k:{a:b for a,b in r.items() if a not in ['by_material','weekly']} for k,r in [('submitted',actual),('naive',naive),('hindsight',oracle)]}}
    (out/'public_audit.json').write_text(json.dumps(metrics,indent=2))
    v.to_csv(out/'public_predictions_audit.csv',index=False)
    mat=v.groupby('material').agg(actual_disruption_rate=('disrupted','mean'),predicted_disruption_rate=('disruption_prob','mean'),brier=('brier','mean'),baseline_brier=('base_brier','mean'),mape=('ape','mean'),baseline_mape=('base_ape','mean'))
    mat.to_csv(out/'public_material_audit.csv')
    for name,r in [('submitted',actual),('naive',naive),('hindsight',oracle)]:
        pd.DataFrame(r['by_material']).to_csv(out/f'public_{name}_policy.csv',index=False)
        pd.DataFrame(r['weekly']).to_csv(out/f'public_{name}_inventory.csv',index=False)
    print(json.dumps(metrics,indent=2))
    n=data['news'];from news_features import _prepare
    signals=pd.DataFrame(_prepare(n));signals[signals.week>=207].to_csv(out/'recent_news_signals.csv',index=False)
    s=data['supply_history'].merge(data['suppliers'],on=['material','country']);s['capacity_ratio']=s.capacity_units/s.normal_capacity_units
    s[s.week>=208].to_csv(out/'recent_supply_conditions.csv',index=False)

if __name__=='__main__':main()
