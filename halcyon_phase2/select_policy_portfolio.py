"""Retain the best-mean challenger and a fixed hedge against signal failures.

This is an explicit robustness decision after retrospective stress testing,
not a claim that the hedge maximizes development mean or an untouched test.
"""
from pathlib import Path
import json
import pandas as pd
from data import load_data
from policy import simulate_policy,naive_policy,_policy_frame
from policy_v2 import blend_policies,score_components
from audit_phase2 import hindsight_policy
from replay_v2 import ORIGINS,DEVELOPMENT

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'outputs_phase2';data=load_data(ROOT/'inputs_phase2')
    selection=json.loads((out/'model_selection.json').read_text())
    challenger=selection.get('policy_maximum_mean_candidate',selection['policy'])
    if challenger!='conditional_flexible':
        raise ValueError('Reassess the robustness portfolio: the historical winner changed.')
    scores=pd.read_csv(out/'policy_replay.csv');orders=pd.read_csv(out/'historical_orders.csv');forecasts=pd.read_csv(out/'selected_forecasts.csv')
    extras=[];new_orders=[]
    for origin in ORIGINS+[208]:
        start,end=origin+1,origin+13
        a=orders[orders.origin.eq(origin)&orders.policy.eq('original')&orders.week.between(start,end)]
        if origin==208:
            a=_policy_frame(pd.read_csv(ROOT/'outputs/submission.csv'))
            a=a[a.week.between(start,end)]
        b=orders[orders.origin.eq(origin)&orders.policy.eq(challenger)&orders.week.between(start,end)]
        if len(a)!=130 or len(b)!=130:raise ValueError(f'Missing policy rows for origin{origin}')
        plan=blend_policies(a,b,.5);actual=simulate_policy(data,plan,start,end,origin)
        naive=simulate_policy(data,naive_policy(data,origin,start,end),start,end,origin)
        oracle=simulate_policy(data,hindsight_policy(data,start,end),start,end,origin)
        f=forecasts[forecasts.origin.eq(origin)]
        extras.append(dict(origin=origin,start_week=start,end_week=end,split=f.iloc[0]['split'],policy='hedged50',**score_components(f,actual,naive,oracle)))
        plan['origin']=origin;plan['policy']='hedged50';new_orders.append(plan)
    scores=scores[scores.policy.ne('hedged50')]
    pd.concat([scores,pd.DataFrame(extras)],ignore_index=True).to_csv(out/'policy_replay.csv',index=False)
    orders=orders[orders.policy.ne('hedged50')]
    pd.concat([orders,*new_orders],ignore_index=True).to_csv(out/'historical_orders.csv',index=False)
    selection['policy']='hedged50';selection['challenger_policy']=challenger
    selection['policy_recommendation_reason']='Explicit fixed50/50 hedge after retrospective stress tests; export development-mean winner separately. The hedge sacrifices some development mean for protection against missed shocks.'
    selection['policy_recommendation_informed_by_retrospective_evaluation']=True
    (out/'model_selection.json').write_text(json.dumps(selection,indent=2))
    print(pd.DataFrame(extras)[['origin','score_proxy','cost_saving','fill_rate']].to_string(index=False))

if __name__=='__main__':main()
