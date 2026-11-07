"""Tune ordering settings on development origins; replay later windows afterward."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from data import load_data
from policy import plan_policy,simulate_policy,naive_policy,_policy_frame
from policy_v2 import plan_v2,CONFIGS,score_components
from audit_phase2 import hindsight_policy
from replay_v2 import ORIGINS,DEVELOPMENT,EVALUATION

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'inputs_phase2')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs_phase2')
    args=ap.parse_args();out=args.output_dir;d=load_data(args.data_dir)
    sel=json.loads((out/'model_selection.json').read_text());pred=pd.read_csv(out/'selected_forecasts.csv')
    results=[];baselines={};saved={}
    def evaluate(origin,names):
        start,end=origin+1,origin+13;f=pred[pred.origin.eq(origin)].copy();sf=pd.read_csv(out/f'candidates_supplier_{origin}.csv')
        naive=simulate_policy(d,naive_policy(d,origin,start,end),start,end,origin)
        oracle=simulate_policy(d,hindsight_policy(d,start,end),start,end,origin)
        for name in names:
            ff=f.copy()
            if name=='original':
                ff['disruption_prob']=ff.original_risk;ff['price_forecast']=ff.original_price
                policy=pd.read_csv(ROOT/'outputs/submission.csv') if origin==208 else plan_policy(d,origin,start,end,ff)
            else:policy=plan_v2(d,origin,start,end,sf,ff,sel['risk'],sel['price'],name)
            actual=simulate_policy(d,policy,start,end,origin)
            metrics=score_components(ff,actual,naive,oracle)
            row=dict(origin=origin,start_week=start,end_week=end,split=f.iloc[0]['split'],policy=name,**metrics)
            results.append(row)
            if name=='original' or origin not in DEVELOPMENT:
                pp=_policy_frame(policy);pp['origin']=origin;pp['policy']=name
                saved[(origin,name)]=pp
            print(f'Policy {name}, origin {origin}: score proxy {metrics["score_proxy"]:.2f}, cost saving {metrics["cost_saving"]:.1%}, fill {metrics["fill_rate"]:.2%}',flush=True)
    for origin in DEVELOPMENT:evaluate(origin,['original',*CONFIGS])
    dev=pd.DataFrame(results)
    ranking=dev[dev.policy.ne('original')].assign(policy_points=lambda x:40*x.F+30*x.R).groupby('policy').agg(
        policy_points=('policy_points','mean'),cost_saving=('cost_saving','mean'),fill=('fill_rate','mean'),shock_fill=('shock_fill','mean'),worst_cost_saving=('cost_saving','min')).sort_values('policy_points',ascending=False)
    winner=ranking.index[0]
    # Within one paired standard error, prefer a 50/50 policy hedge with a
    # stronger worst-window fill. This avoids choosing brittle concentration
    # based on small average-score differences in seven development windows.
    dev['policy_points']=40*dev.F+30*dev.R
    wide=dev.pivot(index='origin',columns='policy',values='policy_points')
    delta=wide[winner]-wide['hedged50'];paired_se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    original_winner=winner
    worst_fill=dev.groupby('policy').fill_rate.min()
    if delta.mean()<=paired_se and worst_fill['hedged50']>=worst_fill[winner]:winner='hedged50'
    sel['policy']=winner;sel['policy_selection_metric']='Mean 40*F+30*R across development origins only'
    sel['policy_maximum_mean_candidate']=original_winner
    sel['policy_hedge_excess_points']=float(delta.mean());sel['policy_paired_standard_error']=paired_se
    sel['policy_robustness_rule']='Prefer hedged50 within one paired standard error when its worst development-window fill is no worse.'
    (out/'model_selection.json').write_text(json.dumps(sel,indent=2))
    ranking.to_csv(out/'policy_selection.csv')
    print('SELECTED POLICY '+winner,flush=True)
    for origin in [o for o in ORIGINS if o not in DEVELOPMENT]+[208]:evaluate(origin,['original',winner])
    pd.DataFrame(results).to_csv(out/'policy_replay.csv',index=False)
    # Generate development-window winner orders too, so the complete historical
    # replay is one explicit file, with no hidden adaptive weekly updates.
    for origin in DEVELOPMENT:
        f=pred[pred.origin.eq(origin)];sf=pd.read_csv(out/f'candidates_supplier_{origin}.csv')
        pp=plan_v2(d,origin,origin+1,origin+13,sf,f,sel['risk'],sel['price'],winner)
        pp['origin']=origin;pp['policy']=winner;saved[(origin,winner)]=pp
    pd.concat(saved.values(),ignore_index=True).to_csv(out/'historical_orders.csv',index=False)
    print(ranking.to_string(),flush=True)

if __name__=='__main__':main()
