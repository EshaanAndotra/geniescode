"""Replay weeks 13-207 from sequential 13-week origins, without future inputs.

Weeks 1-12 are the explicit initialization period. There is no week 0 in the
provided data. The development/evaluation partition is frozen below.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_data,source_manifest
from model_v2 import make_panel,examples,predict_candidates,aggregate_predictions,add_robust_candidates
from price_model import predict_prices
from risk_model import horizon_rows
from risk_challenger import onset_forecast

ROOT=Path(__file__).resolve().parent
ORIGINS=list(range(12,195,13))
DEVELOPMENT=[64,77,90,103,116,129,142]
EVALUATION=[155,168,181,194]

def original_forecasts(data,origin):
    prices=predict_prices(data,origin,range(origin+1,origin+14)).rename(columns={'price_forecast':'original_price'})
    if origin>=26:
        features=pd.read_csv(ROOT/'outputs/risk_features.csv')
        if origin in features.origin.unique():
            t=horizon_rows(features,data);f=t[t.origin.eq(origin)].copy()
            f['original_risk']=onset_forecast(data,t,origin,10,6)
            return prices.merge(f[['week','material','original_risk']],on=['week','material'])
    # The original feature model requires26 weeks; cold-start comparator is
    # an explicitly labeled historical-rate fallback, not a fitted v1 model.
    prior=data['outcomes'].query('week<=@origin').groupby('material').disrupted.mean()
    prices['original_risk']=prices.material.map(prior).clip(.01,.99)
    return prices

def score_table(frame,split):
    score=[]
    for c in [c for c in frame if c.startswith(('tree_','transition_','robust_','scheduled_','gated_'))]+['original_risk','historical_rate']:
        score.append({'split':split,'metric':'brier','model':c,'value':float(((frame[c]-frame.disrupted)**2).mean())})
    for c in [c for c in frame if c.startswith('price_')]+['original_price']:
        score.append({'split':split,'metric':'mape','model':c,'value':float(abs(frame[c]/frame.benchmark_price-1).mean())})
    return score

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'inputs_phase2')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs_phase2')
    ap.add_argument('--refresh',action='store_true',help='Rebuild cached folds.')
    args=ap.parse_args();out=args.output_dir;out.mkdir(exist_ok=True,parents=True)
    d=load_data(args.data_dir);manifest=source_manifest(args.data_dir,d)
    stamp=out/'v2_source_manifest.json'
    if stamp.exists() and json.loads(stamp.read_text())!=manifest:args.refresh=True
    stamp.write_text(json.dumps(manifest,indent=2))
    if args.refresh or not (out/'supplier_panel.pkl').exists():
        p=make_panel(d);p.to_pickle(out/'supplier_panel.pkl');t=examples(p,d);t.to_pickle(out/'training_examples.pkl')
    else:p=pd.read_pickle(out/'supplier_panel.pkl');t=pd.read_pickle(out/'training_examples.pkl')
    frames=[]
    for origin in ORIGINS+[208,221]:
        file=out/f'candidates_supplier_{origin}.csv'
        if args.refresh or not file.exists():
            f=predict_candidates(d,p,t,origin);f.to_csv(file,index=False)
        else:f=pd.read_csv(file)
        a=aggregate_predictions(f)
        a=a.merge(d['outcomes'][['week','material','disrupted','benchmark_price']],on=['week','material'],how='left')
        hist=d['outcomes'].query('week<=@origin').groupby('material').disrupted.mean()
        a['historical_rate']=a.material.map(hist)
        if origin<=208:a=a.merge(original_forecasts(d,origin),on=['week','material'],how='left')
        a=add_robust_candidates(a,p,d)
        a['split']='development' if origin in DEVELOPMENT else 'evaluation' if origin in EVALUATION else 'public_audit' if origin==208 else 'final' if origin==221 else 'cold_start'
        frames.append(a)
        print(f'Forecast fold origin {origin}, target {origin+1}-{origin+13}, {a.iloc[0]["split"]}',flush=True)
    allpred=pd.concat(frames,ignore_index=True);allpred.to_csv(out/'all_forecast_candidates.csv',index=False)
    scores=[]
    for split in ['cold_start','development','evaluation','public_audit']:
        scores.extend(score_table(allpred[allpred.split.eq(split)],split))
    scores=pd.DataFrame(scores);scores.to_csv(out/'model_comparison.csv',index=False)
    dev=scores[scores.split.eq('development')]
    # The original model and naive candidates remain eligible, ensuring a
    # complex candidate is not selected just because it is new.
    risk=dev[dev.metric.eq('brier')].sort_values('value').iloc[0]['model']
    price=dev[dev.metric.eq('mape')].sort_values('value').iloc[0]['model']
    # The minimum-MAPE price model has a tiny edge in seven noisy windows.
    # Prefer a fixed futures/tree ensemble when its excess error is within
    # one paired standard error. This reduces dependence on a single signal.
    development=allpred[allpred.split.eq('development')]
    fold_errors=development.groupby('origin').apply(lambda g:pd.Series({c:float(abs(g[c]/g.benchmark_price-1).mean()) for c in [price,'price_mix']}),include_groups=False)
    difference=fold_errors['price_mix']-fold_errors[price]
    uncertainty=float(difference.std(ddof=1)/np.sqrt(len(difference)))
    raw_price_winner=price
    if difference.mean()<=uncertainty:price='price_mix'
    selection={'risk':risk,'price':price,'development_origins':DEVELOPMENT,'evaluation_origins':EVALUATION,
               'warmup_weeks':[1,12],'replay_weeks':[13,207],'replay_rows':1950,
               'public_used_for_selection':False,'max_forecast_horizon':13,
               'price_minimum_error_candidate':raw_price_winner,
               'price_ensemble_excess_mape':float(difference.mean()),'price_paired_standard_error':uncertainty,
               'price_selection_rule':'Prefer fixed 50/50 futures-tree ensemble if within one paired standard error of development minimum.'}
    (out/'model_selection.json').write_text(json.dumps(selection,indent=2))
    allpred['disruption_prob']=allpred[risk];allpred['price_forecast']=allpred[price]
    allpred.to_csv(out/'selected_forecasts.csv',index=False)
    print(json.dumps(selection,indent=2),flush=True)
    print(scores[scores.model.isin([risk,price,'original_risk','original_price','historical_rate','price_carry'])].to_string(index=False),flush=True)

if __name__=='__main__':main()
