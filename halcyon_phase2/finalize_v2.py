"""Create the final submission and a complete evidence-based Phase2 report."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from data import load_data,validate_submission,source_manifest
from policy_v2 import plan_v2,CONFIGS
from run_pipeline import make_plot

ROOT=Path(__file__).resolve().parent

def diagnostic_plot(data,out,forecast,replay):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=pd.read_csv(ROOT/'outputs/submission.csv')
    pub=data['outcomes'].query('209<=week<=221').copy();pub['id']='w'+pub.week.astype(str)+'_'+pub.material
    pub=pub.merge(old[['id','disruption_prob']],on='id').rename(columns={'disruption_prob':'original'})
    pub=pub.merge(forecast[forecast.origin.eq(208)][['week','material','disruption_prob']],on=['week','material'])
    mats=sorted(pub.material.unique())
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained',sharey=True)
    for ax,col,title in zip(axs,['disrupted','original','disruption_prob'],['What happened','Original probabilities','Revised probabilities']):
        x=pub.pivot(index='material',columns='week',values=col).reindex(mats)
        im=ax.imshow(x,aspect='auto',vmin=0,vmax=1,cmap='YlOrRd')
        ax.set_title(title,loc='left');ax.set_xticks([0,3,6,9,12],[209,212,215,218,221]);ax.set_xlabel('Week')
        ax.set_yticks(range(10),[m.replace('_',' ') for m in mats])
    fig.colorbar(im,ax=axs[-1],fraction=.05,label='Disruption / probability')
    fig.savefig(out/'public_forecast_audit.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for name,label,color in [('original','Original','#64748b'),('revised','Revised','#007f82')]:
        z=replay[replay.version.eq(name)].sort_values('start_week')
        ax.plot(z.start_week,z.score_proxy,'o-',label=label,color=color)
    ax.set_ylim(0,100);ax.set_ylabel('Local score proxy');ax.set_xlabel('First week of each independent 13-week window')
    ax.set_title('Historical replay includes calm and severe periods',loc='left')
    ax.axvspan(156,207,color='#e0f2fe',alpha=.4,label='Chronological evaluation windows')
    ax.legend(frameon=False,ncol=3);fig.savefig(out/'historical_replay_scores.png',dpi=150);plt.close(fig)

def write_report(data,out,selection,forecast,replay,summary):
    audit=json.loads((out/'public_audit.json').read_text());pub=replay[replay.origin.eq(208)].set_index('version')
    old=pub.loc['original'];new=pub.loc['revised'];ev=replay[replay.split.eq('evaluation')].groupby('version').mean(numeric_only=True)
    dev=replay[replay.split.eq('development')].groupby('version').mean(numeric_only=True)
    records=pd.read_csv(out/'public_material_audit.csv').set_index('material')
    latest=data['outcomes'].query('week==221').set_index('material').benchmark_price
    previous=data['outcomes'].query('week==208').set_index('material').benchmark_price
    actual_delta=latest/previous-1
    oldsub=pd.read_csv(ROOT/'outputs/submission.csv');oldend=oldsub[oldsub.id.str.startswith('w221_')].copy();oldend['material']=oldend.id.str.removeprefix('w221_')
    predicted_delta=oldend.set_index('material').price_forecast/previous-1
    direction=float((np.sign(actual_delta)==np.sign(predicted_delta)).mean())
    final=forecast[forecast.origin.eq(221)].groupby('material').agg(mean_risk=('disruption_prob','mean'),first_price=('price_forecast','first'),last_price=('price_forecast','last'))
    rows='\n'.join(f'| {m} | {r.mean_risk:.1%} | {r.first_price:.4f} | {r.last_price:.4f} |' for m,r in final.iterrows())
    text=f'''# Phase 2 audit and final-stage model

## Files to use

`submission_final.csv` is the recommended 260-row submission. `submission_challenger.csv` uses the more aggressive policy that won mean development performance. Both are eligible alternatives for the competition's two final submission slots. Weeks209–221 are a revised replay frozen at origin208; weeks222–234 use the full release through221. Public truth is never copied into forecasts or used to optimize those public orders. The new release is preserved separately in `inputs_phase2`, and the original submission remains in `outputs/submission.csv`.

The empirical risk and ordering changes were developed after seeing the revealed public outcomes. Therefore the revised public result is a diagnostic backcast, not fresh out-of-sample evidence. The private weeks222–234 are still genuinely unknown.

## Why the original scored about55

The local reconstruction is **{audit['local_score_proxy']:.2f}**, compared with the reported Kaggle score55. Its estimated components are:

| Component | Points earned | Points available |
|---|---:|---:|
| Prediction | {audit['prediction_points']:.2f} | 30 |
| Financial | {40*audit['local_financial_component']:.2f} | 40 |
| Resilience | {30*audit['local_resilience_component']:.2f} | 30 |

The simulator reconstruction closely matches the reported total, but these are estimated component values, not a component breakdown supplied by Kaggle. The main lost points were prediction and financial efficiency. Resilience was already nearly saturated.

The public period was **calmer overall but concentrated in a few shocks**:18 of130 material-weeks were disrupted ({audit['public_disruption_rate']:.1%}), compared with{audit['historical_disruption_rate']:.1%} during weeks1–208. A Southern-route strike disrupted lithium, cobalt, copper and natural gas in weeks212–214. The separate lithium supply disruption continued through220. Six materials had no active public-window disruption.

Our original probabilities averaged{audit['predicted_disruption_rate']:.1%}. Lithium was disrupted in9 of13 weeks ({records.loc['lithium','actual_disruption_rate']:.1%}), against an average forecast of{records.loc['lithium','predicted_disruption_rate']:.1%}. Its warnings were recognized, but the selected next-week onset model learned no immediate lift from an early warning. It failed to translate that warning into risk several weeks later. Brier skill was only{audit['brier_skill']:.2%}.

The original price MAPE was{audit['mape']:.2%}, almost identical to carrying the last price forward ({audit['baseline_mape']:.2%}). Its end-window price direction was correct for{direction:.0%} of materials. From208 to221, natural gas rose{actual_delta['natural_gas']:.1%}, polymer resin rose{actual_delta['polymer_resin']:.1%}, lithium rose{actual_delta['lithium']:.1%}, and semiconductors fell{abs(actual_delta['semiconductors']):.1%}. A mild mean-reversion forecast did not capture several of these moves. Large price changes do not necessarily imply a labeled supply disruption.

The original orders met100% of demand in the public replay, including shock weeks. Cost was{audit['local_policy']['submitted']['cost']:,.0f}, versus{audit['local_policy']['naive']['cost']:,.0f} for the sample policy and{audit['local_policy']['hindsight']['cost']:,.0f} for our perfect-information LP. The original policy captured61% of that local attainable saving. Excess ending inventory, holding costs, and rigid country shares left additional financial room. More blanket safety stock would not have addressed the main weakness.

## What changed

1. **Forecast horizon:** direct13-week forecasting, matching the final private window. The original comparison mixed horizons1–26.
2. **Supplier-level signals:** capacity, lead times, futures premiums, tariffs, macroeconomic and shipping conditions, and supplier-scoped news stages/ages are modeled separately for30 suppliers. The observed supplier shock definition reproduces every historical material disruption label exactly. It is used only as a historical training label, never as a future observation.
3. **Risk forecasts:** the selected candidate is `{selection['risk']}`. Two regularized boosted classifiers and a transition fallback produce supplier probabilities. A material-level gate gives the direct models more weight when futures, tension, congestion or tariff announcements corroborate stress, and preserves the transition estimate otherwise. This reduces the tendency of a complex model to become overconfident in quiet conditions. A separate candidate treats future announced tariff starts explicitly, assigning98% probability for a conservative duration estimated only from completed past tariff episodes. This differs from guessing the remaining duration of an already-active tariff; those broader persistence rules did worse. All candidates remain in the comparison table.
4. **Prices:** `{selection['price']}` averages a futures anchor and a regularized model of future supplier price returns, then averages across each material's three suppliers. Its development MAPE was within one paired standard error of the best single model, so the fixed ensemble was preferred to dependence on one signal.
5. **Ordering:** supplier-specific predicted capacity and delay scenarios replace most of the unconditional history, with shared-lane shocks kept correlated. Known tariffs change landed costs from their announced effective week. Leaner, more flexible and blended policies were compared using the actual40/30 financial/resilience weights. The recommended policy is `{selection['policy']}`, an explicit50/50 blend of original and conditional country order quantities. The aggressive challenger `{selection['challenger_policy']}` won mean development performance. The hedge is a judgment made after the retrospective stress tests, sacrificing some development mean to protect against missed shocks. It is not presented as the maximizer of that mean. Configuration details are in `model_selection.json` and `policy_v2.py`.

At221 the inputs contain new Merrow-route tension, Valdmar cobalt export warnings, Kestria rare-earth export warnings, and announced Kestria tariffs on aluminum and semiconductors effective224. The structured tariff table specifies35%; it takes precedence over an approximate25% aluminum-news headline. The lithium resolution at221 clears its prior supplier-specific warning; it does not clear a separate Merrow-route warning. Early warnings remain uncertain, not guaranteed future events.

## Comparison results

| Period and metric | Original | Revised |
|---|---:|---:|
| Public209–221 Brier | {old.brier:.5f} | {new.brier:.5f} |
| Public209–221 price MAPE | {old.mape:.2%} | {new.mape:.2%} |
| Public209–221 local score proxy | {old.score_proxy:.2f} | {new.score_proxy:.2f} |
| Public209–221 demand fill | {old.fill_rate:.2%} | {new.fill_rate:.2%} |
| Development windows mean local score | {dev.loc['original','score_proxy']:.2f} | {dev.loc['revised','score_proxy']:.2f} |
| Later evaluation windows mean local score | {ev.loc['original','score_proxy']:.2f} | {ev.loc['revised','score_proxy']:.2f} |
| Later evaluation windows mean Brier | {ev.loc['original','brier']:.5f} | {ev.loc['revised','brier']:.5f} |
| Later evaluation windows price MAPE | {ev.loc['original','mape']:.2%} | {ev.loc['revised','mape']:.2%} |

The new model does not win everywhere. In particular, the unexpected crisis in169–181 is difficult for signal-driven models, and disruption accuracy deteriorates in the later evaluation group even when other components improve. A more aggressive policy was explicitly compared with a blend that retains the original policy's protection. Every fold, including losses and cold-start performance, is in `historical_score_comparison.csv` and `policy_replay.csv`. The earlier forecast's strong performance on183–208 was too narrow a basis for expecting the same performance in a new regime.

These results are retrospective chronological validation, not an untouched test set: we have now inspected these historical outcomes while developing the approach. No row-level feature or fitted target is allowed past its forecast origin. Numerical forecast selection and the best-mean policy comparison use the designated development folds. The recommended hedge and model architecture were informed by the broader stress tests, so the later evaluation cannot be called independent of model development. No promise of a private leaderboard score is justified by this small synthetic history.

## Replay weeks0–207

There is **no week0** in the files. The new files also leave every original row through208 unchanged. The defensible replay uses weeks1–12 as initialization and produces **1,950 forecast-and-order rows covering weeks13–207**, in15 independent13-week windows. Origins are12,25,38,...,194. Every window starts with five weeks of demand in stock and no transit, as in the competition. The first two origins use an explicitly labeled historical-rate fallback for the original risk model, which required26 weeks.

Development origins are64,77,90,103,116,129,142. Later chronological evaluation origins are155,168,181,194. Earlier folds are reported as cold-start diagnostics. The final historical window195–207 is fully scored without using observations208–221. Each training target must end at or before its origin. Training on all221 weeks and then predicting old weeks would leak their answers; this replay never does that. The published demand schedule is the permitted exception for future covariates.

From the project folder:

```powershell
python run_phase2.py
```

This runs the audit, historical forecasting comparison, policy comparison, final submission generation and tests. Cached supplier forecasts can be reused only with matching data fingerprints. To force refitting:

```powershell
python run_phase2.py --refresh
```

For forecasts alone use `python replay_v2.py --refresh`; `python evaluate_policy_v2.py` evaluates the policies. `historical_replay.csv` contains the complete selected replay with actual labels and orders. `model_comparison.csv` lists all forecast candidates; `policy_selection.csv` lists the policy comparison.

## Private-window forecast snapshot

| Material | Mean disruption probability,222–234 | Price222 | Price234 |
|---|---:|---:|---:|
{rows}

## Limits and verification

The local simulator uses the documented arrival, cancellation, holding, shortage and terminal-salvage mechanics. Initial stock uses five times window-average demand and reference material cost uses window-average landed cost. The hindsight comparator is an exact perfect-information procurement LP under those local assumptions. It is used for evaluation only. A small gap from Kaggle can still reflect unspecified normalization conventions. When the sample already has100% shock fill, the local fill-skill improvement is defined as zero; no-disruption windows earn only the diversification part of resilience.

Source completeness, unique keys, historical prefix equality, supplier eligibility, price positivity, probability bounds, nonnegative orders, and share sums are checked. The final saved submission has260 unique rows and10 columns. Prefix-invariance and future-target exclusion tests exercise the new forecasts. Inventory mass and cost reconciliation are checked separately. Exact verification outputs are stored beside the submission.
'''
    # Add normal prose spacing without changing identifiers inside code spans.
    import re
    pieces=re.split(r'(`[^`]*`)',text)
    text=''.join(piece if piece.startswith('`') else re.sub(r'(?<=[a-zA-Z])(?=\d)',' ',piece) for piece in pieces)
    (out/'REPORT.md').write_text(text,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-dir',type=Path,default=ROOT/'inputs_phase2');ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs_phase2');args=ap.parse_args()
    d=load_data(args.data_dir);out=args.output_dir;selection=json.loads((out/'model_selection.json').read_text());f=pd.read_csv(out/'selected_forecasts.csv')
    parts=[]
    for origin in [208,221]:
        forecast=f[f.origin.eq(origin)].copy();supplier=pd.read_csv(out/f'candidates_supplier_{origin}.csv')
        policy=plan_v2(d,origin,origin+1,origin+13,supplier,forecast,selection['risk'],selection['price'],selection['policy'])
        part=forecast[['week','material','disruption_prob','price_forecast']].merge(policy,on=['week','material'],validate='one_to_one');part['origin']=origin;parts.append(part)
    final=pd.concat(parts,ignore_index=True);final['id']='w'+final.week.astype(str)+'_'+final.material
    submission=d['sample_submission'][['id']].merge(final,on='id',validate='one_to_one')[d['sample_submission'].columns]
    submission.to_csv(out/'submission_final.csv',index=False,float_format='%.10f')
    check=validate_submission(pd.read_csv(out/'submission_final.csv'),d)
    (out/'submission_validation.json').write_text(json.dumps(check,indent=2));final.to_csv(out/'final_decisions.csv',index=False)
    challenger_parts=[]
    for origin in [208,221]:
        ff=f[f.origin.eq(origin)];sf=pd.read_csv(out/f'candidates_supplier_{origin}.csv')
        pp=plan_v2(d,origin,origin+1,origin+13,sf,ff,selection['risk'],selection['price'],selection['challenger_policy'])
        pp=ff[['week','material','disruption_prob','price_forecast']].merge(pp,on=['week','material']);pp['id']='w'+pp.week.astype(str)+'_'+pp.material
        challenger_parts.append(pp)
    cc=pd.concat(challenger_parts,ignore_index=True);cc=d['sample_submission'][['id']].merge(cc,on='id')[d['sample_submission'].columns]
    cc.to_csv(out/'submission_challenger.csv',index=False,float_format='%.10f')
    (out/'challenger_validation.json').write_text(json.dumps(validate_submission(pd.read_csv(out/'submission_challenger.csv'),d),indent=2))
    orders=pd.read_csv(out/'historical_orders.csv');orders=orders[orders.policy.eq(selection['policy'])&orders.origin.lt(208)]
    historical=f[f.origin.lt(208)][['origin','week','material','split','disrupted','benchmark_price','disruption_prob','price_forecast']].merge(orders,on=['origin','week','material'],validate='one_to_one')
    assert len(historical)==1950 and historical.week.min()==13 and historical.week.max()==207
    historical.to_csv(out/'historical_replay.csv',index=False)
    replay=pd.read_csv(out/'policy_replay.csv');replay=replay[replay.policy.isin(['original',selection['policy']])].copy()
    replay['version']=np.where(replay.policy.eq('original'),'original','revised');replay.to_csv(out/'historical_score_comparison.csv',index=False)
    summary={'model_selection':selection,'validation':check,'historical_rows':len(historical),'original_data_cutoff':208,'final_data_cutoff':221,
             'public':replay[replay.origin.eq(208)].to_dict('records'),'mean_by_split':replay.groupby(['split','version'])[['score_proxy','brier','mape','fill_rate','cost_saving']].mean().reset_index().to_dict('records')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));make_plot(submission,out/'final_forecast_overview.png')
    diagnostic_plot(d,out,f,replay);write_report(d,out,selection,f,replay,summary)
    (out/'source_manifest.json').write_text(json.dumps(source_manifest(args.data_dir,d),indent=2))
    print(json.dumps({'submission':str(out/'submission_final.csv'),'validation':check,'policy':selection['policy'],'replay_rows':len(historical)},indent=2))

if __name__=='__main__':main()
