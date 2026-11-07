"""Rebuild forecasts, policies, rolling-origin evaluation and submission."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from data import load_data, source_manifest, validate_submission, COUNTRIES
from risk_model import select_and_forecast, fit_predict_candidates, horizon_rows, build_features
from price_model import predict_prices, SELECTED_MODEL, validate_price_models
from policy import plan_policy, simulate_policy, naive_policy

ROOT=Path(__file__).resolve().parent

def evaluate_policies(data,risk_backtest):
    records=[]
    for origin in [104,130,156,182]:
        forecast=predict_prices(data,origin,range(origin+1,origin+27)).merge(
            risk_backtest.loc[risk_backtest.origin.eq(origin),['week','material','disruption_prob']],
            on=['week','material'],validate='one_to_one')
        for gap in [0,13]:
            start,end=origin+gap+1,origin+gap+13
            plan=plan_policy(data,origin,start,end,forecast)
            baseline=naive_policy(data,origin,start,end)
            actual=simulate_policy(data,plan,start,end,origin)
            naive=simulate_policy(data,baseline,start,end,origin)
            row={'origin':origin,'start_week':start,'end_week':end,'gap':gap,
                 'split':'holdout' if origin==182 else 'tuning',
                 'cost_reduction':1-actual['cost']/naive['cost']}
            for name,metrics in [('model',actual),('naive',naive)]:
                for key in ['cost','fill_rate','mean_material_fill_rate','shock_fill_rate',
                            'mean_material_shock_fill_rate','diversification']:
                    row[name+'_'+key]=metrics[key]
            records.append(row)
            print(f'Policy origin {origin}, gap {gap}: cost change {row["cost_reduction"]:.1%}',flush=True)
    return pd.DataFrame(records)

def make_plot(submission,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=submission.copy()
    f['week']=f.id.str.extract(r'^w(\d+)_').astype(int)
    f['material']=f.id.str.replace(r'^w\d+_','',regex=True)
    fig,axs=plt.subplots(1,2,figsize=(15,5.5),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
    matrix=f.pivot(index='material',columns='week',values='disruption_prob')
    im=axs[0].imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='YlOrRd')
    axs[0].set_yticks(range(len(matrix)),[m.replace('_',' ') for m in matrix.index])
    axs[0].set_xticks([0,6,13,19,25],[209,215,222,228,234])
    axs[0].axvline(12.5,color='#24425b',lw=1.5)
    axs[0].set_title('Disruption probability by week',loc='left',fontweight='bold')
    axs[0].set_xlabel('Week; inventory resets at 222')
    fig.colorbar(im,ax=axs[0],fraction=.04,label='Probability')
    shares=[]
    for m in matrix.index:
        x=f[f.material==m]; q=x.order_units.to_numpy()
        shares.append(np.average(x[['share_'+c for c in COUNTRIES]],axis=0,weights=q))
    shares=np.array(shares); left=np.zeros(len(matrix))
    colors=['#355070','#7096b8','#f4a261','#2a9d8f','#b56576','#8d99ae']
    for j,country in enumerate(COUNTRIES):
        axs[1].barh(range(len(matrix)),shares[:,j],left=left,label=country,color=colors[j]);left+=shares[:,j]
    axs[1].set_yticks(range(len(matrix)),[m.replace('_',' ') for m in matrix.index]);axs[1].invert_yaxis()
    axs[1].set_xlim(0,1);axs[1].set_xlabel('Share of units ordered');axs[1].legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,-.25),frameon=False)
    axs[1].set_title('Country allocation across both windows',loc='left',fontweight='bold')
    fig.savefig(output,dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'inputs')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    ap.add_argument('--reuse-risk',action='store_true',help='Reuse risk files only when source_manifest.json matches; otherwise fail.')
    ap.add_argument('--skip-backtest',action='store_true',help='Generate final submission without rerunning policy/price comparisons.')
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    data=load_data(args.data_dir);manifest=source_manifest(args.data_dir,data)
    if args.reuse_risk:
        saved=json.loads((out/'source_manifest.json').read_text())
        if saved!=manifest: raise ValueError('Cached forecasts do not match these source files; rerun without --reuse-risk')
        risk=pd.read_csv(out/'risk_forecasts.csv')[['week','material','disruption_prob']]
        risk_report=json.loads((out/'risk_validation.json').read_text())
        risk_bt=pd.read_csv(out/'risk_backtest_predictions.csv')
    else:
        risk,risk_report,risk_bt=select_and_forecast(data,out)
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    cutoff=int(data['outcomes'].week.max())
    if cutoff not in [208,221]: raise ValueError(f'Expected Phase1 cutoff208 or Phase2 cutoff221; got {cutoff}')
    # The public policy retains its original information set after Phase2.
    if cutoff==221:
        public_features=build_features({k:(v[v.week<=208].copy() if 'week' in v and k!='demand' else v) for k,v in data.items()})
        public=fit_predict_candidates(data,horizon_rows(public_features,data),208)
        public['disruption_prob']=public[risk_report['selected_model']]
        risk=pd.concat([public.loc[public.week<=221,['week','material','disruption_prob']],risk.loc[risk.week>=222]])
    forecasts=[];policies=[]
    for origin,start,end in [(208,209,221),(min(cutoff,221),222,234)]:
        f=predict_prices(data,origin,range(start,end+1)).merge(risk,on=['week','material'],validate='one_to_one')
        forecasts.append(f)
        policies.append(plan_policy(data,origin,start,end,f))
    forecasts=pd.concat(forecasts,ignore_index=True)
    policy=pd.concat(policies,ignore_index=True)
    joined=forecasts.merge(policy,on=['week','material'],validate='one_to_one')
    joined['id']='w'+joined.week.astype(str)+'_'+joined.material
    submission=data['sample_submission'][['id']].merge(joined.drop(columns=['week','material']),on='id',validate='one_to_one')
    submission=submission[list(data['sample_submission'].columns)]
    submission.to_csv(out/'submission.csv',index=False,float_format='%.10f')
    result=validate_submission(pd.read_csv(out/'submission.csv'),data)
    (out/'submission_validation.json').write_text(json.dumps(result,indent=2))
    joined.to_csv(out/'forecast_policy_detail.csv',index=False)
    make_plot(submission,out/'forecast_overview.png')
    if not args.skip_backtest:
        metrics,pred=validate_price_models(data)
        metrics.to_csv(out/'price_validation.csv',index=False)
        pred.to_csv(out/'price_backtest_predictions.csv',index=False)
        report={'selected_model':SELECTED_MODEL,'tuning_origins':[104,130,156],'holdout_origin':182}
        for split in ['tuning','test']:
            for model in [SELECTED_MODEL,'carry']:
                report[f'{split}_{model}_mape']=float(metrics.query('split==@split and model==@model and window=="all"').mape.mean())
        (out/'price_validation.json').write_text(json.dumps(report,indent=2))
        policy_report=evaluate_policies(data,risk_bt)
        policy_report.to_csv(out/'policy_backtest.csv',index=False)
    print(json.dumps({'submission':str(out/'submission.csv'),'validation':result,'data_cutoff':cutoff,'risk_model':risk_report['selected_model'],'price_model':SELECTED_MODEL},indent=2),flush=True)

if __name__=='__main__': main()
