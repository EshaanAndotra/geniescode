"""Compact, causal news-aware disruption forecasting challengers.

Model selection uses origins 104, 130 and 156 only. The origin-182 block is
reported after selection. Returned probabilities use no later outcomes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from risk_model import horizon_rows,markov_forecast


def compact_features(frame, variant='news'):
    h=frame.horizon.to_numpy(float)
    decay=np.exp(-h/10)
    near=np.exp(-h/4)
    state=frame.disrupted.to_numpy(float)
    x=pd.DataFrame(index=frame.index)
    x['horizon']=h/26
    x['horizon_log']=np.log1p(h)
    x['historical_rate']=frame.historical_rate
    x['current_near']=state*near
    x['current_long']=state*decay
    x['active_duration']=state*np.log1p(frame.state_age)*decay
    x['clear_duration']=(1-state)*np.log1p(frame.state_age)*decay
    x['recent_rate']=frame.recent_rate26
    if variant in ['market','news']:
        x['capacity_loss']=np.maximum(0,1-frame.capacity_ratio_min)*decay
        x['lead_excess']=np.minimum(8,frame.lead_excess_max)*decay
        x['futures_premium']=np.clip(frame.futures_premium_max,0,.3)*np.exp(-((h-4)/6)**2)
        x['tension']=np.clip(frame.tension_index_max,0,100)*decay
        x['announced_tariff']=np.clip(frame.announced_change_max,0,1)*decay
    if variant=='news':
        x['early_warning']=frame.news_early_decayed*decay
        x['late_warning']=frame.news_late_decayed*decay
        x['active_news']=frame.news_active_decayed*decay
        x['resolved_news']=frame.news_resolved_decayed*decay
        x['warning_suppliers']=frame.news_warning_supplier_fraction*decay
        x['active_suppliers']=frame.news_active_supplier_fraction*decay
    return x.replace([np.inf,-np.inf],np.nan).fillna(0)


def challenger_candidates(data,table,origin):
    train=table[(table.week<=origin)&table.target.notna()]
    test=table[table.origin==origin]
    result=test[['origin','week','material','horizon','target']].copy()
    result['markov']=markov_forecast(data,origin,test)
    rates=data['outcomes'].loc[data['outcomes'].week<=origin].groupby('material').disrupted.mean()
    global_rate=float(data['outcomes'].loc[data['outcomes'].week<=origin,'disrupted'].mean())
    result['prior']=test.material.map(rates)
    result['global_prior']=global_rate
    for prior_strength in [.1,.2,.3]:
        result[f'markov_floor{prior_strength}']=result.markov*(1-prior_strength)+.5*prior_strength
    for variant in ['state','market','news']:
        xt=compact_features(test,variant)
        x=compact_features(train,variant)
        for c in [.0001,.001,.01]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=400))
            model.fit(x,train.target)
            p=model.predict_proba(xt)[:,1]
            name=f'{variant}_c{c}'
            result[name]=p
            result[name+'_blend']=.5*p+.5*result.markov
    return result


def _warning_stage(frame):
    # Recent unresolved warnings are categorical; no vocabulary is learned from
    # future articles. Resolved stages have already been cleared by extractor.
    early=frame[[c for c in frame if c.startswith('news_open_') and c.endswith('_early')]].sum(axis=1)
    late=frame[[c for c in frame if c.startswith('news_open_') and c.endswith('_late')]].sum(axis=1)
    active=frame[[c for c in frame if c.startswith('news_open_') and c.endswith('_active')]].sum(axis=1)
    return np.select([late>.1,early>.1,active>.1],[2,1,3],default=0)


def onset_forecast(data,table,origin,shrinkage=20,decay=4):
    """Empirical week-ahead onset hazard, then fixed-origin state propagation."""
    train=table[(table.horizon==1)&(table.week<=origin)&table.target.notna()&table.disrupted.eq(0)].copy()
    train['stage']=_warning_stage(train)
    test=table[table.origin==origin].copy()
    test['stage']=_warning_stage(test)
    hist=data['outcomes'].loc[data['outcomes'].week<=origin].sort_values(['material','week'])
    transitions=[]
    material_hist={m:g.disrupted.to_numpy() for m,g in hist.groupby('material')}
    for y in material_hist.values():
        transitions.extend(zip(y[:-1],y[1:]))
    a=np.asarray(transitions)
    p01=float(a[a[:,0]==0,1].mean())
    p10=float((1-a[a[:,0]==1,1]).mean())
    stage_hazards={}
    for s in [1,2,3]:
        sub=train.loc[train.stage==s]
        stage_hazards[s]=(sub.target.sum()+p01*shrinkage)/(len(sub)+shrinkage)
    stage_hazards[0]=p01
    output=[]
    for row in test.itertuples():
        y=material_hist[row.material]
        n0=(y[:-1]==0).sum(); n1=(y[:-1]==1).sum()
        base=(np.sum((y[:-1]==0)&(y[1:]==1))+p01*45)/(n0+45)
        exit_prob=(np.sum((y[:-1]==1)&(y[1:]==0))+p10*30)/(n1+30)
        warning_lift=max(0,stage_hazards[row.stage]-p01)
        prob=float(y[-1])
        for h in range(1,row.horizon+1):
            onset=np.clip(base+warning_lift*np.exp(-(h-1)/decay),0,.95)
            prob=prob*(1-exit_prob)+(1-prob)*onset
        output.append(prob)
    return np.clip(output,.01,.99)


def evaluate_onset(data,features,output_dir='outputs'):
    table=horizon_rows(features.copy(),data)
    frames=[]
    for origin in [104,130,156]:
        frame=table.loc[table.origin==origin,['origin','week','material','horizon','target']].copy()
        frame['markov']=markov_forecast(data,origin,frame)
        for shrink in [10,30]:
            for decay in [3,6]:
                frame[f'onset_s{shrink}_d{decay}']=onset_forecast(data,table,origin,shrink,decay)
        frames.append(frame)
    tuning=pd.concat(frames)
    models=[c for c in tuning if c.startswith('onset_') or c=='markov']
    scores={c:float(np.mean((tuning[c]-tuning.target)**2)) for c in models}
    winner=min(scores,key=scores.get)
    hold=table.loc[table.origin==182,['origin','week','material','horizon','target']].copy()
    hold['markov']=markov_forecast(data,182,hold)
    final=table.loc[table.origin==int(data['outcomes'].week.max()),['origin','week','material','horizon','target']].copy()
    for shrink in [10,30]:
        for decay in [3,6]:
            hold[f'onset_s{shrink}_d{decay}']=onset_forecast(data,table,182,shrink,decay)
            final[f'onset_s{shrink}_d{decay}']=onset_forecast(data,table,int(data['outcomes'].week.max()),shrink,decay)
    report={'selected_model':winner,'tuning_brier':scores,'holdout_brier':{c:float(np.mean((hold[c]-hold.target)**2)) for c in models}}
    out=Path(output_dir)
    (out/'risk_onset_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.concat([tuning,hold]).to_csv(out/'risk_onset_backtest.csv',index=False)
    final.to_csv(out/'risk_onset_forecasts.csv',index=False)
    return report


def delayed_news_forecast(data,table,origin,shrinkage=20,strength=1):
    """Estimate risk lift from the news stage separately by future horizon.

    A first-stage warning need not cause a shock next week: its learned lift can
    peak later. All future labels contributing to estimates end at the cutoff.
    """
    train=table[(table.week<=origin)&table.target.notna()&table.disrupted.eq(0)].copy()
    test=table[table.origin==origin].copy()
    bins=[0,3,6,10,16,26]
    train['stage']=_warning_stage(train)
    test['stage']=_warning_stage(test)
    train['band']=pd.cut(train.horizon,bins=bins,labels=False)
    test['band']=pd.cut(test.horizon,bins=bins,labels=False)
    lifts={}
    for band in range(5):
        t=train.loc[train.band==band]
        baseline=t.loc[t.stage==0,'target'].mean()
        if not np.isfinite(baseline):
            baseline=t.target.mean()
        for stage in [1,2,3]:
            s=t.loc[t.stage==stage]
            # Repeated horizons in a band are not independent observations.
            width=bins[band+1]-bins[band]
            n=len(s)/width
            posterior=(s.target.sum()/width+shrinkage*baseline)/(n+shrinkage)
            lifts[(band,stage)]=max(0,posterior-baseline)
    base=markov_forecast(data,origin,test)
    lift=np.array([lifts.get((r.band,r.stage),0) for r in test.itertuples()])
    # Only initially-clear materials receive an onset lift; active states are
    # already modeled by persistence and exit probabilities.
    prob=base+(1-base)*lift*strength*(1-test.disrupted.to_numpy())
    return np.clip(prob,.01,.99)


def evaluate_delayed_news(data,features,output_dir='outputs'):
    table=horizon_rows(features.copy(),data)
    frames=[]
    for origin in [104,130,156]:
        frame=table.loc[table.origin==origin,['origin','week','material','horizon','target']].copy()
        frame['markov']=markov_forecast(data,origin,frame)
        for shrink in [10,30]:
            for strength in [.5,1.0]:
                frame[f'delayed_s{shrink}_w{strength}']=delayed_news_forecast(data,table,origin,shrink,strength)
        frames.append(frame)
    tuning=pd.concat(frames)
    models=[c for c in tuning if c.startswith('delayed_') or c=='markov']
    scores={c:float(np.mean((tuning[c]-tuning.target)**2)) for c in models}
    winner=min(scores,key=scores.get)
    hold=table.loc[table.origin==182,['origin','week','material','horizon','target']].copy()
    hold['markov']=markov_forecast(data,182,hold)
    final_origin=int(data['outcomes'].week.max())
    final=table.loc[table.origin==final_origin,['origin','week','material','horizon','target']].copy()
    final['markov']=markov_forecast(data,final_origin,final)
    for shrink in [10,30]:
        for strength in [.5,1.0]:
            name=f'delayed_s{shrink}_w{strength}'
            hold[name]=delayed_news_forecast(data,table,182,shrink,strength)
            final[name]=delayed_news_forecast(data,table,final_origin,shrink,strength)
    report={'selected_model':winner,'tuning_brier':scores,'holdout_brier':{c:float(np.mean((hold[c]-hold.target)**2)) for c in models}}
    out=Path(output_dir)
    (out/'risk_delayed_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.concat([tuning,hold]).to_csv(out/'risk_delayed_backtest.csv',index=False)
    final.to_csv(out/'risk_delayed_forecasts.csv',index=False)
    return report


def joint_state_forecast(data,table,origin,shrinkage=30):
    """Eight states: disruption indicator crossed with observed news stage.

    Future warning, escalation, shock, and resolution transitions are integrated
    through a causal Markov chain. No future news is supplied to the chain.
    """
    history=table.loc[(table.horizon==1)&(table.origin<=origin)].sort_values(['material','origin']).copy()
    history['joint_state']=4*history.disrupted.astype(int)+_warning_stage(history)
    counts=np.zeros((8,8))
    material_counts={}
    last={}
    for material,g in history.groupby('material'):
        c=np.zeros((8,8))
        states=g.joint_state.to_numpy(int)
        for s,t in zip(states[:-1],states[1:]):
            c[s,t]+=1
        counts+=c
        material_counts[material]=c
        last[material]=states[-1]
    # A weak fallback uses empirically estimated disruption transitions and a
    # neutral news state, only stabilizing rare/unobserved joint-state rows.
    base=np.zeros((8,8))
    for risk in [0,1]:
        c=counts[risk*4:risk*4+4].sum(axis=0)
        chance=float(c[4:].sum()/max(c.sum(),1))
        base[risk*4:risk*4+4,0]=1-chance
        base[risk*4:risk*4+4,4]=chance
    pooled=(counts+2*base)/(counts.sum(axis=1,keepdims=True)+2)
    test=table.loc[table.origin==origin]
    answers={}
    for material,c in material_counts.items():
        transition=(c+shrinkage*pooled)/(c.sum(axis=1,keepdims=True)+shrinkage)
        distribution=np.zeros(8)
        distribution[last[material]]=1
        for h in range(1,27):
            distribution=distribution@transition
            answers[(material,h)]=distribution[4:].sum()
    return np.clip([answers[(r.material,r.horizon)] for r in test.itertuples()],.01,.99)


def evaluate_joint(data,features,output_dir='outputs'):
    table=horizon_rows(features.copy(),data)
    frames=[]
    for origin in [104,130,156]:
        frame=table.loc[table.origin==origin,['origin','week','material','horizon','target']].copy()
        frame['markov']=markov_forecast(data,origin,frame)
        for shrink in [10,30,60]:
            frame[f'joint_s{shrink}']=joint_state_forecast(data,table,origin,shrink)
        frames.append(frame)
    tuning=pd.concat(frames)
    models=[c for c in tuning if c.startswith('joint_') or c=='markov']
    scores={c:float(np.mean((tuning[c]-tuning.target)**2)) for c in models}
    winner=min(scores,key=scores.get)
    hold=table.loc[table.origin==182,['origin','week','material','horizon','target']].copy()
    hold['markov']=markov_forecast(data,182,hold)
    final_origin=int(data['outcomes'].week.max())
    final=table.loc[table.origin==final_origin,['origin','week','material','horizon','target']].copy()
    final['markov']=markov_forecast(data,final_origin,final)
    for shrink in [10,30,60]:
        hold[f'joint_s{shrink}']=joint_state_forecast(data,table,182,shrink)
        final[f'joint_s{shrink}']=joint_state_forecast(data,table,final_origin,shrink)
    report={'selected_model':winner,'tuning_brier':scores,'holdout_brier':{c:float(np.mean((hold[c]-hold.target)**2)) for c in models}}
    out=Path(output_dir)
    (out/'risk_joint_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.concat([tuning,hold]).to_csv(out/'risk_joint_backtest.csv',index=False)
    final.to_csv(out/'risk_joint_forecasts.csv',index=False)
    return report


def predict_challenger(data,table,origin,model_name):
    # Public callable supports use after selection without changing risk_model.py.
    pred=challenger_candidates(data,table,origin)
    return pred[['week','material',model_name]].rename(columns={model_name:'disruption_prob'})


def evaluate_challengers(data,features,output_dir='outputs'):
    table=horizon_rows(features,data)
    frames=[]
    ids=['origin','week','material','horizon','target']
    for o in [104,130,156]:
        frames.append(challenger_candidates(data,table,o))
        print(f'compact risk tuning origin {o} done',flush=True)
    tuning=pd.concat(frames,ignore_index=True)
    models=[c for c in tuning if c not in ids]
    scores={c:float(np.mean((tuning[c]-tuning.target)**2)) for c in models}
    winner=min(scores,key=scores.get)
    print('winner before holdout',winner,scores[winner],flush=True)
    # Selection fixed before looking at the holdout outcomes.
    hold=challenger_candidates(data,table,182)
    hold_scores={c:float(np.mean((hold[c]-hold.target)**2)) for c in models}
    final=challenger_candidates(data,table,int(data['outcomes'].week.max()))
    report={'selected_model':winner,'tuning_brier':scores,'holdout_brier':hold_scores,
        'selected_news_model':min([c for c in models if c.startswith('news_')],key=scores.get),
        'tuning_origins':[104,130,156],'holdout_origin':182}
    out=Path(output_dir)
    (out/'risk_challenger_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.concat([tuning,hold],ignore_index=True).to_csv(out/'risk_challenger_backtest.csv',index=False)
    final.to_csv(out/'risk_challenger_forecasts.csv',index=False)
    return report


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='C:/Users/eshaa/Downloads')
    ap.add_argument('--output-dir',default='outputs')
    ap.add_argument('--onset',action='store_true')
    ap.add_argument('--delayed',action='store_true')
    ap.add_argument('--joint',action='store_true')
    args=ap.parse_args()
    data={p.stem:pd.read_csv(p) for p in Path(args.data_dir).glob('*.csv') if p.stem in
        ['suppliers','supply_history','prices','outcomes','news','tariffs','shipping','macro','demand','sample_submission']}
    features=pd.read_csv(Path(args.output_dir)/'risk_features.csv')
    evaluate=evaluate_joint if args.joint else evaluate_delayed_news if args.delayed else evaluate_onset if args.onset else evaluate_challengers
    report=evaluate(data,features,args.output_dir)
    print(json.dumps(report,indent=2))
