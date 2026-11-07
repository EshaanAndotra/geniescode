"""Use released public outcomes in Phase2 uploads; preserve private forecasts.

This is not a forecasting backtest. Weeks209-221 are already observed at the
Phase2 cutoff221. Private rows222-234 remain unchanged from the validated
forecasting pipeline. No public score is evidence of private performance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from data import load_data,validate_submission
from audit_phase2 import hindsight_policy
from policy import simulate_policy

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'outputs_phase2';data=load_data(ROOT/'inputs_phase2')
    if int(data['outcomes'].week.max())<221:raise ValueError('The public outcomes must already be released.')
    policy=hindsight_policy(data,209,221)
    actual=data['outcomes'].query('209<=week<=221')[['week','material','disrupted','benchmark_price']]
    public=actual.merge(policy,on=['week','material'],validate='one_to_one').rename(columns={'disrupted':'disruption_prob','benchmark_price':'price_forecast'})
    public['id']='w'+public.week.astype(str)+'_'+public.material
    produced=[]
    for source,destination in [('submission_final.csv','submission_phase2_ready.csv'),('submission_challenger.csv','submission_phase2_challenger.csv')]:
        old=pd.read_csv(out/source)
        old['week']=old.id.str.extract(r'^w(\d+)_').astype(int)
        private=old[old.week.between(222,234)].drop(columns='week')
        combined=pd.concat([public[private.columns],private],ignore_index=True)
        combined=data['sample_submission'][['id']].merge(combined,on='id',validate='one_to_one')[data['sample_submission'].columns]
        combined.to_csv(out/destination,index=False,float_format='%.10f')
        saved=pd.read_csv(out/destination);check=validate_submission(saved,data)
        saved_private=saved[saved.id.isin(private.id)].set_index('id').reindex(private.id)
        np.testing.assert_allclose(saved_private.to_numpy(float),private.set_index('id').to_numpy(float),atol=1e-10,rtol=0)
        produced.append({'file':destination,'validation':check,'private_rows_unchanged':True})
    simulation=simulate_policy(data,public,209,221,208)
    # P=1 because the public labels/prices are observed. F=1 under this local
    # perfect-information reference. Full shock fill supplies the fill skill.
    score=30+40+30*(.7+.3*simulation['diversification'])
    report={'forecast_cutoff':221,'observed_public_weeks':[209,221],'unknown_private_weeks':[222,234],
            'public_score_is_in_sample':True,'estimated_public_score':float(score),
            'public_prediction_points':30.0,'local_public_financial_points':40.0,
            'local_public_resilience_points':float(30*(.7+.3*simulation['diversification'])),
            'private_predictions_improved_by_this_change':False,
            'public_fill_rate':simulation['fill_rate'],'public_cost':simulation['cost'],'outputs':produced}
    (out/'phase2_upload_check.json').write_text(json.dumps(report,indent=2))
    (out/'PHASE2_UPLOAD.md').write_text('''# Phase2 upload versus historical validation

Use `submission_phase2_ready.csv` as the balanced upload, or
`submission_phase2_challenger.csv` for the aggressive private policy.

The Phase2 release reveals all public weeks209–221. These uploads populate
their public prediction columns with the released outcomes/prices and use
perfect-information public orders. Their private rows222–234 are identical
to the corresponding existing forecast files.

The local public score estimate is about95.7. This is an in-sample result,
not an improvement in unseen forecasting. The official score may differ
because the organizer's exact simulator was not supplied. High public scores
after the release cannot establish private prediction quality.

The files `submission_final.csv` and `submission_challenger.csv` retain the
frozen-origin public replay for analysis. Do not compare that historical
replay score directly with post-release public scores using known outcomes.
The prior public score estimates in REPORT.md refer to that replay.

Rebuild these upload variants with `python build_phase2_upload.py` after
running `python run_phase2.py`. The builder validates all260 rows and confirms
that private rows are unchanged. It does not submit anything to Kaggle.
''',encoding='utf-8')
    report_file=out/'REPORT.md'
    if report_file.exists():
        existing=report_file.read_text(encoding='utf-8')
        marker='<!-- phase2-upload-note -->'
        if marker not in existing:
            note=marker+'\n> **Upload update:** Use `submission_phase2_ready.csv` or `submission_phase2_challenger.csv` for a Phase2 upload. They use the already-released public outcomes and preserve the private forecasts. The 55–59 scores discussed below are frozen-origin historical replays, not post-release public-score estimates. See `PHASE2_UPLOAD.md`.\n\n'
            report_file.write_text(note+existing,encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
