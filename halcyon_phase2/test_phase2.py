"""Meaningful checks for causal forecasting, replay coverage and final schemas."""
from pathlib import Path
import unittest,json
import numpy as np
import pandas as pd
from data import load_data,validate_submission
from model_v2 import make_panel,examples,predict_candidates,aggregate_predictions,add_robust_candidates
from policy_v2 import plan_v2

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs_phase2'

class Phase2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_data(ROOT/'inputs_phase2')
        cls.panel=pd.read_pickle(OUT/'supplier_panel.pkl') if (OUT/'supplier_panel.pkl').exists() else make_panel(cls.data)

    def test_both_submissions(self):
        for name in ['submission_final.csv','submission_challenger.csv']:
            self.assertTrue(validate_submission(pd.read_csv(OUT/name),self.data)['valid'])

    def test_historical_coverage_and_cutoffs(self):
        r=pd.read_csv(OUT/'historical_replay.csv')
        self.assertEqual(len(r),1950);self.assertEqual(set(r.week),set(range(13,208)))
        self.assertFalse(r.duplicated(['week','material']).any())
        self.assertTrue((r.week>r.origin).all());self.assertTrue((r.week-r.origin<=13).all())

    def test_prefix_invariance(self):
        cutoff=155
        prefix={k:(v[v.week<=cutoff].copy() if 'week' in v and k!='demand' else v.copy()) for k,v in self.data.items()}
        original=self.panel;small=make_panel(prefix)
        pd.testing.assert_frame_equal(original[original.week<=cutoff].reset_index(drop=True),small,check_exact=False,atol=1e-12)
        table=examples(small,prefix)
        forecast=predict_candidates(prefix,small,table,cutoff)
        saved=pd.read_csv(OUT/f'candidates_supplier_{cutoff}.csv')
        cols=['tree_small','tree_large','tree_blend','capacity_mean','delay_mean','price_tree','price_futures']
        a=forecast.set_index(['week','material','country']);b=saved.set_index(['week','material','country']).reindex(a.index)
        np.testing.assert_allclose(a[cols],b[cols],rtol=1e-9,atol=1e-9)
        material=aggregate_predictions(forecast)
        full_gate=add_robust_candidates(material,original,self.data)
        prefix_gate=add_robust_candidates(material,small,prefix)
        np.testing.assert_allclose(full_gate.gated_risk,prefix_gate.gated_risk,atol=1e-12)
        np.testing.assert_allclose(full_gate.gated_announced,prefix_gate.gated_announced,atol=1e-12)

    def test_policy_has_no_future_supply_access(self):
        cutoff=155;prefix={k:(v[v.week<=cutoff].copy() if 'week' in v and k!='demand' else v.copy()) for k,v in self.data.items()}
        selection=json.loads((OUT/'model_selection.json').read_text())
        sf=pd.read_csv(OUT/f'candidates_supplier_{cutoff}.csv');f=pd.read_csv(OUT/'selected_forecasts.csv');f=f[f.origin.eq(cutoff)]
        a=plan_v2(self.data,cutoff,156,168,sf,f,selection['risk'],selection['price'],selection['policy'])
        b=plan_v2(prefix,cutoff,156,168,sf,f,selection['risk'],selection['price'],selection['policy'])
        pd.testing.assert_frame_equal(a,b,atol=1e-9,rtol=1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
