"""Checks for submission, baseline reconstruction, leakage and mass balance."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_data,validate_submission
from policy import naive_policy,plan_policy,simulate_policy
from risk_model import horizon_rows
from risk_challenger import onset_forecast
from news_features import get_news_features
from price_model import predict_prices

ROOT=Path(__file__).resolve().parent

class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_data(ROOT/'inputs')
        cls.submission=pd.read_csv(ROOT/'outputs/submission.csv')

    def test_submission_and_invalid_country(self):
        self.assertTrue(validate_submission(self.submission,self.data)['valid'])
        bad=self.submission.copy()
        i=bad.index[bad.id.eq('w209_lithium')][0]
        bad.loc[i,'share_SUN']=.1;bad.loc[i,'share_KES']-=.1
        with self.assertRaises(ValueError): validate_submission(bad,self.data)

    def test_baseline_reconstructs_sample(self):
        actual=naive_policy(self.data,208,209,234)
        actual['id']='w'+actual.week.astype(str)+'_'+actual.material
        cols=['order_units']+[c for c in actual if c.startswith('share_')]
        expected=self.data['sample_submission'].set_index('id')[cols]
        np.testing.assert_allclose(actual.set_index('id').reindex(expected.index)[cols],expected,atol=1e-10)

    def test_future_data_cannot_change_frozen_forecasts(self):
        origin=156
        original=self.data
        altered={k:v.copy() for k,v in original.items()}
        for name,df in altered.items():
            if 'week' not in df or name=='demand': continue
            for col in df.select_dtypes(include='number'):
                if col!='week': df.loc[df.week>origin,col]=0
        a=predict_prices(original,origin,range(157,183))
        b=predict_prices(altered,origin,range(157,183))
        pd.testing.assert_frame_equal(a,b)
        features=pd.read_csv(ROOT/'outputs/risk_features.csv')
        table=horizon_rows(features,original)
        changed=table.copy();changed.loc[changed.week>origin,'target']=1-changed.loc[changed.week>origin,'target']
        np.testing.assert_allclose(onset_forecast(original,table,origin,10,6),onset_forecast(altered,changed,origin,10,6),atol=1e-12)
        for material in original['suppliers'].material.unique():
            a=get_news_features(original['news'],original['suppliers'],origin,material)
            b=get_news_features(original['news'].query('week<=@origin').copy(),original['suppliers'],origin,material)
            self.assertEqual(a,b)

    def test_simulator_conserves_units_and_costs(self):
        plan=plan_policy(self.data,182,183,195)
        result=simulate_policy(self.data,plan,183,195,182)
        for row in result['by_material']:
            d=self.data['demand'].query('material==@row["material"] and 183<=week<=195')
            initial=5*d.demand_units.mean()
            self.assertAlmostEqual(initial+row['accepted_units'],row['filled']+row['ending_inventory']+row['in_transit'],places=6)
            self.assertAlmostEqual(row['cost'],row['purchase_cost']+row['holding_cost']+row['shortage_cost']-row['salvage'],places=6)
        self.assertGreaterEqual(result['fill_rate'],0)
        self.assertLessEqual(result['fill_rate'],1)

if __name__=='__main__': unittest.main(verbosity=2)
