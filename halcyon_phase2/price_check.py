from pathlib import Path
import json
import pandas as pd
import numpy as np
from price_model import predict_prices, SELECTED_MODEL

project=Path(__file__).resolve().parent
root=project/'inputs'
output=project/'outputs'
names=['prices','outcomes','demand','supply_history','tariffs','shipping','macro','suppliers']
data={n:pd.read_csv(root/(n+'.csv')) for n in names}
met=pd.read_csv(output/'price_validation.csv')
tuning=met.loc[(met.split=='tuning')&(met.window=='all')].groupby('model').mape.mean().sort_values()
print(tuning)
print(met.loc[met.model.isin(['carry',SELECTED_MODEL])].to_string(index=False))
best=tuning.index[0]
assert SELECTED_MODEL==best
pred=predict_prices(data,208,range(209,235))
assert pred.shape==(260,3)
assert pred.price_forecast.gt(0).all()
assert not pred.duplicated(['week','material']).any()
pred.to_csv(output/'price_forecasts.csv',index=False)
# Prefix invariance: change future non-demand observations radically.
expected=predict_prices(data,156,range(157,183))
poisoned={k:v.copy() for k,v in data.items()}
for name,frame in poisoned.items():
    if 'week' not in frame or name=='demand':
        continue
    numeric=frame.select_dtypes(include='number').columns.difference(['week'])
    frame.loc[frame.week>156,numeric]=0
actual=predict_prices(poisoned,156,range(157,183))
pd.testing.assert_frame_equal(expected,actual)
results={'selected_model':SELECTED_MODEL,'selection_origins':[104,130,156],
    'holdout_origin':182,'horizons':26,'tuning_mape':float(tuning[SELECTED_MODEL]),
    'tuning_carry_mape':float(tuning['carry']),
    'holdout_mape':float(met.loc[(met.split=='test')&(met.window=='all')&(met.model==SELECTED_MODEL),'mape'].iloc[0]),
    'holdout_carry_mape':float(met.loc[(met.split=='test')&(met.window=='all')&(met.model=='carry'),'mape'].iloc[0]),
    'future_covariate_invariance':True,
    'benchmark_max_rounding_discrepancy':float(data['prices'].groupby(['week','material']).spot_price.mean().sub(data['outcomes'].set_index(['week','material']).benchmark_price).abs().max()),
    'metrics':json.loads(met.to_json(orient='records'))}
(output/'price_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in results.items() if k!='metrics'},indent=2))
