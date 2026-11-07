"""Strict loading and source validation for the ten competition CSVs."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FILES = ['tariffs','supply_history','suppliers','shipping','sample_submission','prices','outcomes','news','macro','demand']
COUNTRIES = ['KES','VAL','SUN','ORI','HAL','AUR']

def load_data(directory):
    directory = Path(directory)
    data = {name: pd.read_csv(directory / (name + '.csv')) for name in FILES}
    keys = dict(tariffs=['week','material','country'], supply_history=['week','material','country'],
                suppliers=['material','country'], shipping=['week','lane'], sample_submission=['id'],
                prices=['week','material','country'], outcomes=['week','material'], news=['article_id'],
                macro=['week','country'], demand=['week','material'])
    for name, frame in data.items():
        if frame.duplicated(keys[name]).any():
            raise ValueError(f'Duplicate source keys in {name}')
        required = frame.drop(columns=['announced_rate','announced_effective_week'], errors='ignore')
        if required.isna().any().any():
            raise ValueError(f'Missing required values in {name}')
    suppliers = data['suppliers']
    if len(suppliers)!=30 or not suppliers.groupby('material').size().eq(3).all():
        raise ValueError('Expected ten materials with three suppliers each')
    cutoff = int(data['outcomes'].week.max())
    expected_pairs = {(w,r.material,r.country) for w in range(1,cutoff+1) for r in suppliers.itertuples()}
    for name in ['prices','supply_history','tariffs']:
        actual_pairs=set(data[name][['week','material','country']].itertuples(index=False,name=None))
        if actual_pairs!=expected_pairs:
            raise ValueError(f'Incomplete or mismatched weekly supplier coverage in {name}')
    for name,key,values in [('shipping','lane',suppliers.lane.unique()),('macro','country',suppliers.country.unique()),('outcomes','material',suppliers.material.unique())]:
        expected_keys={(w,v) for w in range(1,cutoff+1) for v in values}
        actual_keys=set(data[name][['week',key]].itertuples(index=False,name=None))
        if actual_keys!=expected_keys:
            raise ValueError(f'Incomplete weekly coverage in {name}')
    expected_demand={(w,m) for w in range(1,235) for m in suppliers.material.unique()}
    if not expected_demand.issubset(set(data['demand'][['week','material']].itertuples(index=False,name=None))):
        raise ValueError('Known demand schedule must cover weeks 1-234')
    actual = data['prices'].groupby(['week','material']).spot_price.mean()
    target = data['outcomes'].set_index(['week','material']).benchmark_price
    if not np.allclose(actual.reindex(target.index), target, atol=0.0001):
        raise ValueError('Outcome benchmark does not reconcile to average spot price')
    return data

def source_manifest(directory, data):
    return {name: {'rows':len(df), 'columns':list(df.columns),
                   'min_week':int(df.week.min()) if 'week' in df else None,
                   'max_week':int(df.week.max()) if 'week' in df else None,
                   'sha256':hashlib.sha256((Path(directory)/(name+'.csv')).read_bytes()).hexdigest()}
            for name,df in data.items()}

def validate_submission(frame, data):
    expected = data['sample_submission']
    if list(frame.columns) != list(expected.columns):
        raise ValueError('Submission columns do not match sample')
    if len(frame)!=260 or frame.id.duplicated().any() or set(frame.id)!=set(expected.id):
        raise ValueError('Submission must have the 260 unique sample IDs')
    nums = frame.drop(columns='id').to_numpy(dtype=float)
    if not np.isfinite(nums).all():
        raise ValueError('Submission contains blanks or nonfinite values')
    if not frame.disruption_prob.between(0,1).all() or not (frame.price_forecast>0).all() or not (frame.order_units>=0).all():
        raise ValueError('Submission forecast or order range violation')
    shares = frame[['share_'+c for c in COUNTRIES]]
    if not ((shares>=0)&(shares<=1)).all().all() or not np.allclose(shares.sum(axis=1),1,atol=1e-8,rtol=0):
        raise ValueError('Invalid supplier shares')
    allowed = set(zip(data['suppliers'].material,data['suppliers'].country))
    for row in frame.itertuples(index=False):
        material = row.id.split('_',1)[1]
        for country in COUNTRIES:
            if (material,country) not in allowed and getattr(row,'share_'+country)!=0:
                raise ValueError(f'Ineligible country in {row.id}: {country}')
    return {'valid':True,'rows':len(frame),'columns':len(frame.columns),
            'max_share_sum_error':float(abs(shares.sum(axis=1)-1).max())}
