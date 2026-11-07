"""T2: leave-one-event-out check of the onset jump size Model 2 uses (log benchmark at onset minus the mean of
the 4 weeks before). Pooled median (iter4) vs single-supplier-only median, predicting each held-out
single-supplier onset. Under 1 s."""
import os, sys, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "Code Base", "pipelining", "Model 1", "iter4"))
sys.path.insert(0, os.path.join(ROOT, "Code Base", "pipelining", "Model 2", "iter4"))
warnings.filterwarnings("ignore")
import model1, model2
pdata = model2.PriceData(model1.load_data(os.path.join(ROOT, "Data_new")))
P, lb = pdata.P, np.log(pdata.bench)
ev = []
for mi in range(lb.shape[0]):
    ms = pdata.mat_shock[mi]
    for o in np.where(ms[1:] & ~ms[:-1])[0] + 1:
        if o >= 4:
            cty = bool(P.sup_in_country_event[P.mat_units[mi], o].any())
            ev.append((P.materials[mi], P.week(o), cty, float(lb[mi, o] - np.nanmean(lb[mi, o - 4:o]))))
for e in ev:
    print(f"  {e[0]:15s} wk {e[1]:3d} {'country' if e[2] else 'single '} jump {e[3]:+.3f}")
J = np.array([e[3] for e in ev]); C = np.array([e[2] for e in ev])
single = np.where(~C)[0]
err_pool, err_single = [], []
for i in single:
    keep = np.arange(len(J)) != i
    err_pool.append(abs(J[i] - np.median(J[keep])))
    err_single.append(abs(J[i] - np.median(J[keep & ~C])))
print(f"pooled median {np.median(J):.3f}, single-only median {np.median(J[~C]):.3f}, country median {np.median(J[C]):.3f}")
print(f"LOO mean |error| on {len(single)} single onsets: pooled {np.mean(err_pool):.3f}, single-only {np.mean(err_single):.3f}; "
      f"single-only better in {sum(a < b for a, b in zip(err_single, err_pool))}/{len(single)}")
