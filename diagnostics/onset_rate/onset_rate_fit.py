"""Fit Model 1 iter4 (the config Model 3 plans with) at every origin and save what the scenario generator
samples onsets from: q per calm source plus the realized first onset per source in the window.
Usage: python3 onset_rate_fit.py [first last step n_proc]   (default 52 208 1 4; ~4-5 min on 4 processes)"""
import os, sys, time, pickle, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
M1 = os.path.join(ROOT, "Code Base", "pipelining", "Model 1", "iter4")
DATA = os.path.join(ROOT, "Data_new")
H = 13


def init():
    global m1, P
    sys.path.insert(0, M1)
    warnings.filterwarnings("ignore")
    import model1 as _m
    m1 = _m
    P = m1.Panels(m1.load_data(DATA))


def first_onset(active, t0, H):
    """Per row: horizon (1..H) of the first run start in t0+1..t0+H, given calm at t0; 0 if none."""
    out = np.zeros(active.shape[0], int)
    for i in range(active.shape[0]):
        prev = active[i, t0]
        for h in range(1, H + 1):
            t = t0 + h
            if t >= active.shape[1]:
                break
            if active[i, t] and not prev:
                out[i] = h
                break
            prev = active[i, t]
    return out


def first_true(mask, t0, H):
    out = np.zeros(mask.shape[0], int)
    for i in range(mask.shape[0]):
        hit = np.where(mask[i, t0 + 1: t0 + H + 1])[0]
        out[i] = hit[0] + 1 if len(hit) else 0
    return out


def one(o):
    M = m1.DisruptionModel().fit(P, origin=o)
    src = M._source_probs(H)
    t0 = P.idx(o)
    return dict(origin=o, q_sup=src["q_sup"], sup_state=src["sup_state"], q_cty=src["q_cty"], cty_state=src["cty_state"],
                q_lane=src["q_lane"], lane_state=src["lane_state"], tar_state=src["tar_state"], q_tariff=M.q_tariff,
                # realized: single-supplier onsets only (country-typed onsets count under country)
                real_sup=first_true(P.sup_onset_single, t0, H), real_sup_any=first_onset(P.sup_active, t0, H),
                real_cty=first_onset(P.country_active, t0, H), real_lane=first_onset(P.lane_closed, t0, H),
                real_tar=first_onset(P.tar_active, t0, H), pending=P.pending[:, t0].copy(),
                news_open=M.news_open.to_dict("records"))


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:5]] + [52, 208, 1, 4][len(sys.argv[1:5]):]
    first, last, step, nproc = a
    from multiprocessing import Pool
    t = time.time()
    with Pool(nproc, initializer=init) as pool:
        res = pool.map(one, list(range(first, last + 1, step)))
    init()
    meta = dict(suppliers=P.suppliers, countries=list(P.countries), lanes=list(P.lanes))
    with open(os.path.join(HERE, "onset_rate_fits.pkl"), "wb") as f:
        pickle.dump(dict(res=res, meta=meta), f)
    print(f"{len(res)} fits in {time.time() - t:.0f}s -> onset_rate_fits.pkl")
