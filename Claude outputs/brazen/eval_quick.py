"""
eval_tree.py - full score (30 P + 40 F + 21 fill skill + 9 diversification) of candidate files for weeks 222-234
on sampled futures from two independent world generators:
  it8 worlds   Model 1 + Model 3 iter5's make_scenarios (status quo / Merrow war / big war), as in eval_scenarios.py
  tree worlds  the event tree in event_tree.py (calibrated mix, and with single events forced on or off)
Scoring loop copied from scenarios/eval_scenarios.py.

Usage: python eval_tree.py PIPELINE_DIR OUT_DIR name=file.csv ...
"""
import os, sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
PIPE, OUTD = sys.argv[1], sys.argv[2]
files = dict(a.split("=", 1) for a in sys.argv[3:])
sys.argv = [sys.argv[0], PIPE, OUTD]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_tree as et, m1patch
m1patch.apply(et.m1)
m1, m3, W = et.m1, et.m3, et.W
START, END, T = et.START, et.END, et.H
out = W.data["outcomes"]
dmat = out.pivot(index="material", columns="week", values="disrupted").loc[W.materials]
base = dmat.loc[:, :221].mean(axis=1).values
b221 = et.B221; rw_sd = et.RW_SD
STOCK = os.environ.get("STOCK", "sum5")

subs = {}
for n, f in files.items():
    s = pd.read_csv(f); p = s.id.str.extract(r"^w(\d+)_(.+)$"); s["week"], s["material"] = p[0].astype(int), p[1]
    subs[n] = s[s.week.between(START, END)]


def mat_arr(s, col):
    return s.pivot(index="material", columns="week", values=col).loc[W.materials].values


def score(cap_m, lead_a, cost_m, w, dis, noise, price, info, seed):
    S = len(w); masks = m3._window_masks(info, dis, 0, T); calmc = cap_m >= 1 - 1e-9
    rng = np.random.default_rng(seed)
    act = b221[None, :, None] * np.exp(price + np.cumsum(rng.normal(0, rw_sd, (S, len(W.materials), T)), axis=2))
    y = dis.astype(float)
    bs_den = ((base[None, :, None] - y) ** 2).mean(axis=(1, 2))
    mape_c = (np.abs(b221[None, :, None] - act) / act).mean(axis=(1, 2))
    naive_c = np.zeros(S); hind_c = np.zeros(S); naive_u = np.zeros(S); shock_d = np.zeros(S)
    cost = {n: np.zeros(S) for n in subs}; unmet = {n: np.zeros(S) for n in subs}; divs = {n: [] for n in subs}
    for mi, m in enumerate(W.materials):
        units = W.mat_units[mi]; d = W.demand[mi, START - 1:END]; I0 = m3.start_stock(d, STOCK)
        bc = W.normal_cap[units]
        capx = bc[None, :, None] * np.where(calmc[:, units], noise[:, units], cap_m[:, units])
        lead = W.base_lead[units][None, :, None] + lead_a[:, units]
        cst = info.cost_base[units][None, :, :] * cost_m[:, units]
        avg = float(np.mean(info.cost_base[units]))
        k = int(np.where(units == W.naive_unit[mi])[0][0]); on = np.zeros((len(units), T)); on[k] = d
        c, u = m3.simulate_batch(on, capx, lead, cst, d, I0, avg); naive_c += c; naive_u += (u * masks[:, mi]).sum(1)
        shock_d += (d[None] * masks[:, mi]).sum(1)
        fh = m3.hindsight_batch(d, I0, cst, capx, lead, m3.HOLD * avg, m3.PENALTY * avg, m3.SALVAGE * avg)
        hind_c += m3.simulate_batch(fh, capx, lead, cst, d, I0, avg)[0]
        for n, sb in subs.items():
            o = m3._orders_matrix(W, sb, mi, START, END)
            c, u = m3.simulate_batch(o, capx, lead, cst, d, I0, avg)
            cost[n] += c; unmet[n] += (u * masks[:, mi]).sum(1)
            t_ = o.sum(0); sh = np.where(t_ > 0, o / np.maximum(t_, 1e-12), 1 / len(units))
            divs[n].append(m3.diversification(sh.T))
    gap = naive_c - hind_c; short = naive_u > 1e-6
    fn = np.where(shock_d > 0, 1 - naive_u / np.maximum(shock_d, 1e-9), 1.0)
    res = {}
    for n, sb in subs.items():
        q = mat_arr(sb, "disruption_prob"); f = mat_arr(sb, "price_forecast")
        bs = 1 - ((q[None] - y) ** 2).mean(axis=(1, 2)) / np.maximum(bs_den, 1e-12)
        ps = 1 - (np.abs(f[None] - act) / act).mean(axis=(1, 2)) / mape_c
        Pp = np.clip(0.6 * bs + 0.4 * ps, 0, 1)
        F = np.clip((naive_c - cost[n]) / np.maximum(gap, 1e-9), 0, 1)
        fo = np.where(shock_d > 0, 1 - unmet[n] / np.maximum(shock_d, 1e-9), 1.0)
        fs = np.where(short, np.clip((fo - fn) / np.maximum(1 - fn, 1e-9), 0, 1), 0.0)
        dv = float(np.mean(np.concatenate(divs[n])))
        res[n] = dict(total=30 * Pp + 40 * F + 21 * fs + 9 * dv, brier=bs, price=ps, F=F, fs=fs, div=np.full(S, dv))
    return res


results = {}
# --- it8 worlds (Model 1 based), identical seeds to eval_scenarios.py
IT8 = {"it8 status quo": ((), {}), "it8 Merrow war": ((("lane:MERROW", 1.0, 1, 3, 0),), {}),
       "it8 big war": ((("lane:MERROW", 1.0, 1, 3, 6), ("cty:KES", 1.0, 2, 5, 8)), {}),
       "it8 no closure": ((("lane:MERROW", 0.0, 1, 13, 0),), {})}
for wi, (wname, (ovr, extra)) in enumerate(IT8.items()):
    cfg = m3.Config(seed=500 + 37 * wi, n_scen=300, m1_config=m1.Config(source_overrides=ovr, **extra))
    info = m3.OriginInfo(W, 221, T, cfg.m1_config)
    cap_m, lead_a, cost_m, w, dis, noise = et.ORIG_MAKE(W, info, cfg)
    results[wname] = score(cap_m, lead_a, cost_m, w, dis, noise, m3._LAST_PRICE, info, 900 + wi)
    print("done", wname, flush=True)
# --- event-tree worlds
info0 = m3.OriginInfo(W, 221, T)
_bw = et.variant(merrow_p=1.0, kes_p=1.0); _bw["kes_country"]["onset"] = (223, 226)
TREE = {"tree big war": _bw}
for wi, (wname, spec) in enumerate(TREE.items()):
    r = et.gen(spec, 600, 4000 + 101 * wi, info0)
    results[wname] = score(*r, info0, 7000 + wi)
    print("done", wname, flush=True)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
names = list(subs); SAFE = os.environ.get("SAFE", names[0]); THR = float(os.environ.get("THR", 85))
np.savez_compressed(os.path.join(OUTD, f"totals_{STOCK}.npz"), **{f"{w_}|{n}": r[n]["total"] for w_, r in results.items() for n in names})
print("\n### single-file mean total"); print(pd.DataFrame({w_: {n: float(np.mean(r[n]["total"])) for n in names} for w_, r in results.items()}).round(2).to_string())
print(f"\n### single-file P(total > {THR:g})"); print(pd.DataFrame({w_: {n: float(np.mean(r[n]["total"] > THR)) for n in names} for w_, r in results.items()}).round(3).to_string())
for part in ("brier", "price", "F", "fs", "div"):
    print(f"\n### {part}"); print(pd.DataFrame({w_: {n: float(np.mean(r[n][part])) for n in names} for w_, r in results.items()}).round(3).to_string())
print(f"\n### single-file 90th percentile"); print(pd.DataFrame({w_: {n: float(np.quantile(r[n]["total"], .9)) for n in names} for w_, r in results.items()}).round(2).to_string())
print(f"\n### with {SAFE}: P(better of two > {THR:g})")
print(pd.DataFrame({w_: {n: float(np.mean(np.maximum(r[SAFE]["total"], r[n]["total"]) > THR)) for n in names} for w_, r in results.items()}).round(3).to_string())
print(f"\n### with {SAFE}: mean of better of two")
print(pd.DataFrame({w_: {n: float(np.mean(np.maximum(r[SAFE]["total"], r[n]["total"]))) for n in names} for w_, r in results.items()}).round(2).to_string())
