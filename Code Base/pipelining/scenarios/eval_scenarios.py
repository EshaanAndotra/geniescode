"""
eval_scenarios.py - full score (30 P + 40 F + 21 fill skill + 9 diversification) of submission files on
sampled futures for the private window (weeks 222-234, planned from week 221), in three worlds:
  status quo   Model 1's own odds (already closes Merrow about half the time)
  Merrow war   the Merrow Strait closes in weeks 222-224 (certain), fitted closure length
  big war      Merrow closes (at least 6 weeks) and a Kestria-wide event (capacity 20%, at least 8 weeks)
Prices in each future = week-221 benchmark x the scenario's shock jumps x a calm random walk (same noise for
every file). Prints each file's mean score and parts per world, best of two with the first file per world,
and best of two for mixes of the worlds.

  python scenarios/eval_scenarios.py main=../../submissions/submission_main_it8.csv war=../../submissions/submission_war_it8.csv
  STOCK=5avg python scenarios/eval_scenarios.py ...      # score with the 5 x mean starting-stock reading
"""
import sys, os, inspect, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); import run_pipeline_p2 as rp
os.environ["MODEL1_DIR"] = os.path.dirname(rp.PATHS["model1"])
m1 = rp.load_module("model1"); m3 = rp.load_module("model3")
src = inspect.getsource(m3.make_scenarios)
src = src.replace("    dis = np.zeros((S, nM, H), bool)\n", "    dis = np.zeros((S, nM, H), bool)\n    PRICE_LOG = np.zeros((S, nM, H))\n", 1)
src = src.replace("        cost_m[s] *= np.exp(price[sup_mat])\n", "        PRICE_LOG[s] = price\n        cost_m[s] *= np.exp(price[sup_mat])\n", 1)
src = src.replace("    w = np.full(S, 1.0 / S)\n", "    global _LAST_PRICE\n    _LAST_PRICE = PRICE_LOG\n    w = np.full(S, 1.0 / S)\n", 1)
exec(compile(src, "make_scenarios_p", "exec"), m3.__dict__)
DATA = HERE + "/../../Data_new"
W = m3.World(m3.load_data(DATA))
out = pd.read_csv(DATA + "/outcomes.csv")
ORIGIN, START, END = 221, 222, 234; T = 13
dmat = out.pivot(index="material", columns="week", values="disrupted").loc[W.materials]
base = dmat.loc[:, :221].mean(axis=1).values
bench = out.pivot(index="material", columns="week", values="benchmark_price").loc[W.materials]
b221 = bench[221].values
lb = np.log(bench.values); calm = (dmat.values[:, 1:] == 0) & (dmat.values[:, :-1] == 0)
rw_sd = float(np.nanstd(np.diff(lb, axis=1)[calm]))
files = dict(a.split("=", 1) for a in sys.argv[1:])
subs = {}
for n, f in files.items():
    s = pd.read_csv(f); p = s.id.str.extract(r"^w(\d+)_(.+)$"); s["week"], s["material"] = p[0].astype(int), p[1]
    s = s[s.week.between(START, END)]
    subs[n] = s
def mat_arr(s, col):
    return s.pivot(index="material", columns="week", values=col).loc[W.materials].values
WORLDS = {
    "status quo": (),
    "Merrow war": (("lane:MERROW", 1.0, 1, 3, 0),),
    "big war": (("lane:MERROW", 1.0, 1, 3, 6), ("cty:KES", 1.0, 2, 5, 8)),
}
res = {}
for wi, (wname, ovr) in enumerate(WORLDS.items()):
    cfg = m3.Config(seed=500 + 37 * wi, n_scen=300, m1_config=m1.Config(source_overrides=ovr))
    info = m3.OriginInfo(W, ORIGIN, T, cfg.m1_config)
    cap_m, lead_a, cost_m, w, dis, noise = m3.make_scenarios(W, info, cfg)
    price = m3._LAST_PRICE
    S = len(w); masks = m3._window_masks(info, dis, 0, T); calmc = cap_m >= 1 - 1e-9
    rng = np.random.default_rng(900 + wi)
    act = b221[None, :, None] * np.exp(price + np.cumsum(rng.normal(0, rw_sd, (S, len(W.materials), T)), axis=2))
    y = dis.astype(float)
    bs_den = ((base[None, :, None] - y) ** 2).mean(axis=(1, 2))
    mape_c = (np.abs(b221[None, :, None] - act) / act).mean(axis=(1, 2))
    naive_c = np.zeros(S); hind_c = np.zeros(S); naive_u = np.zeros(S); shock_d = np.zeros(S)
    cost = {n: np.zeros(S) for n in subs}; unmet = {n: np.zeros(S) for n in subs}; divs = {n: [] for n in subs}
    for mi, m in enumerate(W.materials):
        units = W.mat_units[mi]; d = W.demand[mi, START - 1:END]; I0 = m3.start_stock(d, os.environ.get("STOCK", "sum5"))
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
    res[wname] = {}
    for n, sb in subs.items():
        q = mat_arr(sb, "disruption_prob"); f = mat_arr(sb, "price_forecast")
        bs = 1 - ((q[None] - y) ** 2).mean(axis=(1, 2)) / np.maximum(bs_den, 1e-12)
        ps = 1 - (np.abs(f[None] - act) / act).mean(axis=(1, 2)) / mape_c
        Pp = np.clip(0.6 * bs + 0.4 * ps, 0, 1)
        F = np.clip((naive_c - cost[n]) / np.maximum(gap, 1e-9), 0, 1)
        fo = np.where(shock_d > 0, 1 - unmet[n] / np.maximum(shock_d, 1e-9), 1.0)
        fs = np.where(short, np.clip((fo - fn) / np.maximum(1 - fn, 1e-9), 0, 1), 0.0)
        dv = float(np.mean(np.concatenate(divs[n])))
        res[wname][n] = dict(total=30 * Pp + 40 * F + 21 * fs + 9 * dv, P=Pp, brier=bs, price=ps, F=F, fs=fs, div=np.full(S, dv))
pd.set_option("display.width", 250)
for wname, r in res.items():
    print(f"\n== {wname}")
    print(pd.DataFrame({n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in r.items()}).T.round(3).to_string())
names = list(subs); ref = names[0]
print(f"\n== best of two with '{ref}', by world")
print(pd.DataFrame({w_: {n: float(np.mean(np.maximum(r[ref]["total"], r[n]["total"]))) for n in names[1:]} for w_, r in res.items()}).round(2).to_string())
for split in (0.5, 0.8):
  print(f"\n== best of two with main; war odds split {split:.0%} Merrow-only / {1-split:.0%} big war")
  rows = {}
  for pi in (0.0, 0.25, 0.5, 0.75):
    wts = {"status quo": 1 - pi, "Merrow war": pi * split, "big war": pi * (1 - split)}
    rows[f"war {pi:.0%}"] = {n: sum(wts[w_] * float(np.mean(np.maximum(res[w_][ref]["total"], res[w_][n]["total"]))) for w_ in res) for n in names[1:]}
  print(pd.DataFrame(rows).T.round(2).to_string())
