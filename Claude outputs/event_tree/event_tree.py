"""
event_tree.py - explicit event-tree model for the private window (weeks 222-234, planned from week 221).

Calm baseline (capacity ~0.984 x normal, base lead, carry-forward prices, landed-cost formula) plus a list of
named events read from the week-221 news and market data, each with its own probability, onset-week
distribution, length distribution and effect sizes, plus a small background hazard for shocks that give no
warning by week 221. The sampled futures replace Model 3 iter5's make_scenarios, so Model 3's score-aware LP,
simulator and scorer are reused unchanged.

Usage:  python event_tree.py PIPELINE_DIR OUT_DIR
"""
import os, sys, copy, inspect, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

PIPE = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PIPE)
import run_pipeline_p2 as rp
os.environ["MODEL1_DIR"] = os.path.dirname(rp.PATHS["model1"])
m1 = rp.load_module("model1"); m3 = rp.load_module("model3")

# price-capturing copy of the original make_scenarios (same hack as scenarios/eval_scenarios.py)
src = inspect.getsource(m3.make_scenarios)
src = src.replace("    dis = np.zeros((S, nM, H), bool)\n", "    dis = np.zeros((S, nM, H), bool)\n    PRICE_LOG = np.zeros((S, nM, H))\n", 1)
src = src.replace("        cost_m[s] *= np.exp(price[sup_mat])\n", "        PRICE_LOG[s] = price\n        cost_m[s] *= np.exp(price[sup_mat])\n", 1)
src = src.replace("    w = np.full(S, 1.0 / S)\n", "    global _LAST_PRICE\n    _LAST_PRICE = PRICE_LOG\n    w = np.full(S, 1.0 / S)\n", 1)
exec(compile(src, "make_scenarios_p", "exec"), m3.__dict__)
ORIG_MAKE = m3.make_scenarios

DATA = os.path.join(PIPE, "..", "..", "Data_new")
W = m3.World(m3.load_data(DATA))
ORIGIN, START, END, H = 221, 222, 234, 13
U, M = len(W.sup), len(W.materials)
SUP = W.sup.reset_index(drop=True)
MAT = W.sup_mat
def unit(mat, cty): return int(np.where((SUP.material == mat) & (SUP.country == cty))[0][0])

# ---------------------------------------------------------------- shipping sensitivity per supplier
ship = W.data["shipping"].pivot(index="lane", columns="week", values="container_rate")
B = np.zeros(U); CR0 = np.zeros(U)
for u in range(U):
    cr = ship.loc[SUP.lane[u], 1:ORIGIN].values
    y = W.landed[u, :ORIGIN] - W.spot[u, :ORIGIN] * (1 + W.tariff[u, :ORIGIN])
    ok = np.isfinite(y) & np.isfinite(cr)
    B[u] = max(np.linalg.lstsq(np.c_[cr[ok], np.ones(ok.sum())], y[ok], rcond=None)[0][0], 0.0)
    CR0[u] = ship.loc[SUP.lane[u], ORIGIN]

# ---------------------------------------------------------------- the event tree (week-221 reading)
SPEC = dict(
    tariff_p=0.97, tariff_len=[10, 12, 18],                      # 3 of 3 announced tariffs took effect; lengths 10/12/18
    bans=[  # export bans: past ones ran 5, 6, 11 weeks at ~5% capacity; benchmark +29%, +29%, +40% while active
        dict(mat="cobalt", cty="VAL", p=0.90, onset={222: .40, 223: .35, 224: .15, 225: .10}),
        dict(mat="rare_earths", cty="KES", p=0.85, onset={222: .10, 223: .35, 224: .30, 225: .15, 226: .10}),
    ],
    ban_len=[5, 6, 8, 11], ban_cap=0.05, ban_jump=(0.22, 0.05),   # log jump ~ +25%
    merrow=dict(p=0.55, onset={222: .10, 223: .35, 224: .30, 225: .15, 226: .10},
                len=(3, 8), tail_p=0.10, tail_len=(9, 13),         # clash closures ran 3 and 8 weeks; 10% longer tail
                cap=0.75, lead=3, lead4_p=0.07, cr_mult=(2.85, 3.2)),
    kes_country=dict(p=0.04, onset=(227, 233), cap=0.2),           # no sanctions chain yet (last one gave 7 weeks' notice)
    background=dict(rate=0.063,                                    # 14 onsets in 221 weeks
                    ramp=(0.2, 0.2),                               # unseen share: 0.2 at h=0, +0.2/week, 1.0 from h=4
                    types=dict(supplier=0.50, lane=0.27, tariff=0.20, country=0.03)),
)
NON_AUR = np.where(SUP.country.values != "AUR")[0]
MERROW_U = np.where(SUP.lane.values == "MERROW")[0]


def gen(spec, S, seed, info):
    """Sample S futures. Returns cap_m, lead_a, cost_m, w, dis, noise (Model 3's format) and log price shifts."""
    rng = np.random.default_rng(seed)
    cap_m = np.ones((S, U, H)); lead_a = np.zeros((S, U, H))
    cost_mul = np.ones((S, U, H)); cost_add = np.zeros((S, U, H))
    dis = np.zeros((S, M, H), bool); price = np.zeros((S, M, H))
    lo = np.maximum(info.cap_mu - 4 * info.cap_sd, 0.8); hi = np.maximum(info.cap_hi, info.cap_mu)
    noise = np.clip(rng.normal(info.cap_mu[None, :, None], info.cap_sd[None, :, None], (S, U, H)),
                    lo[None, :, None], hi[None, :, None])
    tar_units = np.where(info.ann_start >= 0)[0]

    def act(k, L):
        a = np.zeros(H, bool)
        if 0 <= k < H and L > 0:
            a[k:min(k + L, H)] = True
        return a

    def onset(dist):
        ks = np.array(list(dist)); ps = np.array(list(dist.values()), float)
        return int(rng.choice(ks, p=ps / ps.sum())) - START

    def lane_event(s, units, a, cap, extra, crm):
        for u in units:
            lead_a[s, u, a] = np.maximum(lead_a[s, u, a], extra)
            cap_m[s, u, a] *= cap
            cost_add[s, u, a] = np.maximum(cost_add[s, u, a], B[u] * CR0[u] * (crm - 1))
            dis[s, MAT[u], a] = True

    bg = spec["background"]; types = list(bg["types"]); tp = np.array(list(bg["types"].values()))
    for s in range(S):
        # announced tariff (KES aluminum, KES semiconductors -> 35% from week 224)
        for u in tar_units:
            k0 = int(info.ann_start[u]); base = SUP.base_tariff[u]; new = info.tar[u, -1]
            if rng.random() < spec["tariff_p"]:
                L = int(rng.choice(spec["tariff_len"]))
                dis[s, MAT[u]] |= act(k0, L)
                cost_mul[s, u, np.arange(H) >= k0 + L] *= (1 + base) / (1 + new)
            else:
                cost_mul[s, u, k0:] *= (1 + base) / (1 + new)
        # export bans
        for b in spec["bans"]:
            if rng.random() < b["p"]:
                u = unit(b["mat"], b["cty"])
                a = act(onset(b["onset"]), int(rng.choice(spec["ban_len"])) +
                        (int(rng.integers(-1, 2)) if spec.get("ban_jitter", True) else 0))
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], spec["ban_cap"])
                price[s, MAT[u], a] = np.maximum(price[s, MAT[u], a], rng.normal(*spec["ban_jump"]))
                dis[s, MAT[u]] |= a
        # Merrow Strait closure (hits every Kestria and Sunda Coast supplier)
        mr = spec["merrow"]
        if rng.random() < mr["p"]:
            L = int(rng.integers(mr["tail_len"][0], mr["tail_len"][1] + 1)) if rng.random() < mr["tail_p"] \
                else int(rng.integers(mr["len"][0], mr["len"][1] + 1))
            a = act(onset(mr["onset"]), L)
            lane_event(s, MERROW_U, a, mr["cap"], mr["lead"] + (rng.random() < mr["lead4_p"]), rng.uniform(*mr["cr_mult"]))
        # Kestria-wide event (sanctions-style)
        kc = spec["kes_country"]
        if rng.random() < kc["p"]:
            a = act(int(rng.integers(kc["onset"][0], kc["onset"][1] + 1)) - START, H)
            for u in np.where(SUP.country.values == "KES")[0]:
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], kc["cap"])
                price[s, MAT[u], a] = np.maximum(price[s, MAT[u], a], info.j_cty)
                dis[s, MAT[u]] |= a
        # background: shocks with no warning visible at week 221
        for h in range(H):
            lam = bg["rate"] * min(1.0, bg["ramp"][0] + bg["ramp"][1] * h)
            if rng.random() >= lam:
                continue
            t = types[int(rng.choice(len(types), p=tp / tp.sum()))]
            if t == "supplier":
                u = int(rng.choice(NON_AUR)); a = act(h, int(rng.choice([2, 3, 5, 6, 6, 9, 9, 11])))
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], rng.choice(info.sup_caps))
                price[s, MAT[u], a] = np.maximum(price[s, MAT[u], a], rng.choice(info.j_single))
                dis[s, MAT[u]] |= a
            elif t == "lane":
                ln = rng.choice(["MERROW", "ATLANTIC", "SOUTH", "NORTH"], p=[.5, .2, .2, .1])
                lane_event(s, np.where(SUP.lane.values == ln)[0], act(h, int(rng.integers(2, 9))), 0.78,
                           3 + (rng.random() < 0.07), rng.uniform(1.6, 3.0))
            elif t == "tariff" and h >= 2:
                u = int(rng.choice(NON_AUR)); a = act(h, int(rng.choice([10, 12, 18])))
                cost_mul[s, u, a] *= 1.2; dis[s, MAT[u]] |= a
            elif t == "country":
                c = rng.choice(["KES", "VAL", "SUN", "ORI", "HAL"]); a = act(h, H)
                for u in np.where(SUP.country.values == c)[0]:
                    cap_m[s, u, a] = np.minimum(cap_m[s, u, a], 0.2)
                    price[s, MAT[u], a] = np.maximum(price[s, MAT[u], a], info.j_cty)
                    dis[s, MAT[u]] |= a
    cb = info.cost_base[None]
    cost_m = (cb * cost_mul + cost_add) / cb * np.exp(price[:, MAT, :])
    return cap_m, lead_a, cost_m, np.full(S, 1.0 / S), dis, noise, price


def variant(**kw):
    sp = copy.deepcopy(SPEC)
    for k, v in kw.items():
        if k == "merrow_p": sp["merrow"]["p"] = v
        elif k == "merrow_len": sp["merrow"]["len"] = v; sp["merrow"]["tail_p"] = 0.0
        elif k == "merrow_onset": sp["merrow"]["onset"] = v
        elif k == "bans_p": [b.__setitem__("p", v) for b in sp["bans"]]
        elif k == "kes_p": sp["kes_country"]["p"] = v
        else: sp[k] = v
    return sp


# ---------------------------------------------------------------- building files
OUT_RNG_SD = None
out_hist = W.data["outcomes"]
BENCH = out_hist.pivot(index="material", columns="week", values="benchmark_price").loc[W.materials]
B221 = BENCH[ORIGIN].values
DMAT = out_hist.pivot(index="material", columns="week", values="disrupted").loc[W.materials]
_lb = np.log(BENCH.values); _calm = (DMAT.values[:, 1:] == 0) & (DMAT.values[:, :-1] == 0)
RW_SD = float(np.nanstd(np.diff(_lb, axis=1)[_calm]))
TREE_PRICE_MATS = ("cobalt", "rare_earths")


def build_file(name, spec, main, info, n_big=20000, n_lp=300, seed=11):
    t0 = time.time()
    *_, dis, _, price = gen(spec, n_big, seed, info)
    p = dis.mean(0)                                                       # [M, H] P(material disrupted)
    # price: 1/a-weighted median of a = b221 * exp(jump + calm random walk), only where the tree has a named shock
    rng = np.random.default_rng(seed + 1)
    a = B221[None, :, None] * np.exp(price + np.cumsum(rng.normal(0, RW_SD, price.shape), axis=2))
    sub = main.copy()
    wk = sub.id.str.extract(r"^w(\d+)_")[0].astype(int); mat = sub.id.str.split("_", n=1).str[1]
    for mi, m in enumerate(W.materials):
        for h in range(H):
            i = sub.index[(wk == START + h) & (mat == m)][0]
            sub.at[i, "disruption_prob"] = float(np.clip(p[mi, h], 0.005, 0.995))
            if m in TREE_PRICE_MATS:
                x = a[:, mi, h]; o = np.argsort(x); cw = np.cumsum(1 / x[o]); cw /= cw[-1]
                sub.at[i, "price_forecast"] = float(x[o][np.searchsorted(cw, 0.5)])
    # orders: Model 3 iter5's LP planned on event-tree futures
    m3.make_scenarios = lambda W_, info_, cfg_: gen(spec, cfg_.n_scen, cfg_.seed + 7, info_)[:6]
    try:
        cfg = type(m3.MODERATE)(**{**m3.MODERATE.__dict__, "n_scen": n_lp})
        plan = m3.plan_window(W, ORIGIN, START, END, cfg, info=info).set_index(["week", "material"])
    finally:
        m3.make_scenarios = ORIG_MAKE
    for i in sub.index[wk.between(START, END)]:
        r = plan.loc[(int(wk[i]), mat[i])]
        sub.at[i, "order_units"] = int(r.order_units)
        for c in m3.SHARE_COLS:
            sub.at[i, c] = float(r[c])
    sub["order_units"] = sub.order_units.astype(int)
    sh = sub[list(m3.SHARE_COLS)].clip(lower=0).round(6); sub[list(m3.SHARE_COLS)] = sh.div(sh.sum(1), axis=0)
    m3.validate(sub, W.sup)
    path = os.path.join(OUT, f"submission_{name}.csv"); sub.to_csv(path, index=False)
    print(f"  {name}: written in {time.time() - t0:.0f}s -> {path}", flush=True)
    return sub


if __name__ == "__main__":
    info = m3.OriginInfo(W, ORIGIN, H)
    print("shipping b x container rate at week 221 (Merrow suppliers):",
          {f"{SUP.material[u]}/{SUP.country[u]}": round(B[u] * CR0[u] / info.cost_base[u, 0], 3) for u in MERROW_U})
    main = pd.read_csv(os.path.join(PIPE, "..", "..", "submissions", "submission_main_it8.csv"))
    build_file("tree_cal", SPEC, main, info)                    # calibrated: Merrow 0.55
    build_file("tree_war", variant(merrow_p=1.0), main, info)   # sharp on whether Merrow closes, soft on when
    build_file("tree_nowar", variant(merrow_p=0.0), main, info) # sharp the other way
