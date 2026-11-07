"""Step 3 harness: score candidate files on total points (30 P + 40 F + 30 R) per future at origin 221.

Futures are arrays: cap [S,U,H] (absolute units), lead [S,U,H], cost [S,U,H] (landed), bench [S,M,H]
(benchmark price), dis [S,M,H]. Scoring is vectorised over futures and matches score_local.score
(checked in selftest()).
"""
import os, sys, copy, inspect, warnings, dataclasses
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linprog

PL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Code Base/pipelining
ROOT = os.path.normpath(os.path.join(PL, os.pardir, os.pardir))
DATA = os.path.join(ROOT, "Data_new")
sys.path.insert(0, PL)
os.environ["MODEL1_DIR"] = os.path.join(PL, "Model 1", "iter4")
import run_pipeline as rp
rp.PATHS["model2"] = os.path.join(PL, "Model 2", "iter4", "model2.py")
rp.PATHS["model3"] = os.path.join(PL, "Model 3", "iter4", "model3.py")
m3 = rp.load_module("model3")
m3h = rp.load_module("model3_hedge")
import score_local

ORIGIN, S0, E0 = 221, 222, 234
H = E0 - ORIGIN
W = m3.World(m3.load_data(DATA))
INFO = m3.OriginInfo(W, ORIGIN, H)
U, M = len(W.sup), len(W.materials)
SUP_MAT = W.sup_mat
BENCH0 = np.array([np.nanmean(W.spot[u, ORIGIN - 1]) for u in W.mat_units])
BASE_RATE = np.array([np.nanmean(W.disrupted[mi, :ORIGIN]) for mi in range(M)])
DEMAND = W.demand[:, S0 - 1:E0]
PRICE_SD = 0.0265

# make_scenarios with the per-material log price shift exposed (exec'd copy of Model 3 iter4's code)
_src = inspect.getsource(m3.make_scenarios)
_src = _src.replace("def make_scenarios(", "def make_scenarios_px(")
_src = _src.replace("    dis = np.zeros((S, nM, H), bool)\n", "    dis = np.zeros((S, nM, H), bool)\n    PX = np.zeros((S, nM, H))\n", 1)
_src = _src.replace("        cost_m[s] *= np.exp(price[sup_mat])\n", "        PX[s] = price\n        cost_m[s] *= np.exp(price[sup_mat])\n", 1)
_src = _src.replace("    return cap_m, lead_a, cost_m, w, dis, noise", "    return cap_m, lead_a, cost_m, w, dis, noise, PX")
assert "PX[s] = price" in _src
exec(compile(_src, "make_scenarios_px", "exec"), m3.__dict__)
make_scenarios_px = m3.make_scenarios_px


def info_with(boost=None):
    """OriginInfo copy with onset distributions scaled: boost = {("lane"|"cty"|"sup", index): factor}."""
    info = copy.copy(INFO)
    info.src = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in INFO.src.items()}
    for (kind, i), f in (boost or {}).items():
        info.src[f"q_{kind}"][i] = info.src[f"q_{kind}"][i] * f
    return info


def gen(n, seed, cfg=None, info=None, price_noise=True):
    """Generator futures -> dict of truth arrays (plus the raw planning tuple)."""
    cfg = dataclasses.replace(cfg or m3.MODERATE, n_scen=n, seed=seed, use_stress=False)
    info = info or INFO
    cap_m, lead_a, cost_m, w, dis, noise, PX = make_scenarios_px(W, info, cfg)
    calm = cap_m >= 1.0 - 1e-9
    cap = W.normal_cap[None, :, None] * np.where(calm, noise, cap_m)
    rng = np.random.default_rng(seed + 99)
    rw = np.cumsum(rng.normal(0, PRICE_SD, (n, M, H)), axis=2) if price_noise else np.zeros((n, M, H))
    cost = INFO.cost_base[None] * cost_m * np.exp(rw[:, SUP_MAT])
    return dict(cap=cap, lead=W.base_lead[None, :, None] + lead_a, cost=cost,
                bench=BENCH0[None, :, None] * np.exp(PX + rw), dis=dis, w=w,
                raw=(cap_m, lead_a, cost_m, w, dis, noise))


# ----------------------------------------------------------------------------- stress scenarios (57's set)
def _U(pred):
    return [u for u in range(U) if pred(W.sup.loc[u])]


def stress():
    key = {(m, c): u for u, (m, c) in enumerate(zip(W.sup.material, W.sup.country))}
    mat_units = lambda m: list(W.mat_units[W.materials.index(m)])
    mer = _U(lambda r: r.lane == "MERROW")
    kes = _U(lambda r: r.country == "KES")
    kes_mats = sorted(set(W.sup.material[W.sup.country == "KES"]))
    kes_mat_units = [u for m in kes_mats for u in mat_units(m)]
    tar = [W.materials.index(m) for m in ("aluminum", "semiconductors")]

    def ev(units, a, b, cap=None, lead_add=0, cost_mult=1.0, spot_units=None, spot_mult=1.0):
        return dict(units=units, a=a, b=b, cap=cap, lead_add=lead_add, cost_mult=cost_mult,
                    spot_units=spot_units or [], spot_mult=spot_mult)
    merrow = lambda a=223, b=230: ev(mer, a, b, cap=0.78, lead_add=3, cost_mult=1.06)
    val_co = ev([key[("cobalt", "VAL")]], 223, 233, cap=0.30, spot_units=mat_units("cobalt"), spot_mult=1.25)
    kes_re = ev([key[("rare_earths", "KES")]], 223, 230, cap=0.30, spot_units=mat_units("rare_earths"), spot_mult=1.25)
    kes_cty = lambda a=226: ev(kes, a, 234, cap=0.20, spot_units=kes_mat_units, spot_mult=1.10)
    scen = {
        "calm (tariff only)": [],
        "Merrow closure 223-230": [merrow()],
        "VAL cobalt ban 223-233": [val_co],
        "KES rare_earths 223-230": [kes_re],
        "cobalt + rare_earths": [val_co, kes_re],
        "cobalt + RE + Merrow": [val_co, kes_re, merrow()],
        "KES country 226-234": [kes_cty()],
        "cobalt + RE + KES country 226+": [val_co, kes_re, kes_cty()],
        "KES country 223-234": [kes_cty(223)],
        "KES country 223+ + Merrow": [kes_cty(223), merrow()],
        "Merrow 40% cap 223-234": [ev(mer, 223, 234, cap=0.40, lead_add=4, cost_mult=1.15)],
        "Atlantic closure 225-230": [ev(_U(lambda r: r.lane == "ATLANTIC"), 225, 230, cap=0.78, lead_add=3, cost_mult=1.06)],
        "South closure 225-230": [ev(_U(lambda r: r.lane == "SOUTH"), 225, 230, cap=0.78, lead_add=3, cost_mult=1.06)],
    }
    n = len(scen)
    cap = np.repeat((0.984 * W.normal_cap)[None, :, None], n, 0).repeat(H, 2)
    lead = np.repeat(W.base_lead[None, :, None], n, 0).repeat(H, 2).astype(float)
    cost = np.repeat(INFO.cost_base[None], n, 0).copy()
    PX = np.zeros((n, M, H))
    dis = np.zeros((n, M, H), bool)
    dis[:, tar, 224 - S0:] = True
    for s, evs in enumerate(scen.values()):
        for e in evs:
            sl = slice(e["a"] - S0, e["b"] - S0 + 1)
            for u in e["units"]:
                if e["cap"] is not None:
                    cap[s, u, sl] = e["cap"] * W.normal_cap[u]
                lead[s, u, sl] += e["lead_add"]
                cost[s, u, sl] *= e["cost_mult"]
                dis[s, SUP_MAT[u], sl] = True
            for u in e["spot_units"]:
                cost[s, u, sl] *= e["spot_mult"]
            for mi in set(SUP_MAT[e["spot_units"]]):
                PX[s, mi, sl] = np.maximum(PX[s, mi, sl], np.log(e["spot_mult"]))
    return list(scen), dict(cap=cap, lead=lead, cost=cost, bench=BENCH0[None, :, None] * np.exp(PX),
                            dis=dis, w=np.full(n, 1.0 / n))


# ----------------------------------------------------------------------------- scorer
def hindsight_v(d, I0, cost, cap, lead, avg):
    """m3.hindsight_batch with per-future average landed cost (hold/penalty/salvage scale with it)."""
    S, C, T = cost.shape
    hold, pen, salv = m3.HOLD * avg, m3.PENALTY * avg, m3.SALVAGE * avg
    nF = C * T
    per = nF + 2 * T
    N = S * per
    base = np.arange(S) * per
    f_ix = base[:, None, None] + np.arange(C)[None, :, None] * T + np.arange(T)[None, None, :]
    i_ix = base[:, None] + nF + np.arange(T)[None, :]
    u_ix = i_ix + T
    arr_t = np.arange(T)[None, None, :] + lead.astype(int)
    obj = np.zeros(N)
    obj[f_ix] = cost - np.where(arr_t >= T, salv[:, None, None], 0.0)
    obj[i_ix] = hold[:, None]
    obj[i_ix[:, -1]] -= salv
    obj[u_ix] = pen[:, None]
    ub = np.full(N, np.inf)
    ub[f_ix] = np.maximum(cap, 0.0)
    er = [np.arange(S * T), np.arange(S * T)]
    ec = [i_ix.ravel(), u_ix.ravel()]
    ev = [np.ones(S * T), -np.ones(S * T)]
    rr = (np.arange(S)[:, None] * T + np.arange(1, T)[None, :]).ravel()
    er.append(rr); ec.append(i_ix[:, :-1].ravel()); ev.append(-np.ones(len(rr)))
    ok = arr_t < T
    s_i = np.broadcast_to(np.arange(S)[:, None, None], (S, C, T))[ok]
    er.append(s_i * T + arr_t[ok]); ec.append(f_ix[ok]); ev.append(-np.ones(ok.sum()))
    b_eq = np.tile(-np.asarray(d, float), S)
    b_eq[np.arange(S) * T] += I0
    A_eq = sparse.csr_matrix((np.concatenate(ev), (np.concatenate(er), np.concatenate(ec))), shape=(S * T, N))
    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=np.column_stack([np.zeros(N), ub]), method="highs")
    assert res.status == 0, res.message
    return res.x[f_ix]


def sim(orders, cap, lead, cost, d, I0, avg):
    """m3.simulate_batch with per-future avg cost. Returns (total [S], unmet [S, T])."""
    S, C, T = cap.shape
    f = np.minimum(np.maximum(orders, 0.0)[None] if orders.ndim == 2 else np.maximum(orders, 0.0), np.maximum(cap, 0.0))
    purchase = (f * cost).sum(axis=(1, 2))
    a = np.minimum(np.arange(T)[None, None, :] + lead.astype(int), T)
    arrive = np.zeros((S, T + 1))
    np.add.at(arrive, (np.broadcast_to(np.arange(S)[:, None, None], (S, C, T)).ravel(), a.ravel()), f.ravel())
    stock = np.full(S, float(I0)); hold = np.zeros(S); unmet = np.zeros((S, T))
    for t in range(T):
        stock = stock + arrive[:, t]
        served = np.minimum(stock, d[t])
        unmet[:, t] = d[t] - served
        stock = stock - served
        hold += stock
    return purchase + avg * (m3.HOLD * hold - m3.SALVAGE * (stock + arrive[:, T])) + avg * m3.PENALTY * unmet.sum(1), unmet


class Truth:
    """Futures plus everything candidate-independent (naive, hindsight, shock masks), per stock rule."""

    def __init__(self, F, mode):
        self.F, self.mode = F, mode
        S = F["cap"].shape[0]
        self.S = S
        hist = np.broadcast_to(INFO.dis_hist[None], (S, M, INFO.dis_hist.shape[1]))
        full = np.concatenate([hist, F["dis"]], axis=2)
        off = hist.shape[2]
        self.mask = np.stack([full[:, :, max(0, off + t - m3.RECOVERY): off + t + 1].any(2) for t in range(H)], axis=2)
        self.cn = np.zeros((S, M)); self.ch = np.zeros((S, M))
        self.un = np.zeros((S, M)); self.dm = np.zeros((S, M))
        self.per = []
        for mi in range(M):
            units = W.mat_units[mi]
            d = DEMAND[mi]
            cap, lead, cost = F["cap"][:, units], F["lead"][:, units], F["cost"][:, units]
            avg = cost.mean(axis=(1, 2))
            I0 = m3.start_stock(d, mode)
            k = int(np.where(units == W.naive_unit[mi])[0][0])
            o_n = np.zeros((len(units), H)); o_n[k] = d
            self.cn[:, mi], u_n = sim(o_n, cap, lead, cost, d, I0, avg)
            fh = hindsight_v(d, I0, cost, cap, lead, avg)
            self.ch[:, mi], _ = sim(fh, cap, lead, cost, d, I0, avg)
            self.un[:, mi] = (u_n * self.mask[:, mi]).sum(1)
            self.dm[:, mi] = (d[None] * self.mask[:, mi]).sum(1)
            self.per.append((units, d, cap, lead, cost, avg, I0))
        self.brier_base = ((BASE_RATE[None, :, None] - F["dis"]) ** 2).mean(axis=(1, 2))
        self.carry_err = (np.abs(BENCH0[None, :, None] - F["bench"]) / F["bench"]).mean(axis=(1, 2))

    def score(self, sub):
        """sub: DataFrame with id, disruption_prob, price_forecast, order_units, share_* (any weeks).
        Returns DataFrame [S] of total and parts."""
        F, S = self.F, self.S
        x = sub.id.str.extract(r"^w(\d+)_(.+)$")
        df = sub.assign(week=x[0].astype(int), material=x[1])
        df = df[df.week.between(S0, E0)].set_index(["material", "week"])
        p = np.array([[df.loc[(m, w), "disruption_prob"] for w in range(S0, E0 + 1)] for m in W.materials])
        pf = np.array([[df.loc[(m, w), "price_forecast"] for w in range(S0, E0 + 1)] for m in W.materials])
        bs = 1 - ((p[None] - F["dis"]) ** 2).mean(axis=(1, 2)) / self.brier_base
        ps = 1 - (np.abs(pf[None] - F["bench"]) / F["bench"]).mean(axis=(1, 2)) / self.carry_err
        P = np.clip(0.6 * bs + 0.4 * ps, 0, 1)
        co = np.zeros((S, M)); uo = np.zeros((S, M)); divs = []
        for mi, (units, d, cap, lead, cost, avg, I0) in enumerate(self.per):
            g = df.loc[W.materials[mi]].reindex(range(S0, E0 + 1))
            sh = np.stack([g[f"share_{W.sup.country.values[u]}"].values for u in units])
            o = g.order_units.values[None] * sh
            co[:, mi], u_o = sim(o, cap, lead, cost, d, I0, avg)
            uo[:, mi] = (u_o * self.mask[:, mi]).sum(1)
            divs.append(m3.diversification(sh.T))
        gap = self.cn.sum(1) - self.ch.sum(1)
        Fr = np.where(gap > 0, (self.cn.sum(1) - co.sum(1)) / np.where(gap > 0, gap, 1), np.nan)
        Fc = np.where(np.isfinite(Fr), np.clip(Fr, 0, 1), 0.0)
        dm = self.dm.sum(1)
        fo = np.where(dm > 0, 1 - uo.sum(1) / np.maximum(dm, 1e-12), np.nan)
        fn = np.where(dm > 0, 1 - self.un.sum(1) / np.maximum(dm, 1e-12), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            fs = np.where(np.isfinite(fn) & (fn < 1 - 1e-12), (fo - fn) / (1 - fn), np.nan)
        fs0 = np.where(np.isfinite(fs), np.clip(fs, 0, 1), 0.0)
        dv = float(np.mean(np.concatenate(divs)))
        R = np.clip(0.7 * fs0 + 0.3 * dv, 0, 1)
        tot = 100 * (0.3 * P + 0.4 * Fc + 0.3 * R)
        return pd.DataFrame(dict(total=tot, P=P, brier=bs, price=ps, F=Fc, fill=fs, div=dv, R=R, orders70=tot - 30 * P))


def world_of(F, s):
    """A World whose weeks 222-234 hold future s (for checking against score_local.score)."""
    w = copy.copy(W)
    w.cap, w.lead, w.landed, w.disrupted, w.spot = W.cap.copy(), W.lead.copy(), W.landed.copy(), W.disrupted.copy(), W.spot.copy()
    sl = slice(S0 - 1, E0)
    w.cap[:, sl], w.lead[:, sl], w.landed[:, sl] = F["cap"][s], F["lead"][s], F["cost"][s]
    w.disrupted[:, sl] = F["dis"][s]
    for mi, units in enumerate(W.mat_units):
        w.spot[units, sl] = F["bench"][s, mi][None]
    w.last_week = E0
    return w


def read_sub(path):
    return pd.read_csv(path)


def selftest():
    names, F = stress()
    sub = read_sub(os.path.join(ROOT, "submissions", "submission_moderate_p2v2.csv"))
    Fg = gen(4, 5)
    for FF, idx in ((F, [0, 1, 6, 11]), (Fg, [0, 1, 2, 3])):
        for mode in ("sum5", "5avg"):
            T = Truth({k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(FF["w"]) else v)
                       for k, v in FF.items() if k != "raw"}, mode)
            mine = T.score(sub).total.values
            ref = [score_local.score(world_of(T.F, j), sub, S0, E0, mode)["score"] for j in range(len(idx))]
            print(mode, np.round(mine, 3), np.round(ref, 3))
            assert np.allclose(mine, ref, atol=1e-3)
    print("selftest OK")


if __name__ == "__main__":
    selftest()
