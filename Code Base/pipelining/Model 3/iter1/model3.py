"""
model3.py - Model 3: ordering plan (order_units + share_KES..share_AUR).

Pieces:
  simulate()      the competition simulator, written from the rules: each window starts with 5 weeks
                  of demand in stock and nothing in transit; a week's order is split by shares; each
                  slice fills up to that supplier's capacity that week, is paid at that week's landed
                  cost and arrives after that week's lead time; arrivals go into stock, then demand is
                  served; holding costs 2% of the material's average landed cost per unit-week; unmet
                  demand costs 5x; stock + in-transit at the end sells back at 70%.
  score_window()  F (cost vs naive and hindsight), fill-rate skill in shock weeks (disrupted + 4
                  recovery weeks) vs naive, diversification (1 - HHI rescaled), on a past window.
  Scenarios       sampled futures from Model 1's source hazards (supplier, country, lane, tariff) and
                  duration curves, with shock effects measured on weeks 1-208:
                    single supplier  capacity -> its observed ratio (0.05-0.61), material spot +15-31%
                    country event    capacity 0.2 on every supplier of that country, spot ~+8%
                    lane closure     lead time +3 weeks, capacity x0.78
                    tariff           landed cost up on that supplier only
                  plus stress futures (weight omega) bigger than anything in training.
  solve_lp()      two-stage stochastic LP per material: one set of orders (stage 1) that minimizes
                  the weighted cost across all futures (stage 2: fills, stock, shortfall).
  plan_window()   the full plan for one window; fill_submission() writes the CSV columns.
  backtest()      rolling past windows, scored against the naive policy and hindsight.

Backtest (34 past windows, each planned from data up to its origin; points = 40 F + 21 fill skill + 9 div):
  no scenarios, cheapest supplier only   24.7   (36 in windows where naive never runs out, 10 where it does)
  moderate (default)                     39.7
  aggressive (stress futures 0.15)       36.7   (17 where naive never runs out, 61 where it does)

Unknowns in the official scorer (kept as Config options, settle them on the public leaderboard):
  start stock = sum of the window's first 5 weeks of demand (default) or 5 x mean demand;
  average landed cost = mean over the window's weeks and the material's 3 suppliers;
  diversification averaged over rows (also reported weighted by order units).

Requires numpy, pandas, scipy; Model 1 (iter3/model1.py) for the hazards.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

HOLD = 0.02
PENALTY = 5.0
SALVAGE = 0.7
START_WEEKS = 5
COUNTRIES = ("KES", "VAL", "SUN", "ORI", "HAL", "AUR")
SHARE_COLS = tuple(f"share_{c}" for c in COUNTRIES)
FILES = ("outcomes", "suppliers", "supply_history", "tariffs", "prices", "shipping", "macro", "demand")


@dataclass
class Config:
    """Defaults = the "moderate" plan picked on the backtest (34 windows, origins 60-192):
    s_min 0.1, lam 5, hazards x1.5, no stress futures -> about 39.7 of the 70 F+R points.
    AGGRESSIVE adds stress futures (omega 0.15): ~61 points in windows where naive runs out
    of stock in shock weeks, but ~17 in windows where it doesn't (moderate: ~57 and ~26)."""
    n_scen: int = 200            # sampled futures per window
    lam: float = 5.0             # internal stockout penalty, x average landed cost (the real one is 5)
    s_min: float = 0.1           # minimum share of each week's order per supplier
    s_max: float = 1.0           # maximum share per supplier (1 = no cap); a cap diversifies via the
                                 # next-cheapest supplier instead of forcing the most expensive one
    omega: float = 0.15          # total weight of the stress futures (only if use_stress)
    prob_mult: float = 1.5       # scales Model 1's onset hazards
    use_stress: bool = False
    seed: int = 0
    start_stock: str = "sum5"    # "sum5" or "5avg"
    lane_cap: float = 0.78       # capacity multiplier during a lane closure
    lane_lead: int = 3           # extra lead weeks during a lane closure
    country_cap: float = 0.2     # capacity multiplier during a country event


MODERATE = Config()
AGGRESSIVE = Config(use_stress=True, omega=0.15)


# ----------------------------------------------------------------------------- loading
def _find_file(data_dir, name):
    exact = os.path.join(data_dir, f"{name}.csv")
    if os.path.exists(exact):
        return exact
    cands = sorted(f for f in os.listdir(data_dir) if f.endswith(f"{name}.csv"))
    return os.path.join(data_dir, cands[0]) if cands else None


def load_data(data_dir):
    data = {}
    for name in FILES + ("news",):
        p = _find_file(data_dir, name)
        if p is None:
            if name == "news":            # optional: only Model 1 iter4's news layer reads it
                continue
            raise FileNotFoundError(f"{name}.csv not found in {data_dir}")
        data[name] = pd.read_csv(p)
    return data


def import_model1(model1_dir=None):
    if "model1" in sys.modules:
        return sys.modules["model1"]
    here = os.path.dirname(os.path.abspath(__file__))
    model1_dir = model1_dir or os.environ.get("MODEL1_DIR")
    cands = ([model1_dir] if model1_dir else []) + [
        os.path.normpath(os.path.join(here, os.pardir, os.pardir, "Model 1", "iter3")),
        "/Workspace/Repos/DSDG Hackathon/DSDG Hackathon/Code Base/pipelining/Model 1/iter3"]
    for c in cands:
        if c and os.path.isdir(c):
            sys.path.insert(0, c)
            try:
                return importlib.import_module("model1")
            except Exception:
                sys.path.remove(c)
    raise ImportError("model1.py (Model 1 iter3) not found; pass model1_dir")


class World:
    """Dense arrays. Supplier rows follow suppliers.csv; weeks are 1-based (column = week - 1)."""

    def __init__(self, data, model1_dir=None):
        self.data = data
        self.m1 = import_model1(model1_dir)
        self.P = self.m1.Panels(data)                       # Model 1 panels (hazards, states)
        sup = data["suppliers"].reset_index(drop=True)
        self.sup = sup
        self.materials = list(dict.fromkeys(sup.material))
        self.mat_units = [np.where(sup.material.values == m)[0] for m in self.materials]
        self.last_week = int(data["supply_history"].week.max())
        dm = data["demand"]
        self.n_weeks = int(dm.week.max())
        keys = pd.MultiIndex.from_tuples(list(zip(sup.material, sup.country)))
        wk = np.arange(1, self.n_weeks + 1)

        def arr(df, col):
            p = df.pivot_table(index=["material", "country"], columns="week", values=col, aggfunc="first")
            return p.reindex(index=keys, columns=wk).values.astype(float)

        sh = data["supply_history"]
        self.cap = arr(sh, "capacity_units")
        self.lead = arr(sh, "lead_time_weeks")
        self.landed = arr(sh, "landed_cost")
        self.spot = arr(data["prices"], "spot_price")
        self.tariff = arr(data["tariffs"], "tariff_rate")
        self.ann_rate = arr(data["tariffs"], "announced_rate")
        self.ann_eff = arr(data["tariffs"], "announced_effective_week")
        self.normal_cap = sup.normal_capacity_units.values.astype(float)
        self.base_lead = sup.base_lead_weeks.values.astype(int)
        self.demand = dm.pivot_table(index="material", columns="week", values="demand_units",
                                     aggfunc="first").reindex(index=self.materials, columns=wk).values.astype(float)
        o = data["outcomes"]
        self.disrupted = o.pivot_table(index="material", columns="week", values="disrupted",
                                       aggfunc="first").reindex(index=self.materials, columns=wk).values.astype(float)
        self.naive_unit = np.array([u[np.argmax(self.normal_cap[u])] for u in self.mat_units])

    def c(self, week):
        return int(week) - 1


# ----------------------------------------------------------------------------- simulator
def start_stock(demand_window, mode="sum5"):
    if mode == "5avg":
        return START_WEEKS * float(np.mean(demand_window))
    return float(np.sum(demand_window[:START_WEEKS]))


def simulate(orders, cap, lead, cost, demand, avg_cost, mode="sum5"):
    """One material, one window. orders/cap/lead/cost: [3, T] (per supplier, order week).
    Returns dict of cost parts and the served/demand series."""
    T = demand.shape[0]
    filled = np.minimum(np.maximum(orders, 0.0), np.maximum(cap, 0.0))
    purchase = float(np.sum(filled * cost))
    arrive = np.zeros(T + 1)
    transit = 0.0
    for c in range(filled.shape[0]):
        for t in range(T):
            a = t + int(lead[c, t])
            if a < T:
                arrive[a] += filled[c, t]
            else:
                transit += filled[c, t]
    stock = start_stock(demand, mode)
    served = np.zeros(T)
    hold = 0.0
    for t in range(T):
        stock += arrive[t]
        served[t] = min(stock, demand[t])
        stock -= served[t]
        hold += stock
    unmet = demand - served
    parts = dict(purchase=purchase, holding=HOLD * avg_cost * hold, stockout=PENALTY * avg_cost * unmet.sum(),
                 salvage=-SALVAGE * avg_cost * (stock + transit))
    parts["total"] = sum(parts.values())
    parts.update(served=served, demand=demand, end_stock=stock, transit=transit)
    return parts


def shock_mask(world, mi, weeks):
    """Shock weeks of a material inside `weeks`: disrupted, plus the 4 weeks after any disrupted week."""
    d = np.nan_to_num(world.disrupted[mi])
    m = d.copy()
    for k in range(1, 5):
        m[k:] = np.maximum(m[k:], d[:-k])
    return m[[world.c(w) for w in weeks]] > 0


def diversification(shares):
    """shares: [n_rows, 3] -> per-row (1 - HHI) / (1 - 1/3)."""
    s = np.asarray(shares, float)
    return (1.0 - np.sum(s ** 2, axis=1)) / (1.0 - 1.0 / 3.0)


# ----------------------------------------------------------------------------- LP
def solve_lp(d, I0, cost, cap, lead, w, ncap, hold, pen, salv, s_min=0.0, s_max=1.0):
    """Two-stage stochastic LP for one material.
    d [T]; cost/cap/lead [S, 3, T]; w [S] (sums to 1); ncap [3] upper bound on each order.
    Returns orders o [3, T]."""
    S, C, T = cost.shape
    nO = C * T
    nF, nI, nU = C * T, T, T
    per = nF + nI + nU
    N = nO + S * per

    def o_ix(c, t):
        return c * T + t

    def f_ix(s, c, t):
        return nO + s * per + c * T + t

    def i_ix(s, t):
        return nO + s * per + nF + t

    def u_ix(s, t):
        return nO + s * per + nF + nI + t

    obj = np.zeros(N)
    obj[:nO] = 1e-9                                   # tie-breaker: don't order what is never needed
    lb, ub = np.zeros(N), np.full(N, np.inf)
    for c in range(C):
        ub[c * T:(c + 1) * T] = ncap[c]
    rows, cols, vals, rhs_ub = [], [], [], []
    eq_r, eq_c, eq_v, rhs_eq = [], [], [], []
    r_ub = 0
    r_eq = 0
    for s in range(S):
        ws = w[s]
        for c in range(C):
            for t in range(T):
                fi = f_ix(s, c, t)
                a = t + int(lead[s, c, t])
                obj[fi] = ws * cost[s, c, t] - (ws * salv if a >= T else 0.0)
                ub[fi] = max(cap[s, c, t], 0.0)
                # f <= o
                rows += [r_ub, r_ub]; cols += [fi, o_ix(c, t)]; vals += [1.0, -1.0]; rhs_ub.append(0.0); r_ub += 1
                # f >= o - (ncap - cap)   ->   o - f <= ncap - cap
                rows += [r_ub, r_ub]; cols += [o_ix(c, t), fi]; vals += [1.0, -1.0]
                rhs_ub.append(max(ncap[c] - cap[s, c, t], 0.0)); r_ub += 1
        for t in range(T):
            obj[i_ix(s, t)] = ws * hold - (ws * salv if t == T - 1 else 0.0)
            obj[u_ix(s, t)] = ws * pen
        # stock balance: I_t - I_{t-1} - arrivals_t - u_t = -d_t  (I_{-1} = I0)
        arr_by_t = [[] for _ in range(T)]
        for c in range(C):
            for t in range(T):
                a = t + int(lead[s, c, t])
                if a < T:
                    arr_by_t[a].append(f_ix(s, c, t))
        for t in range(T):
            eq_r.append(r_eq); eq_c.append(i_ix(s, t)); eq_v.append(1.0)
            if t > 0:
                eq_r.append(r_eq); eq_c.append(i_ix(s, t - 1)); eq_v.append(-1.0)
            for fi in arr_by_t[t]:
                eq_r.append(r_eq); eq_c.append(fi); eq_v.append(-1.0)
            eq_r.append(r_eq); eq_c.append(u_ix(s, t)); eq_v.append(-1.0)
            rhs_eq.append(-d[t] + (I0 if t == 0 else 0.0))
            r_eq += 1
    if s_min > 0:
        for t in range(T):
            for c in range(C):
                # s_min * sum_c' o_c't - o_ct <= 0
                for c2 in range(C):
                    rows.append(r_ub); cols.append(o_ix(c2, t)); vals.append(s_min - (1.0 if c2 == c else 0.0))
                rhs_ub.append(0.0); r_ub += 1
    if s_max < 1.0:
        for t in range(T):
            for c in range(C):
                # o_ct - s_max * sum_c' o_c't <= 0
                for c2 in range(C):
                    rows.append(r_ub); cols.append(o_ix(c2, t)); vals.append((1.0 if c2 == c else 0.0) - s_max)
                rhs_ub.append(0.0); r_ub += 1
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(r_ub, N))
    A_eq = sparse.csr_matrix((eq_v, (eq_r, eq_c)), shape=(r_eq, N))
    res = linprog(obj, A_ub=A_ub, b_ub=np.array(rhs_ub), A_eq=A_eq, b_eq=np.array(rhs_eq),
                  bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return np.maximum(res.x[:nO].reshape(C, T), 0.0)


# ----------------------------------------------------------------------------- forecasts at an origin
def _parse_age(state):
    try:
        return int(state.split("week")[-1])
    except (ValueError, AttributeError):
        return None


def _sample_len(rng, S_fun, age=0, max_len=80):
    """Number of active weeks: fresh shock (age 0) P(L >= x) = S(x); ongoing P(R >= h) = S(a+h)/S(a)."""
    x = np.arange(1, max_len + 1)
    if age <= 0:
        surv = S_fun(x)
    else:
        surv = S_fun(age + x) / max(float(S_fun(np.array([age]))[0]), 1e-12)
    u = rng.random()
    return int(np.sum(surv >= u))


class OriginInfo:
    """Everything known at an origin week: Model 1 fit, shock states, cost forecast, shock effect sizes."""

    def __init__(self, world, origin, H):
        W, P = world, world.P
        self.origin, self.H = int(origin), int(H)
        self.model = W.m1.DisruptionModel().fit(P, origin=origin)
        self.src = self.model._source_probs(H)
        self.D = self.model.duration
        t1 = P.idx(origin)
        oc = W.c(origin)
        # base landed cost forecast: spot carried forward, tariff on its announced schedule, shipping carried
        U = len(W.sup)
        spot0 = W.spot[:, oc]
        tar0 = W.tariff[:, oc]
        ship0 = W.landed[:, oc] - spot0 * (1 + tar0)
        weeks = np.arange(origin + 1, origin + H + 1)
        tar = np.repeat(tar0[:, None], H, axis=1)
        for u in range(U):
            ar, ae = W.ann_rate[u, oc], W.ann_eff[u, oc]
            if np.isfinite(ar) and np.isfinite(ae) and ae > origin:
                tar[u, weeks >= ae] = ar
        self.spot0, self.ship0, self.tar = spot0, ship0, tar
        self.cost_base = spot0[:, None] * (1 + tar) + ship0[:, None]
        # empirical shock effects up to the origin
        lb = np.log(np.vstack([np.nanmean(W.spot[u, :oc + 1], axis=0) for u in W.mat_units]))
        pm = np.array([P.sup_active[u, :t1 + 1].any(0) for u in P.mat_units])
        single = P.sup_active[:, :t1 + 1] & ~P.sup_in_country_event[:, :t1 + 1]
        cap_r = np.nan_to_num(P.cap_ratio[:, :t1 + 1], nan=1.0)
        self.sup_caps = [float(cap_r[u, s]) for u, s, e, c in W.m1._episodes(single)] or [0.05, 0.33, 0.6]
        jumps_single, jumps_cty = [], []
        for mi in range(len(W.materials)):
            a = pm[mi].astype(int)
            for s in np.where(np.diff(a) == 1)[0] + 1:
                j = float(lb[mi, s] - lb[mi, s - 1])
                if P.sup_in_country_event[P.mat_units[mi], s].any():
                    jumps_cty.append(j)
                else:
                    jumps_single.append(j)
        self.j_single = [j for j in jumps_single if j > 0] or [0.18]
        self.j_cty = float(np.median(jumps_cty)) if jumps_cty else 0.08
        # current price-shock premium per material (to unwind when the shock ends)
        self.cur_jump = np.zeros(len(W.materials))
        self.pm_now = pm[:, t1].copy()
        for mi in range(len(W.materials)):
            if pm[mi, t1]:
                s = t1
                while s > 0 and pm[mi, s - 1]:
                    s -= 1
                self.cur_jump[mi] = float(lb[mi, s] - lb[mi, s - 1]) if s > 0 else 0.0
        self.cap_now = cap_r[:, t1]


def make_scenarios(world, info, cfg):
    """Shared futures for all materials. Returns cap_mult, lead_add, cost_mult [S, U, H] and weights [S]."""
    W, P, src, D = world, world.P, info.src, info.D
    rng = np.random.default_rng(cfg.seed + info.origin)
    U, H, S = len(W.sup), info.H, cfg.n_scen
    nM = len(W.materials)
    sup_mat = np.array([W.materials.index(m) for m in W.sup.material])
    cty_ix = {c: i for i, c in enumerate(P.countries)}
    lane_ix = {l: i for i, l in enumerate(P.lanes)}
    sup_cty = np.array([cty_ix[c] for c in W.sup.country])
    sup_lane = np.array([lane_ix[l] for l in W.sup.lane])

    def draw_path(q, state, typ, pm=1.0):
        """Boolean [H] active path for one source."""
        act = np.zeros(H, bool)
        age = _parse_age(state) if state and state.startswith(typ) else None
        if age is not None:
            R = _sample_len(rng, lambda x: D.S(typ, x), age)
            act[:min(R, H)] = True
            return act
        qq = np.clip(np.asarray(q[:H]) * pm, 0, 1)
        if qq.sum() > 1:
            qq = qq / qq.sum()
        r = rng.random()
        cum = np.cumsum(qq)
        k = int(np.searchsorted(cum, r))
        if k < H:
            L = _sample_len(rng, lambda x: D.S(typ, x))
            act[k:min(k + L, H)] = True
        return act

    if S == 0:                         # deterministic plan: one calm future (plus stress, if on)
        cap_m, lead_a, cost_m = np.ones((1, U, H)), np.zeros((1, U, H)), np.ones((1, U, H))
        S_loop = 0
    else:
        cap_m, lead_a, cost_m = np.ones((S, U, H)), np.zeros((S, U, H)), np.ones((S, U, H))
        S_loop = S
    for s in range(S_loop):
        price = np.zeros((nM, H))          # log price shift per material
        # unwind or keep current price premium according to whether its shock is still on
        # single suppliers
        for u in range(U):
            st = src["sup_state"][u]
            if st == "in country event":
                continue
            a = draw_path(src["q_sup"][u], st, "supplier", cfg.prob_mult)
            ongoing = st.startswith("supplier")
            if ongoing:          # keep today's capacity while it lasts, unwind the price premium after
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], info.cap_now[u])
                price[sup_mat[u], ~a] = np.minimum(price[sup_mat[u], ~a], -info.cur_jump[sup_mat[u]])
            elif a.any():
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], rng.choice(info.sup_caps))
                price[sup_mat[u], a] = np.maximum(price[sup_mat[u], a], rng.choice(info.j_single))
        # countries
        for ci, cname in enumerate(P.countries):
            st = src["cty_state"][ci]
            a = draw_path(src["q_cty"][ci], st, "country", cfg.prob_mult)
            ongoing = st.startswith("country")
            if ongoing or a.any():
                for u in np.where(sup_cty == ci)[0]:
                    cap_m[s, u, a] = np.minimum(cap_m[s, u, a], info.cap_now[u] if ongoing else cfg.country_cap)
                    mi = sup_mat[u]
                    if not ongoing:
                        price[mi, a] = np.maximum(price[mi, a], info.j_cty)
                    else:
                        price[mi, ~a] = np.minimum(price[mi, ~a], -info.cur_jump[mi])
        # lanes
        for li, lname in enumerate(P.lanes):
            a = draw_path(src["q_lane"][li], src["lane_state"][li], "lane", cfg.prob_mult)
            if a.any():
                for u in np.where(sup_lane == li)[0]:
                    lead_a[s, u, a] = np.maximum(lead_a[s, u, a], cfg.lane_lead)
                    cap_m[s, u, a] = cap_m[s, u, a] * cfg.lane_cap
        # tariffs (unannounced hikes, or the end of one in force; announced ones are in cost_base)
        for u in range(U):
            st = src["tar_state"][u]
            if st.startswith("tariff"):
                a = draw_path(None, st, "tariff")
                base = W.sup.base_tariff.values[u]
                cost_m[s, u, ~a] *= (1 + base) / (1 + info.tar[u, 0])
            elif not st.startswith("announced"):
                q = np.full(H, info.model.q_tariff * cfg.prob_mult)
                a = draw_path(q, "", "tariff")
                cost_m[s, u, a] *= 1.2
        cost_m[s] *= np.exp(price[sup_mat])
    w = np.full(len(cap_m), 1.0 / len(cap_m))
    if cfg.use_stress:
        st_cap, st_lead, st_cost = [], [], []
        merrow = np.isin(W.sup.lane.values, ["MERROW"])
        # 1. Merrow Strait closed 8 weeks from week 3
        c1, l1 = np.ones((U, H)), np.zeros((U, H))
        c1[merrow, 2:10] *= cfg.lane_cap
        l1[merrow, 2:10] = cfg.lane_lead
        st_cap.append(c1); st_lead.append(l1); st_cost.append(np.ones((U, H)))
        # 2. each material's cheapest supplier banned for the whole window
        c2 = np.ones((U, H))
        k2 = np.ones((U, H))
        for mi, units in enumerate(W.mat_units):
            cheap = units[np.argmin(info.cost_base[units, 0])]
            c2[cheap] = 0.0
            k2[units] *= np.exp(np.median(info.j_single))
        st_cap.append(c2); st_lead.append(np.zeros((U, H))); st_cost.append(k2)
        # 3. capacity cut 50% across every supplier
        st_cap.append(np.full((U, H), 0.5)); st_lead.append(np.zeros((U, H))); st_cost.append(np.ones((U, H)))
        # 4. Kestria-wide event (as in weeks 99-133) for the whole window
        c4, k4 = np.ones((U, H)), np.ones((U, H))
        kes = W.sup.country.values == "KES"
        c4[kes] = cfg.country_cap
        for mi, units in enumerate(W.mat_units):
            if kes[units].any():
                k4[units] *= np.exp(info.j_cty)
        st_cap.append(c4); st_lead.append(np.zeros((U, H))); st_cost.append(k4)
        n = len(st_cap)
        cap_m = np.concatenate([cap_m, np.stack(st_cap)])
        lead_a = np.concatenate([lead_a, np.stack(st_lead)])
        cost_m = np.concatenate([cost_m, np.stack(st_cost)])
        w = np.concatenate([w * (1 - cfg.omega), np.full(n, cfg.omega / n)])
    return cap_m, lead_a, cost_m, w


# ----------------------------------------------------------------------------- planning
def plan_window(world, origin, start, end, cfg=None, info=None):
    """Orders for weeks start..end, decided with data up to `origin`. Returns DataFrame
    (week, material, order_units, shares..., o_<country> per supplier units)."""
    cfg = cfg or Config()
    W = world
    T = end - start + 1
    lag = start - origin - 1            # weeks between the origin and the window start (0 normally)
    H = lag + T
    info = info or OriginInfo(W, origin, H)
    cap_m, lead_a, cost_m, w = make_scenarios(W, info, cfg)
    S = len(w)
    rows = []
    for mi, m in enumerate(W.materials):
        units = W.mat_units[mi]
        d = W.demand[mi, start - 1:end]
        I0 = start_stock(d, cfg.start_stock)
        # normal capacity, or today's capacity for a supplier already hit (scenarios decide when it recovers)
        base_cap = W.normal_cap[units]
        cap = base_cap[None, :, None] * cap_m[:, units, lag:]
        lead = W.base_lead[units][None, :, None] + lead_a[:, units, lag:]
        cost = info.cost_base[units][None, :, lag:] * cost_m[:, units, lag:]
        avg = float(np.mean(info.cost_base[units, lag:]))
        o = solve_lp(d, I0, cost, cap, lead, w, base_cap, HOLD * avg, cfg.lam * avg, SALVAGE * avg,
                     cfg.s_min, cfg.s_max)
        for t in range(T):
            tot = float(o[:, t].sum())
            units_ord = int(round(tot))
            sh = o[:, t] / tot if tot > 1e-9 else np.full(len(units), 1.0 / len(units))
            row = dict(week=start + t, material=m, order_units=units_ord)
            for c in COUNTRIES:
                row[f"share_{c}"] = 0.0
            for k, u in enumerate(units):
                row[f"share_{W.sup.country.values[u]}"] = float(sh[k])
            rows.append(row)
    return pd.DataFrame(rows)


def naive_plan(world, start, end):
    rows = []
    for mi, m in enumerate(world.materials):
        cty = world.sup.country.values[world.naive_unit[mi]]
        for wk in range(start, end + 1):
            row = dict(week=wk, material=m, order_units=int(world.demand[mi, wk - 1]))
            for c in COUNTRIES:
                row[f"share_{c}"] = 1.0 if c == cty else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _orders_matrix(world, plan, mi, start, end):
    units = world.mat_units[mi]
    p = plan[plan.material == world.materials[mi]].set_index("week").reindex(range(start, end + 1))
    o = np.zeros((len(units), end - start + 1))
    for k, u in enumerate(units):
        o[k] = p.order_units.values * p[f"share_{world.sup.country.values[u]}"].values
    return o


def _truth(world, mi, start, end):
    units = world.mat_units[mi]
    sl = slice(start - 1, end)
    return (world.cap[units, sl], world.lead[units, sl], world.landed[units, sl], world.demand[mi, sl])


def hindsight_orders(world, mi, start, end, mode="sum5"):
    cap, lead, cost, d = _truth(world, mi, start, end)
    avg = float(np.mean(cost))
    o = solve_lp(d, start_stock(d, mode), cost[None], cap[None], lead[None], np.ones(1),
                 np.maximum(cap.max(1), world.normal_cap[world.mat_units[mi]]),
                 HOLD * avg, PENALTY * avg, SALVAGE * avg, 0.0)
    return o


def score_window(world, plan, start, end, mode="sum5"):
    """Score a plan on a past window (truth from supply_history). Returns dict."""
    tot = dict(ours=0.0, naive=0.0, hind=0.0)
    fill = dict(ours=[0.0, 0.0], naive=[0.0, 0.0])
    naive = naive_plan(world, start, end)
    weeks = list(range(start, end + 1))
    divs, divw = [], []
    per_mat = []
    for mi, m in enumerate(world.materials):
        cap, lead, cost, d = _truth(world, mi, start, end)
        avg = float(np.mean(cost))
        res = {}
        for name, pl in (("ours", plan), ("naive", naive)):
            o = _orders_matrix(world, pl, mi, start, end)
            r = simulate(o, cap, lead, cost, d, avg, mode)
            res[name] = r
            tot[name] += r["total"]
            mask = shock_mask(world, mi, weeks)
            fill[name][0] += r["served"][mask].sum()
            fill[name][1] += d[mask].sum()
        oh = hindsight_orders(world, mi, start, end, mode)
        rh = simulate(oh, cap, lead, cost, d, avg, mode)
        tot["hind"] += rh["total"]
        pm = plan[plan.material == m]
        sh = pm[[f"share_{world.sup.country.values[u]}" for u in world.mat_units[mi]]].values
        dv = diversification(sh)
        divs.append(dv)
        divw.append((dv, pm.order_units.values))
        per_mat.append(dict(material=m, ours=res["ours"]["total"], naive=res["naive"]["total"], hind=rh["total"],
                            ours_unmet=float((d - res["ours"]["served"]).sum()),
                            naive_unmet=float((d - res["naive"]["served"]).sum())))
    gap = tot["naive"] - tot["hind"]
    F_raw = (tot["naive"] - tot["ours"]) / gap if gap > 0 else np.nan
    fo = fill["ours"][0] / fill["ours"][1] if fill["ours"][1] > 0 else np.nan
    fn = fill["naive"][0] / fill["naive"][1] if fill["naive"][1] > 0 else np.nan
    if not np.isfinite(fn):
        fs = np.nan                       # no shock weeks in this window
    elif fn >= 1 - 1e-12:
        fs = np.nan                       # naive met all shock-week demand: skill undefined
    else:
        fs = (fo - fn) / (1 - fn)
    dv = float(np.mean(np.concatenate(divs)))
    wv = np.concatenate([x for x, _ in divw]); ww = np.concatenate([y for _, y in divw])
    dvw = float(np.sum(wv * ww) / ww.sum()) if ww.sum() > 0 else dv
    F = float(np.clip(F_raw, 0, 1)) if np.isfinite(F_raw) else 0.0
    fsc = float(np.clip(fs, 0, 1)) if np.isfinite(fs) else np.nan
    return dict(start=start, end=end, cost_ours=tot["ours"], cost_naive=tot["naive"], cost_hind=tot["hind"],
                F_raw=F_raw, F=F, fill_ours=fo, fill_naive=fn, fill_skill_raw=fs, fill_skill=fsc,
                div=dv, div_weighted=dvw,
                points_FR=40 * F + 21 * (fsc if np.isfinite(fsc) else 0.0) + 9 * dv,
                per_material=pd.DataFrame(per_mat))


# ----------------------------------------------------------------------------- backtest
_WORLD = None


def _init_worker(data_dir, model1_dir):
    global _WORLD
    warnings.filterwarnings("ignore")
    _WORLD = World(load_data(data_dir), model1_dir)


def _bt_one(args):
    origin, cfg = args
    W = _WORLD
    start, end = origin + 1, origin + 13
    plan = plan_window(W, origin, start, end, cfg)
    r = score_window(W, plan, start, end, cfg.start_stock)
    r["origin"] = origin
    r.pop("per_material")
    return r


def backtest(data_dir, origins, cfg=None, n_jobs=None, model1_dir=None):
    """Plan each past 13-week window from its origin and score it. Returns DataFrame (one row per window)."""
    from concurrent.futures import ProcessPoolExecutor
    cfg = cfg or Config()
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(data_dir, model1_dir)) as ex:
        out = list(ex.map(_bt_one, [(o, cfg) for o in origins]))
    return pd.DataFrame(out)


def summarize(bt):
    fs = bt.fill_skill
    return pd.Series(dict(windows=len(bt), F_mean=bt.F.mean(), F_min=bt.F.min(), F_raw_mean=bt.F_raw.mean(),
                          fill_ours=bt.fill_ours.mean(), fill_naive=bt.fill_naive.mean(),
                          fill_skill_mean=fs.mean(), fill_skill_windows=int(fs.notna().sum()),
                          div=bt["div"].mean(), points_FR_mean=bt.points_FR.mean(), points_FR_min=bt.points_FR.min()))


# ----------------------------------------------------------------------------- submission
def fill_submission(world, sample, windows=((209, 221), (222, 234)), cfg=None, out_path=None):
    """Fill order_units and share columns. Each window is planned from origin = min(start - 1, last week)."""
    cfg = cfg or Config()
    sub = pd.read_csv(sample) if isinstance(sample, str) else sample.copy()
    parts = sub.id.str.extract(r"^w(\d+)_(.+)$")
    wk, mat = parts[0].astype(int), parts[1]
    for start, end in windows:
        origin = min(start - 1, world.last_week)
        plan = plan_window(world, origin, start, end, cfg).set_index(["week", "material"])
        for i in sub.index[wk.between(start, end)]:
            r = plan.loc[(int(wk[i]), mat[i])]
            sub.at[i, "order_units"] = int(r.order_units)
            for c in SHARE_COLS:
                sub.at[i, c] = float(r[c])
    sub["order_units"] = sub.order_units.astype(int)
    sh = sub[list(SHARE_COLS)].clip(lower=0).round(6)
    sub[list(SHARE_COLS)] = sh.div(sh.sum(1), axis=0)
    validate(sub, world.sup)
    if out_path:
        sub.to_csv(out_path, index=False)
    return sub


def validate(sub, suppliers):
    sells = set(zip(suppliers.material, suppliers.country))
    assert len(sub) == 260 and sub.id.is_unique, "need 260 unique ids"
    assert not sub.isna().any().any(), "blank values"
    assert sub.disruption_prob.between(0, 1).all(), "disruption_prob outside 0-1"
    assert (sub.price_forecast > 0).all(), "price_forecast must be > 0"
    assert (sub.order_units >= 0).all(), "negative order_units"
    assert np.allclose(sub[list(SHARE_COLS)].sum(1), 1.0, atol=1e-9), "shares must sum to 1"
    m = sub.id.str.split("_", n=1).str[1]
    for c in SHARE_COLS:
        bad = (sub[c] > 0) & ~pd.Series([(mm, c[6:]) in sells for mm in m], index=sub.index)
        assert not bad.any(), f"{c} > 0 for a country that doesn't sell the material: {sub.id[bad].iloc[0]}"
    return True


def _main():
    ap = argparse.ArgumentParser(description="Model 3 ordering plan")
    ap.add_argument("--data", required=True)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--submission", help="CSV to fill (sample, or one with Model 1/2 columns already filled)")
    ap.add_argument("--out", default="submission_model3.csv")
    ap.add_argument("--s-min", type=float, default=Config.s_min)
    ap.add_argument("--lam", type=float, default=Config.lam)
    ap.add_argument("--omega", type=float, default=Config.omega)
    ap.add_argument("--n-scen", type=int, default=Config.n_scen)
    ap.add_argument("--aggressive", action="store_true", help="add stress futures (omega)")
    args = ap.parse_args()
    cfg = Config(s_min=args.s_min, lam=args.lam, omega=args.omega, n_scen=args.n_scen,
                 use_stress=args.aggressive)
    if args.backtest:
        W = World(load_data(args.data))
        origins = list(range(60, W.last_week - 13 + 1, 4))
        bt = backtest(args.data, origins, cfg)
        pd.set_option("display.width", 220)
        print(bt.round(3).to_string())
        print(summarize(bt).round(3).to_string())
    if args.submission:
        W = World(load_data(args.data))
        sub = fill_submission(W, args.submission, cfg=cfg, out_path=args.out)
        print(f"wrote {args.out} ({len(sub)} rows)")


if __name__ == "__main__":
    _main()
