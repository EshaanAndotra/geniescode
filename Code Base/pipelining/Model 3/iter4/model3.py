"""
model3.py - Model 3 iter4 (iter3 + "min" starting stock): score-aware ordering plan (order_units + share_KES..share_AUR).

iter3 keeps iter1's two-stage stochastic LP (one set of orders shared by every sampled future)
and changes what it optimizes and how the futures look:

  1. Realistic calm futures. In calm weeks capacity is 0.984 x normal on average (sd 0.012,
     no autocorrelation) and lead time is exactly the base lead. iter1 planned calm weeks at
     1.0 x normal, so an order sized to capacity came up ~1.6% short and every missing unit cost
     5x. Futures now draw capacity noise per supplier-week from the calm weeks before the origin.
     Lane closures add 3 lead weeks (4 with the observed 7% chance).
  2. Points, not dollars. The score is 40 F + 21 fill skill + 9 diversification (+ the P part,
     which orders don't touch), and F divides each window's savings by that window's
     naive-minus-hindsight gap. In a calm future that gap is ~1.5 weeks of purchases per
     material, in a future where naive runs out it is 10-100x bigger, so a dollar is worth far
     more points in calm futures. For every sampled future s the plan computes the naive cost,
     the hindsight cost (block LP over all futures) and naive's shortfall in shock weeks, then
     weights that future's dollars by 40 / gap_s and its shock-week shortfall units by
     21 / naive_shortfall_s. The LP objective is expected points lost.
  3. Diversification per material. Each material's LP is solved for a few minimum-share floors
     (by supplier cost rank); every candidate is scored exactly on the futures with the real
     fill rule (fill = min(order, capacity)) plus 9 x mean row diversification, and the best
     floor is kept. Rows with no order get an even split (diversification 1 for free if the
     scorer averages rows equally; switch with Config(div_mode="units")).

Everything else (simulator, scorer, naive and hindsight plans, validation) matches iter1, so
score_local.py and the full backtest treat both the same way.

iter4 = iter3 planned with start_stock="min": the smaller of sum-of-first-5-weeks and 5 x mean
demand. Neither rule reproduces the public scores exactly (verify_scorer.py: 5avg max error 1.32,
sum5 1.50). On the week-221 stress scenarios the plan built for the wrong rule loses ~2 of 70
points; the "min" plan is within 0.05 of the right-rule plan under either one.

Requires numpy, pandas, scipy; Model 1 (iter4 or iter3 model1.py) for the hazards.
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
RECOVERY = 4
COUNTRIES = ("KES", "VAL", "SUN", "ORI", "HAL", "AUR")
SHARE_COLS = tuple(f"share_{c}" for c in COUNTRIES)
FILES = ("outcomes", "suppliers", "supply_history", "tariffs", "prices", "shipping", "macro", "demand")
# minimum shares for the (2nd cheapest, 3rd cheapest) supplier tried per material
FLOOR_GRID = ((0.0, 0.0), (0.1, 0.05), (0.2, 0.1), (0.3, 0.15), (0.45, 0.0), (1 / 3, 1 / 3))


@dataclass
class Config:
    n_scen: int = 200            # sampled futures per window
    lam: float = 5.0             # stockout penalty, x average landed cost (the real one is 5)
    prob_mult: float = 1.0       # scales Model 1's onset hazards
    use_stress: bool = False     # add iter1's stress futures (weight omega)
    omega: float = 0.15
    seed: int = 0
    start_stock: str = "min"     # "sum5", "5avg" or "min" (plan with the smaller: the scorer's rule is unconfirmed)
    lane_cap: float = 0.78       # capacity multiplier during a lane closure
    lane_lead: int = 3           # extra lead weeks during a lane closure
    lane_lead4_p: float = 0.07   # chance a closure adds 4 weeks instead (12 of 172 closure weeks)
    country_cap: float = 0.2     # capacity multiplier during a country event
    cap_noise: bool = True       # calm capacity noise (fitted on calm weeks before the origin) when scoring candidates
    order_cap_k: float = 2.5     # orders per supplier <= (calm mean - k sd) x normal capacity: never short in calm weeks
    score_weights: bool = True   # weight futures by 40/gap and 21/naive shortfall (False = iter1 dollars)
    w_F: float = 40.0
    w_fill: float = 21.0
    w_div: float = 9.0
    fill_floor: float = 0.05     # naive shortfall floored at this share of shock-week demand (caps 21/shortfall)
    f_mode: str = "pooled"       # F from costs summed over materials ("pooled") or the mean of per-material F
    fill_mode: str = "pooled"    # fill skill from units summed over materials, or the mean per material
    div_mode: str = "rows"       # diversification averaged over rows ("rows") or weighted by units
    floor_grid: tuple = FLOOR_GRID
    s_min: float = None          # a fixed minimum share for every supplier instead of the grid (iter1 style)


MODERATE = Config()
AGGRESSIVE = Config(prob_mult=1.5, use_stress=True, omega=0.10)


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
        os.path.normpath(os.path.join(here, os.pardir, os.pardir, "Model 1", "iter4")),
        os.path.normpath(os.path.join(here, os.pardir, os.pardir, "Model 1", "iter3"))]
    for c in cands:
        if c and os.path.isdir(c):
            sys.path.insert(0, c)
            try:
                return importlib.import_module("model1")
            except Exception:
                sys.path.remove(c)
    raise ImportError("model1.py not found; pass model1_dir")


class World:
    """Dense arrays. Supplier rows follow suppliers.csv; weeks are 1-based (column = week - 1)."""

    def __init__(self, data, model1_dir=None):
        self.data = data
        self.m1 = import_model1(model1_dir)
        self.P = self.m1.Panels(data)
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
        self.sup_mat = np.array([self.materials.index(m) for m in sup.material])

    def c(self, week):
        return int(week) - 1


# ----------------------------------------------------------------------------- simulator
def start_stock(demand_window, mode="sum5"):
    if mode == "5avg":
        return START_WEEKS * float(np.mean(demand_window))
    if mode == "min":
        return min(start_stock(demand_window, "sum5"), start_stock(demand_window, "5avg"))
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


def simulate_batch(orders, cap, lead, cost, demand, I0, avg_cost, pen_mult=PENALTY):
    """simulate() for many futures at once. orders [C, T] (shared) or [S, C, T]; cap/lead/cost [S, C, T].
    Returns (total cost [S], unmet [S, T])."""
    S, C, T = cap.shape
    o = np.maximum(orders, 0.0)
    f = np.minimum(o[None] if o.ndim == 2 else o, np.maximum(cap, 0.0))
    purchase = (f * cost).sum(axis=(1, 2))
    a = np.minimum(np.arange(T)[None, None, :] + lead.astype(int), T)
    arrive = np.zeros((S, T + 1))
    np.add.at(arrive, (np.broadcast_to(np.arange(S)[:, None, None], (S, C, T)).ravel(), a.ravel()), f.ravel())
    stock = np.full(S, float(I0))
    hold = np.zeros(S)
    unmet = np.zeros((S, T))
    for t in range(T):
        stock = stock + arrive[:, t]
        served = np.minimum(stock, demand[t])
        unmet[:, t] = demand[t] - served
        stock = stock - served
        hold += stock
    total = purchase + avg_cost * (HOLD * hold + pen_mult * unmet.sum(1) - SALVAGE * (stock + arrive[:, T]))
    return total, unmet


def shock_mask(world, mi, weeks):
    """Shock weeks of a material inside `weeks`: disrupted, plus the 4 weeks after any disrupted week."""
    d = np.nan_to_num(world.disrupted[mi])
    m = d.copy()
    for k in range(1, RECOVERY + 1):
        m[k:] = np.maximum(m[k:], d[:-k])
    return m[[world.c(w) for w in weeks]] > 0


def diversification(shares):
    """shares: [n_rows, 3] -> per-row (1 - HHI) / (1 - 1/3)."""
    s = np.asarray(shares, float)
    return (1.0 - np.sum(s ** 2, axis=1)) / (1.0 - 1.0 / 3.0)


# ----------------------------------------------------------------------------- LPs
def _merge_futures(w, *arrays):
    """Merge identical futures. Returns (index of each kept future, summed weights, inverse map)."""
    S = len(w)
    key = np.concatenate([np.round(np.asarray(x, float).reshape(S, -1), 6) for x in arrays], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    return first, np.bincount(inv, weights=w, minlength=len(first)), inv


def solve_lp(d, I0, cost, cap, lead, w, ncap, hold, pen, salv, floors=None, alpha=None, beta=None, q=None):
    """Two-stage stochastic LP for one material, built with index arrays.
    d [T]; cost/cap/lead [S, C, T]; w [S] (sums to 1); ncap [C] upper bound on each order;
    floors [C] minimum share of each week's order per supplier; alpha [S] weight per dollar
    (1 = plain expected cost); beta [S, T] extra weight per unit of shortfall.
    q [C]: safe order cap per supplier. Orders stay <= q, so wherever a future's capacity is
    >= q the fill equals the order and needs no variable of its own; only shock cells
    (capacity < q) get fill variables (f <= o, f <= cap, f >= o - (q - cap)). Identical
    futures are merged first. q=None gives iter1's full formulation.
    Returns orders o [C, T]."""
    S, C, T = cost.shape
    alpha = np.ones(S) if alpha is None else np.asarray(alpha, float)
    beta = np.zeros((S, T)) if beta is None else np.asarray(beta, float)
    oub = np.asarray(ncap if q is None else q, float)
    capc = np.minimum(cap, oub[None, :, None]) if q is not None else cap
    keep, w, _ = _merge_futures(np.asarray(w, float), capc, lead, cost, alpha, beta)
    capc, lead, cost, alpha, beta = capc[keep], lead[keep], cost[keep], alpha[keep], beta[keep]
    S = len(keep)
    ws = w * alpha                                            # per-dollar weight of each future
    shock = capc < oub[None, :, None] - 1e-9 if q is not None else np.ones((S, C, T), bool)
    nO, nf = C * T, int(shock.sum())
    f_ix = np.full((S, C, T), -1)
    f_ix[shock] = nO + np.arange(nf)
    i_ix = nO + nf + np.arange(S * T).reshape(S, T)
    u_ix = i_ix + S * T
    N = nO + nf + 2 * S * T
    o_ix = np.arange(C)[:, None] * T + np.arange(T)[None, :]
    arr_t = np.arange(T)[None, None, :] + lead.astype(int)
    unit = ws[:, None, None] * (cost - np.where(arr_t >= T, salv, 0.0))   # value of one filled unit

    obj = np.zeros(N)
    obj[:nO] = 1e-9 + np.where(shock, 0.0, unit).sum(0).ravel()          # calm cells: fill = order
    obj[f_ix[shock]] = unit[shock]
    obj[i_ix] = ws[:, None] * hold
    obj[i_ix[:, -1]] -= ws * salv
    obj[u_ix] = ws[:, None] * pen + w[:, None] * beta
    lb, ub = np.zeros(N), np.full(N, np.inf)
    ub[o_ix] = oub[:, None]
    ub[f_ix[shock]] = np.maximum(capc[shock], 0.0)

    # shock cells: f <= o ; o - f <= q - cap
    oi = np.broadcast_to(o_ix[None], (S, C, T))[shock]
    fi = f_ix[shock]
    r1 = np.arange(nf)
    rows = [r1, r1, nf + r1, nf + r1]
    cols = [fi, oi, oi, fi]
    vals = [np.ones(nf), -np.ones(nf), np.ones(nf), -np.ones(nf)]
    rhs = [np.zeros(nf), np.maximum(oub[None, :, None] - capc, 0.0)[shock]]
    n_ub = 2 * nf
    if floors is not None and np.any(np.asarray(floors) > 0):
        fl = np.asarray(floors, float)
        n0 = n_ub
        for c in range(C):
            if fl[c] <= 0:
                continue
            for c2 in range(C):
                rows.append(n_ub + np.arange(T)); cols.append(o_ix[c2]); vals.append(np.full(T, fl[c] - (c2 == c)))
            n_ub += T
        rhs.append(np.zeros(n_ub - n0))
    A_ub = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n_ub, N)) if n_ub else None

    # stock balance: I_t - I_{t-1} - arrivals_t - u_t = -d_t (+ I0 at t = 0)
    er = [np.arange(S * T), np.arange(S * T)]
    ec = [i_ix.ravel(), u_ix.ravel()]
    ev = [np.ones(S * T), -np.ones(S * T)]
    if T > 1:
        rr = (np.arange(S)[:, None] * T + np.arange(1, T)[None, :]).ravel()
        er.append(rr); ec.append(i_ix[:, :-1].ravel()); ev.append(-np.ones(len(rr)))
    s_all = np.broadcast_to(np.arange(S)[:, None, None], (S, C, T))
    ok = arr_t < T
    var = np.where(shock, f_ix, np.broadcast_to(o_ix[None], (S, C, T)))
    er.append((s_all * T + arr_t)[ok]); ec.append(var[ok]); ev.append(-np.ones(ok.sum()))
    b_eq = np.tile(-np.asarray(d, float), S)
    b_eq[np.arange(S) * T] += I0
    A_eq = sparse.csr_matrix((np.concatenate(ev), (np.concatenate(er), np.concatenate(ec))), shape=(S * T, N))
    res = linprog(obj, A_ub=A_ub, b_ub=np.concatenate(rhs) if n_ub else None, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.column_stack([lb, ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return np.maximum(res.x[:nO].reshape(C, T), 0.0)


def hindsight_batch(d, I0, cost, cap, lead, hold, pen, salv):
    """Perfect-hindsight plan in every future at once (orders = fills, one LP with one block per
    distinct future). Returns fills [S, C, T]; price them with simulate_batch()."""
    keep, _, inv = _merge_futures(np.ones(cost.shape[0]), cap, lead, cost)
    if len(keep) < cost.shape[0]:
        return hindsight_batch(d, I0, cost[keep], cap[keep], lead[keep], hold, pen, salv)[inv]
    S, C, T = cost.shape
    nF = C * T
    per = nF + 2 * T
    N = S * per
    base = np.arange(S) * per
    f_ix = base[:, None, None] + np.arange(C)[None, :, None] * T + np.arange(T)[None, None, :]
    i_ix = base[:, None] + nF + np.arange(T)[None, :]
    u_ix = i_ix + T
    arr_t = np.arange(T)[None, None, :] + lead.astype(int)
    obj = np.zeros(N)
    obj[f_ix] = cost - np.where(arr_t >= T, salv, 0.0)
    obj[i_ix] = hold
    obj[i_ix[:, -1]] -= salv
    obj[u_ix] = pen
    ub = np.full(N, np.inf)
    ub[f_ix] = np.maximum(cap, 0.0)
    er = [np.arange(S * T), np.arange(S * T)]
    ec = [i_ix.ravel(), u_ix.ravel()]
    ev = [np.ones(S * T), -np.ones(S * T)]
    if T > 1:
        rr = (np.arange(S)[:, None] * T + np.arange(1, T)[None, :]).ravel()
        er.append(rr); ec.append(i_ix[:, :-1].ravel()); ev.append(-np.ones(len(rr)))
    ok = arr_t < T
    s_i = np.broadcast_to(np.arange(S)[:, None, None], (S, C, T))[ok]
    er.append(s_i * T + arr_t[ok]); ec.append(f_ix[ok]); ev.append(-np.ones(ok.sum()))
    b_eq = np.tile(-np.asarray(d, float), S)
    b_eq[np.arange(S) * T] += I0
    A_eq = sparse.csr_matrix((np.concatenate(ev), (np.concatenate(er), np.concatenate(ec))), shape=(S * T, N))
    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=np.column_stack([np.zeros(N), ub]), method="highs")
    if res.status != 0:
        raise RuntimeError(f"hindsight LP failed: {res.message}")
    return res.x[f_ix]


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
    """Everything known at an origin week: Model 1 fit, shock states, cost forecast, shock effect sizes,
    calm capacity noise and the disrupted history (for recovery weeks at the start of the window)."""

    def __init__(self, world, origin, H):
        W, P = world, world.P
        self.origin, self.H = int(origin), int(H)
        self.model = W.m1.DisruptionModel().fit(P, origin=origin)
        self.src = self.model._source_probs(H)
        self.D = self.model.duration
        t1 = P.idx(origin)
        oc = W.c(origin)
        U = len(W.sup)
        spot0 = W.spot[:, oc]
        tar0 = W.tariff[:, oc]
        ship0 = W.landed[:, oc] - spot0 * (1 + tar0)
        weeks = np.arange(origin + 1, origin + H + 1)
        tar = np.repeat(tar0[:, None], H, axis=1)
        self.ann_start = np.full(U, -1)
        for u in range(U):
            ar, ae = W.ann_rate[u, oc], W.ann_eff[u, oc]
            if np.isfinite(ar) and np.isfinite(ae) and ae > origin:
                tar[u, weeks >= ae] = ar
                self.ann_start[u] = int(ae - origin - 1)
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
        self.cur_jump = np.zeros(len(W.materials))
        self.pm_now = pm[:, t1].copy()
        for mi in range(len(W.materials)):
            if pm[mi, t1]:
                s = t1
                while s > 0 and pm[mi, s - 1]:
                    s -= 1
                self.cur_jump[mi] = float(lb[mi, s] - lb[mi, s - 1]) if s > 0 else 0.0
        self.cap_now = cap_r[:, t1]
        # calm capacity noise per supplier (weeks with no supplier shock and an open lane)
        calm = ~P.sup_active[:, :t1 + 1] & ~P.lane_closed[P.sup_lane, :t1 + 1] & np.isfinite(P.cap_ratio[:, :t1 + 1])
        r = np.where(calm, P.cap_ratio[:, :t1 + 1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.cap_mu = np.nan_to_num(np.nanmean(r, axis=1), nan=0.984)
            self.cap_sd = np.nan_to_num(np.nanstd(r, axis=1), nan=0.012)
            self.cap_hi = np.nan_to_num(np.nanmax(r, axis=1), nan=1.0)
        # disrupted history up to the origin (recovery weeks spill into the window)
        self.dis_hist = np.nan_to_num(W.disrupted[:, max(0, oc - RECOVERY + 1): oc + 1]).astype(bool)


def make_scenarios(world, info, cfg):
    """Shared futures for all materials. Returns cap_mult, lead_add, cost_mult [S, U, H], weights [S],
    dis [S, M, H] (material disrupted by any source: supplier, country, lane or tariff) and
    noise [S, U, H] (calm capacity / normal, None if cfg.cap_noise is off). cap_mult is 1 in calm
    cells; the noise is kept apart so the LP sees identical calm futures (which merge) while
    candidate plans are scored on the noisy ones."""
    W, P, src, D = world, world.P, info.src, info.D
    rng = np.random.default_rng(cfg.seed + info.origin)
    U, H, S = len(W.sup), info.H, cfg.n_scen
    nM = len(W.materials)
    sup_mat = W.sup_mat
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
        k = int(np.searchsorted(np.cumsum(qq), r))
        if k < H:
            L = _sample_len(rng, lambda x: D.S(typ, x))
            act[k:min(k + L, H)] = True
        return act

    cap_m, lead_a, cost_m = np.ones((S, U, H)), np.zeros((S, U, H)), np.ones((S, U, H))
    dis = np.zeros((S, nM, H), bool)
    noise = None
    if cfg.cap_noise:
        lo = np.maximum(info.cap_mu - 4 * info.cap_sd, 0.8)
        hi = np.maximum(info.cap_hi, info.cap_mu)
        noise = np.clip(rng.normal(info.cap_mu[None, :, None], info.cap_sd[None, :, None], (S, U, H)),
                        lo[None, :, None], hi[None, :, None])
    for s in range(S):
        price = np.zeros((nM, H))          # log price shift per material
        for u in range(U):
            st = src["sup_state"][u]
            if st == "in country event":
                continue
            a = draw_path(src["q_sup"][u], st, "supplier", cfg.prob_mult)
            ongoing = st.startswith("supplier")
            if ongoing:
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], info.cap_now[u])
                price[sup_mat[u], ~a] = np.minimum(price[sup_mat[u], ~a], -info.cur_jump[sup_mat[u]])
            elif a.any():
                cap_m[s, u, a] = np.minimum(cap_m[s, u, a], rng.choice(info.sup_caps))
                price[sup_mat[u], a] = np.maximum(price[sup_mat[u], a], rng.choice(info.j_single))
            dis[s, sup_mat[u]] |= a
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
                    dis[s, mi] |= a
        for li, lname in enumerate(P.lanes):
            a = draw_path(src["q_lane"][li], src["lane_state"][li], "lane", cfg.prob_mult)
            if a.any():
                extra = cfg.lane_lead + (1 if rng.random() < cfg.lane_lead4_p else 0)
                for u in np.where(sup_lane == li)[0]:
                    lead_a[s, u, a] = np.maximum(lead_a[s, u, a], extra)
                    cap_m[s, u, a] = cap_m[s, u, a] * cfg.lane_cap
                    dis[s, sup_mat[u]] |= a
        for u in range(U):
            st = src["tar_state"][u]
            base = W.sup.base_tariff.values[u]
            if st.startswith("tariff"):
                a = draw_path(None, st, "tariff")
                cost_m[s, u, ~a] *= (1 + base) / (1 + info.tar[u, 0])
                dis[s, sup_mat[u]] |= a
            elif st.startswith("announced"):
                k0 = info.ann_start[u]
                if 0 <= k0 < H and rng.random() < info.model.cfg.p_announced:
                    L = _sample_len(rng, lambda x: D.S("tariff", x))
                    a = np.zeros(H, bool)
                    a[k0:min(k0 + L, H)] = True
                    dis[s, sup_mat[u]] |= a
                    after = np.arange(H) >= k0 + L
                    cost_m[s, u, after] *= (1 + base) / (1 + info.tar[u, -1])
                elif 0 <= k0 < H:      # announced but not enacted
                    cost_m[s, u, k0:] *= (1 + base) / (1 + info.tar[u, -1])
            else:
                q = np.full(H, info.model.q_tariff * cfg.prob_mult)
                a = draw_path(q, "", "tariff")
                cost_m[s, u, a] *= 1.2
                dis[s, sup_mat[u]] |= a
        cost_m[s] *= np.exp(price[sup_mat])
    w = np.full(S, 1.0 / S)
    if cfg.use_stress:
        st_cap, st_lead, st_cost, st_dis = [], [], [], []
        merrow = np.isin(W.sup.lane.values, ["MERROW"])
        mer_mats = np.unique(sup_mat[merrow])
        # 1. Merrow Strait closed 8 weeks from week 3
        c1, l1 = np.ones((U, H)), np.zeros((U, H))
        c1[merrow, 2:10] *= cfg.lane_cap
        l1[merrow, 2:10] = cfg.lane_lead
        d1 = np.zeros((nM, H), bool); d1[np.ix_(mer_mats, np.arange(2, min(10, H)))] = True
        st_cap.append(c1); st_lead.append(l1); st_cost.append(np.ones((U, H))); st_dis.append(d1)
        # 2. each material's cheapest supplier banned for the whole window
        c2, k2 = np.ones((U, H)), np.ones((U, H))
        for mi, units in enumerate(W.mat_units):
            cheap = units[np.argmin(info.cost_base[units, 0])]
            c2[cheap] = 0.0
            k2[units] *= np.exp(np.median(info.j_single))
        st_cap.append(c2); st_lead.append(np.zeros((U, H))); st_cost.append(k2); st_dis.append(np.ones((nM, H), bool))
        # 3. capacity cut 50% across every supplier
        st_cap.append(np.full((U, H), 0.5)); st_lead.append(np.zeros((U, H))); st_cost.append(np.ones((U, H)))
        st_dis.append(np.ones((nM, H), bool))
        # 4. Kestria-wide event (as in weeks 99-133) for the whole window
        c4, k4 = np.ones((U, H)), np.ones((U, H))
        kes = W.sup.country.values == "KES"
        c4[kes] = cfg.country_cap
        d4 = np.zeros((nM, H), bool)
        for mi, units in enumerate(W.mat_units):
            if kes[units].any():
                k4[units] *= np.exp(info.j_cty)
                d4[mi] = True
        st_cap.append(c4); st_lead.append(np.zeros((U, H))); st_cost.append(k4); st_dis.append(d4)
        n = len(st_cap)
        cap_m = np.concatenate([cap_m, np.stack(st_cap)])
        if noise is not None:
            noise = np.concatenate([noise, noise[:n]])
        lead_a = np.concatenate([lead_a, np.stack(st_lead)])
        cost_m = np.concatenate([cost_m, np.stack(st_cost)])
        dis = np.concatenate([dis, np.stack(st_dis)])
        w = np.concatenate([w * (1 - cfg.omega), np.full(n, cfg.omega / n)])
    return cap_m, lead_a, cost_m, w, dis, noise


# ----------------------------------------------------------------------------- planning
def _window_masks(info, dis, lag, T):
    """Shock-week masks [S, M, T] for the window: disrupted or within RECOVERY weeks after (history included)."""
    S, M, H = dis.shape
    hist = np.broadcast_to(info.dis_hist[None], (S, M, info.dis_hist.shape[1]))
    full = np.concatenate([hist, dis], axis=2)
    off = hist.shape[2]
    out = np.zeros((S, M, T), bool)
    for t in range(T):
        j = off + lag + t
        out[:, :, t] = full[:, :, max(0, j - RECOVERY): j + 1].any(axis=2)
    return out


def plan_window(world, origin, start, end, cfg=None, info=None, return_detail=False):
    """Orders for weeks start..end, decided with data up to `origin`. Returns DataFrame
    (week, material, order_units, shares...)."""
    cfg = cfg or Config()
    W = world
    T = end - start + 1
    lag = start - origin - 1
    H = lag + T
    info = info or OriginInfo(W, origin, H)
    cap_m, lead_a, cost_m, w, dis, noise = make_scenarios(W, info, cfg)
    S = len(w)
    M = len(W.materials)
    masks = _window_masks(info, dis, lag, T)
    calm = cap_m[:, :, lag:] >= 1.0 - 1e-9
    inp = []
    for mi, m in enumerate(W.materials):
        units = W.mat_units[mi]
        d = W.demand[mi, start - 1:end]
        I0 = start_stock(d, cfg.start_stock)
        base_cap = W.normal_cap[units]
        cap = base_cap[None, :, None] * cap_m[:, units, lag:]                     # LP (calm cells = normal)
        c_ev = cap if noise is None else base_cap[None, :, None] * np.where(
            calm[:, units], noise[:, units, lag:], cap_m[:, units, lag:])           # scoring (calm noise)
        c_h = base_cap[None, :, None] * np.where(calm[:, units], info.cap_mu[units][None, :, None],
                                                 cap_m[:, units, lag:])           # normalizers
        q = np.minimum(info.cap_mu[units] - cfg.order_cap_k * info.cap_sd[units], 1.0) * base_cap
        lead = W.base_lead[units][None, :, None] + lead_a[:, units, lag:]
        cost = info.cost_base[units][None, :, lag:] * cost_m[:, units, lag:]
        avg = float(np.mean(info.cost_base[units, lag:]))
        inp.append((units, d, I0, base_cap, cap, lead, cost, avg, c_ev, c_h, q))

    # per-future normalizers: naive cost and shortfall, hindsight cost
    cn = np.zeros((S, M)); ch = np.zeros((S, M)); un = np.zeros((S, M)); dsh = np.zeros((S, M))
    for mi, (units, d, I0, base_cap, cap, lead, cost, avg, c_ev, c_h, q) in enumerate(inp):
        k = int(np.where(units == W.naive_unit[mi])[0][0])
        o_n = np.zeros((len(units), T)); o_n[k] = d
        cn[:, mi], u_n = simulate_batch(o_n, c_h, lead, cost, d, I0, avg)
        un[:, mi] = (u_n * masks[:, mi]).sum(1)
        dsh[:, mi] = (d[None] * masks[:, mi]).sum(1)
        if cfg.score_weights:
            fh = hindsight_batch(d, I0, cost, c_h, lead, HOLD * avg, PENALTY * avg, SALVAGE * avg)
            ch[:, mi], _ = simulate_batch(fh, c_h, lead, cost, d, I0, avg)
    if cfg.score_weights:
        gap_m = np.maximum(cn - ch, 1e-6 * np.abs(cn))
        if cfg.f_mode == "pooled":
            alpha = np.repeat((cfg.w_F / gap_m.sum(1))[:, None], M, axis=1)
        else:
            alpha = cfg.w_F / M / gap_m
        if cfg.fill_mode == "pooled":
            den = np.maximum(un.sum(1), cfg.fill_floor * dsh.sum(1))
            b = np.where(un.sum(1) > 0, cfg.w_fill / np.maximum(den, 1e-9), 0.0)
            beta_m = np.repeat(b[:, None], M, axis=1)
        else:
            n_sh = np.maximum((un > 0).sum(1, keepdims=True), 1)
            den = np.maximum(un, cfg.fill_floor * dsh)
            beta_m = np.where(un > 0, cfg.w_fill / n_sh / np.maximum(den, 1e-9), 0.0)
    else:
        alpha = np.ones((S, M)); beta_m = np.zeros((S, M))
    n_rows = M * T
    tot_units = sum(x[1].sum() for x in inp)

    rows, detail = [], []
    for mi, m in enumerate(W.materials):
        units, d, I0, base_cap, cap, lead, cost, avg, c_ev, c_h, q = inp[mi]
        C = len(units)
        beta = beta_m[:, mi][:, None] * masks[:, mi].astype(float)       # [S, T]
        rank = np.argsort(info.cost_base[units, lag:].mean(1))            # cheapest first
        grid = [(cfg.s_min,) * 2] if cfg.s_min is not None else list(cfg.floor_grid)
        best = None
        for g in grid:
            fl = np.zeros(C)
            if cfg.s_min is not None:
                fl[:] = cfg.s_min
            else:
                fl[rank[1]], fl[rank[2]] = g
            o = solve_lp(d, I0, cost, cap, lead, w, base_cap, HOLD * avg, cfg.lam * avg, SALVAGE * avg,
                         floors=fl, alpha=alpha[:, mi], beta=beta, q=q)
            o_int = _round_orders(o)
            tc, um = simulate_batch(o_int, c_ev, lead, cost, d, I0, avg)
            loss_F = float(np.sum(w * alpha[:, mi] * tc))
            loss_fill = float(np.sum(w * (beta * um).sum(1)))
            tot = o_int.sum(0)
            sh = np.where(tot > 0, o_int / np.maximum(tot, 1e-12), 1.0 / C)
            dv = diversification(sh.T)
            if cfg.div_mode == "rows":
                gain_div = cfg.w_div * dv.sum() / n_rows
            else:
                gain_div = cfg.w_div * float((dv * tot).sum()) / max(tot_units, 1.0)
            val = -loss_F - loss_fill + gain_div
            if best is None or val > best[0]:
                best = (val, g, o_int)
            detail.append(dict(material=m, floors=g, value=val, loss_F=loss_F, loss_fill=loss_fill, gain_div=gain_div))
        o = best[2]
        for t in range(T):
            tot = float(o[:, t].sum())
            sh = o[:, t] / tot if tot > 0 else np.full(C, 1.0 / C)
            row = dict(week=start + t, material=m, order_units=int(round(tot)))
            for c in COUNTRIES:
                row[f"share_{c}"] = 0.0
            for k, u in enumerate(units):
                row[f"share_{W.sup.country.values[u]}"] = float(sh[k])
            rows.append(row)
    plan = pd.DataFrame(rows)
    if return_detail:
        return plan, pd.DataFrame(detail)
    return plan


def _round_orders(o):
    """Whole units per week; supplier split kept proportional (what the submission can express)."""
    tot = o.sum(0)
    r = np.round(tot)
    sh = np.where(tot > 1e-9, o / np.maximum(tot, 1e-12), 0.0)
    return sh * r[None, :]


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
    return hindsight_batch(d, start_stock(d, mode), cost[None], cap[None], lead[None],
                           HOLD * avg, PENALTY * avg, SALVAGE * avg)[0]


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
        mask = shock_mask(world, mi, weeks)
        for name, pl in (("ours", plan), ("naive", naive)):
            o = _orders_matrix(world, pl, mi, start, end)
            r = simulate(o, cap, lead, cost, d, avg, mode)
            res[name] = r
            tot[name] += r["total"]
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
                            naive_unmet=float((d - res["naive"]["served"]).sum()),
                            shock_weeks=int(mask.sum()),
                            ours_unmet_shock=float((d - res["ours"]["served"])[mask].sum()),
                            naive_unmet_shock=float((d - res["naive"]["served"])[mask].sum()),
                            shock_demand=float(d[mask].sum())))
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
    ap = argparse.ArgumentParser(description="Model 3 iter3 ordering plan")
    ap.add_argument("--data", required=True)
    ap.add_argument("--submission", help="CSV to fill (sample, or one with Model 1/2 columns already filled)")
    ap.add_argument("--out", default="submission_model3.csv")
    ap.add_argument("--aggressive", action="store_true")
    args = ap.parse_args()
    W = World(load_data(args.data))
    sub = fill_submission(W, args.submission, cfg=AGGRESSIVE if args.aggressive else MODERATE, out_path=args.out)
    print(f"wrote {args.out} ({len(sub)} rows)")


if __name__ == "__main__":
    _main()
