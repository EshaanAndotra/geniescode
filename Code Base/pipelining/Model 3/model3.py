"""
model3.py - Model 3: ordering plan (order_units + share_* columns).

Needs model1.py and model2.py importable (shock hazards/durations and landed costs).

Pieces (doc order):
  1. simulate()          the competition's rules: 5 weeks of demand in stock, each country's slice
                         filled up to its capacity (unfilled units cancelled, not charged), paid at
                         that week's landed cost, arriving after that week's lead time; holding 2%
                         and stockouts 5x the average landed cost; stock + in-transit sold back at 70%.
  2. naive_orders()      the sample's policy, inferred from sample_submission: order = that week's
                         demand, 100% from the supplier with the largest normal capacity.
  3. solve_lp()          one LP for one material and one 13-week window over S possible futures
                         (two-stage: shared orders o, per-future fills f, stock I, shortfall u).
                         S = 1 gives the quick LP (with the safety buffer I_t >= k * p_t * d_t) and
                         the perfect-hindsight plan (actual values as the only future).
  4. ScenarioGenerator   futures from Model 1: lane closures, country events, single-supplier
                         shocks and tariffs, each with onset week drawn from Model 1's onset
                         probabilities and length from its duration curves; effects (capacity
                         kappa, extra lead delta, cost) bootstrapped from past episodes. Same draws
                         for all 10 materials, so one Merrow closure hits every Merrow supplier.
                         Plus stress futures (Merrow closed 8 weeks, cheapest supplier banned,
                         capacity halved) with total weight omega.
  5. backtest() / tune() past windows scored like the competition: F (cost vs naive and
                         hindsight), fill rate in shock weeks (+4 recovery), diversification.
  6. fill_submission()   order_units and shares for both windows, with the doc's post-processing.

Backtest (9 past windows from week 92 to 196 + window 183 with an injected 8-week Kestria ban;
score = 0.40 F + 0.21 fill + 0.09 diversification, max 0.70):
    naive 0.175 mean / 0.115 worst | quick LP 0.440 / 0.290 |
    plan A (stochastic, tuned: omega 0.05, s_min 0.1, lambda 5, 400 futures) ~0.515 / ~0.415 |
    plan B (aggressive) 0.482 / 0.300 but best in big-shock windows (fill 100%) | hindsight 0.656
  A wins calm windows (cost ~25-30% below naive), B wins big shocks. Submit both as the two
  final picks: Kaggle keeps the better one.

Quick use:
    import sys; sys.path += [MODEL1_DIR, MODEL2_DIR]
    import model1, model2, model3
    od = model3.OrderData(model1.load_data(DATA))
    res = model3.backtest(od)                   # naive vs quick vs stochastic on past windows
    print(res.summary())
    sub_A = model3.fill_submission(od, "submission.csv", out_path="submission.csv")            # plan A
    sub_B = model3.fill_submission(od, "submission.csv", model3.AGGRESSIVE, out_path="submission_B.csv")

Assumptions to confirm against the official rules (all are Config fields):
  start_stock_mode ("sum_first5" or "mean5"), avg_cost_mode, how F / fill rate /
  diversification are aggregated (see score_window()).
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.optimize import linprog

try:
    import model1
    import model2
except ImportError as exc:  # pragma: no cover
    raise ImportError("model3 needs model1.py and model2.py: add their folders to sys.path first") from exc

__all__ = ["Config", "AGGRESSIVE", "OrderData", "simulate", "naive_orders", "solve_lp", "ScenarioGenerator",
           "plan_window", "score_window", "backtest", "tune", "fill_submission", "inject_shock"]

SHARE_COUNTRIES = ("KES", "VAL", "SUN", "ORI", "HAL", "AUR")
DEFAULT_WINDOWS = (92, 105, 118, 131, 144, 157, 170, 183, 196)   # past 13-week windows (start weeks)


@dataclass
class Config:
    # --- competition rules (simulator)
    hold_rate: float = 0.02
    stockout_mult: float = 5.0
    sellback_rate: float = 0.70
    start_weeks: int = 5
    start_stock_mode: str = "sum_first5"   # "sum_first5" = demand of the window's first 5 weeks, "mean5" = 5 x mean
    avg_cost_mode: str = "window_mean"     # average landed cost = mean over the material's 3 suppliers x window weeks
    window: int = 13
    recovery_weeks: int = 4
    w_fin: float = 0.40
    w_fill: float = 0.21
    w_div: float = 0.09
    # --- policy knobs (doc's tuning table)
    policy: str = "stochastic"             # "stochastic", "quick" or "naive"
    s_min: float = 0.10                    # minimum share per supplier (of ordered units)
    lam: float = 5.0                       # internal stockout penalty, x average landed cost
    omega: float = 0.05                    # total weight of stress futures
    prob_mult: float = 1.0                 # scales Model 1's onset probabilities up
    n_scenarios: int = 400                 # 200 was noticeably seed-dependent; 400 is stable
    seed: int = 0
    buffer_k: float = 3.0                  # quick LP: stock >= k * p_t * demand_t
    stress: tuple = ("merrow_8w", "cheapest_banned", "half_capacity")
    order_cap_mult: float = 1.0            # orders per supplier <= this x its max calm capacity
    model1_config: object = None
    model2_config: object = None


# plan B for the second final pick (doc: higher omega and probability multiplier, lambda 8, s_min 0.33)
AGGRESSIVE = Config(omega=0.15, s_min=0.33, lam=8.0, prob_mult=1.5)


# ----------------------------------------------------------------------------- data
class OrderData:
    """Actuals and forecasts' inputs on top of model2.PriceData / model1.Panels."""

    def __init__(self, data, pdata=None):
        self.data = data
        self.pdata = pdata or model2.PriceData(data)
        P = self.P = self.pdata.P
        sup = P.suppliers
        dem = data["demand"].pivot_table(index="material", columns="week", values="demand_units", aggfunc="first")
        dem = dem.reindex(index=P.materials)
        self.demand_first_week = int(dem.columns.min())
        self.demand = dem.reindex(columns=np.arange(self.demand_first_week, int(dem.columns.max()) + 1)).values.astype(float)
        self.last_demand_week = int(dem.columns.max())
        self.normal = sup.normal_capacity_units.values.astype(float)
        self.base_lead = sup.base_lead_weeks.values.astype(int)
        self.cap = np.nan_to_num(P.cap_ratio, nan=1.0) * self.normal[:, None]
        self.lead = (self.base_lead[:, None] + np.round(np.nan_to_num(P.lt_excess))).astype(int)
        self.cost = self.pdata.landed
        hit = P.sup_active | P.lane_closed[P.sup_lane]
        self._calm_ratio_all = np.where(hit, np.nan, np.nan_to_num(P.cap_ratio, nan=1.0))

    def calm_stats(self, origin):
        """Median and max calm capacity / normal per supplier, using weeks up to `origin` only."""
        r = self._calm_ratio_all[:, : self.P.idx(origin) + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, mx = np.nanmedian(r, axis=1), np.nanmax(r, axis=1)
        return np.nan_to_num(med, nan=0.985), np.nan_to_num(mx, nan=1.0)

    def d(self, mi, w0, T):
        i = w0 - self.demand_first_week
        return self.demand[mi, i: i + T]

    def actuals(self, mi, w0, T):
        """Actual capacity, lead time, landed cost [3, T] for the material's suppliers."""
        P = self.P
        units = P.mat_units[mi]
        if w0 + T - 1 > P.last_week:
            raise ValueError(f"no actuals after week {P.last_week}")
        sl = slice(P.idx(w0), P.idx(w0) + T)
        return self.cap[units, sl].copy(), self.lead[units, sl].copy(), self.cost[units, sl].copy()

    def shock_mask(self, mi, w0, T, recovery):
        """Weeks in the window that are disrupted or within `recovery` weeks after a disrupted week."""
        P = self.P
        lo = P.idx(w0) - recovery
        y = np.nan_to_num(P.disrupted[mi, max(0, lo): P.idx(w0) + T]).astype(bool)
        y = np.r_[np.zeros(max(0, -lo), bool), y]
        mask = np.zeros(T, bool)
        for k in range(T):
            mask[k] = y[k: k + recovery + 1].any()
        return mask

    def start_stock(self, mi, w0, cfg):
        T = cfg.window
        d = self.d(mi, w0, T)
        return float(d[: cfg.start_weeks].sum()) if cfg.start_stock_mode == "sum_first5" else float(cfg.start_weeks * d.mean())


# ----------------------------------------------------------------------------- simulator
def simulate(orders, cap, lead, cost, demand, start_stock, avg_cost, cfg):
    """Competition rules for one material and one window.
    orders/cap/lead/cost: [C, T] (orders = units ordered from each supplier each week)."""
    orders = np.maximum(np.asarray(orders, float), 0)
    C, T = orders.shape
    filled = np.minimum(orders, np.maximum(cap, 0))
    arrivals = np.zeros(T)
    in_transit = 0.0
    for c in range(C):
        for t in range(T):
            a = t + int(lead[c, t])
            if a < T:
                arrivals[a] += filled[c, t]
            else:
                in_transit += filled[c, t]
    stock, used, unmet, stock_path = start_stock, np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        stock += arrivals[t]
        used[t] = min(stock, demand[t])
        unmet[t] = demand[t] - used[t]
        stock -= used[t]
        stock_path[t] = stock
    purchase = float((filled * cost).sum())
    holding = cfg.hold_rate * avg_cost * stock_path.sum()
    stockout = cfg.stockout_mult * avg_cost * unmet.sum()
    sellback = cfg.sellback_rate * avg_cost * (stock + in_transit)
    return dict(total=purchase + holding + stockout - sellback, purchase=purchase, holding=holding,
                stockout=stockout, sellback=sellback, served=used, demand=np.asarray(demand, float),
                unmet_units=float(unmet.sum()), filled=filled, end_stock=stock, in_transit=in_transit)


def naive_orders(od, mi, w0, T):
    """Sample policy: order this week's demand, all from the largest-capacity supplier."""
    units = od.P.mat_units[mi]
    o = np.zeros((len(units), T))
    o[int(np.argmax(od.normal[units]))] = od.d(mi, w0, T)
    return o


# ----------------------------------------------------------------------------- LP
def solve_lp(demand, start_stock, cap, lead, cost, weights, o_ub, avg_cost, s_min, lam, cfg,
             buffer=None, return_value=False):
    """Two-stage LP for one material and one window.
    cap, lead, cost: [S, C, T]; weights [S]; o_ub [C]. Returns orders o [C, T]."""
    cap = np.minimum(np.maximum(np.asarray(cap, float), 0), o_ub[None, :, None])
    S, C, T = cap.shape
    w = np.asarray(weights, float) / np.sum(weights)
    h, pi, r = cfg.hold_rate * avg_cost, lam * avg_cost, cfg.sellback_rate * avg_cost
    nO = C * T
    n = nO + S * C * T + S * 2 * T
    # variable index helpers
    fo = lambda s: nO + s * C * T                     # noqa: E731  f block start
    io = lambda s: nO + S * C * T + s * 2 * T         # noqa: E731  I block start (u follows)
    cvec = np.zeros(n)
    cvec[:nO] = 1e-7 * avg_cost
    lb, ub = np.zeros(n), np.full(n, np.inf)
    ub[:nO] = np.repeat(o_ub, T)
    eq_r, eq_c, eq_v, beq = [], [], [], np.zeros(S * T)
    in_r, in_c, in_v, bin_ = [], [], [], []
    row = 0
    for s in range(S):
        f0, i0 = fo(s), io(s)
        u0 = i0 + T
        arrive = np.arange(T)[None, :] + lead[s].astype(int)            # [C, T]
        fc = w[s] * (cost[s] - r * (arrive >= T))
        cvec[f0: f0 + C * T] = fc.ravel()
        ub[f0: f0 + C * T] = cap[s].ravel()
        cvec[i0: i0 + T] = w[s] * h
        cvec[i0 + T - 1] += -w[s] * r
        cvec[u0: u0 + T] = w[s] * pi
        if buffer is not None:
            lb[i0: i0 + T] = buffer
        # stock balance: I_t - I_{t-1} - arrivals_t - u_t = -d_t   (I_{-1} = start stock)
        for t in range(T):
            rr = s * T + t
            eq_r += [rr, rr]
            eq_c += [i0 + t, u0 + t]
            eq_v += [1.0, -1.0]
            if t > 0:
                eq_r.append(rr); eq_c.append(i0 + t - 1); eq_v.append(-1.0)
            beq[rr] = -demand[t] + (start_stock if t == 0 else 0.0)
        cc, tt = np.nonzero(arrive < T)
        eq_r += list(s * T + arrive[cc, tt])
        eq_c += list(f0 + cc * T + tt)
        eq_v += [-1.0] * len(cc)
        # f <= o  and  o - f <= cap_ref - cap   (fills can't fall below what the simulator would fill)
        for c in range(C):
            for t in range(T):
                fi, oi = f0 + c * T + t, c * T + t
                in_r += [row, row]; in_c += [fi, oi]; in_v += [1.0, -1.0]; bin_.append(0.0); row += 1
                in_r += [row, row]; in_c += [oi, fi]; in_v += [1.0, -1.0]; bin_.append(o_ub[c] - cap[s, c, t]); row += 1
    if s_min > 0:
        for t in range(T):
            for c in range(C):
                for c2 in range(C):
                    in_r.append(row); in_c.append(c2 * T + t)
                    in_v.append(-(1 - s_min) if c2 == c else s_min)
                bin_.append(0.0)
                row += 1
    A_eq = sps.csr_matrix((eq_v, (eq_r, eq_c)), shape=(S * T, n))
    A_ub = sps.csr_matrix((in_v, (in_r, in_c)), shape=(row, n))
    args = dict(A_ub=A_ub, b_ub=np.array(bin_), A_eq=A_eq, b_eq=beq, bounds=np.column_stack([lb, ub]))
    res = linprog(cvec, method="highs-ipm", **args)
    if res.status != 0:
        res = linprog(cvec, method="highs", **args)
    if res.status != 0:
        if buffer is not None:   # buffer infeasible (capacity too low): drop it
            return solve_lp(demand, start_stock, cap, lead, cost, weights, o_ub, avg_cost, s_min, lam, cfg,
                            buffer=None, return_value=return_value)
        raise RuntimeError(f"LP failed: {res.message}")
    o = res.x[:nO].reshape(C, T)
    return (o, res.fun) if return_value else o


def _dedupe(cap, lead, cost, w):
    """Merge identical futures (same capacity, lead and cost for this material's suppliers), adding weights."""
    S = cap.shape[0]
    key = np.concatenate([np.round(cap.reshape(S, -1), 6), lead.reshape(S, -1).astype(float),
                          np.round(cost.reshape(S, -1), 6)], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    wsum = np.bincount(inv, weights=w, minlength=len(first))
    return cap[first], lead[first], cost[first], wsum


def postprocess(o):
    """Doc's post-processing: whole-number totals, shares from orders, even split when total is 0."""
    C, T = o.shape
    o = np.maximum(o, 0)
    total = np.round(o.sum(axis=0))
    shares = np.where(o.sum(axis=0) > 1e-9, o / np.maximum(o.sum(axis=0), 1e-12), 1.0 / C)
    shares = shares / shares.sum(axis=0, keepdims=True)
    return total, shares, shares * total[None, :]


# ----------------------------------------------------------------------------- scenarios
class ScenarioGenerator:
    """Futures for every supplier over weeks origin+1 .. origin+H, shared by all materials."""

    def __init__(self, od, m1, pm, H, cfg):
        self.od, self.m1, self.pm, self.H, self.cfg = od, m1, pm, H, cfg
        P = od.P
        self.t0 = P.idx(m1.origin)
        self.src = m1._source_probs(H)
        self._profiles()

    # -- effect sizes from past episodes (up to the origin)
    def _profiles(self):
        od, P, t0 = self.od, self.od.P, self.t0
        lb = np.log(od.pdata.bench)
        prof = {"supplier": [], "country": [], "lane": []}
        for u, s, e, _ in model1._episodes(P.sup_active[:, : t0 + 1]):
            typ = model1._sup_onset_type(P, u, s)
            mi = P.sup_mat[u]
            jump = lb[mi, s] - np.nanmean(lb[mi, max(0, s - 4): s]) if s >= 1 else 0.2
            kappa = float(np.nanmedian(P.cap_ratio[u, s: e + 1]))
            prof[typ].append((kappa, 0, float(jump)))
        for l, s, e, _ in model1._episodes(P.lane_closed[:, : t0 + 1]):
            units = np.where(P.sup_lane == l)[0]
            kappa = float(np.nanmedian(P.cap_ratio[units, s: e + 1]))
            delta = int(np.round(np.nanmedian(P.lt_excess[units, s: e + 1])))
            ratio = float(np.nanmedian(od.pdata.container[l, s: e + 1]) / self.pm.normal_container[l])
            prof["lane"].append((kappa, delta, ratio))
        defaults = {"supplier": [(0.3, 0, 0.2)], "country": [(0.2, 0, 0.1)], "lane": [(0.75, 3, 2.5)]}
        self.profiles = {k: (v if v else defaults[k]) for k, v in prof.items()}

    @staticmethod
    def _draw_onset(q, rng, S, mult):
        """First-onset week index (0-based, H = none) for S futures from per-week probabilities q [H]."""
        q = np.clip(np.asarray(q) * mult, 0, None)
        cum = np.cumsum(q)
        if cum[-1] > 1:
            cum = cum / cum[-1]
        u = rng.random(S)
        return np.searchsorted(cum, u, side="left")    # == len(q) when no onset

    def _draw_length(self, typ, rng, S, age=None, xmax=400):
        x = np.arange(1, xmax + 1)
        Sx = self.m1.duration.S(typ, x)
        u = rng.random(S)
        if age is None:
            return (Sx[None, :] >= u[:, None]).sum(1)                      # total length
        cond = self.m1.duration.S(typ, age + x) / max(self.m1.duration.S(typ, age), 1e-12)
        return (cond[None, :] >= u[:, None]).sum(1)                        # remaining weeks after origin

    def _pick(self, typ, rng, S):
        p = self.profiles[typ]
        return [p[i] for i in rng.integers(0, len(p), S)]

    def generate(self, S, seed=0):
        """Returns dict of arrays over (S, U, H): kappa, delta, spot_mult, tariff; container (S, L, H)."""
        od, P, H, cfg, t0, src = self.od, self.od.P, self.H, self.cfg, self.t0, self.src
        rng = np.random.default_rng(seed)
        U, L, Cn = len(P.suppliers), len(P.lanes), len(P.countries)
        kappa = np.ones((S, U, H))
        delta = np.zeros((S, U, H), dtype=int)
        jump = np.zeros((S, len(P.materials), H))
        tariff = np.repeat(P.base_tariff[None, :, None], S, 0).repeat(H, 2).astype(float)
        container = np.repeat(self.pm.normal_container[None, :, None], S, 0).repeat(H, 2).astype(float)
        weeks = np.arange(H)

        def window_mask(start, length):
            return (weeks[None, :] >= start[:, None]) & (weeks[None, :] < (start + length)[:, None])

        def run_start(row):
            s = t0
            while s > 0 and row[s - 1]:
                s -= 1
            return s

        # lanes
        for l in range(L):
            if P.lane_closed[l, t0]:
                age = t0 - run_start(P.lane_closed[l]) + 1
                start, length = np.zeros(S, int), self._draw_length("lane", rng, S, age)
                cur_ratio = od.pdata.container[l, t0] / self.pm.normal_container[l]
                prof = [(np.nanmedian(P.cap_ratio[P.sup_lane == l, t0]), int(np.nanmedian(P.lt_excess[P.sup_lane == l, t0])),
                         cur_ratio)] * S
            else:
                start = self._draw_onset(src["q_lane"][l], rng, S, cfg.prob_mult)
                length = self._draw_length("lane", rng, S)
                prof = self._pick("lane", rng, S)
            m = window_mask(start, length)
            k_ = np.array([p[0] for p in prof])[:, None]
            d_ = np.array([p[1] for p in prof])[:, None]
            c_ = np.array([p[2] for p in prof])[:, None] * self.pm.normal_container[l]
            for u in np.where(P.sup_lane == l)[0]:
                kappa[:, u] = np.where(m, np.minimum(kappa[:, u], k_), kappa[:, u])
                delta[:, u] = np.where(m, np.maximum(delta[:, u], d_), delta[:, u])
            container[:, l] = np.where(m, np.maximum(container[:, l], c_), container[:, l])
        # countries
        lb = np.log(od.pdata.bench)
        for ci in range(Cn):
            units = np.where(P.sup_country == ci)[0]
            if P.country_active[ci, t0]:
                age = t0 - run_start(P.country_active[ci]) + 1
                start, length = np.zeros(S, int), self._draw_length("country", rng, S, age)
                prof = [(float(np.nanmedian(P.cap_ratio[units, t0])), 0, None)] * S
            else:
                start = self._draw_onset(src["q_cty"][ci], rng, S, cfg.prob_mult)
                length = self._draw_length("country", rng, S)
                prof = self._pick("country", rng, S)
            m = window_mask(start, length)
            k_ = np.array([p[0] for p in prof])[:, None]
            for u in units:
                kappa[:, u] = np.where(m, np.minimum(kappa[:, u], k_), kappa[:, u])
                mi = P.sup_mat[u]
                if prof[0][2] is None:     # active now: keep today's premium
                    o_ = self.od.pdata.shock_start(mi, t0) if self.od.pdata.mat_shock[mi, t0] else t0
                    j_ = np.full((S, 1), lb[mi, t0] - np.nanmean(lb[mi, max(0, o_ - 4): o_ or 1]))
                else:
                    j_ = np.array([p[2] for p in prof])[:, None]
                jump[:, mi] = np.where(m, np.maximum(jump[:, mi], j_), jump[:, mi])
        # single-supplier shocks
        for u in range(U):
            mi = P.sup_mat[u]
            if P.sup_active[u, t0] and not P.sup_in_country_event[u, t0]:
                age = t0 - run_start(P.sup_active[u]) + 1
                start, length = np.zeros(S, int), self._draw_length("supplier", rng, S, age)
                o_ = self.od.pdata.shock_start(mi, t0)
                cur_jump = lb[mi, t0] - np.nanmean(lb[mi, max(0, o_ - 4): o_ or 1])
                prof = [(float(P.cap_ratio[u, t0]), 0, cur_jump)] * S
            elif P.sup_in_country_event[u, t0]:
                continue
            else:
                start = self._draw_onset(src["q_sup"][u], rng, S, cfg.prob_mult)
                length = self._draw_length("supplier", rng, S)
                prof = self._pick("supplier", rng, S)
            m = window_mask(start, length)
            k_ = np.array([p[0] for p in prof])[:, None]
            j_ = np.array([p[2] for p in prof])[:, None]
            kappa[:, u] = np.where(m, np.minimum(kappa[:, u], k_), kappa[:, u])
            jump[:, mi] = np.where(m, np.maximum(jump[:, mi], j_), jump[:, mi])
        # tariffs (active now or announced)
        for u in range(U):
            if P.tar_active[u, t0]:
                age = t0 - run_start(P.tar_active[u]) + 1
                length = self._draw_length("tariff", rng, S, age)
                m = window_mask(np.zeros(S, int), length)
                tariff[:, u] = np.where(m, P.tariff_rate[u, t0], tariff[:, u])
            elif P.pending[u, t0]:
                k_eff = int(P.ann_eff[u, t0]) - self.m1.origin - 1
                happens = rng.random(S) < self.m1.cfg.p_announced
                length = self._draw_length("tariff", rng, S)
                m = window_mask(np.full(S, k_eff), length) & happens[:, None]
                tariff[:, u] = np.where(m, P.ann_rate[u, t0], tariff[:, u])
        return dict(kappa=kappa, delta=delta, jump=jump, tariff=tariff, container=container,
                    weights=np.full(S, 1.0 / S))

    def stress(self, kinds):
        """Stress futures bigger than anything in training (each one array set)."""
        P, H = self.od.P, self.H
        U = len(P.suppliers)
        base_lc = self.pm_landed_calm_mean
        out = []
        for kind in kinds:
            kappa = np.ones((1, U, H))
            delta = np.zeros((1, U, H), dtype=int)
            jump = np.zeros((1, len(P.materials), H))
            tariff = P.base_tariff[None, :, None].repeat(H, 2).astype(float)
            container = self.pm.normal_container[None, :, None].repeat(H, 2).astype(float)
            if kind == "merrow_8w":
                l = P.lanes.index("MERROW") if "MERROW" in P.lanes else 0
                wk = slice(2, min(H, 10))
                for u in np.where(P.sup_lane == l)[0]:
                    kappa[0, u, wk] = 0.5
                    delta[0, u, wk] = 4
                container[0, l, wk] = self.pm.normal_container[l] * 3.0
            elif kind == "cheapest_banned":
                for mi, units in enumerate(P.mat_units):
                    u = units[int(np.argmin(base_lc[units]))]
                    kappa[0, u, :] = 0.0
                    jump[0, mi, :] = 0.25
            elif kind == "half_capacity":
                kappa[:] = 0.5
                jump[:] = 0.25
            elif kind == "kestria_ban":
                ci = P.countries.index("KES") if "KES" in P.countries else 0
                for u in np.where(P.sup_country == ci)[0]:
                    kappa[0, u, 1:] = 0.0
                    jump[0, P.sup_mat[u], 1:] = 0.2
            else:
                raise ValueError(f"unknown stress scenario {kind}")
            out.append(dict(kappa=kappa, delta=delta, jump=jump, tariff=tariff, container=container))
        return out


# ----------------------------------------------------------------------------- planning
@dataclass
class WindowPlan:
    origin: int
    start: int
    orders: dict            # material -> o [3, T] after post-processing
    totals: dict
    shares: dict
    info: dict = field(default_factory=dict)


def _material_inputs(od, pm, sup_df, mi, start, T):
    """Calm / expected landed-cost forecasts [3, T] for the window weeks, plus cost components."""
    P = od.P
    units = P.mat_units[mi]
    rows = sup_df[(sup_df.material == P.materials[mi]) & sup_df.week.between(start, start + T - 1)]
    piv = lambda col: rows.pivot(index="country", columns="week", values=col).reindex(P.suppliers.country.values[units]).values  # noqa: E731
    return dict(expected=piv("landed_cost_expected"), calm=piv("landed_cost_calm"), spot_calm=piv("spot_calm"))


def plan_window(od, start, cfg=None, origin=None, m1=None, pm=None, verbose=False):
    """Orders for all materials for the 13-week window starting at `start`, using data up to
    `origin` (default: start - 1, capped at the last data week)."""
    cfg = cfg or Config()
    P = od.P
    T = cfg.window
    origin = int(origin if origin is not None else min(start - 1, P.last_week))
    off = start - origin - 1                      # weeks between origin and window start
    H = off + T
    m1 = m1 or model1.DisruptionModel(cfg.model1_config).fit(P, origin=origin)
    pm = pm or model2.PriceModel(cfg.model2_config).fit(od.pdata, origin=origin, m1=m1)
    _, sup_df = pm.predict(H)
    plan = WindowPlan(origin=origin, start=start, orders={}, totals={}, shares={})
    t_start = time.time()

    if cfg.policy == "stochastic":
        gen = ScenarioGenerator(od, m1, pm, H, cfg)
        gen.pm_landed_calm_mean = sup_df.groupby(["material", "country"], sort=False).landed_cost_calm.mean() \
            .reindex(list(zip(P.suppliers.material, P.suppliers.country))).values
        sc = gen.generate(cfg.n_scenarios, seed=cfg.seed)
        stress = gen.stress(cfg.stress) if cfg.omega > 0 and cfg.stress else []
    if cfg.policy == "quick":
        dp = m1.predict(H, raw=True)

    calm_med, calm_max = od.calm_stats(origin)
    for mi, m in enumerate(P.materials):
        units = P.mat_units[mi]
        d = od.d(mi, start, T)
        s0 = od.start_stock(mi, start, cfg)
        inp = _material_inputs(od, pm, sup_df, mi, start, T)
        avg = float(np.nanmean(inp["expected"]))
        o_ub = od.normal[units] * calm_max[units] * cfg.order_cap_mult
        base_cap = (od.normal[units] * calm_med[units])[:, None].repeat(T, 1)
        base_lead = od.base_lead[units][:, None].repeat(T, 1)
        if cfg.policy == "naive":
            o = naive_orders(od, mi, start, T)
        elif cfg.policy == "quick":
            p = dp[dp.material == m].set_index("week").p_raw.reindex(range(start, start + T)).values
            buf = np.minimum(cfg.buffer_k * np.clip(p * cfg.prob_mult, 0, 1) * d, 0.9 * s0 + d.sum())
            o = solve_lp(d, s0, base_cap[None], base_lead[None], inp["expected"][None], [1.0], o_ub, avg,
                         cfg.s_min, cfg.lam, cfg, buffer=buf)
        else:
            sl = slice(off, off + T)
            fut = [(sc["kappa"][:, units, sl], sc["delta"][:, units, sl], sc["jump"][:, mi, sl],
                    sc["tariff"][:, units, sl], sc["container"][:, P.sup_lane[units], sl], sc["weights"] * (1 - cfg.omega))]
            for st in stress:
                fut.append((st["kappa"][:, units, sl], st["delta"][:, units, sl], st["jump"][:, mi, sl],
                            st["tariff"][:, units, sl], st["container"][:, P.sup_lane[units], sl],
                            np.array([cfg.omega / len(stress)])))
            caps, leads, costs, ws = [], [], [], []
            b, c0 = pm.land_b[units][:, None], pm.land_c[units][:, None]
            for k_, d_, j_, t_, c_, w_ in fut:
                caps.append(base_cap[None] * k_)
                leads.append(base_lead[None] + d_)
                spot = inp["spot_calm"][None] * np.exp(j_)[:, None, :]
                costs.append(spot * (1 + t_) + b[None] * c_ + c0[None])
                ws.append(w_)
            caps, leads, costs, ws = _dedupe(np.concatenate(caps), np.concatenate(leads),
                                             np.concatenate(costs), np.concatenate(ws))
            plan.info.setdefault("unique_futures", {})[m] = len(ws)
            o = solve_lp(d, s0, caps, leads, costs, ws, o_ub, avg, cfg.s_min, cfg.lam, cfg)
        total, shares, o_pp = postprocess(o)
        plan.orders[m], plan.totals[m], plan.shares[m] = o_pp, total, shares
    plan.info["seconds"] = time.time() - t_start
    plan.info["m1"], plan.info["pm"] = m1, pm
    if verbose:
        print(f"window {start}: planned in {plan.info['seconds']:.1f}s")
    return plan


# ----------------------------------------------------------------------------- scoring
def hindsight_orders(od, mi, start, cfg):
    T = cfg.window
    cap, lead, cost = od.actuals(mi, start, T)
    units = od.P.mat_units[mi]
    o_ub = np.maximum(od.normal[units] * od.calm_stats(start + T - 1)[1][units], cap.max(axis=1))
    avg = _avg_cost(cost, cfg)
    return solve_lp(od.d(mi, start, T), od.start_stock(mi, start, cfg), cap[None], lead[None], cost[None], [1.0],
                    o_ub, avg, 0.0, cfg.stockout_mult, cfg)


def _avg_cost(actual_cost, cfg):
    return float(np.nanmean(actual_cost))


def diversification(shares):
    """1 - HHI rescaled so single sourcing = 0 and an even split over 3 suppliers = 1 (per week)."""
    C = shares.shape[0]
    hhi = (shares ** 2).sum(axis=0)
    return (1 - hhi) / (1 - 1 / C)


def score_window(od, start, plans, cfg=None):
    """Simulate each plan (dict name -> WindowPlan) plus naive and hindsight against actuals.
    F = (naive - ours) / (naive - hindsight) on total cost over all materials, clipped to [0, 1];
    fill = mean over materials with shock weeks of (served / demand) in shock + recovery weeks
    (all weeks if the window has no shock); diversification = mean over rows."""
    cfg = cfg or Config()
    P, T = od.P, cfg.window
    rows = []
    for mi, m in enumerate(P.materials):
        cap, lead, cost = od.actuals(mi, start, T)
        d = od.d(mi, start, T)
        s0 = od.start_stock(mi, start, cfg)
        avg = _avg_cost(cost, cfg)
        mask = od.shock_mask(mi, start, T, cfg.recovery_weeks)
        pols = {"naive": naive_orders(od, mi, start, T), "hindsight": postprocess(hindsight_orders(od, mi, start, cfg))[2]}
        pols.update({name: p.orders[m] for name, p in plans.items()})
        for name, o in pols.items():
            sim = simulate(o, cap, lead, cost, d, s0, avg, cfg)
            tot = o.sum(axis=0)
            sh = np.where(tot > 0, o / np.maximum(tot, 1e-12), 1 / o.shape[0])
            rows.append(dict(window=start, material=m, policy=name, cost=sim["total"], purchase=sim["purchase"],
                             holding=sim["holding"], stockout=sim["stockout"], sellback=sim["sellback"],
                             unmet=sim["unmet_units"], fill_all=sim["served"].sum() / d.sum(),
                             fill_shock=(sim["served"][mask].sum() / d[mask].sum()) if mask.any() else np.nan,
                             has_shock=bool(mask.any()), divers=float(diversification(sh).mean())))
    return pd.DataFrame(rows)


def _summarize(df, cfg):
    out = []
    for (w, pol), g in df.groupby(["window", "policy"]):
        nv = df[(df.window == w) & (df.policy == "naive")].cost.sum()
        hs = df[(df.window == w) & (df.policy == "hindsight")].cost.sum()
        F = np.clip((nv - g.cost.sum()) / (nv - hs), 0, 1) if nv > hs else np.nan
        fill = g.fill_shock.mean() if g.has_shock.any() else g.fill_all.mean()
        div = g.divers.mean()
        out.append(dict(window=w, policy=pol, F=F, fill=fill, divers=div,
                        score=cfg.w_fin * F + cfg.w_fill * fill + cfg.w_div * div,
                        cost=g.cost.sum(), unmet=g.unmet.sum(), cost_vs_naive=g.cost.sum() / nv - 1))
    return pd.DataFrame(out)


@dataclass
class Model3Backtest:
    detail: pd.DataFrame
    windows: pd.DataFrame
    config: Config

    def summary(self):
        """Mean and worst window per policy (score out of 0.70 = this model's share of the total)."""
        g = self.windows.groupby("policy")
        return pd.DataFrame({"score_mean": g.score.mean(), "score_worst": g.score.min(), "F_mean": g.F.mean(),
                             "fill_mean": g.fill.mean(), "div_mean": g.divers.mean(),
                             "cost_vs_naive": g.cost_vs_naive.mean(), "unmet_units": g.unmet.sum()}).sort_values("score_mean", ascending=False)


def inject_shock(od, kind="kestria_ban", start=None, length=8, window_start=183, cfg=None):
    """A copy of `od` whose actuals contain a big shock inside a past window (for stress-testing
    plans): 'kestria_ban' (capacity 0, prices +25%) or 'merrow_closure' (capacity 50%, lead +4,
    container x3). The forecasts (fit at the origin) don't see it."""
    import copy
    cfg = cfg or Config()
    new = copy.copy(od)
    new.cap, new.lead, new.cost = od.cap.copy(), od.lead.copy(), od.cost.copy()
    P = od.P
    s = P.idx(window_start) + (start if start is not None else 3)
    sl = slice(s, s + length)
    if kind == "kestria_ban":
        units = np.where(P.suppliers.country.values == "KES")[0]
        new.cap[units, sl] = 0.0
        for mi in np.unique(P.sup_mat[units]):
            new.cost[P.mat_units[mi], sl] *= 1.25
    elif kind == "merrow_closure":
        units = np.where(P.suppliers.lane.values == "MERROW")[0]
        new.cap[units, sl] *= 0.5
        new.lead[units, sl] += 4
        new.cost[units, sl] *= 1.15
    else:
        raise ValueError(kind)
    # mark the shock weeks as disrupted for the fill-rate score
    P2 = copy.copy(P)
    P2.disrupted = P.disrupted.copy()
    for mi in np.unique(P.sup_mat[units]):
        P2.disrupted[mi, sl] = 1.0
    new.P = P2
    return new


def backtest(od, windows=DEFAULT_WINDOWS, configs=None, cfg=None, inject=("kestria_ban",), verbose=False):
    """Score policies on past windows (and on an injected big shock). configs: dict name -> Config.
    Model 1 / Model 2 are fit once per window origin and shared by all configs."""
    cfg = cfg or Config()
    configs = configs or {"quick": replace(cfg, policy="quick"), "stochastic": replace(cfg, policy="stochastic")}
    tests = [(w, od, str(w)) for w in windows if w + cfg.window - 1 <= od.P.last_week]
    for kind in inject or ():
        w = 183 if 183 + cfg.window - 1 <= od.P.last_week else windows[-1]
        tests.append((w, inject_shock(od, kind, window_start=w, cfg=cfg), f"{w}+{kind}"))
    details = []
    fits = {}
    for w, od_t, label in tests:
        if w not in fits:
            m1 = model1.DisruptionModel(cfg.model1_config).fit(od.P, origin=w - 1)
            pm = model2.PriceModel(cfg.model2_config).fit(od.pdata, origin=w - 1, m1=m1)
            fits[w] = (m1, pm)
        m1, pm = fits[w]
        plans = {name: plan_window(od, w, c, m1=m1, pm=pm) for name, c in configs.items()}
        det = score_window(od_t, w, plans, cfg)
        det["window"] = label
        details.append(det)
        if verbose:
            print(f"window {label} done")
    detail = pd.concat(details, ignore_index=True)
    return Model3Backtest(detail=detail, windows=_summarize(detail, cfg), config=cfg)


def tune(od, grid, windows=DEFAULT_WINDOWS, base=None, inject=("kestria_ban",), verbose=False):
    """grid: dict of Config field -> list of values (stochastic policy). Returns one row per setting,
    sorted by the worst window's score (the doc: robustness beats the best average)."""
    base = base or Config()
    import itertools
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    configs = {", ".join(f"{k}={v}" for k, v in c.items()): replace(base, policy="stochastic", **c) for c in combos}
    configs["quick (reference)"] = replace(base, policy="quick")
    bt = backtest(od, windows, configs, base, inject=inject, verbose=verbose)
    return bt.summary().sort_values("score_worst", ascending=False), bt


# ----------------------------------------------------------------------------- submission
def fill_submission(od, sample, cfg=None, windows=((209, 221), (222, 234)), out_path=None, verbose=False):
    """Fill order_units and share_* for both windows (each planned from min(start - 1, last week))."""
    cfg = cfg or Config()
    P = od.P
    sub = pd.read_csv(sample) if isinstance(sample, str) else sample.copy()
    parts = sub.id.str.extract(r"^w(\d+)_(.+)$")
    wk, mat = parts[0].astype(int), parts[1]
    share_cols = [f"share_{c}" for c in SHARE_COUNTRIES]
    for start, end in windows:
        if start - 1 > P.last_week:
            warnings.warn(f"window {start}-{end} is planned from week {P.last_week} (data ends there). "
                          "Re-run after Phase 2.")
        plan = plan_window(od, start, cfg, verbose=verbose)
        for mi, m in enumerate(P.materials):
            units = P.mat_units[mi]
            countries = P.suppliers.country.values[units]
            for k in range(cfg.window):
                idx = sub.index[(wk == start + k) & (mat == m)]
                if len(idx) != 1:
                    raise ValueError(f"expected one row for w{start + k}_{m}")
                i = idx[0]
                sub.loc[i, "order_units"] = int(plan.totals[m][k])
                sub.loc[i, share_cols] = 0.0
                for j, c in enumerate(countries):
                    sub.loc[i, f"share_{c}"] = float(plan.shares[m][j, k])
    sub["order_units"] = sub.order_units.astype(int)
    sub[share_cols] = sub[share_cols].div(sub[share_cols].sum(axis=1), axis=0)
    if out_path:
        sub.to_csv(out_path, index=False)
    return sub


def _main():
    ap = argparse.ArgumentParser(description="Model 3 ordering plan")
    ap.add_argument("--data", required=True)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--submission")
    ap.add_argument("--out", default="submission_model3.csv")
    args = ap.parse_args()
    od = OrderData(model1.load_data(args.data))
    if args.backtest:
        print(backtest(od, verbose=True).summary().round(3).to_string())
    if args.submission:
        fill_submission(od, args.submission, out_path=args.out, verbose=True)
        print("wrote", args.out)


if __name__ == "__main__":
    _main()
