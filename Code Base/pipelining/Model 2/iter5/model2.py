"""
model2.py - Model 2: price forecast + landed costs.

iter5 = iter4 + Config(jump_by_source=True): the onset jump is the median of single-supplier onsets (0.201 log
up to week 221) or of country-event onsets (0.094), weighted per week by which source drives P(price shock),
instead of the pooled median (0.179, pulled down by the 7 country onsets at week 99). Leave-one-event-out on
the 8 single-supplier onsets: mean |error| 0.051 vs 0.069 (diagnostics/t2_jump). Default False = iter4.
Works with Model 1 iter4 or iter5; with iter5 it reads the plan p_sup unless Config(cal_onset=True).

Fills `price_forecast` (benchmark = mean spot of the material's 3 suppliers) and produces
per-supplier landed-cost forecasts for Model 3. Needs model1.py importable (it supplies the
shock states and probabilities).

What the data says (weeks 1-208), and what the model does about it:
  * Calm weeks: spot is close to a random walk (weekly moves 3-5%, no autocorrelation) with
    a mild pull back to its recent calm average. Futures carry NO information in calm weeks
    (futures_check(): futures lose to carry-forward when |premium| < 3%), so the doc's
    "follow the futures" baseline scores below carry-forward.
      -> calm forecast = carry-forward + drift - beta_h * (deviation from 26-week calm mean),
         beta_h and drift fitted per horizon by least absolute error.
  * Supplier-level shocks (capacity collapse or country ban): ALL suppliers of the material
    jump together, +15-35% in the onset week, stay up while the shock lasts, and drop back
    when it ends. Lane closures and tariffs barely move spot (they hit landed cost instead).
      -> each forecast is the median of a two-scenario mix (shock on / shock off, weighted
         by Model 1's probability) plus normal price noise. Close to the doc's 0.5 rule when
         the scenarios are far apart, but no all-or-nothing flips when they're close.
         In a shock now: scenarios are "today's level" vs "pre-shock level".
         Calm now: scenarios are "calm forecast" vs "calm forecast + typical jump", jumps
         only up to 4 weeks ahead (Model 1's signals don't reach further), unless an onset is
         already likely within those 4 weeks: then the jump is held while p > 0.5 (iter4).
  * Landed cost = spot x (1 + tariff) + shipping, where shipping = b x container_rate + c per
    supplier (b = 0 for suppliers on NORTH, SOUTH, RAIL). Fit error ~0.06% (max ~0.9%).
      -> landed_cost_expected: probability-weighted (Model 1) spot, tariff, container. Use in
         the quick deterministic LP.
      -> landed_cost_calm: what it would cost with no shock at all (calm spot level, base
         tariff, normal container). Use in the scenario LP, which applies its own shock
         multipliers; using the expected column there would count shocks twice.

Backtest (26 origins every 4 weeks from 104, h = 1..13), MAPE skill vs carry-forward:
  v1 futures-implied -0.007 | v2 mean reversion 0.022 | v3 + shock reversion 0.093 |
  v4 + jump rule 0.096 (doc's hard 0.5 switch instead: 0.046). Landed cost MAPE 5.7% vs 6.4%
  carry-forward. Run ablation(pdata) to reproduce.

Quick use:
    import sys; sys.path.append("<folder with model1.py>")
    import model1, model2
    data = model1.load_data(DATA)
    pdata = model2.PriceData(data)                     # builds model1.Panels internally
    pm = model2.PriceModel().fit(pdata)                # origin = last week in the data
    bench, sup = pm.predict()                          # benchmark forecast, supplier-level detail
    bt = model2.backtest(pdata); print(bt.summary())   # MAPE skill vs carry-forward
    print(model2.ablation(pdata))                      # v0 carry ... v4 + jump rule
    sub = model2.fill_submission(pdata, "submission.csv", out_path="submission.csv")
    lc = model2.export_for_model3(pdata, out_path="landed_costs.csv")
    model2.futures_check(pdata); model2.landed_fit_report(pdata)   # diagnostics

Requires: numpy, pandas, scipy, scikit-learn (through model1).
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

try:
    import model1
except ImportError as exc:  # pragma: no cover
    raise ImportError("model2 needs model1.py: add the folder that contains it to sys.path "
                      "before importing model2") from exc

__all__ = ["Config", "PriceData", "PriceModel", "PriceBacktest", "backtest", "ablation",
           "fill_submission", "export_for_model3", "futures_check", "landed_fit_report"]

DOC_ORIGINS = (156, 169, 182, 195)


@dataclass
class Config:
    horizon: int = 13
    use_mean_reversion: bool = True     # calm: drift + pull toward the 26-week calm mean
    use_futures: bool = False           # calm: move toward futures_4w (doc baseline; off: no signal in calm)
    use_shock_state: bool = True        # in a shock: keep the premium while P(continue) > threshold
    use_jump: bool = True               # calm: add a jump where P(price-moving shock) > threshold
    jump_max_h: int = 4                 # only add jumps for coming shocks up to this many weeks ahead
                                        # (Model 1's warning signals don't reach further than ~4 weeks)
    jump_hold: bool = True              # iter4: if an onset is likely within jump_max_h (p > threshold),
                                        # keep the jump beyond it while p stays > threshold (bans last 5-11 wk)
    threshold: float = 0.5
    jump_mode: str = "mixture"          # "mixture" = median of the shock / no-shock mix incl. price noise (default)
                                        # "median"  = doc's hard switch at threshold; "expected" = probability-weighted
    ref_window: int = 4                 # weeks before a shock onset that define the pre-shock price
    calm_window: int = 26               # trailing weeks for the calm anchor
    min_calm_weeks: int = 8
    beta_grid: tuple = tuple(np.round(np.arange(0.0, 0.81, 0.02), 2))
    min_train_week: int = 27
    default_jump: float = 0.20          # log jump used when no onset has been seen yet
    jump_by_source: bool = False        # iter5: separate single-supplier / country-event jump medians
    cal_onset: bool = False             # iter5: True = use Model 1 iter5's calibrated p_sup (premium_mix); False
                                        # = its plan p_sup (the mix cost 0.04 pt of premium-chain MAPE)
    model1_config: object = None        # passed to model1.DisruptionModel


# ----------------------------------------------------------------------------- data
class PriceData:
    """Dense (supplier x week) price arrays on top of model1.Panels."""

    def __init__(self, data, panels=None, news_features=None):
        self.data = data
        self.P = P = panels or model1.Panels(data, news_features=news_features)
        keys = pd.MultiIndex.from_tuples(list(zip(P.suppliers.material, P.suppliers.country)))

        def sup_arr(df, col):
            p = df.pivot_table(index=["material", "country"], columns="week", values=col, aggfunc="first")
            return p.reindex(index=keys, columns=P.weeks).values.astype(float)

        self.spot = sup_arr(data["prices"], "spot_price")
        self.fut = sup_arr(data["prices"], "futures_4w")
        self.landed = sup_arr(data["supply_history"], "landed_cost")
        self.logspot = np.log(self.spot)
        self.container = P.container
        M = len(P.materials)
        # material price-shock state: a supplier-level (capacity / country) shock on any of its suppliers
        self.mat_shock = np.zeros((M, P.W), dtype=bool)
        for mi, units in enumerate(P.mat_units):
            self.mat_shock[mi] = P.sup_active[units].any(axis=0)
        self.bench = np.vstack([np.nanmean(self.spot[u], axis=0) for u in P.mat_units])
        self.dev = None
        self._dev_params = None

    def deviation(self, window, min_weeks):
        """log spot minus its mean over the last `window` calm weeks (0 if too few calm weeks)."""
        if self._dev_params == (window, min_weeks):
            return self.dev
        P = self.P
        calm = ~self.mat_shock[P.sup_mat]
        ls = np.nan_to_num(self.logspot)
        cs = np.cumsum(np.where(calm, ls, 0.0), axis=1)
        cn = np.cumsum(calm, axis=1)
        cs = np.hstack([np.zeros((cs.shape[0], 1)), cs])
        cn = np.hstack([np.zeros((cn.shape[0], 1)), cn])
        W = ls.shape[1]
        hi = np.arange(1, W + 1)
        lo = np.maximum(hi - window, 0)
        s = cs[:, hi] - cs[:, lo]
        n = cn[:, hi] - cn[:, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            anchor = s / n
        dev = np.where(n >= min_weeks, ls - anchor, 0.0)
        self.dev, self._dev_params = np.nan_to_num(dev), (window, min_weeks)
        return self.dev

    def shock_start(self, mi, t):
        s = t
        while s > 0 and self.mat_shock[mi, s - 1]:
            s -= 1
        return s


# ----------------------------------------------------------------------------- model
class PriceModel:
    def __init__(self, config=None, **overrides):
        self.cfg = replace(config or Config(), **overrides)

    def fit(self, pdata, origin=None, m1=None):
        cfg, P = self.cfg, pdata.P
        self.pdata = pdata
        self.origin = int(origin if origin is not None else P.last_week)
        if self.origin > P.last_week:
            raise ValueError(f"origin {self.origin} is after the last week with data ({P.last_week})")
        self.t0 = t0 = P.idx(self.origin)
        self.m1 = m1 if m1 is not None else model1.DisruptionModel(cfg.model1_config).fit(P, origin=self.origin)
        self.dev = pdata.deviation(cfg.calm_window, cfg.min_calm_weeks)
        self._fit_calm(t0)
        self._fit_jump(t0)
        self._fit_landed(t0)
        return self

    # -- calm drift + mean reversion, fitted per horizon by least absolute error
    def _fit_calm(self, t0):
        cfg, pd_ = self.cfg, self.pdata
        P = pd_.P
        K = cfg.horizon
        self.beta = np.zeros(K + 1)
        self.drift = np.zeros(K + 1)
        self.sigma = 0.035 * np.sqrt(np.arange(K + 1))     # fallback noise scale (log price)
        t_min = max(0, P.idx(cfg.min_train_week))
        calm = ~pd_.mat_shock[P.sup_mat]
        ls = pd_.logspot
        betas = np.asarray(cfg.beta_grid)
        for h in range(1, K + 1):
            t_hi = t0 - h
            if t_hi <= t_min:
                continue
            sl = slice(t_min, t_hi + 1)
            ok = calm[:, sl] & calm[:, t_min + h: t_hi + h + 1]
            y = (ls[:, t_min + h: t_hi + h + 1] - ls[:, sl])[ok]
            x = self.dev[:, sl][ok]
            good = np.isfinite(y)
            y, x = y[good], x[good]
            if len(y) < 50:
                continue
            if cfg.use_mean_reversion:
                r = y[None, :] + betas[:, None] * x[None, :]
                d = np.median(r, axis=1)
                loss = np.abs(r - d[:, None]).mean(axis=1)
                i = int(np.argmin(loss))
                self.beta[h], self.drift[h] = betas[i], d[i]
            resid = y + self.beta[h] * x - self.drift[h]
            self.sigma[h] = 1.4826 * np.median(np.abs(resid - np.median(resid)))

    def _calm_params(self, h):
        K = self.cfg.horizon
        if h <= K:
            return self.beta[h], self.drift[h]
        return self.beta[K], self.drift[K] * h / K

    def _sigma(self, hs):
        K = self.cfg.horizon
        return np.array([self.sigma[h] if h <= K else self.sigma[K] * np.sqrt(h / K) for h in hs])

    @staticmethod
    def _mixture_median(a, b, p, sigma, iters=60):
        """Median of (1-p) N(a, s^2) + p N(b, s^2), elementwise, by bisection."""
        from scipy.stats import norm
        lo = np.minimum(a, b) - 6 * sigma
        hi = np.maximum(a, b) + 6 * sigma
        for _ in range(iters):
            mid = (lo + hi) / 2
            cdf = (1 - p) * norm.cdf((mid - a) / sigma) + p * norm.cdf((mid - b) / sigma)
            lo = np.where(cdf < 0.5, mid, lo)
            hi = np.where(cdf < 0.5, hi, mid)
        return (lo + hi) / 2

    # -- typical log jump of the benchmark in the onset week of a price-moving shock
    def _fit_jump(self, t0):
        cfg, pd_ = self.cfg, self.pdata
        r = cfg.ref_window
        lb = np.log(pd_.bench)
        jumps, is_cty = [], []
        for mi in range(lb.shape[0]):
            ms = pd_.mat_shock[mi, : t0 + 1]
            for o in np.where(ms[1:] & ~ms[:-1])[0] + 1:
                if o >= r:
                    jumps.append(lb[mi, o] - np.nanmean(lb[mi, o - r: o]))
                    is_cty.append(bool(pd_.P.sup_in_country_event[pd_.P.mat_units[mi], o].any()))
        self.jumps = np.array(jumps)
        self.jump = float(np.median(jumps)) if len(jumps) else cfg.default_jump
        c = np.array(is_cty, dtype=bool)
        self.jump_single = float(np.median(self.jumps[~c])) if (~c).any() else self.jump
        self.jump_cty = float(np.median(self.jumps[c])) if c.any() else self.jump

    # -- landed cost = spot * (1 + tariff) + b * container + c, per supplier
    def _fit_landed(self, t0):
        pd_, P = self.pdata, self.pdata.P
        U = len(P.suppliers)
        self.land_b, self.land_c = np.zeros(U), np.zeros(U)
        for u in range(U):
            cont = pd_.container[P.sup_lane[u], : t0 + 1]
            resid = pd_.landed[u, : t0 + 1] - pd_.spot[u, : t0 + 1] * (1 + P.tariff_rate[u, : t0 + 1])
            ok = np.isfinite(cont) & np.isfinite(resid)
            A = np.column_stack([cont[ok], np.ones(ok.sum())])
            (b, c), *_ = np.linalg.lstsq(A, resid[ok], rcond=None)
            self.land_b[u], self.land_c[u] = b, c
        L = len(P.lanes)
        self.normal_container = np.zeros(L)
        ratios = []
        for l in range(L):
            cont = pd_.container[l, : t0 + 1]
            closed = P.lane_closed[l, : t0 + 1]
            self.normal_container[l] = np.nanmedian(cont[~closed])
            if closed.any():
                ratios.extend(cont[closed] / self.normal_container[l])
        self.closure_ratio = float(np.median(ratios)) if ratios else 2.5

    # -- forecasting
    def predict(self, horizon=None):
        """Returns (benchmark_df, supplier_df) for weeks origin+1 .. origin+horizon."""
        cfg, pd_, P = self.cfg, self.pdata, self.pdata.P
        H = int(horizon or cfg.horizon)
        t0 = self.t0
        src = self.m1._source_probs(H)
        hs = np.arange(1, H + 1)
        thr = cfg.threshold

        # P(price-moving shock active at T+h) per material: supplier + country sources
        M = len(P.materials)
        p_ps = np.zeros((M, H))
        jump_mh = np.full((M, H), self.jump)
        p_sup = src["p_sup"] if cfg.cal_onset else src.get("p_sup_plan", src["p_sup"])
        for mi, units in enumerate(P.mat_units):
            ctys = np.unique(P.sup_country[units])
            ps = 1 - np.prod(1 - p_sup[units], axis=0)
            pc = 1 - np.prod(1 - src["p_cty"][ctys], axis=0)
            p_ps[mi] = 1 - (1 - ps) * (1 - pc)
            if cfg.jump_by_source:
                jump_mh[mi] = (ps * self.jump_single + pc * self.jump_cty + 1e-12 * self.jump_single) / (ps + pc + 1e-12)

        rows, brows = [], []
        for mi, (m, units) in enumerate(zip(P.materials, P.mat_units)):
            in_shock = bool(pd_.mat_shock[mi, t0])
            o = pd_.shock_start(mi, t0) if in_shock else None
            f_med = np.zeros((len(units), H))
            f_exp = np.zeros((len(units), H))
            f_calm = np.zeros((len(units), H))
            for j, u in enumerate(units):
                ls0 = pd_.logspot[u, t0]
                beta_drift = [self._calm_params(h) for h in hs]
                beta = np.array([b for b, _ in beta_drift])
                drift = np.array([d for _, d in beta_drift])
                base = ls0 + drift - beta * self.dev[u, t0]
                if cfg.use_futures and np.isfinite(pd_.fut[u, t0]):
                    base = base + np.log(pd_.fut[u, t0] / pd_.spot[u, t0]) * np.minimum(hs, 4) / 4
                p = p_ps[mi]
                sig = np.maximum(self._sigma(hs), 1e-4)
                if in_shock and cfg.use_shock_state:
                    r0 = max(0, o - cfg.ref_window)
                    ref = np.nanmean(pd_.logspot[u, r0:o]) if o > 0 else ls0
                    per_week = self.drift[cfg.horizon] / cfg.horizon if cfg.horizon else 0.0
                    noshock = ref + per_week * (t0 - o + hs)
                    shocked = np.full(H, ls0)
                    lo_lvl, hi_lvl = noshock, shocked
                    calm_lvl = noshock
                else:
                    on = hs <= cfg.jump_max_h
                    if cfg.jump_hold and (p[on] > thr).any():
                        on = on | (p > thr)
                    J = np.where(on, jump_mh[mi], 0.0) if cfg.use_jump else np.zeros(H)
                    lo_lvl, hi_lvl = base, base + J
                    calm_lvl = base
                exp_ = np.log((1 - p) * np.exp(lo_lvl) + p * np.exp(hi_lvl))
                if cfg.jump_mode == "expected":
                    med = exp_
                elif cfg.jump_mode == "median":
                    med = np.where(p > thr, hi_lvl, lo_lvl)
                else:
                    med = self._mixture_median(lo_lvl, hi_lvl, p, sig)
                f_med[j] = med
                f_exp[j], f_calm[j] = exp_, calm_lvl
            spot_med, spot_exp, spot_calm = np.exp(f_med), np.exp(f_exp), np.exp(f_calm)
            bench = spot_med.mean(axis=0)
            for h in hs:
                brows.append((self.origin, h, self.origin + h, m, bench[h - 1], p_ps[mi, h - 1], in_shock))
            for j, u in enumerate(units):
                rows.extend(self._landed_rows(u, m, src, hs, spot_med[j], spot_exp[j], spot_calm[j]))
        bench_df = pd.DataFrame(brows, columns=["origin", "h", "week", "material", "price_forecast",
                                                "p_price_shock", "in_shock_at_origin"])
        return bench_df, pd.DataFrame(rows)

    def _landed_rows(self, u, m, src, hs, s_med, s_exp, s_calm):
        pd_, P, t0 = self.pdata, self.pdata.P, self.t0
        # tariffs
        cur, base = P.tariff_rate[u, t0], P.base_tariff[u]
        if P.tar_active[u, t0]:
            rate_shock = cur
        elif P.pending[u, t0]:
            rate_shock = P.ann_rate[u, t0]
        else:
            rate_shock = base
        p_tar = src["p_tar"][u]
        t_exp = base + p_tar * (rate_shock - base)
        t_med = np.where(p_tar > self.cfg.threshold, rate_shock, base if P.tar_active[u, t0] else cur)
        # container rate of the supplier's lane
        l = P.sup_lane[u]
        c_now, normal = pd_.container[l, t0], self.normal_container[l]
        closed = P.lane_closed[l, t0]
        open_lvl = normal if closed else c_now
        if_closed = c_now if closed else normal * self.closure_ratio
        p_lane = src["p_lane"][l]
        c_exp = open_lvl + p_lane * (if_closed - open_lvl)
        c_med = np.where(p_lane > self.cfg.threshold, if_closed, open_lvl)
        b, c = self.land_b[u], self.land_c[u]
        lc_med = s_med * (1 + t_med) + b * c_med + c
        lc_exp = s_exp * (1 + t_exp) + b * c_exp + c
        lc_calm = s_calm * (1 + base) + b * open_lvl + c
        country, lane = P.suppliers.country[u], P.lanes[l]
        return [dict(origin=self.origin, week=self.origin + h, h=h, material=m, country=country, lane=lane,
                     spot_forecast=s_med[i], spot_expected=s_exp[i], spot_calm=s_calm[i],
                     tariff_forecast=t_med[i], tariff_expected=t_exp[i],
                     container_forecast=c_med[i], container_expected=c_exp[i],
                     landed_cost=lc_med[i], landed_cost_expected=lc_exp[i], landed_cost_calm=lc_calm[i])
                for i, h in enumerate(hs)]


# ----------------------------------------------------------------------------- backtest
def _mape(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return float(np.mean(np.abs(f - a) / a))


@dataclass
class PriceBacktest:
    rows: pd.DataFrame
    landed: pd.DataFrame
    config: Config = field(default_factory=Config)

    @staticmethod
    def _skill(d):
        cf = _mape(d.carry, d.actual)
        return 1 - _mape(d.price_forecast, d.actual) / cf if cf > 0 else np.nan

    def summary(self):
        r = self.rows
        hb = pd.cut(r.h, [0, 4, 8, 13, 10**6], labels=["h1-4", "h5-8", "h9-13", "h14+"])
        out = {"overall": self._skill(r),
               "calm_at_origin": self._skill(r[~r.in_shock_at_origin]),
               "in_shock_at_origin": self._skill(r[r.in_shock_at_origin]) if r.in_shock_at_origin.any() else np.nan}
        for b in hb.cat.categories:
            if (hb == b).any():
                out[b] = self._skill(r[hb == b])
        out["doc_origins"] = self._skill(r[r.origin.isin(DOC_ORIGINS)])
        out["mape"] = _mape(r.price_forecast, r.actual)
        out["mape_carry"] = _mape(r.carry, r.actual)
        if len(self.landed):
            L = self.landed
            out["landed_mape_expected"] = _mape(L.landed_cost_expected, L.actual)
            out["landed_mape_carry"] = _mape(L.carry, L.actual)
        out["n_rows"] = len(r)
        out["n_origins"] = r.origin.nunique()
        return pd.Series(out)

    def by(self, key):
        r = self.rows.assign(e=lambda d: (d.price_forecast - d.actual).abs() / d.actual,
                             ec=lambda d: (d.carry - d.actual).abs() / d.actual)
        g = r.groupby(key).agg(mape=("e", "mean"), mape_carry=("ec", "mean"), n=("e", "size"))
        g["skill"] = 1 - g.mape / g.mape_carry
        return g


def default_origins(pdata, start=104, step=4):
    return list(range(start, pdata.P.last_week, step))


def backtest(pdata, origins=None, config=None, m1_cache=None, verbose=False, **overrides):
    """Rolling-origin backtest. Every fit (Model 1 and Model 2) uses only data up to its origin.
    Benchmark actuals = mean spot of the 3 suppliers (identical to outcomes.benchmark_price)."""
    cfg = replace(config or Config(), **overrides)
    P = pdata.P
    origins = sorted(origins or default_origins(pdata))
    m1_cache = {} if m1_cache is None else m1_cache
    rows, lrows = [], []
    for T in origins:
        if T >= P.last_week:
            continue
        if T not in m1_cache:
            m1_cache[T] = model1.DisruptionModel(cfg.model1_config).fit(P, origin=T)
        pm = PriceModel(cfg).fit(pdata, origin=T, m1=m1_cache[T])
        H = min(cfg.horizon, P.last_week - T)
        bench, sup = pm.predict(H)
        t0 = P.idx(T)
        mi = {m: i for i, m in enumerate(P.materials)}
        bench["actual"] = [pdata.bench[mi[m], P.idx(w)] for m, w in zip(bench.material, bench.week)]
        bench["carry"] = [pdata.bench[mi[m], t0] for m in bench.material]
        rows.append(bench)
        ui = {(m, c): i for i, (m, c) in enumerate(zip(P.suppliers.material, P.suppliers.country))}
        sup["actual"] = [pdata.landed[ui[(m, c)], P.idx(w)] for m, c, w in zip(sup.material, sup.country, sup.week)]
        sup["carry"] = [pdata.landed[ui[(m, c)], t0] for m, c in zip(sup.material, sup.country)]
        lrows.append(sup)
        if verbose:
            print(f"origin {T} done")
    r = pd.concat(rows, ignore_index=True)
    lr = pd.concat(lrows, ignore_index=True)
    return PriceBacktest(rows=r[np.isfinite(r.actual)], landed=lr[np.isfinite(lr.actual)], config=cfg)


def ablation(pdata, origins=None, config=None):
    """The doc's ladder: carry-forward, futures-implied, + mean reversion, + shock state, + jump rule."""
    cfg = config or Config()
    off = dict(use_mean_reversion=False, use_futures=False, use_shock_state=False, use_jump=False)
    versions = [
        ("v0 carry-forward", off),
        ("v1 futures-implied (doc baseline)", {**off, "use_futures": True}),
        ("v2 drift + mean reversion", {**off, "use_mean_reversion": True}),
        ("v3 + active-shock reversion", {**off, "use_mean_reversion": True, "use_shock_state": True,
                                         "jump_mode": "mixture"}),
        ("v4 alt: doc's 0.5 switch (not used)", dict(use_mean_reversion=True, use_shock_state=True, use_jump=True,
                                                    jump_mode="median")),
        ("v4 alt: probability-weighted (not used)", dict(use_mean_reversion=True, use_shock_state=True, use_jump=True,
                                                       jump_mode="expected")),
        ("v4 + jump rule, mixture median (submitted)", dict(use_mean_reversion=True, use_shock_state=True,
                                                            use_jump=True, jump_mode="mixture")),
    ]
    cache = {}
    out = {name: backtest(pdata, origins, replace(cfg, **ov), m1_cache=cache).summary() for name, ov in versions}
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- outputs
def fill_submission(pdata, sample, windows=((209, 221), (222, 234)), config=None, out_path=None):
    """Fill price_forecast. Each window is forecast from min(window start - 1, last data week).
    Other columns are left untouched."""
    cfg = config or Config()
    P = pdata.P
    sub = pd.read_csv(sample) if isinstance(sample, str) else sample.copy()
    parts = sub.id.str.extract(r"^w(\d+)_(.+)$")
    wk, mat = parts[0].astype(int), parts[1]
    vals = pd.Series(np.nan, index=sub.index)
    for start, end in windows:
        origin = min(start - 1, P.last_week)
        H = end - origin
        if H > cfg.horizon:
            warnings.warn(f"window {start}-{end}: data ends at week {P.last_week}; weeks beyond "
                          f"{origin + cfg.horizon} are forecast {cfg.horizon + 1}+ weeks ahead. Re-run after Phase 2.")
        bench, _ = PriceModel(cfg).fit(pdata, origin=origin).predict(H)
        lut = {(int(w), m): v for w, m, v in zip(bench.week, bench.material, bench.price_forecast)}
        mask = wk.between(start, end)
        vals[mask] = [lut.get((int(w), m), np.nan) for w, m in zip(wk[mask], mat[mask])]
    if vals.isna().any():
        raise ValueError(f"no forecast for {int(vals.isna().sum())} rows, e.g. {sub.id[vals.isna()].head(3).tolist()}")
    sub["price_forecast"] = vals.round(4).values
    if out_path:
        sub.to_csv(out_path, index=False)
    return sub


def export_for_model3(pdata, windows=((209, 221), (222, 234)), config=None, out_path=None):
    """Per-supplier, per-week landed costs for each ordering window (each from its own origin).
    Columns: window, origin, week, h, material, country, lane, spot_*, tariff_*, container_*,
    landed_cost (most likely path), landed_cost_expected (probability-weighted, for the quick LP),
    landed_cost_calm (no shocks, for the scenario LP that applies its own multipliers)."""
    cfg = config or Config()
    P = pdata.P
    parts = []
    for start, end in windows:
        origin = min(start - 1, P.last_week)
        _, sup = PriceModel(cfg).fit(pdata, origin=origin).predict(end - origin)
        sup = sup[sup.week.between(start, end)].copy()
        sup.insert(0, "window", f"{start}-{end}")
        parts.append(sup)
    out = pd.concat(parts, ignore_index=True)
    if out_path:
        out.to_csv(out_path, index=False)
    return out


# ----------------------------------------------------------------------------- diagnostics
def futures_check(pdata, until=None):
    """The doc's check: does futures_4w at T predict spot at T+4 better than spot at T?
    Split by whether the futures premium is inside normal noise (|premium| < 3%)."""
    t_hi = pdata.P.idx(until or pdata.P.last_week) - 4
    s, f = pdata.spot[:, : t_hi + 1], pdata.fut[:, : t_hi + 1]
    a = pdata.spot[:, 4: t_hi + 5]
    prem = np.abs(f / s - 1)
    rows = []
    for name, mask in (("all weeks", np.isfinite(a)), ("|premium| < 3%", prem < 0.03), ("|premium| >= 3%", prem >= 0.03)):
        rows.append(dict(subset=name, n=int(mask.sum()), mape_futures=_mape(f[mask], a[mask]),
                         mape_carry=_mape(s[mask], a[mask])))
    df = pd.DataFrame(rows)
    df["futures_better"] = df.mape_futures < df.mape_carry
    return df


def landed_fit_report(pdata, until=None):
    """Per-supplier landed-cost formula: landed = spot*(1+tariff) + b*container + c, and its error."""
    pm = PriceModel().fit(pdata, origin=until or pdata.P.last_week)
    P, t0 = pdata.P, pm.t0
    rows = []
    for u in range(len(P.suppliers)):
        cont = pdata.container[P.sup_lane[u], : t0 + 1]
        pred = pdata.spot[u, : t0 + 1] * (1 + P.tariff_rate[u, : t0 + 1]) + pm.land_b[u] * cont + pm.land_c[u]
        act = pdata.landed[u, : t0 + 1]
        ok = np.isfinite(pred) & np.isfinite(act)
        rows.append(dict(material=P.suppliers.material[u], country=P.suppliers.country[u], lane=P.lanes[P.sup_lane[u]],
                         b_container=pm.land_b[u], c_const=pm.land_c[u],
                         mape=_mape(pred[ok], act[ok]), max_error=float(np.max(np.abs(pred[ok] - act[ok]) / act[ok]))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser(description="Model 2 price forecast")
    ap.add_argument("--data", required=True)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--submission", help="submission csv to fill (price_forecast only)")
    ap.add_argument("--out", default="submission_model2.csv")
    ap.add_argument("--landed-out", help="write landed costs for Model 3 to this CSV")
    args = ap.parse_args()
    pdata = PriceData(model1.load_data(args.data))
    print(f"data weeks {pdata.P.first_week}-{pdata.P.last_week}")
    print(futures_check(pdata).round(4).to_string(index=False))
    if args.ablation:
        pd.set_option("display.width", 220)
        print(ablation(pdata).round(3).to_string())
    if args.backtest:
        print(backtest(pdata).summary().round(4).to_string())
    if args.submission:
        sub = fill_submission(pdata, args.submission, out_path=args.out)
        print(f"wrote {args.out} ({len(sub)} rows)")
    if args.landed_out:
        lc = export_for_model3(pdata, out_path=args.landed_out)
        print(f"wrote {args.landed_out} ({len(lc)} rows)")


if __name__ == "__main__":
    _main()
