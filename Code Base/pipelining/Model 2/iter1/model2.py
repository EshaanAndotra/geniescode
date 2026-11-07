"""
model2.py - Model 2: price forecast.

Fills the `price_forecast` submission column (the benchmark price: the average
spot price across a material's 3 suppliers, per week; worth 12% of the total
score) and produces the per-supplier landed cost forecasts that the ordering
model (Model 3) uses to decide who to buy from and when.

Scoring: MAPE skill against carrying the origin-week benchmark price forward
unchanged,  skill = 1 - our MAPE / carry-forward MAPE.  Carry-forward scores 0
by definition, and it is a tough baseline over 1-13 week horizons, so the goal
is a simple model that reliably beats it.

Forecasters - each predicts the % change from the last known spot, per supplier:
  carry-forward   no change from today's price (the 0-skill baseline)
  futures-implied spot_now * (futures_4w / spot_now) ** (min(h, 4) / 4), flat after
  mean-reversion  drift back toward the trailing 26-week average (half-life 8 w)
  boosted trees   v3, off by default: learns the % change from the futures
                  premium, momentum, the Model 1 disruption probability and the
                  tariff schedule, with absolute-error loss; kept only if it
                  beats the blend without it
Blend weights are fitted separately for horizons 1-4, 5-8 and 9-13 on the
rolling backtest, then shrunk toward carry-forward when the evidence is weak
(a bucket needs 10%+ relative MAPE gain over carry-forward before its blend
is trusted at all). The backtest reports the blend with leave-one-origin-out
weights so the skill is honest.

Shock jump rule: the forecast is multiplied by a fitted jump only in weeks
where Model 1's P(disruption) for that material is above 0.5. MAPE targets the
median outcome, so adding an expected jump (p * size) costs more than it
saves; when futures already price a shock in (premium above ~6%), the
futures-implied member follows on its own - that is the "futures agree" branch
of the rule. Model 1 (iter 3, model1.py) is imported for this; without it the
jump is skipped with a warning.

Landed cost: fit  landed_cost ~ spot * (1 + tariff) + a * container_rate + b
per supplier on supply_history (landed_fit reports the fit quality - there may
be extra components), then forecast it per supplier and week with the blended
spot forecast, tariffs switched to announced_rate from announced_effective_week
(bookkeeping, not modeling) and the container rate carried forward.

Backtest: the same origins as Model 1 - 156, 169, 182, 195 (208 joins once
Phase 2 data lands) - each 1 to 13 weeks ahead, scored on the material-week
benchmark price. MAPE skill is reported overall, per material and per horizon
bucket. futures_check verifies the v1 assumption that futures at T beat spot at
T at predicting spot at T+4 (if they don't, the shrink rule falls back to
carry-forward on its own). futures_premium_scan is the big-shock early warning
for the Phase 2 weeks: a futures premium well above its training range means
the market is pricing something in and the forecast should follow; flat
futures = stay close to carry-forward.

Quick use:
    import model2
    data = model2.load_data("path/to/data")            # folder with the competition CSVs
    panels = model2.PricePanels(data)
    model2.futures_check(panels)                        # v1 assumption check
    bt = model2.backtest(panels); print(bt.summary())   # MAPE skill, LOO blend
    print(model2.ladder(panels))                        # v0 / v1 / v2 / v2+jump
    sub, landed = model2.fill_submission(panels, "sample_submission.csv",
                                         out_path="submission_model2.csv")
    landed.to_csv("landed_model2.csv", index=False)     # for the ordering model
    model2.futures_premium_scan(panels)                 # Phase 2 big-shock check

Command line:
    python model2.py --data DATA_DIR --backtest --ladder --futures-check \
        --submission sample_submission.csv --out submission_model2.csv \
        --landed-out landed_model2.csv [--trees] [--model1-dir DIR]

Requires: numpy, pandas; scipy + scikit-learn for the blend optimizer and the
optional boosted trees.
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

__all__ = [
    "Config", "PricePanels", "PriceBacktestResult", "TreeMember", "load_data",
    "backtest", "ladder", "fill_submission", "landed_fit", "landed_forecast",
    "futures_check", "futures_premium_scan", "fit_jump", "import_model1",
    "mape", "mape_skill", "forecast", "fit_weights",
]

# ----------------------------------------------------------------------------- constants
HORIZON = 13                         # forecast weeks per window
BUCKETS = ((1, 4), (5, 8), (9, 13))   # horizon buckets, each with its own blend weights
MEAN_WINDOW = 26                     # mean-reversion anchor
REVERT_HALFLIFE = 8.0                # weeks to close half the gap to the anchor
JUMP_P = 0.5                         # add a shock jump only above this P(disruption)
PREM_SIGNAL = 0.06                   # futures premium above this = market pricing something in
SHRINK_GAIN = 0.10                   # relative MAPE gain a bucket needs before its blend is trusted
DOC_ORIGINS = (156, 169, 182, 195)    # same origins as Model 1; 208 joins after Phase 2
SUBMISSION_WINDOWS = ((209, 221), (222, 234))
MEMBERS = ("carry", "futures", "meanrev", "trees")
FILES = ("prices", "supply_history", "shipping", "tariffs", "suppliers")
OPTIONAL_FILES = ("outcomes", "macro")  # outcomes: jump fit + Model 1 hook; macro: Model 1 hook


@dataclass
class Config:
    horizon: int = HORIZON
    mean_window: int = MEAN_WINDOW
    revert_halflife: float = REVERT_HALFLIFE
    jump_p: float = JUMP_P
    shrink_gain: float = SHRINK_GAIN
    use_meanrev: bool = True
    use_jump: bool = True
    use_trees: bool = False           # v3: only if it beats the blend without it


# ----------------------------------------------------------------------------- loading
def _find_file(data_dir, name):
    exact = os.path.join(data_dir, f"{name}.csv")
    if os.path.exists(exact):
        return exact
    cands = sorted(f for f in os.listdir(data_dir) if f.endswith(f"{name}.csv"))
    if not cands:
        return None
    return os.path.join(data_dir, cands[0])


def load_data(data_dir):
    """Read the competition CSVs from a folder (file names may carry a prefix)."""
    data = {}
    for name in FILES + OPTIONAL_FILES:
        path = _find_file(data_dir, name)
        if path is None:
            if name in FILES:
                raise FileNotFoundError(f"{name}.csv not found in {data_dir}")
            continue
        data[name] = pd.read_csv(path)
    return data


def _ffill(x):
    """Forward-fill a [N, W] array along weeks (leaves leading NaNs)."""
    return pd.DataFrame(x).ffill().values


# ----------------------------------------------------------------------------- panels
class PricePanels:
    """Dense (supplier x week) arrays for prices, tariffs, shipping and landed costs."""

    def __init__(self, data, config=None):
        self.data = data
        self.cfg = config or Config()
        sup = data["suppliers"].reset_index(drop=True)
        feat_dfs = [data[k] for k in ("prices", "supply_history", "shipping", "tariffs")]  # weekly files only
        self.first_week = int(min(d.week.min() for d in feat_dfs))
        self.last_week = int(min(d.week.max() for d in feat_dfs))
        self.weeks = np.arange(self.first_week, self.last_week + 1)
        self.W = len(self.weeks)

        self.suppliers = sup
        self.materials = list(dict.fromkeys(sup.material))
        self.lanes = sorted(sup.lane.unique())
        self.sup_lane = sup.lane.map({l: i for i, l in enumerate(self.lanes)}).values
        self.sup_mat = sup.material.map({m: i for i, m in enumerate(self.materials)}).values
        self.mat_units = [np.where(self.sup_mat == i)[0] for i in range(len(self.materials))]
        keys = list(zip(sup.material, sup.country))

        def sup_arr(df, col):
            p = df.pivot_table(index=["material", "country"], columns="week", values=col, aggfunc="first")
            p = p.reindex(index=pd.MultiIndex.from_tuples(keys), columns=self.weeks)
            return p.values.astype(float)

        def lane_arr(df, col):
            p = df.pivot_table(index="lane", columns="week", values=col, aggfunc="first")
            return p.reindex(index=self.lanes, columns=self.weeks).values.astype(float)

        self.spot = sup_arr(data["prices"], "spot_price")
        self.fut = sup_arr(data["prices"], "futures_4w")
        self.landed = sup_arr(data["supply_history"], "landed_cost")
        self.tariff = sup_arr(data["tariffs"], "tariff_rate")
        self.ann_rate = sup_arr(data["tariffs"], "announced_rate")
        self.ann_eff = sup_arr(data["tariffs"], "announced_effective_week")
        self.container = lane_arr(data["shipping"], "container_rate")

        # forward-filled views: the origin's last known value even if a week is missing
        self.spot_ff = _ffill(self.spot)
        self.fut_ff = _ffill(self.fut)
        self.tariff_ff = _ffill(self.tariff)
        self.ann_rate_ff = _ffill(self.ann_rate)
        self.ann_eff_ff = _ffill(self.ann_eff)
        self.container_ff = _ffill(self.container)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.prem = self.fut_ff / np.maximum(self.spot_ff, 1e-9) - 1.0

        # trailing mean of spot (past only, incl. current week)
        s = pd.DataFrame(self.spot_ff)
        self.mean26 = s.rolling(window=self.cfg.mean_window, min_periods=8).mean().values

        # benchmark price: average spot across the material's suppliers
        self.bench = np.vstack([np.nanmean(self.spot[u], axis=0) for u in self.mat_units])

        # material x week disrupted (NaN where unknown) - for the jump fit
        if "outcomes" in data:
            p = data["outcomes"].pivot_table(index="material", columns="week", values="disrupted", aggfunc="first")
            self.disrupted = p.reindex(index=self.materials, columns=self.weeks).values.astype(float)
        else:
            self.disrupted = np.full((len(self.materials), self.W), np.nan)

        self._model1_cache = None       # (module, Panels) for the Model 1 hook

    def idx(self, week):
        return int(week) - self.first_week

    def week(self, idx):
        return int(idx) + self.first_week


# ----------------------------------------------------------------------------- members
def member_changes(panels, t0, H, use_meanrev=True):
    """Per-supplier % change from the origin spot for each member. dict name -> [U, H]."""
    U = len(panels.suppliers)
    h = np.arange(1, H + 1)
    spot = panels.spot_ff[:, t0]
    safe = np.maximum(spot, 1e-9)
    ok = np.isfinite(spot) & (spot > 0)

    out = {"carry": np.zeros((U, H))}

    ratio = np.where(ok, panels.fut_ff[:, t0] / safe, 1.0)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    out["futures"] = ratio[:, None] ** (np.minimum(h, 4)[None, :] / 4.0) - 1.0

    if use_meanrev:
        gap = np.where(ok, panels.mean26[:, t0] / safe - 1.0, 0.0)
        gap = np.nan_to_num(gap)
        rho = 0.5 ** (1.0 / panels.cfg.revert_halflife)
        out["meanrev"] = gap[:, None] * (1.0 - rho ** h[None, :])
    return out


def _bucket(h):
    for i, (lo, hi) in enumerate(BUCKETS):
        if h <= hi:
            return i
    return len(BUCKETS) - 1


def _bench_from_chg(panels, chg, t0):
    """[M, H] benchmark price implied by a per-supplier % change matrix [U, H]."""
    base = panels.spot_ff[:, t0]
    prices = base[:, None] * (1.0 + chg)
    return np.vstack([np.nanmean(prices[u], axis=0) for u in panels.mat_units])


# ----------------------------------------------------------------------------- metrics
def mape(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    m = np.isfinite(f) & np.isfinite(a) & (a != 0)
    return float(np.mean(np.abs(f[m] - a[m]) / np.abs(a[m]))) if m.any() else np.nan


def mape_skill(f, a, base):
    """1 - our MAPE / baseline MAPE (baseline = carry-forward -> 0 by definition)."""
    mb, mf = mape(base, a), mape(f, a)
    if not (np.isfinite(mb) and mb > 0):
        return np.nan
    return 1.0 - mf / mb


# ----------------------------------------------------------------------------- blend
def fit_weights(rows, shrink_gain=SHRINK_GAIN, members=None):
    """Blend weights per horizon bucket, fitted on material-week backtest rows.
    Simplex weights (sum 1, >= 0) minimizing MAPE, then shrunk toward carry-forward
    by how much the bucket actually beats carry-forward (no gain -> pure carry)."""
    from scipy.optimize import minimize
    members = members or [c for c in MEMBERS if c in rows.columns]
    k = len(members)
    ic = members.index("carry") if "carry" in members else None
    weights, diag = [], []
    for lo, hi in BUCKETS:
        d = rows[rows.h.between(lo, hi)]
        if len(d) < 10:
            w = np.zeros(k)
            if ic is not None:
                w[ic] = 1.0
            else:
                w[:] = 1.0 / k
            weights.append(w)
            diag.append(dict(bucket=f"h{lo}-{hi}", n=len(d), mape_carry=np.nan, mape_blend=np.nan,
                             gain=0.0, shrink=1.0))
            continue
        X = np.column_stack([d[c].values for c in members]).astype(float)
        a = d["actual"].values
        m_carry = mape(X[:, ic] if ic is not None else X.mean(1), a)

        def obj(wv):
            return mape(X @ wv, a)

        w0 = np.full(k, 1.0 / k)
        try:
            res = minimize(obj, w0, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                           constraints=[{"type": "eq", "fun": lambda wv: wv.sum() - 1.0}],
                           options={"maxiter": 300, "ftol": 1e-12})
            w_opt = np.clip(res.x, 0.0, 1.0)
            w_opt = w_opt / max(w_opt.sum(), 1e-9)
        except Exception:
            w_opt = w0.copy()
        m_opt = obj(w_opt)
        gain = (m_carry - m_opt) / m_carry if (np.isfinite(m_carry) and m_carry > 0) else 0.0
        gain = max(float(gain), 0.0)
        lam = float(np.clip(1.0 - gain / max(shrink_gain, 1e-9), 0.0, 1.0))
        w_carry = np.eye(k)[ic] if ic is not None else np.full(k, 1.0 / k)
        w = (1.0 - lam) * w_opt + lam * w_carry
        weights.append(w)
        diag.append(dict(bucket=f"h{lo}-{hi}", n=len(d), mape_carry=m_carry, mape_blend=m_opt,
                         gain=gain, shrink=lam,
                         **{f"w_{c}": w[i] for i, c in enumerate(members)}))
    return weights, pd.DataFrame(diag)


# ----------------------------------------------------------------------------- jump fit
def fit_jump(panels, t_end=None):
    """Median spot move in the first two weeks of a disruption, per material,
    fitted only on weeks <= t_end (leak-free inside a backtest). A disruption
    never lowers the forecast, so negative jumps are clipped to 0."""
    t_end = panels.W - 1 if t_end is None else t_end
    pooled, rows = [], []
    for mi, m in enumerate(panels.materials):
        rowv = np.nan_to_num(panels.disrupted[mi, : t_end + 1], nan=0.0)
        evs = []
        for t in range(len(rowv)):
            if rowv[t] == 1 and (t == 0 or rowv[t - 1] == 0):
                hi = min(t + 3, t_end + 1)
                if hi - 1 <= t:
                    continue
                r = []
                for u in panels.mat_units[mi]:
                    base = panels.spot[u, t]
                    nxt = panels.spot[u, t + 1: hi]
                    if np.isfinite(base) and base > 0 and np.isfinite(nxt).any():
                        r.append(float(np.nanmean(nxt) / base - 1.0))
                if r:
                    evs.append(float(np.mean(r)))
        pooled += evs
        rows.append(dict(material=m, n_events=len(evs),
                         jump=float(np.median(evs)) if len(evs) >= 2 else np.nan))
    pooled_jump = float(np.median(pooled)) if pooled else 0.0
    df = pd.DataFrame(rows).set_index("material")
    df["jump"] = df.jump.fillna(pooled_jump).clip(lower=0.0)
    return df


# ----------------------------------------------------------------------------- Model 1 hook
def import_model1(model1_dir=None):
    """Import Model 1 (iter 3, model1.py) for the jump rule and tree features.
    Returns the module, or None if it cannot be found."""
    if "model1" in sys.modules:
        return sys.modules["model1"]
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if model1_dir:
        cands.append(model1_dir)
    cands += [os.path.normpath(os.path.join(here, os.pardir, os.pardir, "Model 1", "iter3")),
              "/Workspace/Repos/DSDG Hackathon/DSDG Hackathon/Code Base/pipelining/Model 1/iter3"]
    for c in cands:
        if not c or not os.path.isdir(c):
            continue
        sys.path.insert(0, c)
        try:
            return importlib.import_module("model1")
        except Exception:
            sys.path.remove(c)
    return None


def model1_probs(panels, origin, H, model1_dir=None):
    """P(disruption) per (week, material) from Model 1, or None if Model 1 or its
    inputs (outcomes, macro) are unavailable. Model 1's Panels is built once."""
    if panels._model1_cache is None:
        try:
            m1 = import_model1(model1_dir)
            p1 = m1.Panels(panels.data) if m1 is not None else None
            panels._model1_cache = (m1, p1)
        except Exception:
            panels._model1_cache = (None, None)
    m1, p1 = panels._model1_cache
    if m1 is None or p1 is None:
        return None
    try:
        pred = m1.DisruptionModel().fit(p1, origin=origin).predict(H)
    except Exception as e:
        warnings.warn(f"Model 1 predict failed at origin {origin}: {e}")
        return None
    return {(int(w), mm): float(p) for w, mm, p in zip(pred.week, pred.material, pred.disruption_prob)}


# ----------------------------------------------------------------------------- v3 trees
class TreeMember:
    """v3: boosted trees on scale-free features, absolute-error loss. Learns the
    % change from the last spot from the futures premium, momentum, the Model 1
    disruption probability and the tariff schedule. Off by default; keep only if
    it beats the blend without it."""

    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05):
        from sklearn.ensemble import GradientBoostingRegressor
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, loss="absolute_error", random_state=0)
        self.clf = GradientBoostingRegressor(**self.params)
        self.with_prob = False

    @staticmethod
    def _origin_feats(panels, t0):
        spot = panels.spot_ff[:, t0]
        safe = np.maximum(spot, 1e-9)
        ok = np.isfinite(spot) & (spot > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            prem = np.where(ok, panels.fut_ff[:, t0] / safe - 1.0, 0.0)
            mom = lambda lag: np.where(ok and t0 >= lag, spot / panels.spot_ff[:, t0 - lag] - 1.0, 0.0) \
                if t0 >= lag else np.zeros(len(spot))
            dev = np.where(ok, panels.mean26[:, t0] / safe - 1.0, 0.0)
            cong_chg = panels.container_ff[panels.sup_lane, t0] / \
                panels.container_ff[panels.sup_lane, max(t0 - 1, 0)] - 1.0 if t0 >= 1 else np.zeros(len(spot))
        pend = np.nan_to_num(panels.ann_eff_ff[:, t0], nan=0.0)
        pend = np.where(np.isnan(panels.ann_rate_ff[:, t0]) | (pend <= panels.weeks[t0]), 0.0,
                        np.clip(pend - panels.weeks[t0], 0.0, 13.0))
        F = np.column_stack([np.nan_to_num(prem), np.nan_to_num(mom(1)), np.nan_to_num(mom(4)),
                             np.nan_to_num(mom(13)), np.nan_to_num(dev), pend, np.nan_to_num(cong_chg)])
        return F, ok

    def fit(self, panels, origins, probs=None):
        """origins: training origins (their target weeks must be known at test
        time). probs: {origin: {(week, material): p}} from Model 1, or None."""
        X, y = [], []
        mats = np.asarray(panels.materials)
        self.with_prob = bool(probs) and any(probs.get(o) for o in origins)
        for o in origins:
            t0 = panels.idx(o)
            F, ok0 = self._origin_feats(panels, t0)
            spot = panels.spot_ff[:, t0]
            pr = (probs or {}).get(o) or {}
            for h in range(1, panels.cfg.horizon + 1):
                t = t0 + h
                if t >= panels.W:
                    break
                tgt = panels.spot_ff[:, t] / np.maximum(spot, 1e-9) - 1.0
                ok = ok0 & np.isfinite(tgt)
                if not ok.any():
                    continue
                pcol = np.array([pr.get((panels.week(t), mats[panels.sup_mat[u]]), 0.0) for u in range(len(spot))])
                cols = [F[ok], np.full(int(ok.sum()), float(h))]
                if self.with_prob:
                    cols.append(pcol[ok])
                X.append(np.column_stack(cols))
                y.append(tgt[ok])
        self.clf.fit(np.vstack(X), np.concatenate(y))
        return self

    def changes(self, panels, t0, H, probs=None):
        """[U, H] % change from the origin spot."""
        F, _ = self._origin_feats(panels, t0)
        U = F.shape[0]
        mats = np.asarray(panels.materials)
        origin = panels.week(t0)
        out = np.zeros((U, H))
        for h in range(1, H + 1):
            cols = [F, np.full(U, float(h))]
            if self.with_prob:
                pcol = np.array([((probs or {}).get((origin + h, mats[panels.sup_mat[u]])) or 0.0)
                                 for u in range(U)])
                cols.append(pcol)
            out[:, h - 1] = self.clf.predict(np.column_stack(cols))
        return out


def _fit_tree(panels, T, cfg, model1_dir):
    """Fit the v3 trees on origins strictly before T (targets within T)."""
    train = [o for o in range(40, T - cfg.horizon, 4) if o > panels.first_week]
    if len(train) < 5:
        return None
    probs = {o: model1_probs(panels, o, cfg.horizon, model1_dir) for o in train}
    probs = {o: p for o, p in probs.items() if p is not None}
    try:
        return TreeMember().fit(panels, train, probs or None)
    except Exception as e:
        warnings.warn(f"trees skipped at origin {T}: {e}")
        return None


# ----------------------------------------------------------------------------- backtest rows
def _member_rows(panels, origins, config=None, model1_dir=None, use_trees=False, verbose=False):
    """Backtest rows: one per material-week with each member's benchmark price,
    the actual, the Model 1 probability (if available) and the jump multiplier."""
    cfg = config or panels.cfg
    rows = []
    for T in sorted(set(o for o in origins if panels.first_week < o <= panels.last_week)):
        t0 = panels.idx(T)
        H = min(cfg.horizon, panels.W - 1 - t0)
        if H < 1:
            continue
        chg = member_changes(panels, t0, H, use_meanrev=cfg.use_meanrev)
        prices = {n: _bench_from_chg(panels, c, t0) for n, c in chg.items()}
        probs = model1_probs(panels, T, H, model1_dir) if cfg.use_jump else None
        jump = fit_jump(panels, t_end=t0) if cfg.use_jump else None
        if cfg.use_jump and (probs is None or jump is None):
            warnings.warn(f"origin {T}: Model 1 or outcomes unavailable - shock jump skipped")
        if use_trees:
            tree = _fit_tree(panels, T, cfg, model1_dir)
            if tree is not None:
                prices["trees"] = _bench_from_chg(panels, tree.changes(panels, t0, H, probs), t0)
        n_rows_T = 0
        for mi, m in enumerate(panels.materials):
            base = panels.bench[mi, t0]
            if not np.isfinite(base) or base <= 0:
                continue
            jm = float(jump.jump.get(m, 0.0)) if jump is not None else 0.0
            for h in range(1, H + 1):
                a = panels.bench[mi, t0 + h]
                if not np.isfinite(a) or a <= 0:
                    continue
                p = probs.get((T + h, m), np.nan) if probs else np.nan
                mult = 1.0 + jm if (np.isfinite(p) and p > cfg.jump_p and jm > 0) else 1.0
                row = dict(origin=T, h=h, week=T + h, material=m, base=base, actual=a,
                           p_disruption=p, jump_mult=mult)
                for n, P in prices.items():
                    row[n] = P[mi, h - 1]
                rows.append(row)
                n_rows_T += 1
        if verbose:
            print(f"origin {T}: {n_rows_T} rows")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- result
class PriceBacktestResult:
    def __init__(self, rows, weights, members, weight_diag):
        self.rows = rows
        self.weights = weights      # full-fit weights (all origins)
        self.members = members
        self.weight_diag = weight_diag

    def _skill(self, df, col):
        if len(df) == 0 or col not in df:
            return np.nan
        return mape_skill(df[col], df.actual, df.carry)

    def summary(self):
        r = self.rows
        out = {"n_rows": len(r), "n_origins": r.origin.nunique(), "mape_carry": mape(r.carry, r.actual)}
        for c in self.members + ["blend", "blend_jump"]:
            if c in r:
                out[f"skill_{c}"] = self._skill(r, c)
                out[f"mape_{c}"] = mape(r[c], r.actual)
        return pd.Series(out)

    def by_material(self, col="blend_jump"):
        return self.rows.groupby("material").apply(
            lambda d: pd.Series({"skill": self._skill(d, col), "mape": mape(d[col], d.actual),
                                 "mape_carry": mape(d.carry, d.actual), "n": len(d)}),
            include_groups=False)

    def by_bucket(self, col="blend_jump"):
        hb = pd.cut(self.rows.h, [0, 4, 8, 13, 10**6], labels=["h1-4", "h5-8", "h9-13", "h14+"])
        return self.rows.groupby(hb, observed=True).apply(
            lambda d: pd.Series({"skill": self._skill(d, col), "mape": mape(d[col], d.actual),
                                 "mape_carry": mape(d.carry, d.actual), "n": len(d)}),
            include_groups=False)


# ----------------------------------------------------------------------------- backtest
def backtest(panels, origins=None, config=None, model1_dir=None, use_trees=None,
             verbose=False, **overrides):
    """Rolling-origin backtest on the material-week benchmark price. Same origins
    as Model 1 by default (156, 169, 182, 195; 208 joins once Phase 2 data lands).
    The blend column uses leave-one-origin-out weights: each origin's forecast
    is blended with weights fitted on the other origins."""
    cfg = replace(config or panels.cfg, **overrides)
    if use_trees is None:
        use_trees = cfg.use_trees
    origins = list(origins) if origins is not None else [o for o in DOC_ORIGINS + (208,) if o < panels.last_week]
    rows = _member_rows(panels, origins, cfg, model1_dir, use_trees, verbose)
    if rows.empty:
        raise ValueError("no backtest rows: check origins vs the data weeks")
    members = [c for c in MEMBERS if c in rows.columns]

    rows["blend"] = np.nan
    per_origin_diag = []
    for T in rows.origin.unique():
        w, d = fit_weights(rows[rows.origin != T], cfg.shrink_gain, members)
        per_origin_diag.append(d.assign(origin=T))
        te = rows[rows.origin == T]
        for i in te.index:
            b = _bucket(int(rows.at[i, "h"]))
            rows.at[i, "blend"] = float(sum(w[b][j] * rows.at[i, c] for j, c in enumerate(members)))
    rows["blend_jump"] = rows["blend"] * rows["jump_mult"]

    w_full, _ = fit_weights(rows, cfg.shrink_gain, members)
    return PriceBacktestResult(rows=rows, weights=w_full, members=members,
                               weight_diag=pd.concat(per_origin_diag, ignore_index=True))


def ladder(panels, **kw):
    """The spec's build order as MAPE skill: v0 carry-forward, v1 futures-implied,
    v2 blend, v2 + shock jump (v3 trees only when use_trees=True)."""
    bt = backtest(panels, **kw)
    cols = [("v0 carry-forward", "carry"), ("v1 futures-implied", "futures"),
            ("v2 blend", "blend"), ("v2 + shock jump", "blend_jump")]
    if "trees" in bt.rows:
        cols.append(("v3 + trees", "trees"))
    out = {}
    for name, c in cols:
        bb = bt.by_bucket(c)
        row = {"overall": bt._skill(bt.rows, c)}
        for b in ("h1-4", "h5-8", "h9-13"):
            row[b] = bb.loc[b, "skill"] if b in bb.index else np.nan
        out[name] = row
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- checks
def futures_check(panels):
    """v1 assumption: at week T, does futures_4w beat spot_T at predicting spot
    at T+4? If not, fall back to carry-forward (the shrink rule does anyway)."""
    e_f = e_s = 0.0
    n = 0
    for t in range(1, panels.W - 4):
        f, s, a = panels.fut_ff[:, t], panels.spot_ff[:, t], panels.spot_ff[:, t + 4]
        ok = np.isfinite(f) & np.isfinite(s) & np.isfinite(a) & (a > 0) & (s > 0)
        if ok.any():
            e_f += float(np.sum(np.abs(f[ok] - a[ok]) / a[ok]))
            e_s += float(np.sum(np.abs(s[ok] - a[ok]) / a[ok]))
            n += int(ok.sum())
    if n == 0:
        return pd.Series(dict(n_supplier_weeks=0))
    mf, ms = e_f / n, e_s / n
    return pd.Series(dict(n_supplier_weeks=n, mape_futures=mf, mape_spot=ms,
                          skill=1.0 - mf / ms if ms > 0 else np.nan))


def futures_premium_scan(panels, train_end=208, recent_start=209):
    """Big-shock early warning: the futures premium in the recent (Phase 2) weeks
    against its max up to train_end, per supplier. Use when Phase 2 data lands."""
    a = panels.idx(train_end)
    b = min(panels.idx(recent_start), panels.W - 1)
    rows = []
    for u, r in panels.suppliers.iterrows():
        prem = panels.prem[u]
        tr, rc = prem[: a + 1], prem[b:] if b > a else np.array([])
        tmax = float(np.nanmax(tr)) if np.isfinite(tr).any() else np.nan
        rmax = float(np.nanmax(rc)) if len(rc) and np.isfinite(rc).any() else np.nan
        last = float(prem[-1]) if np.isfinite(prem[-1]) else np.nan
        rows.append(dict(material=r.material, country=r.country, train_max=tmax, recent_max=rmax,
                        last=last,
                        exceeds=bool(np.isfinite(rmax) and np.isfinite(tmax) and rmax > tmax),
                        signals=bool(np.isfinite(last) and last > PREM_SIGNAL)))
    return (pd.DataFrame(rows).sort_values(["signals", "recent_max"], ascending=False)
            .reset_index(drop=True))


# ----------------------------------------------------------------------------- landed cost
def landed_fit(panels):
    """Fit landed_cost = spot * (1 + tariff) + a * container_rate + b per supplier
    on supply_history. Reports the fit quality - there may be extra components."""
    rows = []
    for u, r in panels.suppliers.iterrows():
        x, lc, tf = panels.spot[u], panels.landed[u], panels.tariff[u]
        cr = panels.container[panels.sup_lane[u]]
        ok = np.isfinite(x) & np.isfinite(lc) & np.isfinite(tf) & np.isfinite(cr) & (x > 0)
        if ok.sum() < 20:
            rows.append(dict(material=r.material, country=r.country, lane=r.lane,
                             a=np.nan, b=np.nan, r2=np.nan, mae=np.nan, n=int(ok.sum())))
            continue
        resid = lc[ok] - x[ok] * (1.0 + tf[ok])
        A = np.column_stack([cr[ok], np.ones(int(ok.sum()))])
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        fitv = A @ coef
        ss_res = float(((resid - fitv) ** 2).sum())
        ss_tot = float(((resid - resid.mean()) ** 2).sum())
        rows.append(dict(material=r.material, country=r.country, lane=r.lane,
                         a=float(coef[0]), b=float(coef[1]),
                         r2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                         mae=float(np.mean(np.abs(resid - fitv))), n=int(ok.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- forecast
def forecast(panels, origin, H, weights, config=None, probs=None, jump=None, tree=None):
    """Blend the members at `origin` for h = 1..H and apply the shock jump.
    h > HORIZON keeps the last bucket's weights (members extend naturally:
    futures flat after 4 weeks, mean reversion keeps converging).
    Returns (supplier_df, material_df)."""
    cfg = config or panels.cfg
    t0 = panels.idx(origin)
    H = int(H)
    chg = member_changes(panels, t0, H, use_meanrev=cfg.use_meanrev)
    if tree is not None:
        chg["trees"] = tree.changes(panels, t0, H, probs)
    names = [c for c in MEMBERS if c in chg]

    base = panels.spot_ff[:, t0]
    U = len(panels.suppliers)
    sup = np.zeros((U, H))
    for h in range(1, H + 1):
        w = weights[_bucket(h)]
        for j, n in enumerate(names):
            sup[:, h - 1] += w[j] * base * (1.0 + chg[n][:, h - 1])

    # shock jump: per material-week, only where Model 1's P(disruption) > jump_p
    jmult = np.ones((len(panels.materials), H))
    if jump is not None:
        for mi, m in enumerate(panels.materials):
            jm = float(jump.jump.get(m, 0.0))
            if jm <= 0:
                continue
            for h in range(1, H + 1):
                p = (probs or {}).get((origin + h, m))
                if p is not None and np.isfinite(p) and p > cfg.jump_p:
                    jmult[mi, h - 1] = 1.0 + jm
    for mi, u in enumerate(panels.mat_units):
        sup[u] = sup[u] * jmult[mi][None, :]

    srows, mrows = [], []
    for u, r in panels.suppliers.iterrows():
        for h in range(1, H + 1):
            srows.append(dict(origin=origin, week=origin + h, h=h, material=r.material,
                              country=r.country, lane=r.lane,
                              spot_forecast=float(sup[u, h - 1]),
                              p_disruption=(probs or {}).get((origin + h, r.material), np.nan)))
    for mi, m in enumerate(panels.materials):
        for h in range(1, H + 1):
            mrows.append(dict(origin=origin, week=origin + h, h=h, material=m,
                              price_forecast=float(np.mean(sup[panels.mat_units[mi], h - 1]))))
    return pd.DataFrame(srows), pd.DataFrame(mrows)


# ----------------------------------------------------------------------------- submission
def _prepare(panels, windows, cfg, use_trees, model1_dir):
    """Fit blend weights on the Model-1-origin backtest and forecast every window.
    Returns (weights, fits, [per-window frames])."""
    rows = _member_rows(panels, [o for o in DOC_ORIGINS + (208,) if o < panels.last_week],
                        cfg, model1_dir, use_trees)
    members = [c for c in MEMBERS if c in rows.columns]
    weights, _ = fit_weights(rows, cfg.shrink_gain, members)
    fits = landed_fit(panels).set_index(["material", "country"])
    out = []
    for start, end in windows:
        origin = min(start - 1, panels.last_week)
        H = end - origin
        probs = model1_probs(panels, origin, H, model1_dir) if cfg.use_jump else None
        jump = fit_jump(panels, t_end=panels.idx(origin)) if cfg.use_jump else None
        if cfg.use_jump and (probs is None or jump is None):
            warnings.warn(f"window {start}-{end}: Model 1 or outcomes unavailable - shock jump skipped")
        tree = _fit_tree(panels, origin, cfg, model1_dir) if use_trees else None
        sup_df, mat_df = forecast(panels, origin, H, weights, cfg, probs, jump, tree)
        out.append(dict(start=start, end=end, origin=origin, H=H,
                        sup_df=sup_df, mat_df=mat_df, fits=fits))
    return weights, out


def fill_submission(panels, sample, windows=SUBMISSION_WINDOWS, config=None, use_trees=None,
                    model1_dir=None, out_path=None):
    """Fill price_forecast in a submission file and build the per-supplier landed
    cost forecasts. Each window is forecast from origin = min(window start - 1,
    last data week), like Model 1's fill_submission; other columns are untouched.
    Returns (submission, landed_costs DataFrame)."""
    cfg = replace(config or panels.cfg, use_jump=(config or panels.cfg).use_jump)
    if use_trees is None:
        use_trees = cfg.use_trees
    sub = pd.read_csv(sample) if isinstance(sample, str) else sample.copy()
    parts = sub.id.str.extract(r"^w(\d+)_(.+)$")
    sub_week, sub_mat = parts[0].astype(int), parts[1]

    _, windows_out = _prepare(panels, windows, cfg, use_trees, model1_dir)
    landed_rows = []
    for w in windows_out:
        origin, H, sup_df = w["origin"], w["H"], w["sup_df"]
        lut = {(int(x["week"]), x["material"]): x["price_forecast"]
               for _, x in w["mat_df"].iterrows()}
        mask = sub_week.between(w["start"], w["end"])
        for i in sub.index[mask]:
            sub.at[i, "price_forecast"] = lut.get((int(sub_week[i]), sub_mat[i]), np.nan)

        # landed costs for the ordering model: spot forecast * (1 + tariff in
        # force) + a * container_rate (carried forward) + b, tariffs switched on
        # their announced schedule
        t0 = panels.idx(origin)
        spot_arr = sup_df.spot_forecast.values.reshape(len(panels.suppliers), H)
        for u, r in panels.suppliers.iterrows():
            cur = panels.tariff_ff[u, t0]
            ann_r = panels.ann_rate_ff[u, t0]
            ann_w = panels.ann_eff_ff[u, t0]
            try:
                f = w["fits"].loc[(r.material, r.country)]
                a_c, b_c = float(f["a"]), float(f["b"])
            except KeyError:
                a_c, b_c = np.nan, np.nan
            cr = panels.container_ff[panels.sup_lane[u], t0]
            for h in range(1, H + 1):
                wk = origin + h
                rate = ann_r if (np.isfinite(ann_r) and np.isfinite(ann_w) and ann_w <= wk) else cur
                sf = float(spot_arr[u, h - 1])
                landed_rows.append(dict(window=f"{w['start']}-{w['end']}", origin=origin, week=wk, h=h,
                                        material=r.material, country=r.country, lane=r.lane,
                                        spot_forecast=sf, tariff_rate=float(rate),
                                        container_rate=float(cr),
                                        landed_cost=sf * (1.0 + rate) + a_c * cr + b_c))

    if sub.price_forecast.isna().any():
        missing = sub.id[sub.price_forecast.isna()].tolist()[:5]
        raise ValueError(f"no forecast for {sub.price_forecast.isna().sum()} rows, e.g. {missing}")
    sub["price_forecast"] = sub.price_forecast.astype(float).round(4)
    if out_path:
        sub.to_csv(out_path, index=False)
    return sub, pd.DataFrame(landed_rows)


def landed_forecast(panels, windows=SUBMISSION_WINDOWS, **kw):
    """Per-supplier, per-week landed cost forecasts for the ordering model (thin
    wrapper over fill_submission - see there for the method)."""
    empty = pd.DataFrame({"id": pd.Series([], dtype=str)})
    _, landed = fill_submission(panels, empty, windows=windows, **kw)
    return landed


# ----------------------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser(description="Model 2 price forecast")
    ap.add_argument("--data", required=True, help="folder with the competition CSVs")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--futures-check", action="store_true", help="v1 assumption check")
    ap.add_argument("--premium-scan", action="store_true", help="Phase 2 big-shock early warning")
    ap.add_argument("--submission", help="sample_submission.csv to fill")
    ap.add_argument("--out", default="submission_model2.csv")
    ap.add_argument("--landed-out", help="write per-supplier landed cost forecasts for Model 3")
    ap.add_argument("--model1-dir", help="folder with Model 1's model1.py (for the jump rule)")
    ap.add_argument("--trees", action="store_true", help="v3 boosted trees")
    args = ap.parse_args()

    panels = PricePanels(load_data(args.data))
    print(f"data weeks {panels.first_week}-{panels.last_week}")
    if args.futures_check:
        print("futures vs spot at T+4:\n", futures_check(panels).round(4).to_string())
    if args.premium_scan:
        print("futures premium scan (Phase 2):\n", futures_premium_scan(panels).head(20).to_string())
    if args.ladder:
        pd.set_option("display.width", 200)
        print("MAPE skill ladder:\n", ladder(panels, use_trees=args.trees,
                                             model1_dir=args.model1_dir).round(3).to_string())
    if args.backtest:
        bt = backtest(panels, use_trees=args.trees, model1_dir=args.model1_dir)
        print("summary:\n", bt.summary().round(4).to_string())
        print("by horizon bucket:\n", bt.by_bucket().round(4).to_string())
        print("by material:\n", bt.by_material().round(4).to_string())
    if args.submission:
        sub, landed = fill_submission(panels, args.submission, use_trees=args.trees,
                                      model1_dir=args.model1_dir, out_path=args.out)
        print(f"wrote {args.out} ({len(sub)} rows)")
        if args.landed_out:
            landed.to_csv(args.landed_out, index=False)
            print(f"wrote {args.landed_out} ({len(landed)} rows)")


if __name__ == "__main__":
    _main()
