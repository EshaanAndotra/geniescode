"""
model1.py - Model 1: disruption forecast (iter 4 = iter 3 + news story layer).

iter4 change: news_stories.py turns headline story chains (shortage, export ban, strike, clash,
sanctions warnings and their rumor endings) into onset timing for each supplier, lane and country,
estimated only from stories already closed at the origin. For a calm source with an open story,
first-onset curves combine as F = 1 - (1 - F_model)(1 - F_news), then go through the same duration
curves. Config(use_news_stories=False) gives back iter3 exactly. See README.md, section 5.


Fills `disruption_prob`: P(material disrupted at week T+h), where disrupted means at least
one of its 3 suppliers is hit by an active shock. Also produces the supplier / country /
lane hazards and duration curves that Model 3's scenario generator needs.

How it works (built from the event study of weeks 1-208):
  Every past shock comes from one of four sources, and together they rebuild the
  `disrupted` column exactly (check_labels() shows 0 mismatches on 2,080 rows):
    supplier  one (material, country) supplier: capacity < 70% of normal, or lead time
              >= base + 1 week while its lane is open. Warning: that supplier's futures
              premium ramps up 1-4 weeks before (premium > 0.06 preceded a shock 13/13 times).
    country   2+ suppliers of one country hit in the same week (Kestria, weeks 99-133).
              Warning: tension, fx depreciation, PMI drop.
    lane      shipping lane closed (transit days >= 1.5x normal); hits every supplier on
              it (Kestria and Sunda Coast share MERROW). Warning: congestion, ~1 week ahead.
    tariff    tariff above base. Announced 2-8 weeks ahead in tariffs.csv, so scheduled.
  For each source: P(active at T+h)
    active now -> S(a+h) / S(a): type-specific discrete Weibull duration curve, fitted on
                  de-duplicated events (15 real events, not 31 material-episodes), type
                  scales shrunk toward a pooled scale, mixed with a long exponential tail
                  (tail_weight) for a shock longer than anything in training.
    calm now   -> onset model: logistic for P(first onset within k weeks) with a free rate
                  per k and signal effects per lead bucket; increments q_k are convolved
                  with the duration curve: P(active at T+h) = sum_k q_k * S(h-k+1).
    tariff     -> announced: active from the effective week with prob p_announced.
  Material probability = 1 - product over its distinct sources of (1 - p), each lane and
  country counted once. Then clip to [0.02, 0.98]. Platt calibration is available but off by
  default (fit on earlier folds it made the backtest worse: too few events).

Backtest (26 origins every 4 weeks from 104, h = 1..13, weeks 1-208), Brier skill vs base rate:
  v1 duration + calm hazard 0.32 | v2 + tariff schedule 0.35 | v4 + signals ~0.43
  Run ablation(panels) to reproduce; numbers move when Phase 2 data is added.

Quick use:
    import model1
    data = model1.load_data("path/to/data")          # folder with the competition CSVs
    panels = model1.Panels(data)                      # optional: news_features=df (see below)
    m = model1.DisruptionModel().fit(panels)          # origin = last week in the data
    probs = m.predict()                               # material x week probabilities
    bt = model1.backtest(panels); print(bt.summary()) # rolling-origin Brier skill
    print(model1.ablation(panels))                    # ablation ladder for the video
    sub = model1.fill_submission(panels, "sample_submission.csv", out_path="submission.csv")
    sup_hz, src_hz = m.hazards()                      # hazards for Model 3
    curves = m.duration.curves()                      # duration curves for Model 3
    model1.signal_extremes(panels, train_end=208, recent_start=209)   # Phase 2 check

Command line:
    python model1.py --data DATA_DIR --backtest --ablation \
        --submission sample_submission.csv --out submission_model1.csv --hazards-out hazards.csv

News hook: pass a DataFrame to Panels(news_features=...) keyed by 'week' plus one of
'lane' | ('material','country') | 'country' | 'material'. Every other numeric column becomes a
feature named news_<col>: lane-keyed columns feed the lane model, the others the supplier
model, and country-keyed columns also feed the country model. Build them leak-free (only
articles and labels known at that week). ablation() adds a "v4 + news" row automatically.

Requires: numpy, pandas, scipy, scikit-learn.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # news_stories.py lives next to this file

__all__ = [
    "Config", "Panels", "DisruptionModel", "DurationModel", "BacktestResult",
    "load_data", "backtest", "ablation", "fill_submission", "signal_extremes",
    "check_labels", "brier_skill",
]

# ----------------------------------------------------------------------------- constants
CAP_HIT = 0.70            # capacity / normal below this = supplier hit
LEAD_HIT = 1              # lead time >= base + this = hit
TARIFF_EPS = 0.005        # tariff above base + this = tariff shock
LANE_TRANSIT_MULT = 1.5   # transit >= this x normal = lane closed
BASELINE_WEEKS = 52       # weeks used to define "normal" transit per lane
PREM_NOISE = 0.02         # futures premium inside +/- this is normal noise
CONG_NOISE = 38.0         # congestion below this is normal (calm 99th pct ~36-43 outside MERROW/ATLANTIC)
TYPES = ("lane", "supplier", "country", "tariff")
LEAD_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 10**6))
SUPPLIER_FEATURES = ("prem_excess", "prem_chg2", "tension", "tension_chg3", "tension_z")
COUNTRY_FEATURES = ("tension", "tension_chg3", "tension_z", "fx_chg4", "pmi_chg4")
LANE_FEATURES = ("cong_excess", "cong_chg1", "cong_chg2", "cong_z")
DOC_ORIGINS = (156, 169, 182, 195)
FILES = ("outcomes", "suppliers", "supply_history", "tariffs", "prices", "shipping", "macro")
OPTIONAL_FILES = ("news", "demand")


@dataclass
class Config:
    horizon: int = 13
    use_signals: bool = True            # False = calm hazard only (ablation v1/v2)
    use_tariff_schedule: bool = True    # False = ignore tariff announcements (ablation v1)
    use_news: bool = True               # use news_* columns if Panels has them
    supplier_features: tuple = SUPPLIER_FEATURES
    country_features: tuple = COUNTRY_FEATURES
    lane_features: tuple = LANE_FEATURES
    C_supplier: float = 0.3             # L2 strength (smaller = more shrinkage)
    C_country: float = 0.1
    C_lane: float = 0.3
    p_announced: float = 0.95           # chance an announced tariff actually takes effect
    tail_weight: float = 0.15           # share of a long-tail component in every duration curve
    tail_mean: float = 40.0             # mean length (weeks) of that tail component
    shrink_sd: float = 0.5              # how far type curves may move from the pooled curve (log scale)
    floor: float = 0.02
    ceil: float = 0.98
    min_train_week: int = 27            # first origin week used for onset training rows
    lead_buckets: tuple = LEAD_BUCKETS   # lead-time groups that get their own rate and signal effects
    use_news_stories: bool = True        # iter4 news story layer
    story_timeout: int = 10              # a story with no new warning for this many weeks is closed
    story_prior_p: float = 0.5           # prior P(story becomes a shock)
    story_prior_n: float = 2.0           # prior strength (pseudo-stories)
    story_lead_prior_max: int = 6        # prior on weeks from first warning to onset: flat over 1..this
    story_types: tuple = ("shortage", "export", "strike", "clash", "sanction")


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


# ----------------------------------------------------------------------------- panels
def _trailing_z(x, window=26, min_periods=8):
    """z-score of x[:, t] against x[:, t-window:t] (past only)."""
    out = np.zeros_like(x, dtype=float)
    for t in range(x.shape[1]):
        lo = max(0, t - window)
        w = x[:, lo:t]
        if w.shape[1] >= min_periods:
            mu = np.nanmean(w, axis=1)
            sd = np.nanstd(w, axis=1)
            out[:, t] = (x[:, t] - mu) / np.maximum(sd, 1e-6)
    return np.nan_to_num(out)


def _lag_diff(x, lag):
    out = np.zeros_like(x, dtype=float)
    out[:, lag:] = x[:, lag:] - x[:, :-lag]
    return np.nan_to_num(out)


def _lag_ratio(x, lag):
    out = np.zeros_like(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, lag:] = x[:, lag:] / x[:, :-lag] - 1.0
    return np.nan_to_num(out)


class Panels:
    """Dense (unit x week) arrays for suppliers, lanes and materials."""

    def __init__(self, data, news_features=None):
        out = data["outcomes"]
        sup = data["suppliers"].reset_index(drop=True)
        feat_dfs = [data[k] for k in ("supply_history", "tariffs", "prices", "shipping", "macro")]
        self.first_week = int(min(d.week.min() for d in feat_dfs + [out]))
        # forecasts only need the feature files (labels are rebuilt from them); outcomes are for scoring
        self.last_week = int(min(d.week.max() for d in feat_dfs))
        self.last_outcome_week = int(out.week.max())
        self.weeks = np.arange(self.first_week, self.last_week + 1)
        W = len(self.weeks)
        self.W = W

        self.suppliers = sup
        self.materials = list(dict.fromkeys(sup.material))
        self.lanes = sorted(sup.lane.unique())
        lane_ix = {l: i for i, l in enumerate(self.lanes)}
        self.sup_lane = sup.lane.map(lane_ix).values
        self.sup_mat = sup.material.map({m: i for i, m in enumerate(self.materials)}).values
        self.mat_units = [np.where(self.sup_mat == i)[0] for i in range(len(self.materials))]
        self.mat_lanes = [np.unique(self.sup_lane[u]) for u in self.mat_units]
        keys = list(zip(sup.material, sup.country))

        def sup_arr(df, col):
            p = df[df.week.between(self.first_week, self.last_week)].pivot_table(
                index=["material", "country"], columns="week", values=col, aggfunc="first")
            p = p.reindex(index=pd.MultiIndex.from_tuples(keys), columns=self.weeks)
            return p.values.astype(float)

        def lane_arr(col):
            p = data["shipping"].pivot_table(index="lane", columns="week", values=col, aggfunc="first")
            return p.reindex(index=self.lanes, columns=self.weeks).values.astype(float)

        sh, tar, pr, mac = data["supply_history"], data["tariffs"], data["prices"], data["macro"]
        normal = sup.normal_capacity_units.values[:, None].astype(float)
        self.cap_ratio = sup_arr(sh, "capacity_units") / normal
        self.lt_excess = sup_arr(sh, "lead_time_weeks") - sup.base_lead_weeks.values[:, None]
        self.tariff_rate = sup_arr(tar, "tariff_rate")
        self.ann_rate = sup_arr(tar, "announced_rate")
        self.ann_eff = sup_arr(tar, "announced_effective_week")
        self.base_tariff = sup.base_tariff.values.astype(float)
        spot, fut = sup_arr(pr, "spot_price"), sup_arr(pr, "futures_4w")

        # lanes
        self.transit = lane_arr("transit_days")
        self.congestion = lane_arr("congestion_index")
        self.container = lane_arr("container_rate")
        nb = min(BASELINE_WEEKS, W)
        self.normal_transit = np.nanmedian(self.transit[:, :nb], axis=1)
        self.lane_closed = np.nan_to_num(self.transit) >= LANE_TRANSIT_MULT * self.normal_transit[:, None]

        # source states
        lane_closed_sup = self.lane_closed[self.sup_lane]
        cap = np.nan_to_num(self.cap_ratio, nan=1.0)
        lt = np.nan_to_num(self.lt_excess, nan=0.0)
        self.sup_active = (cap < CAP_HIT) | ((lt >= LEAD_HIT) & ~lane_closed_sup)
        self.tar_active = np.nan_to_num(self.tariff_rate, nan=0.0) > self.base_tariff[:, None] + TARIFF_EPS
        with np.errstate(invalid="ignore"):
            self.pending = (~np.isnan(self.ann_rate)) & (self.ann_rate > self.base_tariff[:, None] + TARIFF_EPS) \
                & (self.ann_eff > self.weeks[None, :])

        # material outcomes (NaN where unknown)
        p = out.pivot_table(index="material", columns="week", values="disrupted", aggfunc="first")
        self.disrupted = p.reindex(index=self.materials, columns=self.weeks).values.astype(float)

        # country features (macro), then supplier features (own futures premium + its country's macro)
        self.countries = sorted(sup.country.unique())
        self.sup_country = sup.country.map({c: i for i, c in enumerate(self.countries)}).values

        def cty_arr(col):
            p = mac.pivot_table(index="country", columns="week", values=col, aggfunc="first")
            return p.reindex(index=self.countries, columns=self.weeks).values.astype(float)

        tension = cty_arr("tension_index")
        self.cty_feat = {
            "tension": np.nan_to_num(tension, nan=np.nanmedian(tension)),
            "tension_chg3": _lag_diff(tension, 3),
            "tension_z": _trailing_z(tension),
            "fx_chg4": _lag_ratio(cty_arr("fx_rate"), 4),
            "pmi_chg4": _lag_diff(cty_arr("pmi"), 4),
        }
        self.sup_feat = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            prem = fut / spot - 1.0
        self.sup_feat["prem"] = np.nan_to_num(prem)
        self.sup_feat["prem_chg2"] = _lag_diff(np.nan_to_num(prem), 2)
        self.sup_feat["prem_excess"] = np.maximum(np.nan_to_num(prem) - PREM_NOISE, 0.0)
        for f, arr in self.cty_feat.items():
            self.sup_feat[f] = arr[self.sup_country]

        # split supplier-level activity into single-supplier shocks and whole-country events
        # (type is decided in the onset week, so this uses no future information)
        U = len(sup)
        self.sup_in_country_event = np.zeros_like(self.sup_active)
        self.sup_onset_single = np.zeros_like(self.sup_active)   # onset weeks of single-supplier shocks
        for u, s, e, _ in _episodes(self.sup_active):
            if _sup_onset_type(self, u, s) == "country":
                self.sup_in_country_event[u, s: e + 1] = True
            else:
                self.sup_onset_single[u, s] = True
        self.country_active = np.zeros((len(self.countries), W), dtype=bool)
        for u in range(U):
            self.country_active[self.sup_country[u]] |= self.sup_in_country_event[u]

        # lane features
        self.lane_feat = {
            "congestion": np.nan_to_num(self.congestion, nan=np.nanmedian(self.congestion)),
            "cong_chg1": _lag_diff(self.congestion, 1),
            "cong_chg2": _lag_diff(self.congestion, 2),
            "cong_z": _trailing_z(self.congestion),
            "cong_excess": np.maximum(np.nan_to_num(self.congestion, nan=0.0) - CONG_NOISE, 0.0),
        }
        self.news_sup, self.news_lane, self.news_cty = [], [], []
        if news_features is not None:
            self._add_news(news_features)
        # iter4: tagged story headlines (news.csv), stories built lazily
        self.news_tags = None
        self._stories = {}
        if data.get("news") is not None:
            from news_stories import parse_news
            nw = data["news"]
            self.news_tags = parse_news(nw[nw.week <= self.last_week], sup)

    def stories(self, timeout=10):
        """Story table (news_stories.build_stories) on all weeks in the data; cached."""
        if self.news_tags is None:
            return None
        if timeout not in self._stories:
            from news_stories import build_stories
            lane_ix = {l: i for i, l in enumerate(self.lanes)}
            cty_ix = {c: i for i, c in enumerate(self.countries)}
            off = self.first_week - 1                  # array column = week - 1 - off

            def active_of(src, key):
                if src == "sup":
                    row = self.sup_active[int(key)]
                elif src == "lane":
                    row = self.lane_closed[lane_ix[key]]
                else:
                    row = self.country_active[cty_ix[key]]
                return np.r_[np.zeros(off, bool), row]
            self._stories[timeout] = build_stories(self.news_tags, active_of, self.last_week, timeout)
        return self._stories[timeout]

    def _add_news(self, nf):
        nf = nf.copy()
        if "lane" in nf.columns:
            keys, target = ["lane"], "lane"
        elif {"material", "country"} <= set(nf.columns):
            keys, target = ["material", "country"], "sup"
        elif "country" in nf.columns:
            keys, target = ["country"], "sup"
        elif "material" in nf.columns:
            keys, target = ["material"], "sup"
        else:
            raise ValueError("news_features needs 'week' plus 'lane', 'country', 'material' or both of the last two")
        cols = [c for c in nf.columns if c not in keys + ["week", "date"] and pd.api.types.is_numeric_dtype(nf[c])]
        for c in cols:
            name = f"news_{c}"
            p = nf.pivot_table(index=keys, columns="week", values=c, aggfunc="first")
            if target == "lane":
                arr = p.reindex(index=self.lanes, columns=self.weeks).values
                self.lane_feat[name] = np.nan_to_num(arr.astype(float))
                self.news_lane.append(name)
            else:
                if keys == ["material", "country"]:
                    idx = pd.MultiIndex.from_tuples(list(zip(self.suppliers.material, self.suppliers.country)))
                else:
                    idx = self.suppliers[keys[0]].values
                arr = p.reindex(index=idx, columns=self.weeks).values
                self.sup_feat[name] = np.nan_to_num(arr.astype(float))
                self.news_sup.append(name)
                if keys == ["country"]:   # country-level news also feeds the country-event model
                    carr = p.reindex(index=self.countries, columns=self.weeks).values
                    self.cty_feat[name] = np.nan_to_num(carr.astype(float))
                    self.news_cty.append(name)

    def idx(self, week):
        return int(week) - self.first_week

    def week(self, idx):
        return int(idx) + self.first_week


def check_labels(panels):
    """Confusion table: 'any source active' (rebuilt from supply/tariff/shipping) vs disrupted."""
    rows = []
    for mi, units in enumerate(panels.mat_units):
        any_src = panels.sup_active[units].any(0) | panels.tar_active[units].any(0) \
            | panels.lane_closed[panels.mat_lanes[mi]].any(0)
        y = panels.disrupted[mi]
        ok = ~np.isnan(y)
        rows.append(pd.DataFrame({"rebuilt": any_src[ok].astype(int), "disrupted": y[ok].astype(int)}))
    df = pd.concat(rows, ignore_index=True)
    return pd.crosstab(df.rebuilt, df.disrupted)


# ----------------------------------------------------------------------------- episodes / events
def _episodes(active):
    """Runs of True per row: list of (unit, start_idx, end_idx, censored)."""
    eps = []
    U, T = active.shape
    for u in range(U):
        a = active[u].astype(int)
        d = np.diff(np.r_[0, a, 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
        for s, e in zip(starts, ends):
            eps.append((u, int(s), int(e), bool(e == T - 1)))
    return eps


def _sup_onset_type(panels, u, s):
    """'country' if 2+ suppliers of the same country start a supplier-level shock in week s
    (a ban/sanction hitting the whole country), else 'supplier'."""
    act = panels.sup_active
    same = np.where(panels.suppliers.country.values == panels.suppliers.country.values[u])[0]
    starts = act[same, s] & (~act[same, s - 1] if s > 0 else True)
    return "country" if starts.sum() >= 2 else "supplier"


def build_events(panels, t0):
    """De-duplicated shock events up to index t0 (inclusive), with type, length, censoring."""
    sl = slice(0, t0 + 1)
    countries = panels.suppliers.country.values
    rows = []
    for u, s, e, c in _episodes(panels.lane_closed[:, sl]):
        rows.append(dict(type="lane", key=f"lane:{panels.lanes[u]}:{s}", length=e - s + 1, censored=c, start=s))
    for u, s, e, c in _episodes(panels.sup_active[:, sl]):
        if _sup_onset_type(panels, u, s) == "country":
            rows.append(dict(type="country", key=f"country:{countries[u]}:{s}", length=e - s + 1, censored=c, start=s))
        else:
            rows.append(dict(type="supplier", key=f"supplier:{u}:{s}", length=e - s + 1, censored=c, start=s))
    for u, s, e, c in _episodes(panels.tar_active[:, sl]):
        rows.append(dict(type="tariff", key=f"tariff:{countries[u]}:{s}", length=e - s + 1, censored=c, start=s))
    if not rows:
        return pd.DataFrame(columns=["type", "key", "length", "censored", "start", "n_units"])
    ev = pd.DataFrame(rows)
    ev = ev.groupby("key", as_index=False).agg(type=("type", "first"), length=("length", "max"),
                                                censored=("censored", "max"), start=("start", "first"),
                                                n_units=("length", "size"))
    # a source that started in the very first week may have started earlier: treat as censored on the left
    # (kept, since it only lengthens; small effect)
    return ev


# ----------------------------------------------------------------------------- duration
def _weib_S(x, lam, k):
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        return np.exp(-np.power(np.maximum(x - 1.0, 0.0) / lam, k))


class DurationModel:
    """Type-specific discrete Weibull survival S(x) = P(duration >= x), shared shape,
    type scales shrunk toward a pooled scale, mixed with a long exponential tail."""

    def __init__(self, shrink_sd=0.5, tail_weight=0.10, tail_mean=40.0):
        self.shrink_sd, self.tail_weight, self.tail_mean = shrink_sd, tail_weight, tail_mean
        self.k, self.lam0, self.lam = 1.0, 8.0, {t: 8.0 for t in TYPES}
        self.n_events = {t: 0 for t in TYPES}

    def fit(self, events):
        ev = events
        if len(ev) == 0:
            return self
        ti = ev.type.map({t: i for i, t in enumerate(TYPES)}).values
        L = ev.length.values.astype(float)
        C = ev.censored.values.astype(bool)
        sd = self.shrink_sd

        def nll(theta):
            logk, loglam0, loglam = theta[0], theta[1], theta[2:]
            k = np.exp(logk)
            lam = np.exp(loglam[ti])
            s_d = _weib_S(L, lam, k)
            s_d1 = _weib_S(L + 1, lam, k)
            ll = np.where(C, np.log(np.maximum(s_d, 1e-300)), np.log(np.maximum(s_d - s_d1, 1e-300)))
            pen = np.sum((loglam - loglam0) ** 2) / (2 * sd ** 2) + logk ** 2 / 2.0 \
                + (loglam0 - np.log(8.0)) ** 2 / (2 * 2.0 ** 2)
            return -ll.sum() + pen

        theta0 = np.r_[0.0, np.log(8.0), np.full(len(TYPES), np.log(8.0))]
        res = minimize(nll, theta0, method="L-BFGS-B")
        th = res.x
        self.k = float(np.exp(th[0]))
        self.lam0 = float(np.exp(th[1]))
        self.lam = {t: float(np.exp(th[2 + i])) for i, t in enumerate(TYPES)}
        self.n_events = ev.type.value_counts().reindex(TYPES, fill_value=0).to_dict()
        return self

    def S(self, typ, x):
        x = np.asarray(x, dtype=float)
        base = _weib_S(x, self.lam[typ], self.k)
        tail = np.exp(-np.maximum(x - 1.0, 0.0) / self.tail_mean)
        return (1 - self.tail_weight) * base + self.tail_weight * tail

    def S_mix(self, weights, x):
        return sum(w * self.S(t, x) for t, w in weights.items())

    def p_cont(self, typ, age, h):
        """P(still active h weeks ahead | currently in week `age` of the shock)."""
        h = np.asarray(h, dtype=float)
        return self.S(typ, age + h) / np.maximum(self.S(typ, age), 1e-12)

    def curves(self, max_x=52):
        x = np.arange(1, max_x + 1)
        return pd.DataFrame({"weeks_into_shock": x, **{t: self.S(t, x) for t in TYPES}})


# ----------------------------------------------------------------------------- onset
def _bucket(k, buckets=LEAD_BUCKETS):
    k = np.asarray(k)
    b = np.zeros(k.shape, dtype=int)
    for i, (lo, hi) in enumerate(buckets):
        b[(k >= lo) & (k <= hi)] = i
    return b


def _next_active(active):
    """next_act[u, t] = smallest t' > t with active[u, t'] (or a large number)."""
    U, T = active.shape
    nxt = np.full((U, T), 10**9, dtype=np.int64)
    run = np.full(U, 10**9, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        nxt[:, t] = run
        run = np.where(active[:, t], t, run)
    return nxt


class OnsetModel:
    """Onset timing for one source type. Fits F_k = P(first onset within k weeks | calm at T,
    signals at T) for k = 1..K (logistic: free rate per k, signal effects per lead bucket),
    then returns q_k = F_k - F_(k-1) = P(first onset exactly k weeks ahead)."""

    def __init__(self, features, C=0.3, buckets=LEAD_BUCKETS, K=13):
        self.features, self.C, self.buckets, self.K = list(features), C, tuple(buckets), int(K)
        self.clf, self.mu, self.sd, self.const = None, None, None, None

    def _design(self, Xz, k):
        n = len(k)
        kk = np.clip(np.asarray(k), 1, self.K)
        ohk = np.zeros((n, self.K))
        ohk[np.arange(n), kk - 1] = 1.0
        cols = [ohk[:, 1:] * 10.0]   # scaled so the L2 penalty barely touches the per-k rates
        if Xz.shape[1]:
            b = _bucket(kk, self.buckets)
            ohb = np.zeros((n, len(self.buckets)))
            ohb[np.arange(n), b] = 1.0
            cols.append((Xz[:, :, None] * ohb[:, None, :]).reshape(n, -1))
        return np.hstack(cols)

    def fit(self, active, feats, t0, K, t_min, positive=None):
        """active: [U, W] bool; feats: dict name -> [U, W]; train on origins t in [t_min, t0-1].
        positive: optional [U, W] bool; only onsets in weeks marked True count as events."""
        self.K = int(K)
        nxt = _next_active(active[:, : t0 + 1])
        uu, tt = np.where(~active[:, t_min:t0])
        tt = tt + t_min
        X = np.stack([feats[f][uu, tt] for f in self.features], axis=1) if self.features else np.zeros((len(uu), 0))
        self.mu = X.mean(0) if X.shape[1] else np.zeros(0)
        self.sd = X.std(0) + 1e-9 if X.shape[1] else np.zeros(0)
        Xz = (X - self.mu) / self.sd if X.shape[1] else X
        first = nxt[uu, tt]
        counts = first <= t0
        if positive is not None:
            ok = np.zeros(len(first), dtype=bool)
            ok[counts] = positive[uu[counts], first[counts]]
            counts &= ok
        rep = np.repeat(np.arange(len(uu)), K)
        k = np.tile(np.arange(1, K + 1), len(uu))
        t_lab = tt[rep] + k
        keep = t_lab <= t0
        rep, k, t_lab = rep[keep], k[keep], t_lab[keep]
        y = (counts[rep] & (first[rep] <= t_lab)).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            self.const = (y.sum() + 0.5) / (len(y) + 1.0)
            return self
        self.clf = LogisticRegression(C=self.C, max_iter=3000)
        self.clf.fit(self._design(Xz[rep], k), y)
        return self

    def predict_q(self, X_now, H):
        """X_now: [n, F] raw features at the origin. Returns q [n, H] (first-onset probability by week)."""
        n = X_now.shape[0]
        Kp = min(H, self.K)
        if self.clf is None:
            F = 1 - (1 - self.const / max(self.K, 1)) ** np.arange(1, Kp + 1)[None, :].repeat(n, 0)
        else:
            Xz = (X_now - self.mu) / self.sd if X_now.shape[1] else X_now
            k = np.tile(np.arange(1, Kp + 1), n)
            F = self.clf.predict_proba(self._design(np.repeat(Xz, Kp, axis=0), k))[:, 1].reshape(n, Kp)
        F = np.maximum.accumulate(np.clip(F, 0, 1), axis=1)
        q = np.diff(F, axis=1, prepend=0.0)
        if H > Kp:   # beyond the trained horizon: keep the late weekly rate
            tail = q[:, max(0, Kp - 4):].mean(1, keepdims=True)
            q = np.hstack([q, np.repeat(tail, H - Kp, axis=1)])
            q = np.minimum(q, np.maximum(1 - np.cumsum(q, axis=1) + q, 0))
        return q


def _convolve(q, S_vals):
    """P(active at h) = sum_{k<=h} q_k * S(h-k+1). q [n, H], S_vals [H] = S(1..H)."""
    n, H = q.shape
    P = np.zeros((n, H))
    for h in range(1, H + 1):
        P[:, h - 1] = (q[:, :h] * S_vals[h - 1::-1][None, :]).sum(1)
    return np.clip(P, 0.0, 1.0)


# ----------------------------------------------------------------------------- calibration
class Platt:
    """logit(p_cal) = a + b*logit(p) + c*h"""

    def __init__(self):
        self.a, self.b, self.c = 0.0, 1.0, 0.0
        self.fitted = False

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def fit(self, p, h, y):
        X = np.column_stack([self._logit(np.asarray(p)), np.asarray(h, dtype=float)])
        clf = LogisticRegression(C=1e4, max_iter=1000).fit(X, np.asarray(y).astype(int))
        self.a, (self.b, self.c) = float(clf.intercept_[0]), clf.coef_[0].astype(float)
        self.fitted = True
        return self

    def apply(self, p, h):
        if not self.fitted:
            return np.asarray(p)
        z = self.a + self.b * self._logit(np.asarray(p)) + self.c * np.asarray(h, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------- main model
class DisruptionModel:
    def __init__(self, config=None, **overrides):
        self.cfg = replace(config or Config(), **overrides)
        self.duration = None
        self.calibrator = Platt()

    # -- fitting
    def fit(self, panels, origin=None):
        cfg = self.cfg
        self.panels = P = panels
        self.origin = int(origin if origin is not None else P.last_week)
        if self.origin > P.last_week:
            raise ValueError(f"origin {self.origin} is after the last week with data ({P.last_week})")
        t0 = P.idx(self.origin)
        self.t0 = t0
        t_min = max(0, P.idx(cfg.min_train_week))

        self.events = build_events(P, t0)
        self.duration = DurationModel(cfg.shrink_sd, cfg.tail_weight, cfg.tail_mean).fit(self.events)

        sup_f = list(cfg.supplier_features) if cfg.use_signals else []
        cty_f = list(cfg.country_features) if cfg.use_signals else []
        lane_f = list(cfg.lane_features) if cfg.use_signals else []
        if cfg.use_news:
            sup_f += P.news_sup
            cty_f += P.news_cty
            lane_f += P.news_lane
        self.sup_onset = OnsetModel(sup_f, cfg.C_supplier, cfg.lead_buckets, cfg.horizon).fit(
            P.sup_active, P.sup_feat, t0, cfg.horizon, t_min, positive=P.sup_onset_single)
        self.cty_onset = OnsetModel(cty_f, cfg.C_country, cfg.lead_buckets, cfg.horizon).fit(P.country_active, P.cty_feat, t0, cfg.horizon, t_min)
        self.lane_onset = OnsetModel(lane_f, cfg.C_lane, cfg.lead_buckets, cfg.horizon).fit(P.lane_closed, P.lane_feat, t0, cfg.horizon, t_min)

        # tariff base hazard (unannounced onsets per calm supplier-week)
        ta = P.tar_active[:, : t0 + 1]
        onsets = (ta[:, 1:] & ~ta[:, :-1]).sum()
        calm_weeks = (~ta[:, :-1]).sum()
        self.q_tariff = float((onsets + 0.5) / (calm_weeks + 1.0))
        return self

    # -- source probabilities
    def _source_probs(self, H):
        P, cfg, D = self.panels, self.cfg, self.duration
        t0 = self.t0
        h = np.arange(1, H + 1)
        U = len(P.suppliers)

        def run_start(active_row):
            s = t0
            while s > 0 and active_row[s - 1]:
                s -= 1
            return s

        def calm_or_active(active_now, model, feats, typ):
            n = len(active_now)
            p, q, state = np.zeros((n, H)), np.zeros((n, H)), [""] * n
            calm = np.where(~active_now)[0]
            if len(calm):
                X = np.stack([feats[f][calm, t0] for f in model.features], axis=1) \
                    if model.features else np.zeros((len(calm), 0))
                q[calm] = model.predict_q(X, H)
                p[calm] = _convolve(q[calm], D.S(typ, h))
            return p, q, state, np.where(active_now)[0]

        # single-supplier shocks
        p_sup, q_sup, sup_state, act = calm_or_active(P.sup_active[:, t0], self.sup_onset, P.sup_feat, "supplier")
        for u in act:
            if P.sup_in_country_event[u, t0]:
                sup_state[u] = "in country event"      # covered by the country source
                continue
            age = t0 - run_start(P.sup_active[u]) + 1
            p_sup[u] = D.p_cont("supplier", age, h)
            sup_state[u] = f"supplier:week{age}"

        # whole-country events
        p_cty, q_cty, cty_state, act = calm_or_active(P.country_active[:, t0], self.cty_onset, P.cty_feat, "country")
        for c in act:
            age = t0 - run_start(P.country_active[c]) + 1
            p_cty[c] = D.p_cont("country", age, h)
            cty_state[c] = f"country:week{age}"

        # lane closures
        p_lane, q_lane, lane_state, act = calm_or_active(P.lane_closed[:, t0], self.lane_onset, P.lane_feat, "lane")
        for l in act:
            age = t0 - run_start(P.lane_closed[l]) + 1
            p_lane[l] = D.p_cont("lane", age, h)
            lane_state[l] = f"lane:week{age}"

        # iter4: news stories for calm sources
        news_open = []
        if cfg.use_news_stories and P.news_tags is not None:
            from news_stories import story_q
            st = P.stories(cfg.story_timeout)
            lane_ix = {l: i for i, l in enumerate(P.lanes)}
            cty_ix = {c: i for i, c in enumerate(P.countries)}
            for item in story_q(st, self.origin, H, cfg.story_prior_p, cfg.story_prior_n,
                                lead_prior_max=cfg.story_lead_prior_max, types=cfg.story_types):
                if item["source"] == "sup":
                    i, qq, pp, states, typ = int(item["key"]), q_sup, p_sup, sup_state, "supplier"
                elif item["source"] == "lane":
                    i, qq, pp, states, typ = lane_ix[item["key"]], q_lane, p_lane, lane_state, "lane"
                else:
                    i, qq, pp, states, typ = cty_ix[item["key"]], q_cty, p_cty, cty_state, "country"
                if states[i]:
                    continue                            # source already in a shock
                F = 1 - (1 - np.cumsum(qq[i])) * (1 - np.cumsum(item["q"]))
                qq[i] = np.diff(np.clip(F, 0, 1), prepend=0.0)
                pp[i] = _convolve(qq[i][None, :], D.S(typ, h))[0]
                news_open.append({k: v for k, v in item.items() if k != "q"} | {"p_active_max": float(pp[i].max())})
        self.news_open = pd.DataFrame(news_open)

        # tariff sources
        p_tar = np.zeros((U, H))
        tar_state = [""] * U
        S_tar = D.S("tariff", h)
        base_q = np.full((1, H), self.q_tariff)
        base_p = _convolve(base_q, S_tar)[0]
        for u in range(U):
            if P.tar_active[u, t0]:
                s = t0
                while s > 0 and P.tar_active[u, s - 1]:
                    s -= 1
                age = t0 - s + 1
                p_tar[u] = D.p_cont("tariff", age, h)
                tar_state[u] = f"tariff:week{age}"
            elif cfg.use_tariff_schedule and P.pending[u, t0]:
                kE = int(P.ann_eff[u, t0]) - self.origin
                sched = np.where(h >= kE, cfg.p_announced * D.S("tariff", h - kE + 1), 0.0)
                p_tar[u] = 1 - (1 - sched) * (1 - base_p)
                tar_state[u] = f"announced:effective_w{int(P.ann_eff[u, t0])}"
            else:
                p_tar[u] = base_p
        # country source mapped onto each supplier
        p_cty_u = p_cty[P.sup_country]
        return dict(p_sup=p_sup, q_sup=q_sup, sup_state=sup_state, p_lane=p_lane, q_lane=q_lane,
                    lane_state=lane_state, p_tar=p_tar, tar_state=tar_state,
                    p_cty=p_cty, q_cty=q_cty, cty_state=cty_state, p_cty_u=p_cty_u)

    # -- prediction
    def predict(self, horizon=None, calibrate=True, raw=False):
        """Material-level probabilities for weeks origin+1 .. origin+horizon."""
        H = int(horizon or self.cfg.horizon)
        P = self.panels
        src = self._source_probs(H)
        rows = []
        for mi, m in enumerate(P.materials):
            units, lanes = P.mat_units[mi], P.mat_lanes[mi]
            ctys = np.unique(P.sup_country[units])
            none = np.prod(1 - src["p_sup"][units], axis=0) * np.prod(1 - src["p_tar"][units], axis=0) \
                * np.prod(1 - src["p_lane"][lanes], axis=0) * np.prod(1 - src["p_cty"][ctys], axis=0)
            p_raw = 1 - none
            in_shock = bool(P.sup_active[units, self.t0].any() or P.tar_active[units, self.t0].any()
                            or P.lane_closed[lanes, self.t0].any())
            for h in range(1, H + 1):
                rows.append((self.origin, h, self.origin + h, m, p_raw[h - 1], in_shock))
        df = pd.DataFrame(rows, columns=["origin", "h", "week", "material", "p_raw", "in_shock_at_origin"])
        p = self.calibrator.apply(df.p_raw.values, df.h.values) if (calibrate and not raw) else df.p_raw.values
        df["disruption_prob"] = p if raw else np.clip(p, self.cfg.floor, self.cfg.ceil)
        self._last_src = src
        return df

    def hazards(self, horizon=None):
        """Per-supplier and per-lane hazards for Model 3's scenario generator.

        Returns (suppliers_df, sources_df):
          suppliers_df: origin, week, h, material, country, lane,
                        p_supplier  P(single-supplier shock active)   q_supplier  P(one starts that week)
                        p_country   P(country-wide event active)       q_country   P(one starts that week)
                        p_lane      P(its lane closed)                 q_lane      P(a closure starts that week)
                        p_tariff    P(tariff shock active)
                        p_any       1 - product of (1 - each of the four)
                        state_*     what is active at the origin and for how long
          sources_df:   origin, week, h, source ('lane' or 'country'), name, p_active, q_onset, state
                        (draw lane/country events once per future and apply them to every supplier
                        on that lane / in that country, so correlated hits stay correlated)
        Duration curves for sampling lengths: self.duration.curves().
        """
        H = int(horizon or self.cfg.horizon)
        P = self.panels
        src = self._source_probs(H)
        rows = []
        for u, r in P.suppliers.iterrows():
            l, c = P.sup_lane[u], P.sup_country[u]
            for h in range(1, H + 1):
                ps, pc = src["p_sup"][u, h - 1], src["p_cty"][c, h - 1]
                pl, pt = src["p_lane"][l, h - 1], src["p_tar"][u, h - 1]
                rows.append(dict(origin=self.origin, week=self.origin + h, h=h, material=r.material,
                                 country=r.country, lane=r.lane,
                                 p_supplier=ps, q_supplier=src["q_sup"][u, h - 1],
                                 p_country=pc, q_country=src["q_cty"][c, h - 1],
                                 p_lane=pl, q_lane=src["q_lane"][l, h - 1], p_tariff=pt,
                                 p_any=1 - (1 - ps) * (1 - pc) * (1 - pl) * (1 - pt),
                                 state_supplier=src["sup_state"][u], state_country=src["cty_state"][c],
                                 state_lane=src["lane_state"][l], state_tariff=src["tar_state"][u]))
        srows = []
        for kind, names, pk, qk, sk in (("lane", P.lanes, "p_lane", "q_lane", "lane_state"),
                                        ("country", P.countries, "p_cty", "q_cty", "cty_state")):
            for i, name in enumerate(names):
                for h in range(1, H + 1):
                    srows.append(dict(origin=self.origin, week=self.origin + h, h=h, source=kind, name=name,
                                      p_active=src[pk][i, h - 1], q_onset=src[qk][i, h - 1], state=src[sk][i]))
        return pd.DataFrame(rows), pd.DataFrame(srows)

    def coefficients(self):
        """Fitted onset coefficients (standardized features, per lead bucket) for inspection."""
        out = []
        for name, om in (("supplier", self.sup_onset), ("country", self.cty_onset), ("lane", self.lane_onset)):
            if om.clf is None or not om.features:
                continue
            nb = len(om.buckets)
            coef = om.clf.coef_[0][om.K - 1:].reshape(len(om.features), nb)
            df = pd.DataFrame(coef, index=om.features, columns=[f"k{lo}-{hi if hi < 999 else '+'}" for lo, hi in om.buckets])
            df.insert(0, "model", name)
            out.append(df)
        return pd.concat(out) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- scoring
def brier_skill(p, y, baseline):
    p, y, b = map(np.asarray, (p, y, baseline))
    bs, bb = np.mean((p - y) ** 2), np.mean((b - y) ** 2)
    return 1 - bs / bb if bb > 0 else np.nan


@dataclass
class BacktestResult:
    rows: pd.DataFrame
    config: Config = field(default_factory=Config)

    def _skill(self, df, col):
        return brier_skill(df[col], df.y, df.baseline) if len(df) else np.nan

    def summary(self, col=None):
        """Brier skill overall, by state at origin, horizon bucket, plus the doc's 4 origins."""
        col = col or ("p_cal" if "p_cal" in self.rows else "p")
        r = self.rows
        hb = pd.cut(r.h, [0, 4, 8, 13, 10**6], labels=["h1-4", "h5-8", "h9-13", "h14+"])
        out = {"overall": self._skill(r, col),
               "calm_at_origin": self._skill(r[~r.in_shock_at_origin], col),
               "in_shock_at_origin": self._skill(r[r.in_shock_at_origin], col)}
        for b in hb.cat.categories:
            if (hb == b).any():
                out[b] = self._skill(r[hb == b], col)
        out["doc_origins"] = self._skill(r[r.origin.isin(DOC_ORIGINS)], col)
        out["brier"] = float(np.mean((r[col] - r.y) ** 2))
        out["brier_baseline"] = float(np.mean((r.baseline - r.y) ** 2))
        out["n_rows"] = len(r)
        out["n_origins"] = r.origin.nunique()
        return pd.Series(out)

    def by(self, key, col=None):
        col = col or ("p_cal" if "p_cal" in self.rows else "p")
        return self.rows.groupby(key).apply(
            lambda d: pd.Series({"skill": self._skill(d, col), "n": len(d), "events": d.y.sum()}),
            include_groups=False)

    def reliability(self, col=None, bins=10):
        col = col or ("p_cal" if "p_cal" in self.rows else "p")
        r = self.rows
        b = pd.cut(r[col], np.linspace(0, 1, bins + 1), include_lowest=True)
        return r.groupby(b, observed=True).agg(mean_pred=(col, "mean"), actual_rate=("y", "mean"), n=("y", "size"))


def default_origins(panels, start=104, step=4):
    """Every `step` weeks from `start` to the last week with outcomes (more folds than the doc's 4,
    because the doc's origins only contain ~4 new shocks)."""
    last = min(panels.last_week, panels.last_outcome_week)
    return list(range(start, last, step))


def backtest(panels, origins=None, config=None, calibrate=False, min_cal_pos=15, verbose=False, **overrides):
    """Rolling-origin backtest. Every fit uses only data up to its origin. If calibrate=True,
    each origin's Platt calibrator is fitted on earlier origins' predictions whose label week
    is <= the current origin (never on the fold being scored)."""
    cfg = replace(config or Config(), **overrides)
    origins = sorted(origins or default_origins(panels))
    all_rows = []
    last_scored = min(panels.last_week, panels.last_outcome_week)
    for T in origins:
        if T >= last_scored:
            continue
        m = DisruptionModel(cfg).fit(panels, origin=T)
        H = min(cfg.horizon, last_scored - T)
        df = m.predict(H, calibrate=False)
        df["p"] = df.disruption_prob
        ti = panels.idx(T)
        base = np.nanmean(panels.disrupted[:, : ti + 1], axis=1)
        mi = {mm: i for i, mm in enumerate(panels.materials)}
        df["baseline"] = [base[mi[mm]] for mm in df.material]
        df["y"] = [panels.disrupted[mi[mm], panels.idx(w)] for mm, w in zip(df.material, df.week)]
        if calibrate:
            prev = pd.concat(all_rows) if all_rows else pd.DataFrame()
            df["p_cal"] = df.p
            if len(prev):
                pool = prev[prev.week <= T]
                if pool.y.sum() >= min_cal_pos and pool.origin.nunique() >= 3 and (1 - pool.y).sum() > 0:
                    cal = Platt().fit(pool.p_raw, pool.h, pool.y)
                    df["p_cal"] = np.clip(cal.apply(df.p_raw, df.h), cfg.floor, cfg.ceil)
        all_rows.append(df)
        if verbose:
            print(f"origin {T}: rows {len(df)}, events {int(df.y.sum())}")
    rows = pd.concat(all_rows, ignore_index=True)
    rows = rows[~rows.y.isna()]
    return BacktestResult(rows=rows, config=cfg)


def ablation(panels, origins=None, config=None):
    """Ablation ladder (doc's v0..v4). Returns a table of Brier skill per version."""
    cfg = config or Config()
    versions = [
        ("v1 duration + calm hazard", dict(use_signals=False, use_tariff_schedule=False, use_news=False), False),
        ("v2 + tariff schedule", dict(use_signals=False, use_tariff_schedule=True, use_news=False), False),
        ("v4 + market/shipping/macro signals", dict(use_signals=True, use_tariff_schedule=True, use_news=False), False),
    ]
    if panels.news_sup or panels.news_lane:
        versions.append(("v4 + news", dict(use_news=True), False))
    versions.append(("final + Platt calibration", dict(use_news=bool(panels.news_sup or panels.news_lane)), True))
    out = {}
    first = None
    for name, ov, cal in versions:
        bt = backtest(panels, origins, replace(cfg, **ov), calibrate=cal)
        out[name] = bt.summary()
        first = first or bt
    base = first.summary().copy()
    for c in base.index:
        if c not in ("brier_baseline", "n_rows", "n_origins"):
            base[c] = 0.0
    base["brier"] = base["brier_baseline"]
    out = {"v0 base rate": base, **out}
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------- submission
def fill_submission(panels, sample, windows=((209, 221), (222, 234)), config=None, calibrate=False,
                    cal_origins=None, out_path=None):
    """Fill disruption_prob in a submission file. Each window is forecast from
    origin = min(window start - 1, last data week). Other columns are left untouched.
    calibrate=True fits Platt on a backtest up to each window's origin (off by default:
    with ~15 events it made the backtest worse)."""
    cfg = config or Config()
    sub = pd.read_csv(sample) if isinstance(sample, str) else sample.copy()
    parts = sub.id.str.extract(r"^w(\d+)_(.+)$")
    sub_week, sub_mat = parts[0].astype(int), parts[1]
    probs = pd.Series(np.nan, index=sub.index)
    for start, end in windows:
        origin = min(start - 1, panels.last_week)
        H = end - origin
        if H > cfg.horizon:
            warnings.warn(f"window {start}-{end}: data ends at week {panels.last_week}, so weeks beyond "
                          f"{origin + cfg.horizon} are forecast {cfg.horizon + 1}+ weeks ahead (hazard prior only). "
                          "Re-run once the newer data is loaded.")
        m = DisruptionModel(cfg).fit(panels, origin=origin)
        if calibrate:
            bt = backtest(panels, cal_origins or [o for o in default_origins(panels) if o < origin], cfg)
            pool = bt.rows[bt.rows.week <= origin]
            if pool.y.sum() >= 15:
                m.calibrator = Platt().fit(pool.p_raw, pool.h, pool.y)
        pred = m.predict(H)
        lut = {(int(w), mm): p for w, mm, p in zip(pred.week, pred.material, pred.disruption_prob)}
        mask = sub_week.between(start, end)
        probs[mask] = [lut.get((int(w), mm), np.nan) for w, mm in zip(sub_week[mask], sub_mat[mask])]
    if probs.isna().any():
        missing = sub.id[probs.isna()].tolist()[:5]
        raise ValueError(f"no forecast for {probs.isna().sum()} rows, e.g. {missing}")
    sub["disruption_prob"] = probs.round(4).values
    if out_path:
        sub.to_csv(out_path, index=False)
    return sub


# ----------------------------------------------------------------------------- diagnostics
def signal_extremes(panels, train_end, recent_start, recent_end=None):
    """Compare signals in [recent_start, recent_end] with their max up to train_end.
    Use it when Phase 2 data lands to spot signals past their training range."""
    recent_end = recent_end or panels.last_week
    a, b, c = panels.idx(train_end), panels.idx(recent_start), panels.idx(recent_end)
    rows = []
    for kind, feats, names in (("supplier", panels.sup_feat,
                                [f"{m}/{k}" for m, k in zip(panels.suppliers.material, panels.suppliers.country)]),
                               ("lane", panels.lane_feat, panels.lanes)):
        for f, arr in feats.items():
            tmax = np.nanmax(arr[:, : a + 1])
            rmax = np.nanmax(arr[:, b: c + 1], axis=1)
            for i, v in enumerate(rmax):
                rows.append(dict(kind=kind, unit=names[i], feature=f, recent_max=v, train_max=tmax,
                                 exceeds=bool(v > tmax)))
    df = pd.DataFrame(rows)
    return df.sort_values(["exceeds", "recent_max"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser(description="Model 1 disruption forecast")
    ap.add_argument("--data", required=True, help="folder with the competition CSVs")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="Platt-calibrate the submission column")
    ap.add_argument("--submission", help="sample_submission.csv (or your current submission) to fill")
    ap.add_argument("--out", default="submission_model1.csv")
    ap.add_argument("--hazards-out", help="write supplier hazards for Model 3 to this CSV")
    args = ap.parse_args()

    panels = Panels(load_data(args.data))
    print(f"data weeks {panels.first_week}-{panels.last_week}")
    print("label check (rebuilt vs disrupted):\n", check_labels(panels))
    if args.ablation:
        pd.set_option("display.width", 200)
        print(ablation(panels).round(3).to_string())
    if args.backtest:
        bt = backtest(panels, calibrate=args.calibrate)
        print(bt.summary().round(3).to_string())
    if args.submission:
        sub = fill_submission(panels, args.submission, calibrate=args.calibrate, out_path=args.out)
        print(f"wrote {args.out} ({len(sub)} rows)")
    if args.hazards_out:
        m = DisruptionModel().fit(panels)
        sup_h, src_h = m.hazards()
        sup_h.to_csv(args.hazards_out, index=False)
        root, ext = os.path.splitext(args.hazards_out)
        src_h.to_csv(f"{root}_sources{ext}", index=False)
        m.duration.curves().to_csv(f"{root}_duration_curves{ext}", index=False)
        print(f"wrote {args.hazards_out}, {root}_sources{ext}, {root}_duration_curves{ext}")


if __name__ == "__main__":
    _main()
