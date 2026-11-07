"""
score_local.py - score a submission locally on any window whose truth is in the data.

Score = 100 x (0.30 P + 0.40 F + 0.30 R), each part clipped to 0-1:
  P = 0.6 Brier skill (vs each material's historical disruption rate up to the origin)
    + 0.4 price skill (1 - MAPE / MAPE of carrying the origin's benchmark price forward)
  F = (naive cost - our cost) / (naive cost - hindsight cost)       (Model 3's simulator)
  R = 0.7 fill-rate skill in shock weeks + 0.3 diversification
Use it after Phase 2 lands: score your public-window submissions (weeks 209-221, origin 208)
and compare with the Kaggle public scores to check the simulator's assumptions.

The rules leave a few aggregation details open. `score` uses the defaults below; the other
`score_*` keys recompute the total under one alternative each, so a Kaggle score can be matched:
  score_fs_undef1   fill skill = 1 (not 0) when naive meets all shock-week demand and we do too
  score_R_div       R = diversification alone when fill skill is undefined
  score_F_mat       F = mean of per-material F (each clipped) instead of pooled costs
  score_fill_mat    fill skill = mean of per-material skills instead of pooled units
  score_div_units   diversification weighted by order units instead of averaged over rows
  score_base_all    Brier baseline = each material's rate over every released week (not to the origin)

  python score_local.py SUBMISSION.csv [--start 209 --end 221] [--start-stock sum5|5avg]
  python score_local.py --self-test       # the sample policy on a past window must score 0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline import find_data, load_module  # noqa: E402


def _fill_skill(fo, fn):
    if not np.isfinite(fn) or fn >= 1 - 1e-12:
        return np.nan
    return (fo - fn) / (1 - fn)


def score(world, sub, start, end, mode="sum5", fill_skill_if_naive_perfect=0.0):
    m3 = sys.modules["model3"]
    origin = start - 1
    df = sub.copy()
    parts = df.id.str.extract(r"^w(\d+)_(.+)$")
    df["week"], df["material"] = parts[0].astype(int), parts[1]
    df = df[df.week.between(start, end)]
    if df.week.max() > world.last_week:
        raise ValueError(f"truth only goes to week {world.last_week}")
    mi = {m: i for i, m in enumerate(world.materials)}
    y = np.array([world.disrupted[mi[m], w - 1] for w, m in zip(df.week, df.material)])
    base = np.array([np.nanmean(world.disrupted[mi[m], :origin]) for m in df.material])
    base_all = np.array([np.nanmean(world.disrupted[mi[m], :world.last_week]) for m in df.material])
    bs = 1 - np.mean((df.disruption_prob.values - y) ** 2) / np.mean((base - y) ** 2)
    bs_all = 1 - np.mean((df.disruption_prob.values - y) ** 2) / np.mean((base_all - y) ** 2)
    bench = np.vstack([np.nanmean(world.spot[u], axis=0) for u in world.mat_units])
    act = np.array([bench[mi[m], w - 1] for w, m in zip(df.week, df.material)])
    carry = np.array([bench[mi[m], origin - 1] for m in df.material])
    ps = 1 - np.mean(np.abs(df.price_forecast.values - act) / act) / np.mean(np.abs(carry - act) / act)
    P = float(np.clip(0.6 * bs + 0.4 * ps, 0, 1))
    P_all = float(np.clip(0.6 * bs_all + 0.4 * ps, 0, 1))
    r = m3.score_window(world, df, start, end, mode)
    fs = r["fill_skill"] if np.isfinite(r["fill_skill"]) else fill_skill_if_naive_perfect
    R = float(np.clip(0.7 * fs + 0.3 * r["div"], 0, 1))

    def total(P_=P, F_=r["F"], R_=R):
        return 100 * (0.3 * P_ + 0.4 * F_ + 0.3 * R_)

    out = dict(score=total(), P=P, brier_skill=bs, price_skill=ps, F=r["F"], F_raw=r["F_raw"],
               fill_ours=r["fill_ours"], fill_naive=r["fill_naive"], fill_skill=r["fill_skill"],
               div=r["div"], div_weighted=r["div_weighted"], R=R)
    # alternative conventions (one change each)
    pm = r["per_material"]
    undefined = not np.isfinite(r["fill_skill"])
    perfect = np.isfinite(r["fill_ours"]) and r["fill_ours"] >= 1 - 1e-12
    fs1 = 1.0 if (undefined and np.isfinite(r["fill_naive"]) and perfect) else fs
    out["score_fs_undef1"] = total(R_=float(np.clip(0.7 * fs1 + 0.3 * r["div"], 0, 1)))
    out["score_R_div"] = total(R_=float(r["div"]) if undefined else R)
    gap = pm.naive - pm.hind
    Fm = np.clip(np.where(gap > 0, (pm.naive - pm.ours) / gap.where(gap > 0, 1), 0.0), 0, 1)
    out["score_F_mat"] = total(F_=float(np.mean(Fm)))
    if "shock_demand" in pm:
        sk = [_fill_skill(1 - a / dd, 1 - b / dd) for a, b, dd in
              zip(pm.ours_unmet_shock, pm.naive_unmet_shock, pm.shock_demand) if dd > 0]
        sk = [np.clip(x, 0, 1) for x in sk if np.isfinite(x)]
        fsm = float(np.mean(sk)) if sk else fill_skill_if_naive_perfect
        out["score_fill_mat"] = total(R_=float(np.clip(0.7 * fsm + 0.3 * r["div"], 0, 1)))
    out["score_div_units"] = total(R_=float(np.clip(0.7 * fs + 0.3 * r["div_weighted"], 0, 1)))
    out["score_base_all"] = total(P_=P_all)
    return out


def self_test(world):
    """Naive policy + base-rate probabilities + carried price on a past window must score 0."""
    m3 = sys.modules["model3"]
    start, end = 183, 195
    origin = start - 1
    rows = m3.naive_plan(world, start, end)
    bench = np.vstack([np.nanmean(world.spot[u], axis=0) for u in world.mat_units])
    mi = {m: i for i, m in enumerate(world.materials)}
    rows["id"] = [f"w{w}_{m}" for w, m in zip(rows.week, rows.material)]
    rows["disruption_prob"] = [np.nanmean(world.disrupted[mi[m], :origin]) for m in rows.material]
    rows["price_forecast"] = [bench[mi[m], origin - 1] for m in rows.material]
    s0 = score(world, rows, start, end)
    plan = m3.plan_window(world, origin, start, end, m3.MODERATE)
    rows2 = rows.drop(columns=["order_units"] + list(m3.SHARE_COLS)).merge(plan, on=["week", "material"])
    s1 = score(world, rows2, start, end)
    print("sample policy (must be 0):", {k: round(v, 4) for k, v in s0.items()})
    print("Model 3 moderate orders:  ", {k: round(v, 4) for k, v in s1.items()})
    assert abs(s0["score"]) < 1e-9, "self-test failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", nargs="?")
    ap.add_argument("--data")
    ap.add_argument("--start", type=int, default=209)
    ap.add_argument("--end", type=int, default=221)
    ap.add_argument("--start-stock", default="sum5", choices=["sum5", "5avg"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    load_module("model1")
    m3 = load_module("model3")
    world = m3.World(m3.load_data(find_data(args.data)))
    if args.self_test:
        self_test(world)
        return
    sub = pd.read_csv(args.submission)
    for mode in (["sum5", "5avg"] if args.start_stock == "sum5" else [args.start_stock]):
        s = score(world, sub, args.start, args.end, mode)
        print(f"start stock {mode}:", {k: round(v, 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
