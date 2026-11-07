"""
verify_scorer.py - find which scorer conventions reproduce the known Kaggle public scores.

Once Phase 2 is out, weeks 209-221 are in the data, so every file already scored on the public
leaderboard can be re-scored locally. The rules leave six details open; this tries every
combination and ranks them by how closely they reproduce all known public scores:
  stock   starting stock = sum of the window's first 5 weeks of demand, or 5 x the window mean
  F       pooled costs over materials, or the mean of per-material F (each clipped to 0-1)
  fill    pooled units over materials, or the mean of per-material fill skills
  undef   fill skill when naive meets all shock-week demand: 0, 1 if we do too, or R = diversification
  div     averaged over rows, or weighted by order units
  base    Brier baseline = each material's disruption rate to week 208, or over all released weeks

  python verify_scorer.py                     # the scored files listed in KNOWN
  python verify_scorer.py --add file.csv=63.10 --add other.csv=58.2

The best combination's settings go into score_local.py / Model 3's Config (f_mode, fill_mode,
div_mode, start_stock) and into the backtest summary.
"""
import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, HERE)
from run_pipeline import find_data, load_module  # noqa: E402

KNOWN = {   # public leaderboard (weeks 209-221), all built from Phase 1 data
    os.path.join(ROOT, "submissions", "submission_moderate_iter5.csv"): 68.40,
    os.path.join(ROOT, "submissions", "submission_moderate_wild.csv"): 65.29,
    os.path.join(ROOT, "submissions", "C_full_moderate.csv"): 62.33,
    os.path.join(ROOT, "submissions", "submission_B_iter2.csv"): 44.15,
}
START, END = 209, 221


def _fs(fo, fn):
    return (fo - fn) / (1 - fn) if np.isfinite(fn) and fn < 1 - 1e-12 else np.nan


def components(world, sub, mode):
    """Every ingredient needed by every convention, for one file and one starting-stock rule."""
    m3 = sys.modules["model3"]
    origin = START - 1
    df = sub.copy()
    parts = df.id.str.extract(r"^w(\d+)_(.+)$")
    df["week"], df["material"] = parts[0].astype(int), parts[1]
    df = df[df.week.between(START, END)]
    mi = {m: i for i, m in enumerate(world.materials)}
    y = np.array([world.disrupted[mi[m], w - 1] for w, m in zip(df.week, df.material)])
    out = {}
    for name, upto in (("to208", origin), ("released", world.last_week)):
        b = np.array([np.nanmean(world.disrupted[mi[m], :upto]) for m in df.material])
        out[f"bs_{name}"] = 1 - np.mean((df.disruption_prob.values - y) ** 2) / np.mean((b - y) ** 2)
    bench = np.vstack([np.nanmean(world.spot[u], axis=0) for u in world.mat_units])
    act = np.array([bench[mi[m], w - 1] for w, m in zip(df.week, df.material)])
    carry = np.array([bench[mi[m], origin - 1] for m in df.material])
    out["ps"] = 1 - np.mean(np.abs(df.price_forecast.values - act) / act) / np.mean(np.abs(carry - act) / act)
    r = m3.score_window(world, df, START, END, mode)
    pm = r["per_material"]
    gap = (pm.naive - pm.hind).values
    out["F_pooled"] = r["F"]
    out["F_mat"] = float(np.mean(np.clip(np.where(gap > 0, (pm.naive - pm.ours).values / np.where(gap > 0, gap, 1), 0), 0, 1)))
    out["fill_pooled"] = r["fill_skill_raw"]
    out["fo"], out["fn"] = r["fill_ours"], r["fill_naive"]
    sk = [_fs(1 - a / d, 1 - b / d) for a, b, d in zip(pm.ours_unmet_shock, pm.naive_unmet_shock, pm.shock_demand) if d > 0]
    sk = [float(np.clip(x, 0, 1)) for x in sk if np.isfinite(x)]
    out["fill_mat"] = float(np.mean(sk)) if sk else np.nan
    out["div_rows"], out["div_units"] = r["div"], r["div_weighted"]
    return out


def total(c, F, fill, undef, div, base):
    P = float(np.clip(0.6 * c[f"bs_{base}"] + 0.4 * c["ps"], 0, 1))
    fs = c[f"fill_{fill}"]
    dv = c[f"div_{div}"]
    if np.isfinite(fs):
        R = 0.7 * float(np.clip(fs, 0, 1)) + 0.3 * dv
    elif undef == "zero":
        R = 0.3 * dv
    elif undef == "one_if_perfect":
        R = 0.3 * dv + (0.7 if np.isfinite(c["fn"]) and c["fo"] >= 1 - 1e-12 else 0.0)
    else:                                     # "div_only"
        R = dv
    return 100 * (0.3 * P + 0.4 * c[f"F_{F}"] + 0.3 * float(np.clip(R, 0, 1)))


def main():
    global START, END
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--add", action="append", default=[], help="path=public_score")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--start", type=int, default=209, help="window to re-score (testing only)")
    ap.add_argument("--end", type=int, default=221)
    ap.add_argument("--no-known", action="store_true", help="use only the --add files")
    args = ap.parse_args()
    START, END = args.start, args.end
    warnings.filterwarnings("ignore")
    load_module("model1")
    m3 = load_module("model3")
    world = m3.World(m3.load_data(find_data(args.data)))
    if world.last_week < END:
        sys.exit(f"truth only goes to week {world.last_week}: needs Phase 2 data (weeks 1-221)")
    known = {} if args.no_known else dict(KNOWN)
    for a in args.add:
        p, s = a.rsplit("=", 1)
        known[os.path.abspath(p)] = float(s)
    comps = {}
    for path in known:
        sub = pd.read_csv(path)
        for mode in ("sum5", "5avg"):
            comps[(path, mode)] = components(world, sub, mode)
    print("ingredients per file (sum5 stock):")
    print(pd.DataFrame({os.path.basename(p): comps[(p, "sum5")] for p in known}).round(4).to_string())
    rows = []
    for mode, F, fill, undef, div, base in itertools.product(
            ("sum5", "5avg"), ("pooled", "mat"), ("pooled", "mat"), ("zero", "one_if_perfect", "div_only"),
            ("rows", "units"), ("to208", "released")):
        pred = {os.path.basename(p): total(comps[(p, mode)], F, fill, undef, div, base) for p in known}
        err = [pred[os.path.basename(p)] - s for p, s in known.items()]
        rows.append(dict(stock=mode, F=F, fill=fill, undef=undef, div=div, base=base,
                         max_abs_err=float(np.max(np.abs(err))), **{k: round(v, 2) for k, v in pred.items()}))
    res = pd.DataFrame(rows).sort_values("max_abs_err")
    pd.set_option("display.width", 250)
    print("\nknown public scores:", {os.path.basename(p): s for p, s in known.items()})
    print(res.head(args.top).round(3).to_string(index=False))
    best = res.iloc[0]
    print(f"\nbest match: max error {best.max_abs_err:.3f} points. "
          + ("Conventions confirmed." if best.max_abs_err < 0.05 else
             "No combination is exact: the simulator differs somewhere else too (hindsight plan, average cost, ...)."))


if __name__ == "__main__":
    main()
