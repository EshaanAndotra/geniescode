"""
backtest_full.py - score the whole pipeline (Models 1, 2 and 3) on past 13-week windows.

Every window T+1..T+13 is planned from origin T with data up to T only, which is exactly how
the private window (222-234) is planned from week 221 once Phase 2 is out. Each window gets
the full competition score (P, F, R and all their parts) from score_local.score(), including
the alternative scorer conventions (score_fs_undef1, score_F_mat, ...).

  python backtest_full.py                                   # current pipeline defaults
  python backtest_full.py --m3 iter1 --tag iter1            # older Model 3 for comparison
  python backtest_full.py --cfg '{"prob_mult": 1.5}' --tag pm15
  python backtest_full.py --m3 iter1 --preset aggressive --tag hedge   # the hedge pick
  python backtest_full.py --m3 iter1 --sharpen 1 --tag old_main        # the pre-iteration-5 main pick

Writes backtests/bt_<tag>.csv (one row per window) and prints the averages, split into windows
where naive runs short in shock weeks ("shock") and the rest ("calm").
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_pipeline  # noqa: E402

G = {}


def _init(paths, data_dir, sharpen):
    warnings.filterwarnings("ignore")
    G["sharpen"] = sharpen
    run_pipeline.PATHS.update(paths)
    os.environ["MODEL1_DIR"] = os.path.dirname(paths["model1"])
    m1, m2, m3 = (run_pipeline.load_module(n) for n in ("model1", "model2", "model3"))
    import score_local
    data = m1.load_data(data_dir)
    world = m3.World(m3.load_data(data_dir))
    G.update(m1=m1, m2=m2, m3=m3, sl=score_local, world=world,
             pdata=m2.PriceData(data, panels=world.P))


def window_rows(origin, cfg, start=None, end=None):
    """Submission rows for one window planned from `origin` (all three models)."""
    m1, m2, m3, W = G["m1"], G["m2"], G["m3"], G["world"]
    start = start or origin + 1
    end = end or origin + 13
    H = end - origin
    sharpen = {"sharpen_k": G["sharpen"]} if "sharpen_k" in m1.Config.__dataclass_fields__ else {}
    mod = m1.DisruptionModel(m1.Config(**sharpen)).fit(W.P, origin=origin)
    pr = mod.predict(H)
    bench, _ = m2.PriceModel().fit(G["pdata"], origin=origin, m1=mod).predict(H)
    plan = m3.plan_window(W, origin, start, end, cfg)
    df = plan.merge(pr[["week", "material", "disruption_prob"]], on=["week", "material"]) \
             .merge(bench[["week", "material", "price_forecast"]], on=["week", "material"])
    df["id"] = [f"w{w}_{m}" for w, m in zip(df.week, df.material)]
    return df


def _one(args):
    origin, preset, overrides = args
    m3 = G["m3"]
    base = getattr(m3, preset.upper())
    cfg = type(base)(**{**base.__dict__, **overrides})
    t = time.time()
    df = window_rows(origin, cfg)
    s = G["sl"].score(G["world"], df, origin + 1, origin + 13)
    s.update(origin=origin, secs=time.time() - t,
             naive_short=bool(np.isfinite(s["fill_skill"])))
    return s


def run(origins, preset="moderate", overrides=None, paths=None, data_dir=None, jobs=None,
        sharpen=run_pipeline.SHARPEN_K):
    paths = {**run_pipeline.PATHS, **(paths or {})}
    data_dir = run_pipeline.find_data(data_dir)
    jobs = jobs or max(1, (os.cpu_count() or 2))
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(paths, data_dir, sharpen)) as ex:
        res = list(ex.map(_one, [(o, preset, overrides or {}) for o in origins]))
    return pd.DataFrame(res)


def summarize(bt):
    cols = ["score", "P", "brier_skill", "price_skill", "F", "fill_skill", "div", "R"] + \
           [c for c in bt.columns if c.startswith("score_")]
    out = pd.DataFrame({"all": bt[cols].mean(),
                        "shock": bt[bt.naive_short][cols].mean(),
                        "calm": bt[~bt.naive_short][cols].mean()})
    out.loc["windows"] = [len(bt), int(bt.naive_short.sum()), int((~bt.naive_short).sum())]
    out.loc["worst score"] = [bt.score.min(), bt[bt.naive_short].score.min(), bt[~bt.naive_short].score.min()]
    return out


def main():
    ap = argparse.ArgumentParser(description="Full-score backtest on past windows")
    ap.add_argument("--data")
    ap.add_argument("--m1", default=None, help="Model 1 iteration folder, e.g. iter4")
    ap.add_argument("--m2", default=None, help="Model 2 iteration folder, e.g. iter3")
    ap.add_argument("--m3", default=None, help="Model 3 iteration folder, e.g. iter3")
    ap.add_argument("--preset", default="moderate", choices=["moderate", "aggressive"])
    ap.add_argument("--cfg", default="{}", help="JSON overrides for the Model 3 Config")
    ap.add_argument("--origins", default="52:196:3", help="start:stop:step (stop exclusive)")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--sharpen", type=float, default=run_pipeline.SHARPEN_K, help="Model 1 sharpening k (1 = off)")
    args = ap.parse_args()
    paths = {}
    for k, it in (("model1", args.m1), ("model2", args.m2), ("model3", args.m3)):
        if it:
            paths[k] = os.path.join(HERE, f"Model {k[-1]}", it, f"{k}.py")
    a, b, c = (int(x) for x in args.origins.split(":"))
    t = time.time()
    bt = run(list(range(a, b, c)), args.preset, json.loads(args.cfg), paths, args.data, args.jobs, args.sharpen)
    os.makedirs(os.path.join(HERE, "backtests"), exist_ok=True)
    out = os.path.join(HERE, "backtests", f"bt_{args.tag}.csv")
    bt.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(summarize(bt).round(3).to_string())
    print(f"\n{out}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
