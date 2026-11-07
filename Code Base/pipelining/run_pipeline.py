"""
run_pipeline.py - build the final submissions from the competition data in one go.

  Model 1 (iter4)  disruption_prob            P(material disrupted), per week (iter3 + news stories),
                                              sharpened: sigmoid(1.5 logit p)
  Model 2 (iter3)  price_forecast             benchmark price (calm mean reversion + shock on/off mix)
  Model 3          order_units + share_*      main:  iter3 moderate, score-aware stochastic LP (lean)
                                              hedge: iter1 aggressive, stress futures (heavily hedged)
Kaggle ranks a team by the better of its 2 selected files, so the two plans are built to win in
different futures: on 48 past windows the pair averages 56.1 best-of-two (old pair: 49.4).
Every window is forecast from min(window start - 1, last week in the data): with Phase 2 data
(weeks 1-221) the private window 222-234 is planned from week 221.

Writes to --out-dir:
  submission_moderate.csv     main pick (Model 3 iter3 MODERATE)
  submission_aggressive.csv   hedge pick (Model 3 iter1 AGGRESSIVE)
  landed_costs.csv            Model 2's per-supplier landed cost forecasts (for inspection, not uploaded)
and prints checks worth reading before uploading: data weeks, shocks active at the last week,
signals past their training range, futures premiums, and the order volume of each plan.

Usage:
  python run_pipeline.py --data ../../Data --out-dir ../../submissions
  python run_pipeline.py                    # data found automatically (repo Data/ or /kaggle/input)
"""
import argparse
import glob
import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "model1": os.path.join(HERE, "Model 1", "iter4", "model1.py"),
    "model2": os.path.join(HERE, "Model 2", "iter3", "model2.py"),
    "model3": os.path.join(HERE, "Model 3", "iter3", "model3.py"),
    "model3_hedge": os.path.join(HERE, "Model 3", "iter1", "model3.py"),
}
SHARPEN_K = 1.5
WINDOWS = ((209, 221), (222, 234))


def load_module(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PATHS[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # Models 2 and 3 import "model1" by name and reuse this one
    spec.loader.exec_module(mod)
    return mod


def find_data(arg):
    cands = [arg] if arg else []
    cands += [os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "Data"))]
    cands += sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/suppliers.csv", recursive=True))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "suppliers.csv")) or (c and glob.glob(os.path.join(c, "*suppliers.csv"))):
            return c
    raise FileNotFoundError("competition data not found; pass --data")


def find_sample(data_dir):
    hits = sorted(glob.glob(os.path.join(data_dir, "*sample_submission.csv")))
    if not hits:
        raise FileNotFoundError("sample_submission.csv not found next to the data")
    return hits[0]


def checks(m1, m2, data, last_week):
    """Things to look at before trusting the run."""
    P = m1.Panels(data)
    print(f"\n[data] weeks {P.first_week}-{P.last_week} (outcomes to week {P.last_outcome_week})")
    if P.last_week < 221:
        print("  NOTE: Phase 1 data. The private window (222-234) is forecast 14-26 weeks ahead. "
              "Rerun once Phase 2 (weeks 1-221) is out.")
    m = m1.DisruptionModel().fit(P)
    src = m._source_probs(13)
    active = [f"{P.suppliers.material[u]}/{P.suppliers.country[u]}: {s}" for u, s in enumerate(src["sup_state"]) if s]
    active += [f"country {c}: {s}" for c, s in zip(P.countries, src["cty_state"]) if s]
    active += [f"lane {l}: {s}" for l, s in zip(P.lanes, src["lane_state"]) if s]
    active += [f"tariff {P.suppliers.material[u]}/{P.suppliers.country[u]}: {s}"
               for u, s in enumerate(src["tar_state"]) if s]
    print(f"[shocks active or announced at week {P.last_week}]" + ("".join(f"\n  {a}" for a in active) or " none"))
    no = getattr(m, "news_open", None)
    if no is not None:
        if len(no):
            no = no.assign(target=[f"{P.suppliers.material[int(k)]}/{P.suppliers.country[int(k)]}" if s == "sup" else k
                                   for s, k in zip(no.source, no.key)])
            print("[open news stories (iter4)]\n" + no[["story", "target", "first", "p_type", "p_now", "p_active_max"]]
                  .round(3).to_string(index=False))
        else:
            print("[open news stories (iter4)] none")
    if P.last_week > 208:
        ext = m1.signal_extremes(P, train_end=208, recent_start=209)
        ext = ext[ext.exceeds]
        print(f"[signals in weeks 209-{P.last_week} above their weeks 1-208 maximum] "
              + ("none" if ext.empty else "\n" + ext.head(15).to_string(index=False)))
    prem = P.sup_feat["prem"]
    hot = pd.DataFrame({"material": P.suppliers.material, "country": P.suppliers.country,
                        "premium_now": prem[:, -1], "premium_last_week": prem[:, -2]})
    hot = hot[hot.premium_now > 0.03].sort_values("premium_now", ascending=False)
    print("[own futures premium > 3% at the last week (real warnings climb to 7-25% within 1-2 weeks; "
          "one-week 4% spikes were mostly false alarms)] "
          + ("none" if hot.empty else "\n" + hot.round(3).to_string(index=False)))


def main():
    ap = argparse.ArgumentParser(description="Build final submissions (Models 1-3)")
    ap.add_argument("--data", help="folder with the competition CSVs")
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "submissions")))
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--model1-iter", default="iter4", choices=["iter3", "iter4"],
                    help="iter4 (default) adds the news story layer, +2.96 on the public LB vs iter3; "
                         "see README.md")
    ap.add_argument("--tag", default="", help="suffix for output file names, e.g. _wild")
    ap.add_argument("--sharpen", type=float, default=SHARPEN_K,
                    help="Model 1 sharpening k (1 = off); needs Model 1 iter4")
    args = ap.parse_args()
    PATHS["model1"] = os.path.join(HERE, "Model 1", args.model1_iter, "model1.py")
    os.environ["MODEL1_DIR"] = os.path.dirname(PATHS["model1"])
    warnings.filterwarnings("ignore")
    t0 = time.time()

    data_dir = find_data(args.data)
    sample = find_sample(data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data: {data_dir}\nout:  {args.out_dir}\nModel 1: {args.model1_iter}")
    if args.model1_iter == "iter4" and not glob.glob(os.path.join(data_dir, "*news.csv")):
        print("  WARNING: news.csv not found - the iter4 news layer will be off")

    m1, m2, m3, m3h = (load_module(n) for n in ("model1", "model2", "model3", "model3_hedge"))
    data = m1.load_data(data_dir)
    world = m3.World(m3.load_data(data_dir))
    world_h = m3h.World(m3h.load_data(data_dir))
    cfg1 = m1.Config(sharpen_k=args.sharpen) if "sharpen_k" in m1.Config.__dataclass_fields__ else m1.Config()
    if not args.skip_checks:
        checks(m1, m2, data, world.last_week)

    print("\n[1/3] Model 1: disruption_prob")
    sub = m1.fill_submission(m1.Panels(data), sample, windows=WINDOWS, config=cfg1)
    print("[2/3] Model 2: price_forecast + landed costs")
    pdata = m2.PriceData(data)
    sub = m2.fill_submission(pdata, sub, windows=WINDOWS)
    m2.export_for_model3(pdata, windows=WINDOWS, out_path=os.path.join(args.out_dir, f"landed_costs{args.tag}.csv"))

    print("[3/3] Model 3: order_units + shares (main: iter3 moderate, hedge: iter1 aggressive)")
    for name, mod, w_, cfg in (("moderate", m3, world, m3.MODERATE), ("aggressive", m3h, world_h, m3h.AGGRESSIVE)):
        out = mod.fill_submission(w_, sub.copy(), windows=WINDOWS, cfg=cfg)
        path = os.path.join(args.out_dir, f"submission_{name}{args.tag}.csv")
        out.to_csv(path, index=False)
        wk = out.id.str.extract(r"^w(\d+)_")[0].astype(int)
        mat = out.id.str.split("_", n=1).str[1]
        d = pd.Series([float(world.demand[world.materials.index(mm), w - 1]) for w, mm in zip(wk, mat)])
        ratio = {f"{a}-{b}": round(out.order_units[wk.between(a, b)].sum() / d[wk.between(a, b)].sum(), 3)
                 for a, b in WINDOWS}
        div = float(m3.diversification(out[list(m3.SHARE_COLS)].values).mean())
        print(f"  {path}\n    validated; units ordered / window demand {ratio}; diversification {div:.3f}")
    print(f"\ndone in {time.time() - t0:.0f}s. Upload submission_moderate.csv (main) and "
          f"submission_aggressive.csv (hedge); select both as final submissions on Kaggle.")


if __name__ == "__main__":
    main()
