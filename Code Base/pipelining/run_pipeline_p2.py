"""
run_pipeline_p2.py - Phase 2 final submissions (iteration 8): the main pick and the war hedge.

Built from the week-221 data, for the private window (weeks 222-234). See docs/PHASE2_ITERATION8.md.

  Model 1 (iter6)  disruption_prob     iter4 news layer + calm-only sharpening (sigmoid(1.5 logit p) only
                                       when the window's mean p < 0.15), tariff minimum length, and a
                                       scenario layer (Config.source_overrides) used by the war hedge
  Model 2 (iter6)  price_forecast      half-strength calm mean reversion; shock jumps at every horizon
  Model 3 (iter5)  order_units, share_* iter3 score-aware LP, hazards x 1.5, plans on the smaller of the two
                                       starting-stock readings; Config.m1_config plans on a scenario

Writes to --out-dir:
  submission_main_it8.csv   main pick: Model 1's own odds
  submission_war_it8.csv    war hedge: Merrow Strait closes in weeks 222-224 with probability 0.9 and stays
                            closed at least 6 weeks (WAR_SCENARIO); weeks 209-221 copied from the main pick
  landed_costs_it8.csv      Model 2's per-supplier landed costs (inspection only)
Upload both submission files and select both as final submissions (Kaggle keeps the better one).

Usage:
  python run_pipeline_p2.py                      # finds Data_new/ (Phase 2, weeks 1-221) on its own
  python run_pipeline_p2.py --data ../../Data_new --out-dir ../../submissions --tag _x
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
    "model1": os.path.join(HERE, "Model 1", "iter6", "model1.py"),
    "model2": os.path.join(HERE, "Model 2", "iter6", "model2.py"),
    "model3": os.path.join(HERE, "Model 3", "iter5", "model3.py"),
}
SHARPEN_K = 1.5
# War hedge: Model 1's scenario layer assumes the Merrow Strait closes in weeks 222-224 with
# probability 0.9 and stays closed at least 6 weeks; Models 2 and 3 plan on that. It matches the week-171
# clash pattern in the week-221 news (Kestria tension 10 -> 41 -> 54, "Tensions rise between Kestria and
# neighbors", Merrow congestion 27 -> 40 -> 44). See docs/PHASE2_ITERATION8.md.
WAR_SCENARIO = (("lane:MERROW", 0.9, 1, 3, 6),)
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
    cands += [os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, d)) for d in ("Data_new", "Data")]
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
    ap = argparse.ArgumentParser(description="Phase 2 final submissions: main pick + war hedge")
    ap.add_argument("--data", help="folder with the competition CSVs (default: Data_new, then Data)")
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "submissions")))
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--sharpen", type=float, default=SHARPEN_K, help="Model 1 sharpening k for calm outlooks (1 = off)")
    args = ap.parse_args()
    os.environ["MODEL1_DIR"] = os.path.dirname(PATHS["model1"])
    warnings.filterwarnings("ignore")
    t0 = time.time()

    data_dir = find_data(args.data)
    sample = find_sample(data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data: {data_dir}\nout:  {args.out_dir}")
    m1, m2, m3 = (load_module(n) for n in ("model1", "model2", "model3"))
    data = m1.load_data(data_dir)
    world = m3.World(m3.load_data(data_dir))
    if not args.skip_checks:
        checks(m1, m2, data, world.last_week)

    def report(out, path):
        m3.validate(out, world.sup)
        wk = out.id.str.extract(r"^w(\d+)_")[0].astype(int)
        mat = out.id.str.split("_", n=1).str[1]
        d = pd.Series([float(world.demand[world.materials.index(mm), w - 1]) for w, mm in zip(wk, mat)])
        ratio = {f"{a}-{b}": round(out.order_units[wk.between(a, b)].sum() / d[wk.between(a, b)].sum(), 3)
                 for a, b in WINDOWS}
        div = float(m3.diversification(out[list(m3.SHARE_COLS)].values).mean())
        print(f"  {path}\n    validated; units ordered / window demand {ratio}; diversification {div:.3f}")

    print("\n[main] Model 1 -> Model 2 -> Model 3, Model 1's own odds")
    cfg1 = m1.Config(sharpen_k=args.sharpen)
    sub = m1.fill_submission(m1.Panels(data), sample, windows=WINDOWS, config=cfg1)
    pdata = m2.PriceData(data)
    sub = m2.fill_submission(pdata, sub, windows=WINDOWS)
    m2.export_for_model3(pdata, windows=WINDOWS, out_path=os.path.join(args.out_dir, f"landed_costs_it8{args.tag}.csv"))
    main_sub = m3.fill_submission(world, sub, windows=WINDOWS, cfg=m3.MODERATE)
    path = os.path.join(args.out_dir, f"submission_main_it8{args.tag}.csv")
    main_sub.to_csv(path, index=False)
    report(main_sub, path)

    print("[war hedge] scenario " + str(WAR_SCENARIO))
    cw = m1.Config(sharpen_k=args.sharpen, source_overrides=WAR_SCENARIO)
    subw = m1.fill_submission(m1.Panels(data), main_sub.copy(), windows=WINDOWS, config=cw)
    subw = m2.fill_submission(pdata, subw, windows=WINDOWS, config=m2.Config(model1_config=cw))
    cfgw = type(m3.MODERATE)(**{**m3.MODERATE.__dict__, "m1_config": m1.Config(source_overrides=WAR_SCENARIO)})
    war = m3.fill_submission(world, subw, windows=WINDOWS, cfg=cfgw)
    pub = war.id.str.extract(r"^w(\d+)_")[0].astype(int) < WINDOWS[1][0]      # weeks 209-221: same as main
    war.loc[pub, :] = main_sub.loc[pub, war.columns].values
    war["order_units"] = war.order_units.astype(int)
    path = os.path.join(args.out_dir, f"submission_war_it8{args.tag}.csv")
    war.to_csv(path, index=False)
    report(war, path)
    print(f"\ndone in {time.time() - t0:.0f}s. Upload submission_main_it8.csv and submission_war_it8.csv, "
          f"then select both as final submissions on Kaggle.")


if __name__ == "__main__":
    main()
