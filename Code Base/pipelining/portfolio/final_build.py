"""final_build.py - the two final files.

P columns (both files): Model 1 iter5 Config(sharpen_k=1.5, premium_mix=0.5, tariff_floor=0.95) and
Model 2 iter5 Config(jump_by_source=True). Model 1 iter5 with default flags gives Model 3 exactly iter4's
hazards, so orders are unchanged by iter5.
Orders: main = Model 3 iter4 MODERATE with start_stock="sum5" (the public score confirmed the scorer uses sum5);
hedge = Model 3 iter1 AGGRESSIVE. Both at n_scen = N_SCEN (seed noise: sd 2.0 at 200 vs 0.9 at 1000).

    python3 final_build.py [--n-scen 1000] [--tag _final]
Writes submissions/submission_{moderate,aggressive}<tag>.csv (refuses to overwrite) and validates both.
"""
import argparse, dataclasses, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
PL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.normpath(os.path.join(PL, os.pardir, os.pardir))
DATA = os.path.join(ROOT, "Data_new")
sys.path.insert(0, PL)
sys.path.insert(0, os.path.join(PL, "Model 1", "iter5"))
os.environ["MODEL1_DIR"] = os.path.join(PL, "Model 1", "iter5")
import run_pipeline as rp
rp.PATHS.update(model1=os.path.join(PL, "Model 1", "iter5", "model1.py"),
                model2=os.path.join(PL, "Model 2", "iter5", "model2.py"),
                model3=os.path.join(PL, "Model 3", "iter4", "model3.py"),
                model3_hedge=os.path.join(PL, "Model 3", "iter1", "model3.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=1000)
    ap.add_argument("--names", default="final_main,final_hedge")
    a = ap.parse_args()
    outs = dict(zip(("moderate", "aggressive"), (os.path.join(ROOT, "submissions", f"{n}.csv") for n in a.names.split(","))))
    for p in outs.values():
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    t0 = time.time()
    m1, m2, m3, m3h = (rp.load_module(n) for n in ("model1", "model2", "model3", "model3_hedge"))
    data = m1.load_data(DATA)
    sample = rp.find_sample(DATA)
    sub = m1.fill_submission(m1.Panels(data), sample, windows=rp.WINDOWS,
                             config=m1.Config(sharpen_k=1.5, premium_mix=0.5, tariff_floor=0.95))
    sub = m2.fill_submission(m2.PriceData(data), sub, windows=rp.WINDOWS, config=m2.Config(jump_by_source=True))
    print(f"P columns done {time.time() - t0:.0f}s")
    plans = (("moderate", m3, m3.World(m3.load_data(DATA)), dataclasses.replace(m3.MODERATE, start_stock="sum5", n_scen=a.n_scen)),
             ("aggressive", m3h, m3h.World(m3h.load_data(DATA)), dataclasses.replace(m3h.AGGRESSIVE, n_scen=a.n_scen)))
    for name, mod, w, cfg in plans:
        out = mod.fill_submission(w, sub.copy(), windows=rp.WINDOWS, cfg=cfg)
        out.to_csv(outs[name], index=False)
        print(f"{outs[name]} written {time.time() - t0:.0f}s")
    r = subprocess.run([sys.executable, os.path.join(PL, "validate_submission.py"), *outs.values(), "--data", DATA],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-500:])


if __name__ == "__main__":
    main()
