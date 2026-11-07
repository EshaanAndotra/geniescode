"""
build_scenarios.py - build candidate submissions planned on a Model 1 scenario (weeks 222-234 from week 221).
Usage: python scenarios/build_scenarios.py NAME '<json spec>' [NAME '<json spec>' ...]  -> scenarios/out/cand_NAME.csv
Example (the war hedge): python scenarios/build_scenarios.py war '{"ovr": [["lane:MERROW", 0.9, 1, 3, 6]]}'
Weeks 209-221 are copied from the main pick, which must exist (run run_pipeline_p2.py first).
spec: {"ovr": [[source, p, first_h, last_h, min_len], ...], "m3": "iter5", "preset": "moderate"|"aggressive",
       "m3cfg": {...}, "p_from_war": true, "price_from_main": false}
  p_from_war false keeps the main pick's disruption_prob and price_forecast; price_from_main keeps only its prices.
"""
import sys, os, json, warnings, importlib.util, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); import run_pipeline_p2 as rp
os.environ["MODEL1_DIR"] = os.path.dirname(rp.PATHS["model1"])
m1, m2, m3 = (rp.load_module(n) for n in ("model1", "model2", "model3"))
DATA = HERE + "/../../Data_new"
data = m1.load_data(DATA); P = m1.Panels(data); pdata = m2.PriceData(data, panels=P)
worlds = {"iter5": m3.World(m3.load_data(DATA))}
mods = {"iter5": m3}
main = pd.read_csv(HERE + "/../../submissions/final/submission_main_it8.csv")
OUT = os.path.join(HERE, "scenarios", "out"); os.makedirs(OUT, exist_ok=True)
WIN = ((209, 221), (222, 234))
for name, spec in zip(sys.argv[1::2], sys.argv[2::2]):
    spec = json.loads(spec)
    ovr = tuple(tuple(x) for x in spec.get("ovr", []))
    c1 = m1.Config(sharpen_k=rp.SHARPEN_K, source_overrides=ovr)
    sub = main.copy()
    if spec.get("p_from_war", True):
        sub = m1.fill_submission(P, sub, windows=WIN, config=c1)
        sub = m2.fill_submission(pdata, sub, windows=WIN, config=m2.Config(model1_config=c1))
    mod = mods[spec.get("m3", "iter5")]
    base = getattr(mod, spec.get("preset", "moderate").upper())
    cfg = type(base)(**{**base.__dict__, **spec.get("m3cfg", {}), "m1_config": m1.Config(source_overrides=ovr)})
    out = mod.fill_submission(worlds[spec.get("m3", "iter5")], sub, windows=WIN, cfg=cfg)
    if spec.get("price_from_main"):
        out["price_forecast"] = main["price_forecast"].values
    pub = out.id.str.extract(r"^w(\d+)_")[0].astype(int) <= 221        # public rows: keep the main file's
    out.loc[pub, :] = main.loc[pub, out.columns].values
    out["order_units"] = out.order_units.astype(int)
    out.to_csv(os.path.join(OUT, f"cand_{name}.csv"), index=False)
    print(name, "written")
