"""
build_certain.py - scenarios/build_scenarios.py plus (a) m1patch's rescaled overrides and (b) extra Model 1 config keys
(e.g. p_announced). Writes OUT_DIR/cand_NAME.csv. Weeks 209-221 are copied from the main pick.

Usage: python build_certain.py PIPELINE_DIR OUT_DIR NAME '<json spec>' [NAME '<json spec>' ...]
spec: {"ovr": [[source, p, first_h, last_h, min_len], ...], "m1": {"p_announced": 1.0}}
"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
PIPE, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PIPE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_pipeline_p2 as rp, m1patch
os.environ["MODEL1_DIR"] = os.path.dirname(rp.PATHS["model1"])
m1, m2, m3 = (rp.load_module(n) for n in ("model1", "model2", "model3"))
m1patch.apply(m1)
DATA = os.path.join(PIPE, "..", "..", "Data_new")
data = m1.load_data(DATA); P = m1.Panels(data); pdata = m2.PriceData(data, panels=P)
W = m3.World(m3.load_data(DATA))
main = pd.read_csv(os.path.join(PIPE, "..", "..", "submissions", "submission_main_it8.csv"))
WIN = ((209, 221), (222, 234))

# onset odds Model 1 gives the three events at week 221 (before any override)
src = m1.DisruptionModel(m1.Config(sharpen_k=rp.SHARPEN_K)).fit(P, origin=221)._source_probs(13)
sup = P.suppliers.reset_index(drop=True)
for mat, cty in (("cobalt", "VAL"), ("rare_earths", "KES")):
    u = int(np.where((sup.material == mat) & (sup.country == cty))[0][0])
    q = src["q_sup"][u]
    print(f"Model 1 at week 221, {mat}/{cty}: P(onset in 222-234) = {q.sum():.3f}; "
          f"x1.5 in Model 3 = {min(1.5 * q.sum(), 1):.3f}; by week: {np.round(q, 3).tolist()}")
print("p_announced (tariff) =", m1.Config().p_announced)

for name, spec in zip(sys.argv[3::2], sys.argv[4::2]):
    spec = json.loads(spec)
    ovr = tuple(tuple(x) for x in spec.get("ovr", []))
    extra = spec.get("m1", {})
    c1 = m1.Config(sharpen_k=rp.SHARPEN_K, source_overrides=ovr, **extra)
    sub = m1.fill_submission(P, main.copy(), windows=WIN, config=c1)
    sub = m2.fill_submission(pdata, sub, windows=WIN, config=m2.Config(model1_config=c1))
    cfg = type(m3.MODERATE)(**{**m3.MODERATE.__dict__, "m1_config": m1.Config(source_overrides=ovr, **extra)})
    out = m3.fill_submission(W, sub, windows=WIN, cfg=cfg)
    pub = out.id.str.extract(r"^w(\d+)_")[0].astype(int) <= 221
    out.loc[pub, :] = main.loc[pub, out.columns].values
    out["order_units"] = out.order_units.astype(int)
    m3.validate(out, W.sup)
    out.to_csv(os.path.join(OUT, f"cand_{name}.csv"), index=False)
    print(name, "written", flush=True)
