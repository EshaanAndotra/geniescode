"""iter5 end to end: (1) all flags off reproduces iter4's disruption_prob and price_forecast exactly;
(2) the recommended flags fill a valid file (naive orders from the sample; ae supplies the real orders)."""
import os, sys, subprocess, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PIPE = os.path.join(ROOT, "Code Base", "pipelining")
DATA = os.path.join(ROOT, "Data_new")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(it, cfg1, cfg2=None):
    for k in ("model1", "model2", "news_stories"):
        sys.modules.pop(k, None)
    sys.path[:] = [p for p in sys.path if not p.startswith(os.path.join(PIPE, "Model 1"))]
    sys.path.insert(0, os.path.join(PIPE, "Model 1", it))     # model1 imports news_stories lazily
    m1 = load("model1", os.path.join(PIPE, "Model 1", it, "model1.py"))
    m2 = load("model2", os.path.join(PIPE, "Model 2", it, "model2.py"))
    data = m1.load_data(DATA)
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = m1.fill_submission(m1.Panels(data), sample, config=m1.Config(**cfg1))
    return m2.fill_submission(m2.PriceData(data), sub, config=m2.Config(**(cfg2 or {})))


a = build("iter4", dict(sharpen_k=1.5))
b = build("iter5", dict(sharpen_k=1.5))
print("flags off == iter4:", bool(np.allclose(a.disruption_prob, b.disruption_prob, atol=0, rtol=0)),
      bool(np.allclose(a.price_forecast, b.price_forecast, atol=0, rtol=0)))
c = build("iter5", dict(sharpen_k=1.5, premium_mix=0.5, lane_floor_m=104.0, tariff_floor=0.95), dict(jump_by_source=True))
out = os.path.join(HERE, "iter5_P_columns_naive_orders.csv")
c.to_csv(out, index=False)
r = subprocess.run([sys.executable, os.path.join(PIPE, "validate_submission.py"), out, "--data", DATA], capture_output=True, text=True)
print(r.stdout.strip()[-400:], r.stderr.strip()[-400:])
priv = c.id.str.extract(r"^w(\d+)_")[0].astype(int) >= 222
d = c[priv].assign(dp=(c.disruption_prob - a.disruption_prob)[priv], dprice=(c.price_forecast / a.price_forecast - 1)[priv])
d["material"] = d.id.str.split("_", n=1).str[1]
print("private rows changed: P", int((d.dp.abs() > 1e-9).sum()), "price", int((d.dprice.abs() > 1e-9).sum()))
print(d.groupby("material")[["dp", "dprice"]].agg(lambda s: s.abs().max()).round(3).to_string())
