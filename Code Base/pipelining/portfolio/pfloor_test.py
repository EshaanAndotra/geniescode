"""pfloor_test.py - hedge-only P variants, re-scored on backtest_pair's 18 origins (orders unchanged, sum5).
(a) hedge P = main P; (b)/(c) disruption_prob floored at 0.30 / 0.50 on every material with a KES or SUN supplier.
main's scores come from results/bt_pair_lean.csv (same procedure). Writes pfloor_test.csv in the current folder."""
import os, sys
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest_pair as bp

FLOORS = {"a_mainP": None, "b_floor030": 0.30, "c_floor050": 0.50}


def run(o):
    g = vars(bp)
    if "W" not in g:
        bp._init()
    bp._set_origin(o)
    W, m3h, np = bp.W, bp.m3h, bp.np
    pr = bp.m1.DisruptionModel(bp.m1.Config(sharpen_k=bp.rp.SHARPEN_K)).fit(W.P, origin=o).predict(13)
    bench, _ = bp.m2.PriceModel().fit(bp.PDATA, origin=o).predict(13)
    plan = m3h.plan_window(bp.WH, o, o + 1, o + 13, m3h.AGGRESSIVE)
    df = plan.merge(pr[["week", "material", "disruption_prob"]], on=["week", "material"]) \
             .merge(bench[["week", "material", "price_forecast"]], on=["week", "material"])
    df["id"] = [f"w{w}_{m}" for w, m in zip(df.week, df.material)]
    exposed = set(W.sup.material[W.sup.country.isin(["KES", "SUN"])])
    out = dict(origin=o)
    for k, f in FLOORS.items():
        d = df.copy()
        if f is not None:
            sel = d.material.isin(exposed)
            d.loc[sel, "disruption_prob"] = np.maximum(d.loc[sel, "disruption_prob"], f)
        out[k] = bp.score_local.score(W, d, o + 1, o + 13, "sum5")["score"]
    return out


if __name__ == "__main__":
    import pandas as pd
    origins = [o for v in bp.CATS.values() for o in v]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        res = pd.DataFrame(list(ex.map(run, origins)))
    bt = pd.read_csv(os.path.join(HERE, "results", "bt_pair_lean.csv")).query("stock == 'sum5'")[["origin", "cat", "main", "hedge"]]
    df = bt.merge(res, on="origin")
    print("check: a_mainP == backtest hedge:", bool((df.a_mainP - df.hedge).abs().max() < 1e-6), round(float((df.a_mainP - df.hedge).abs().max()), 4))
    for k in FLOORS:
        df[f"max_{k}"] = df[["main", k]].max(1)
    df.to_csv("pfloor_test.csv", index=False)
    cols = [f"max_{k}" for k in FLOORS]
    pd.set_option("display.width", 200)
    print(df.groupby("cat")[cols].mean().round(2).to_string())
    print("ALL mean:", df[cols].mean().round(2).to_dict())
    print("ALL worst:", df[cols].min().round(2).to_dict())
