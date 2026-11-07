"""backtest_pair.py - historical check of the second-file procedure.

For each origin o (window o+1..o+13, planned with data up to o only):
  main  = Model 3 iter4 orders + Model 1 iter4 (sharpened) disruption_prob + Model 2 iter4 price_forecast
  hedge = Model 3 iter1 aggressive orders (sum5 planning) + main's P columns          (the current second file)
  B     = the E[max] procedure from iterB.py: start from onset hazards x0.5 with its own P shift, then twice
          replan on the generator futures (2,000, from origin o) where B beats main
and every file is scored on the real window with score_local.score under both starting-stock rules.

    cd <scratch folder>; python3 <repo>/Code Base/pipelining/portfolio/backtest_pair.py [--origins 55,70,...] [--jobs 6]
Writes bt_pair.csv in the current folder and prints per-category means and worst windows.
"""
import os, sys, time, argparse
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = {"calm": [55, 70, 85, 190, 193], "merrow": [166, 169, 172], "premium_chain": [127, 130, 133, 148, 151],
        "kestria": [94, 97], "recent": [199, 202, 208]}
N_POOL, N_ITER = 2000, 2


def _init():
    g = globals()
    exec(open(os.path.join(HERE, "cands.py")).read().rsplit("\nif __name__", 1)[0], g)
    m1 = sys.modules["model1"]
    m2 = rp.load_module("model2")
    g["m1"], g["m2"] = m1, m2
    g["PDATA"] = m2.PriceData(m1.load_data(DATA), panels=W.P)
    g["WH"] = m3h.World(m3h.load_data(DATA))


def _set_origin(o):
    g = globals()
    g["ORIGIN"], g["S0"], g["E0"] = o, o + 1, o + 13
    g["INFO"] = m3.OriginInfo(W, o, H)
    g["BENCH0"] = np.array([np.nanmean(W.spot[u, o - 1]) for u in W.mat_units])
    g["BASE_RATE"] = np.array([np.nanmean(W.disrupted[mi, :o]) for mi in range(M)])
    g["DEMAND"] = W.demand[:, o:o + 13]


def _main_sub(o):
    pr = m1.DisruptionModel(m1.Config(sharpen_k=rp.SHARPEN_K)).fit(W.P, origin=o).predict(13)
    bench, _ = m2.PriceModel().fit(PDATA, origin=o).predict(13)
    plan = m3.plan_window(W, o, o + 1, o + 13, m3.MODERATE, info=INFO)
    df = plan.merge(pr[["week", "material", "disruption_prob"]], on=["week", "material"]) \
             .merge(bench[["week", "material", "price_forecast"]], on=["week", "material"])
    df["id"] = [f"w{w}_{m}" for w, m in zip(df.week, df.material)]
    return df[["id", "disruption_prob", "price_forecast", "order_units"] + list(m3.SHARE_COLS)].reset_index(drop=True)


def _with_orders(base, plan, p_shift=None, px_shift=None):
    """assemble() for a 130-row window frame (cands.assemble validates 260-row files)."""
    sub = base.copy()
    x = sub.id.str.extract(r"^w(\d+)_(.+)$"); wk, mat = x[0].astype(int), x[1]
    pl = plan.set_index(["week", "material"])
    for i in sub.index:
        r = pl.loc[(int(wk[i]), mat[i])]
        sub.at[i, "order_units"] = int(r.order_units)
        for c in m3.SHARE_COLS:
            sub.at[i, c] = float(r[c])
        mi, t = W.materials.index(mat[i]), int(wk[i]) - S0
        if p_shift is not None:
            sub.at[i, "disruption_prob"] = float(np.clip(sub.at[i, "disruption_prob"] + p_shift[mi, t], 0.001, 0.999))
        if px_shift is not None:
            sub.at[i, "price_forecast"] = float(sub.at[i, "price_forecast"] * np.exp(px_shift[mi, t]))
    sh = sub[list(m3.SHARE_COLS)].clip(lower=0)
    sub[list(m3.SHARE_COLS)] = sh.div(sh.sum(1), axis=0)
    return sub


def run_origin(o):
    t0 = time.time()
    _set_origin(o)
    main = _main_sub(o)
    ph = m3h.plan_window(WH, o, o + 1, o + 13, m3h.AGGRESSIVE)
    hedge = _with_orders(main, ph)
    Gp = gen(N_POOL, 1000 + o)
    ref = dict(zip(("cap_m", "lead_a", "cost_m", "w", "dis", "noise"), Gp["raw"]))
    ref["PX"] = np.log(Gp["bench"] / BENCH0[None, :, None])
    Gp.pop("raw")
    Tp = Truth(Gp, "5avg")
    main_p = Tp.score(main).total.values
    cfg = dataclasses.replace(m3.MODERATE, prob_mult=0.5)
    p05 = pool(N_POOL, 1000 + o, cfg)
    B = _with_orders(main, m3.plan_window(W, o, o + 1, o + 13, cfg, info=INFO), *shifts(p05, ref))
    for it in range(N_ITER):
        win = Tp.score(B).total.values > main_p
        if win.sum() < 20:
            break
        p = subset(ref, np.where(win)[0][:400])
        B = _with_orders(main, plan_on(p), *shifts(p, ref))
    rows = []
    for mode in ("sum5", "5avg"):
        s = {k: score_local.score(W, f, o + 1, o + 13, mode) for k, f in (("main", main), ("hedge", hedge), ("B", B))}
        rows.append(dict(origin=o, stock=mode, main=s["main"]["score"], hedge=s["hedge"]["score"], B=s["B"]["score"],
                         main_P=s["main"]["P"], B_P=s["B"]["P"], naive_short=bool(np.isfinite(s["main"]["fill_skill"])),
                         secs=time.time() - t0))
    return rows


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default=",".join(str(o) for v in CATS.values() for o in v))
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    import numpy as np, pandas as pd
    origins = [int(x) for x in a.origins.split(",")]
    t = time.time()
    with ProcessPoolExecutor(a.jobs, initializer=_init) as ex:
        res = [r for rr in ex.map(run_origin, origins) for r in rr]
    df = pd.DataFrame(res)
    cat = {o: c for c, v in CATS.items() for o in v}
    df["cat"] = df.origin.map(cat).fillna("other")
    df["max_hedge"] = df[["main", "hedge"]].max(1)
    df["max_B"] = df[["main", "B"]].max(1)
    df.to_csv("bt_pair.csv", index=False)
    pd.set_option("display.width", 220)
    for mode in ("sum5", "5avg"):
        d = df[df.stock == mode]
        print(f"\n== stock {mode}")
        print(d[["origin", "cat", "main", "hedge", "B", "max_hedge", "max_B", "main_P", "B_P"]].round(1).to_string(index=False))
        g = d.groupby("cat")[["main", "hedge", "B", "max_hedge", "max_B"]].agg(["mean", "min"]).round(1)
        print(g.to_string())
        print("ALL mean:", d[["main", "hedge", "B", "max_hedge", "max_B"]].mean().round(2).to_dict())
        print("ALL worst:", d[["main", "hedge", "B", "max_hedge", "max_B"]].min().round(2).to_dict())
    print(f"\n{time.time() - t:.0f}s")


if __name__ == "__main__":
    main_cli()
