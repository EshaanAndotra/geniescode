"""Combined measurement of the Model 1 / Model 2 iter5 options against iter4 behaviour (all flags off).
One Model 1 fit per origin (the flags only act in _source_probs / predict), sharpening k = 1.5 as in the pipeline.
Usage: python3 iter5_eval.py [origins...]   (default: premium-chain + tariff + every-26th calm origins)"""
import os, sys, time, warnings
from dataclasses import replace
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "Code Base", "pipelining", "Model 1", "iter5"))
sys.path.insert(0, os.path.join(ROOT, "Code Base", "pipelining", "Model 2", "iter5"))
warnings.filterwarnings("ignore")
import model1, model2

PREMIUM = [35, 36, 128, 129, 130, 131, 133, 134, 135, 151, 152, 156, 157, 210, 211, 212, 213, 214]
TARIFF = [13, 119, 178]
V1 = {"base": {}, "mix": dict(premium_mix=0.5), "lane": dict(lane_floor_m=104.0), "tariff": dict(tariff_floor=0.95)}
V1["all"] = {k: v for d in list(V1.values())[1:] for k, v in d.items()}
V2 = {"base": ("base", False), "mix": ("mix", False), "jump": ("base", True), "mix+jump": ("mix", True)}

CALM = list(range(52, 209, 26))
NPROC = int(os.environ.get("NPROC", "2"))


def init():
    global pdata, P, last, cfg0
    pdata = model2.PriceData(model1.load_data(os.path.join(ROOT, "Data_new")))
    P = pdata.P
    last = P.last_outcome_week
    cfg0 = model1.Config(sharpen_k=1.5)


def one(o):
    brows, prows = [], []
    grp = "premium" if o in PREMIUM else "tariff" if o in TARIFF else "calm"
    M = model1.DisruptionModel(cfg0).fit(P, origin=o)
    H = min(13, last - o)
    mi = {m: i for i, m in enumerate(P.materials)}
    for v, ov in V1.items():
        M.cfg = replace(cfg0, **ov)
        df = M.predict(H)
        df["y"] = [P.disrupted[mi[m], P.idx(w)] for m, w in zip(df.material, df.week)]
        brows.append(df.assign(variant=v, group=grp)[["origin", "h", "week", "material", "disruption_prob", "y", "variant", "group"]])
    pm = model2.PriceModel(model2.Config()).fit(pdata, origin=o, m1=M)
    for v, (v1, jb) in V2.items():
        M.cfg = replace(cfg0, **V1[v1])
        pm.cfg = replace(pm.cfg, jump_by_source=jb)
        b, _ = pm.predict(H)
        b["actual"] = [pdata.bench[mi[m], P.idx(w)] for m, w in zip(b.material, b.week)]
        b["carry"] = [pdata.bench[mi[m], P.idx(o)] for m in b.material]
        b["dis"] = [P.disrupted[mi[m], P.idx(w)] for m, w in zip(b.material, b.week)]
        prows.append(b.assign(variant=v, group=grp))
    return brows, prows


if __name__ == "__main__":
    from multiprocessing import Pool
    origins = [int(x) for x in sys.argv[1:]] or sorted(set(PREMIUM + TARIFF + CALM))
    t = time.time()
    with Pool(NPROC, initializer=init) as pool:
        out = pool.map(one, origins)
    brows = [x for b, _ in out for x in b]
    prows = [x for _, p in out for x in p]
    print(f"{len(origins)} origins in {time.time() - t:.0f}s")
    B = pd.concat(brows, ignore_index=True)
    Pr = pd.concat(prows, ignore_index=True)
    B.to_csv(os.path.join(HERE, "brier_rows.csv"), index=False)
    Pr.to_csv(os.path.join(HERE, "price_rows.csv"), index=False)

    key = ["origin", "week", "material"]
    base = B[B.variant == "base"].set_index(key)
    print("\nBRIER (sharpened disruption_prob; lower is better). changed = rows whose p moved by > 1e-9 vs base")
    for v in V1:
        if v == "base":
            continue
        x = B[B.variant == v].set_index(key)
        ch = (x.disruption_prob - base.disruption_prob).abs() > 1e-9
        for g in ["premium", "tariff", "calm", "ALL"]:
            sel = (x.group == g) if g != "ALL" else np.ones(len(x), bool)
            b0 = ((base.disruption_prob - base.y) ** 2)[sel]
            b1 = ((x.disruption_prob - x.y) ** 2)[sel]
            c = ch[sel]
            print(f"  {v:7s} {g:8s} rows {sel.sum():5d} brier base {b0.mean():.4f} -> {b1.mean():.4f}   changed {c.sum():4d}: "
                  f"{b0[c].mean() if c.any() else float('nan'):.4f} -> {b1[c].mean() if c.any() else float('nan'):.4f}; "
                  f"unchanged rows identical: {bool((b0[~c] == b1[~c]).all())}")
    pb = Pr[Pr.variant == "base"].set_index(key)
    print("\nPRICE MAPE (benchmark). skill = 1 - MAPE / carry MAPE; calm = not in shock at origin and not disrupted that week")
    for v in V2:
        x = Pr[Pr.variant == v].set_index(key)
        e = (x.price_forecast - x.actual).abs() / x.actual
        e0 = (pb.price_forecast - pb.actual).abs() / pb.actual
        ec = (x.carry - x.actual).abs() / x.actual
        calm = (~x.in_shock_at_origin) & (x.dis == 0)
        ch = (x.price_forecast - pb.price_forecast).abs() > 1e-9
        out = []
        for g in ["premium", "tariff", "calm", "ALL"]:
            sel = (x.group == g) if g != "ALL" else np.ones(len(x), bool)
            out.append(f"{g} {e[sel].mean()*100:.3f}% (skill {1 - e[sel].mean() / ec[sel].mean():+.3f})")
        print(f"  {v:8s} " + " | ".join(out) + f" | calm rows {e[calm].mean()*100:.3f}% vs base {e0[calm].mean()*100:.3f}% | changed rows {int(ch.sum())}")
