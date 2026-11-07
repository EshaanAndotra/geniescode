"""
backtest_phase2.py - walk-forward test of the pipeline on the Phase 2 data (weeks 1-221).

Every window T+1..T+13 is planned from origin T with data up to week T only (demand excepted:
it is known for every week), which is how the private window 222-234 is planned from week 221.
Origins run from 52 to 208, so weeks 53-221 are all used as test weeks. Origin 208 is the Kaggle
public window (weeks 209-221): no model choice before Phase 2 saw those weeks, so it is the one
true holdout. Windows that reach past week 208 are reported as their own group too.

Two modes:
  --mode P      Models 1 + 2 only: the P part (Brier skill and price skill), every origin.
                Fast (about 1 s per window). Use it to tune Model 1 and Model 2.
  --mode full   all three models, the full score (P, F, R and every part) from score_local.score().

Each plan is a JSON object; any key left out uses the pipeline default:
  name     label in the output
  m1, m2, m3            iteration folders, e.g. "iter6", "iter6", "iter5" (defaults: run_pipeline_p2.py)
  preset   Model 3 preset: "moderate" or "aggressive"
  sharpen  Model 1 sharpening k
  m1cfg, m2cfg, m3cfg   Config overrides for each model, e.g. {"tail_weight": 0.1}

Groups in the summary:
  all       every window
  design    windows ending by week 208: the weeks every pre-Phase-2 design choice was made on
  new       windows reaching into weeks 209-221
  public    origin 208 only (weeks 209-221)
  alert     origins where Model 1 already expected trouble: mean predicted p over the window
            >= --alert (judged at the origin, so no hindsight). The groups are fixed by
            --alert-ref (the first long run's values) so every plan is compared on the same windows.
  warn      alert windows with no material disrupted at the origin: calm now, trouble forecast.
            That is week 221's situation.
  shock / calm  (full mode) whether naive runs short in shock weeks

Examples:
  python backtest_phase2.py --mode P --plans '[{"name":"old","m1":"iter4","m2":"iter3"},{"name":"new"}]'
  python backtest_phase2.py --mode full --plans '[{"name":"main"},{"name":"hedge","m3":"iter1","preset":"aggressive"}]'
  python backtest_phase2.py --mode full --origins 208:209:1          # the public window only

Writes backtests/p2_<mode>_<name>.csv per plan and prints the summary table, plus best-of-two
for every pair of plans in full mode (Kaggle keeps the better of a team's 2 final picks).
"""
import argparse
import itertools
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
import run_pipeline_p2 as run_pipeline  # noqa: E402

DEFAULT_DATA = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir, "Data_new"))
PUBLIC_ORIGIN = 208
G = {}


def plan_paths(plan):
    paths = dict(run_pipeline.PATHS)
    for k in ("m1", "m2", "m3"):
        if plan.get(k):
            n = k[1]
            paths[f"model{n}"] = os.path.join(HERE, f"Model {n}", plan[k], f"model{n}.py")
    return paths


def _init(paths, data_dir, plan, mode):
    warnings.filterwarnings("ignore")
    run_pipeline.PATHS.update(paths)
    os.environ["MODEL1_DIR"] = os.path.dirname(paths["model1"])
    m1 = run_pipeline.load_module("model1")
    m2 = run_pipeline.load_module("model2")
    data = m1.load_data(data_dir)
    G.update(m1=m1, m2=m2, plan=plan, mode=mode)
    if mode == "full":
        m3 = run_pipeline.load_module("model3")
        import score_local
        world = m3.World(m3.load_data(data_dir))
        G.update(m3=m3, sl=score_local, world=world, P=world.P, pdata=m2.PriceData(data, panels=world.P))
    else:
        P = m1.Panels(data)
        G.update(P=P, pdata=m2.PriceData(data, panels=P))


def _m1_config(m1, plan):
    kw = dict(plan.get("m1cfg") or {})
    fields = m1.Config.__dataclass_fields__
    k = plan.get("sharpen", run_pipeline.SHARPEN_K)
    if "sharpen_k" in fields:
        kw.setdefault("sharpen_k", k)
    return m1.Config(**{a: (tuple(b) if isinstance(b, list) else b) for a, b in kw.items() if a in fields})


def forecasts(origin):
    """disruption_prob and price_forecast rows for weeks origin+1..origin+13, plus the alert level."""
    m1, m2, P, plan = G["m1"], G["m2"], G["P"], G["plan"]
    mod = m1.DisruptionModel(_m1_config(m1, plan)).fit(P, origin=origin)
    pr = mod.predict(13)
    raw = m1.DisruptionModel(_m1_config(m1, {**plan, "sharpen": 1.0})).fit(P, origin=origin).predict(13)
    m2cfg = m2.Config(**(plan.get("m2cfg") or {}))
    bench, _ = m2.PriceModel(m2cfg).fit(G["pdata"], origin=origin, m1=mod).predict(13)
    df = pr[["week", "material", "disruption_prob"]].merge(bench[["week", "material", "price_forecast"]],
                                                           on=["week", "material"])
    return df, float(raw.disruption_prob.mean())


def p_part(df, origin, end):
    """The P part exactly as score_local.score() computes it."""
    P, pdata = G["P"], G["pdata"]
    mi = {m: i for i, m in enumerate(P.materials)}
    d = df[df.week.between(origin + 1, end)]
    y = np.array([G["truth_dis"][mi[m], w - 1] for w, m in zip(d.week, d.material)])
    base = np.array([np.nanmean(G["truth_dis"][mi[m], :origin]) for m in d.material])
    bs = 1 - np.mean((d.disruption_prob.values - y) ** 2) / np.mean((base - y) ** 2)
    act = np.array([pdata.bench[mi[m], P.idx(w)] for w, m in zip(d.week, d.material)])
    carry = np.array([pdata.bench[mi[m], P.idx(origin)] for m in d.material])
    ps = 1 - np.mean(np.abs(d.price_forecast.values - act) / act) / np.mean(np.abs(carry - act) / act)
    return dict(P=float(np.clip(0.6 * bs + 0.4 * ps, 0, 1)), brier_skill=bs, price_skill=ps,
                n_disrupted=int(y.sum()))


def _one(origin):
    t = time.time()
    end = origin + 13
    if G["mode"] == "full":
        m3, W, plan = G["m3"], G["world"], G["plan"]
        base = getattr(m3, plan.get("preset", "moderate").upper())
        cfg = type(base)(**{**base.__dict__, **(plan.get("m3cfg") or {})})
        df, alert = forecasts(origin)
        rows = m3.plan_window(W, origin, origin + 1, end, cfg)
        rows = rows.merge(df, on=["week", "material"])
        rows["id"] = [f"w{w}_{m}" for w, m in zip(rows.week, rows.material)]
        s = G["sl"].score(W, rows, origin + 1, end)
        s["naive_short"] = bool(np.isfinite(s["fill_skill"]))
    else:
        df, alert = forecasts(origin)
        s = p_part(df, origin, end)
    calm_now = not bool(G["truth_dis"][:, origin - 1].any())
    s.update(origin=origin, alert_p=alert, calm_origin=calm_now, secs=time.time() - t)
    return s


def _init_p(paths, data_dir, plan, mode):
    _init(paths, data_dir, plan, mode)
    o = pd.read_csv(G["m1"]._find_file(data_dir, "outcomes"))
    W = int(o.week.max())
    G["truth_dis"] = np.vstack([o[o.material == m].set_index("week").disrupted.reindex(range(1, W + 1)).values
                                for m in G["P"].materials]).astype(float)


def run(plan, origins, data_dir, mode, jobs):
    paths = plan_paths(plan)
    with ProcessPoolExecutor(jobs, initializer=_init_p, initargs=(paths, data_dir, plan, mode)) as ex:
        res = list(ex.map(_one, origins))
    return pd.DataFrame(res).sort_values("origin").reset_index(drop=True)


def groups(bt, alert):
    ref = G.get("alert_ref")
    if ref is not None:
        bt = bt.drop(columns=["alert_p"]).merge(ref, on="origin", how="left")
    g = {"all": np.ones(len(bt), bool),
         "design": (bt.origin + 13 <= 208).values,
         "new": (bt.origin + 13 > 208).values,
         "public": (bt.origin == PUBLIC_ORIGIN).values,
         "alert": (bt.alert_p >= alert).values,
         "warn": ((bt.alert_p >= alert) & bt.calm_origin).values}
    if "naive_short" in bt:
        g["shock"] = bt.naive_short.values.astype(bool)
        g["calm"] = ~g["shock"]
    return g


def summarize(bt, alert, cols):
    out = {}
    for name, m in groups(bt, alert).items():
        if m.any():
            s = bt.loc[m, cols].mean()
            s["windows"] = int(m.sum())
            if "score" in bt:
                s["worst score"] = bt.loc[m, "score"].min()
            out[name] = s
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest on the Phase 2 data")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--mode", choices=["P", "full"], default="full")
    ap.add_argument("--plans", default='[{"name": "main"}]', help="JSON list of plans (see the docstring)")
    ap.add_argument("--origins", default="52:209:3", help="start:stop:step, stop exclusive (52:209:3 ends at 208)")
    ap.add_argument("--alert", type=float, default=0.15, help="alert group: mean predicted p at the origin >= this")
    ap.add_argument("--alert-ref", default=os.path.join(HERE, "backtests", "p2_alert_ref.csv"),
                    help="CSV (origin, alert_p) that fixes the alert groups across plans; written by the first run")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    plans = json.loads(args.plans)
    a, b, c = (int(x) for x in args.origins.split(":"))
    origins = sorted(set(range(a, b, c)))
    os.makedirs(os.path.join(HERE, "backtests"), exist_ok=True)
    pd.set_option("display.width", 250)
    results = {}
    if os.path.exists(args.alert_ref):
        G["alert_ref"] = pd.read_csv(args.alert_ref)[["origin", "alert_p"]]
    for plan in plans:
        name = plan.get("name", "plan")
        t = time.time()
        bt = run(plan, origins, args.data, args.mode, args.jobs)
        results[name] = bt
        if "alert_ref" not in G and len(bt) > 100:
            bt[["origin", "alert_p"]].to_csv(args.alert_ref, index=False)
            G["alert_ref"] = bt[["origin", "alert_p"]]
        path = os.path.join(HERE, "backtests", f"p2_{args.mode}_{name}{args.tag}.csv")
        bt.to_csv(path, index=False)
        cols = ["P", "brier_skill", "price_skill"] if args.mode == "P" else \
            ["score", "P", "brier_skill", "price_skill", "F", "fill_skill", "div", "R"]
        print(f"\n=== {name}  {json.dumps({k: v for k, v in plan.items() if k != 'name'})}  ({time.time() - t:.0f}s)")
        print(summarize(bt, args.alert, cols).round(3).to_string())
    if len(results) > 1:
        key = "score" if args.mode == "full" else "P"
        tab = pd.DataFrame({n: r.set_index("origin")[key] for n, r in results.items()})
        ref = next(iter(results.values())).set_index("origin")
        gmask = groups(ref.reset_index(), args.alert)
        print(f"\n=== mean {key} by group")
        print(pd.DataFrame({g: tab[m].mean() for g, m in gmask.items() if m.any()}).round(3).to_string())
        if args.mode == "full":
            rows = []
            for x, y in itertools.combinations(tab.columns, 2):
                best = np.maximum(tab[x], tab[y])
                rows.append({"pair": f"{x} | {y}", **{g: best[m].mean() for g, m in gmask.items() if m.any()}})
            print("\n=== best of two (Kaggle keeps the better final pick)")
            print(pd.DataFrame(rows).set_index("pair").round(3).sort_values("all", ascending=False).to_string())


if __name__ == "__main__":
    main()
