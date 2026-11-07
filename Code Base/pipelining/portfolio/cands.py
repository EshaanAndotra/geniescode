"""Build step-3 candidate second files (each a full 260-row submission; only rows 222-234 differ from main).
Each scenario-cluster candidate is planned by Model 3 iter4 on futures conditioned on its cluster, and gets
its own P columns: main's disruption_prob shifted by (cluster mean of disrupted - pool mean), and main's
price_forecast scaled by exp(cluster median log price shift - pool median). Writes cands/<name>.csv plus
cands/<name>_mainP.csv (same orders, main's P columns)."""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e.py")).read().split("if __name__")[0])

OUT = "cands"
os.makedirs(OUT, exist_ok=True)
MAIN = pd.read_csv(os.path.join(ROOT, "submissions", "submission_moderate_p2v2.csv"))
HEDGE = pd.read_csv(os.path.join(ROOT, "submissions", "submission_aggressive_p2v2.csv"))
N_PLAN = 200
KES_C = list(W.P.countries).index("KES")
MER_L = list(W.P.lanes).index("MERROW")
MER_U = np.array([u for u in range(U) if W.sup.lane[u] == "MERROW"])
KES_U = np.array([u for u in range(U) if W.sup.country[u] == "KES"])


def pool(n, seed, cfg=None, boost=None):
    cfg = dataclasses.replace(cfg or m3.MODERATE, n_scen=n, seed=seed, use_stress=False)
    out = make_scenarios_px(W, info_with(boost), cfg)
    return dict(zip(("cap_m", "lead_a", "cost_m", "w", "dis", "noise", "PX"), out))


def subset(p, idx):
    q = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    q["w"] = np.full(len(idx), 1.0 / len(idx))
    return q


def concat(*ps):
    q = {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
    q["w"] = np.full(len(q["w"]), 1.0 / len(q["w"]))
    return q


def plan_on(p, cfg=None):
    raw = (p["cap_m"], p["lead_a"], p["cost_m"], p["w"], p["dis"], p["noise"])
    orig = m3.make_scenarios
    m3.make_scenarios = lambda *a, **k: raw
    try:
        return m3.plan_window(W, ORIGIN, S0, E0, cfg or m3.MODERATE, info=INFO)
    finally:
        m3.make_scenarios = orig


def assemble(plan, p_shift=None, px_shift=None, base=MAIN):
    sub = base.copy()
    x = sub.id.str.extract(r"^w(\d+)_(.+)$"); wk, mat = x[0].astype(int), x[1]
    pl = plan.set_index(["week", "material"])
    for i in sub.index[wk.between(S0, E0)]:
        r = pl.loc[(int(wk[i]), mat[i])]
        sub.at[i, "order_units"] = int(r.order_units)
        for c in m3.SHARE_COLS:
            sub.at[i, c] = float(r[c])
        mi, t = W.materials.index(mat[i]), int(wk[i]) - S0
        if p_shift is not None:
            sub.at[i, "disruption_prob"] = round(float(np.clip(sub.at[i, "disruption_prob"] + p_shift[mi, t], 0.001, 0.999)), 4)
        if px_shift is not None:
            sub.at[i, "price_forecast"] = round(float(sub.at[i, "price_forecast"] * np.exp(px_shift[mi, t])), 4)
    sub["order_units"] = sub.order_units.astype(int)
    sh = sub[list(m3.SHARE_COLS)].clip(lower=0).round(6)
    sub[list(m3.SHARE_COLS)] = sh.div(sh.sum(1), axis=0)
    m3.validate(sub, W.sup)
    return sub


def shifts(p_cluster, p_ref):
    return (p_cluster["dis"].mean(0) - p_ref["dis"].mean(0),
            np.median(p_cluster["PX"], 0) - np.median(p_ref["PX"], 0))


def describe(name, sub):
    x = sub.id.str.extract(r"^w(\d+)_(.+)$"); d = sub[x[0].astype(int) >= S0].assign(material=x[1])
    dem = sum(DEMAND[W.materials.index(m)].sum() for m in W.materials)
    mer_share = 0.0
    for m in W.materials:
        g = d[d.material == m]
        for u in W.mat_units[W.materials.index(m)]:
            if W.sup.lane[u] == "MERROW":
                mer_share += float((g.order_units * g[f"share_{W.sup.country[u]}"]).sum())
    tot = float(d.order_units.sum())
    print(f"  {name}: units/demand {tot / dem:.2f} (by count), Merrow-lane share of units {mer_share / max(tot, 1):.2f}, "
          f"mean p {d.disruption_prob.mean():.3f}")


def save(name, plan, ps=None, pxs=None):
    s = assemble(plan, ps, pxs)
    s.to_csv(f"{OUT}/{name}.csv", index=False)
    describe(name, s)
    if ps is not None:
        assemble(plan).to_csv(f"{OUT}/{name}_mainP.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    MAIN.to_csv(f"{OUT}/main.csv", index=False); describe("main", MAIN)
    HEDGE.to_csv(f"{OUT}/hedge_sum5.csv", index=False); describe("hedge_sum5", HEDGE)
    # current hedge with min-stock planning (iter1 has no "min" mode: patch its start_stock for planning)
    Wh = m3h.World(m3h.load_data(DATA))
    _ss = m3h.start_stock
    m3h.start_stock = lambda d, mode="sum5": min(_ss(d, "sum5"), _ss(d, "5avg")) if mode == "min" else _ss(d, mode)
    ph = m3h.plan_window(Wh, ORIGIN, S0, E0, dataclasses.replace(m3h.AGGRESSIVE, start_stock="min"))
    m3h.start_stock = _ss
    save("hedge_min", ph)

    ref = pool(2000, 1000)
    mer_any = ref["lead_a"][:, MER_U].max(axis=(1, 2)) > 0
    print("pool: Merrow closure in", mer_any.mean())
    # Merrow specialist
    p_mer = subset(ref, np.where(mer_any)[0][:N_PLAN])
    save("merrow", plan_on(p_mer), *shifts(p_mer, ref))
    # Kestria country specialist (onset boosted so it happens, timing as Model 1 says)
    kb = pool(600, 2000, boost={("cty", KES_C): 1.0 / max(INFO.src["q_cty"][KES_C].sum(), 1e-9)})
    kes_any = (kb["cap_m"][:, KES_U] <= m3.MODERATE.country_cap + 1e-6).all(1).any(1)
    print("boosted pool: KES country event in", kes_any.mean())
    p_kes = subset(kb, np.where(kes_any)[0][:N_PLAN])
    save("kes", plan_on(p_kes), *shifts(p_kes, ref))
    # Kestria/Merrow cluster: half each
    p_km = concat(subset(ref, np.where(mer_any)[0][:N_PLAN // 2]), subset(kb, np.where(kes_any)[0][:N_PLAN // 2]))
    save("kesmer", plan_on(p_km), *shifts(p_km, ref))
    # calm / lean specialist: futures where naive meets all demand
    G = gen(2000, 1000, price_noise=False)
    nu = np.zeros(2000)
    for mi in range(M):
        units = W.mat_units[mi]; d = DEMAND[mi]
        k = int(np.where(units == W.naive_unit[mi])[0][0])
        o = np.zeros((len(units), H)); o[k] = d
        _, un = sim(o, G["cap"][:, units], G["lead"][:, units], G["cost"][:, units], d, m3.start_stock(d, "min"),
                    G["cost"][:, units].mean(axis=(1, 2)))
        nu += un.sum(1)
    calm = nu <= 1e-6
    print("pool: naive never short in", calm.mean())
    p_calm = subset(ref, np.where(calm)[0][:N_PLAN])
    save("calm", plan_on(p_calm), *shifts(p_calm, ref))
    # onset hazard multipliers (make_scenarios applies prob_mult to new onsets only; ongoing and announced untouched)
    for pm in (0.5, 1.5, 2.5):
        cfg = dataclasses.replace(m3.MODERATE, prob_mult=pm)
        plan = m3.plan_window(W, ORIGIN, S0, E0, cfg, info=INFO)
        save(f"pm{pm}", plan, *shifts(pool(2000, 1000, cfg), ref))
    print(f"done {time.time() - t0:.0f}s")
