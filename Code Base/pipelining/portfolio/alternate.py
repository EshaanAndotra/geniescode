"""alternate.py - step 3b: update both files of the pair (2-cluster split of the futures).

  B-step:    B    <- Model 3 iter4 planned on the pool futures where B beats main (B gets its own P shift)
  main-step: main <- Model 3 iter4 planned on the pool futures where main >= B, plus explicit big-shock futures at
                     total weight OMEGA (iter1's 4 stress futures + Merrow at 40% capacity + an early Kestria-wide
                     event), so shocks the generator under-weights stay priced in. main keeps Model 1's P columns.
Start: main = Model 3 iter4 (moderate), B = onset hazards x0.5 with its P shift. Rounds: B, main, B, main, B.

    cd <scratch folder>; python3 <repo>/Code Base/pipelining/portfolio/alternate.py [omega]
evaluates the week-221 pair on fresh generator futures and the stress set (both stock rules) and writes
alt_main.csv / alt_B.csv in the current folder.
"""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cands.py")).read().rsplit("\nif __name__", 1)[0])

OMEGA = 0.10
N_POOL = 2000
ROUNDS = 2


def big_shock_futures(noise_src):
    """Planning futures (cap_m, lead_a, cost_m, dis, noise) for shocks bigger or earlier than the generator draws."""
    mer = W.sup.lane.values == "MERROW"
    kes = W.sup.country.values == "KES"
    mer_mats = np.unique(SUP_MAT[mer]); kes_mats = np.unique(SUP_MAT[kes])
    caps, leads, costs, diss = [], [], [], []

    def add(c, l, k, d):
        caps.append(c); leads.append(l); costs.append(k); diss.append(d)
    one, zero = np.ones((U, H)), np.zeros((U, H))
    # iter1 stress 1: Merrow closed 8 weeks from week 3
    c, l, d = one.copy(), zero.copy(), np.zeros((M, H), bool)
    c[mer, 2:10] *= m3.MODERATE.lane_cap; l[mer, 2:10] = m3.MODERATE.lane_lead; d[np.ix_(mer_mats, np.arange(2, 10))] = True
    add(c, l, one.copy(), d)
    # iter1 stress 2: each material's cheapest supplier banned all window
    c, k = one.copy(), one.copy()
    for mi, units in enumerate(W.mat_units):
        c[units[np.argmin(INFO.cost_base[units, 0])]] = 0.0
        k[units] *= np.exp(np.median(INFO.j_single))
    add(c, zero.copy(), k, np.ones((M, H), bool))
    # iter1 stress 3: every supplier at 50%
    add(np.full((U, H), 0.5), zero.copy(), one.copy(), np.ones((M, H), bool))
    # iter1 stress 4 and an early variant: Kestria-wide event (as weeks 99-133)
    for a in (0, 1):
        c, k, d = one.copy(), one.copy(), np.zeros((M, H), bool)
        c[kes, a:] = m3.MODERATE.country_cap
        for mi in kes_mats:
            k[W.mat_units[mi], a:] *= np.exp(INFO.j_cty)
        d[kes_mats, a:] = True
        add(c, zero.copy(), k, d)
    # Merrow at 40% capacity with 4 extra lead weeks from week 2 to the end (bigger than training)
    c, l, d = one.copy(), zero.copy(), np.zeros((M, H), bool)
    c[mer, 1:] = 0.40; l[mer, 1:] = 4; d[mer_mats, 1:] = True
    add(c, l, one.copy(), d)
    n = len(caps)
    return dict(cap_m=np.stack(caps), lead_a=np.stack(leads), cost_m=np.stack(costs), dis=np.stack(diss),
                noise=noise_src[:n], PX=np.zeros((n, M, H)), w=np.full(n, 1.0 / n))


def with_extra(p, extra, omega):
    q = {k: np.concatenate([p[k], extra[k]]) for k in ("cap_m", "lead_a", "cost_m", "dis", "noise", "PX")}
    q["w"] = np.concatenate([np.full(len(p["w"]), (1 - omega) / len(p["w"])), np.full(len(extra["w"]), omega / len(extra["w"]))])
    return q


def alternate(main, B, ref, Tp, assemble_fn, omega=OMEGA, rounds=ROUNDS, log=print):
    """main, B: submission frames (main's P columns are kept). ref: pool futures dict; Tp: Truth on the same pool."""
    extra = big_shock_futures(ref["noise"])
    plan_main_P = main.copy()

    def b_step(main, B):
        win = Tp.score(B).total.values > Tp.score(main).total.values
        if win.sum() < 20:
            return B
        p = subset(ref, np.where(win)[0][:400])
        return assemble_fn(plan_main_P, plan_on(p), *shifts(p, ref))

    def m_step(main, B):
        keep = Tp.score(main).total.values >= Tp.score(B).total.values
        if keep.sum() < 20:
            return main
        p = with_extra(subset(ref, np.where(keep)[0][:400]), extra, omega)
        return assemble_fn(plan_main_P, plan_on(p))

    for r in range(rounds):
        B = b_step(main, B)
        main = m_step(main, B)
        if log:
            m, b = Tp.score(main).total.values, Tp.score(B).total.values
            log(f"  round {r}: pool E[main] {m.mean():.2f} E[B] {b.mean():.2f} E[max] {np.maximum(m, b).mean():.2f}")
    B = b_step(main, B)
    return main, B


if __name__ == "__main__":
    t0 = time.time()
    omega = float(sys.argv[1]) if len(sys.argv) > 1 else OMEGA
    Gp = gen(N_POOL, 1000)
    ref = dict(zip(("cap_m", "lead_a", "cost_m", "w", "dis", "noise"), Gp["raw"]))
    ref["PX"] = np.log(Gp["bench"] / BENCH0[None, :, None])
    Gp.pop("raw")
    Tp = Truth(Gp, "5avg")
    cfg = dataclasses.replace(m3.MODERATE, prob_mult=0.5)
    B0 = assemble(m3.plan_window(W, ORIGIN, S0, E0, cfg, info=INFO), *shifts(pool(N_POOL, 1000, cfg), ref), base=MAIN)
    main, B = alternate(MAIN, B0, ref, Tp, lambda base, plan, *a: assemble(plan, *a, base=base), omega)
    main.to_csv("alt_main.csv", index=False); B.to_csv("alt_B.csv", index=False)
    describe("alt_main", main); describe("alt_B", B)
    Fe = gen(500, 777); Fe.pop("raw")
    names, Fs = stress()
    old_B = pd.read_csv(os.path.join(os.getcwd(), "cands_iter", "pm0.5_it2.csv")) if os.path.exists("cands_iter/pm0.5_it2.csv") else B0
    pairs = {"v2 main + hedge": (MAIN, HEDGE), "v2 main + lean B (pm0.5_it2)": (MAIN, old_B), "3b main + 3b B": (main, B)}
    for setname, F in (("gen", Fe), ("stress", Fs)):
        for mode in ("sum5", "5avg"):
            T = Truth(F, mode)
            line = []
            for k, (a, b) in pairs.items():
                sa, sb = T.score(a).total.values, T.score(b).total.values
                line.append(f"{k}: E[max] {np.maximum(sa, sb).mean():.2f} (A {sa.mean():.1f}, B {sb.mean():.1f})")
            print(f"{setname} {mode} | " + " | ".join(line))
    T = Truth(Fs, "5avg")
    tab = pd.DataFrame({k: np.maximum(T.score(a).total.values, T.score(b).total.values) for k, (a, b) in pairs.items()}, index=names)
    tab["3b main"] = T.score(main).total.values; tab["3b B"] = T.score(B).total.values
    tab["v2 main"] = T.score(MAIN).total.values; tab["hedge"] = T.score(HEDGE).total.values
    print(tab.round(1).to_string())
    print(f"{time.time() - t0:.0f}s")
