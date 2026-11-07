"""alternate.py - step 3b: update both files of the pair (2-cluster split of the futures).

  B-step:    B    <- Model 3 iter4 planned on the pool futures where B beats main (B gets its own P shift)
  main-step: main <- Model 3 iter4 planned on the pool futures where main >= B, plus generic no-warning shocks
                     (blindside_futures: random lane / country / supplier, random onset, signal-independent) at
                     total weight OMEGA, so shocks the generator under-weights stay priced in. main keeps Model 1's P.
Start: main = Model 3 iter4 (moderate), B = onset hazards x0.5 with its P shift. Rounds: B, main, B, main, B.

    cd <scratch folder>; python3 <repo>/Code Base/pipelining/portfolio/alternate.py [omega]
evaluates the week-221 pair on fresh generator futures and the stress set (both stock rules) and writes
alt_main.csv / alt_B.csv in the current folder.
"""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cands.py")).read().rsplit("\nif __name__", 1)[0])

OMEGA = 0.15                 # a priori share of main's planning weight on blindside futures (0.15 and 0.20 reported)
N_POOL = 2000
ROUNDS = 2
N_BLIND = 60
CAP = 400                    # max pool futures per cluster plan (1000 for the week-221 build: seed noise)


def blindside_futures(noise_src, n=N_BLIND, seed=4242):
    """Generic no-warning shocks, independent of the signals at the origin: each future holds one event, its type
    drawn evenly from {lane closure, country event, single-supplier shock}, its source uniformly within the type
    (lanes: by realized closure frequency up to the origin),
    onset uniform on window weeks 1-8, length from Model 1's duration curve for the type. Effects follow the
    generator: lane cap x lane_cap and +lane_lead weeks; country cap = country_cap and the country price jump;
    supplier cap and price jump drawn from the single-supplier episodes seen up to the origin."""
    rng = np.random.default_rng(seed + ORIGIN)
    cfg, D = m3.MODERATE, INFO.D
    lanes, ctys = list(W.P.lanes), list(W.P.countries)
    # lane mix = realized closure onsets up to the origin (+0.5 each so no lane is impossible): Merrow had 3 of 5
    closed = W.P.lane_closed[:, :W.P.idx(ORIGIN) + 1]
    lane_w = (closed[:, 1:] & ~closed[:, :-1]).sum(1) + 0.5
    lane_w = lane_w / lane_w.sum()
    cap_m, lead_a, cost_m = np.ones((n, U, H)), np.zeros((n, U, H)), np.ones((n, U, H))
    dis = np.zeros((n, M, H), bool)
    for s in range(n):
        typ = ("lane", "country", "supplier")[rng.integers(3)]
        k = int(rng.integers(0, 8))
        L = m3._sample_len(rng, lambda x: D.S(typ, x))
        a = np.zeros(H, bool); a[k:min(k + max(L, 1), H)] = True
        if typ == "lane":
            units = np.where(W.sup.lane.values == lanes[rng.choice(len(lanes), p=lane_w)])[0]
            for u in units:
                cap_m[s, u, a] *= cfg.lane_cap
                lead_a[s, u, a] = cfg.lane_lead + (1 if rng.random() < cfg.lane_lead4_p else 0)
        elif typ == "country":
            units = np.where(W.sup.country.values == ctys[rng.integers(len(ctys))])[0]
            for u in units:
                cap_m[s, u, a] = cfg.country_cap
            for mi in np.unique(SUP_MAT[units]):
                cost_m[s, W.mat_units[mi]][:, a] *= np.exp(INFO.j_cty)
        else:
            units = np.array([rng.integers(U)])
            cap_m[s, units[0], a] = rng.choice(INFO.sup_caps)
            mi = SUP_MAT[units[0]]
            j = rng.choice(INFO.j_single)
            for u in W.mat_units[mi]:
                cost_m[s, u, a] *= np.exp(j)
        for mi in np.unique(SUP_MAT[units]):
            dis[s, mi] |= a
    return dict(cap_m=cap_m, lead_a=lead_a, cost_m=cost_m, dis=dis, noise=noise_src[:n],
                PX=np.zeros((n, M, H)), w=np.full(n, 1.0 / n))


def with_extra(p, extra, omega):
    q = {k: np.concatenate([p[k], extra[k]]) for k in ("cap_m", "lead_a", "cost_m", "dis", "noise", "PX")}
    q["w"] = np.concatenate([np.full(len(p["w"]), (1 - omega) / len(p["w"])), np.full(len(extra["w"]), omega / len(extra["w"]))])
    return q


def alternate(main, B, ref, Tp, assemble_fn, omega=OMEGA, rounds=ROUNDS, cap=CAP, log=print):
    """main, B: submission frames (main's P columns are kept). ref: pool futures dict; Tp: Truth on the same pool."""
    extra = blindside_futures(ref["noise"])
    plan_main_P = main.copy()

    def b_step(main, B):
        win = Tp.score(B).total.values > Tp.score(main).total.values
        if win.sum() < 20:
            return B
        p = subset(ref, np.where(win)[0][:cap])
        return assemble_fn(plan_main_P, plan_on(p), *shifts(p, ref))

    def m_step(main, B):
        keep = Tp.score(main).total.values >= Tp.score(B).total.values
        if keep.sum() < 20:
            return main
        p = with_extra(subset(ref, np.where(keep)[0][:cap]), extra, omega)
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
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    Gp = gen(N_POOL, 1000)
    ref = dict(zip(("cap_m", "lead_a", "cost_m", "w", "dis", "noise"), Gp["raw"]))
    ref["PX"] = np.log(Gp["bench"] / BENCH0[None, :, None])
    Gp.pop("raw")
    Tp = Truth(Gp, "5avg")
    cfg = dataclasses.replace(m3.MODERATE, prob_mult=0.5)
    B0 = assemble(m3.plan_window(W, ORIGIN, S0, E0, cfg, info=INFO), *shifts(pool(N_POOL, 1000, cfg), ref), base=MAIN)
    main, B = alternate(MAIN, B0, ref, Tp, lambda base, plan, *a: assemble(plan, *a, base=base), omega, cap=cap)
    main.to_csv(f"alt_main_w{omega}.csv", index=False); B.to_csv(f"alt_B_w{omega}.csv", index=False)
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
