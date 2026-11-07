"""E[max]-directed B: start from a candidate, find the pool futures where B beats main, replan B (orders + its own
P columns) on those futures, repeat. Iterates on the planning pool (seed 1000), reports on the eval set (seed 777)."""
import os, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cands.py")).read().rsplit('\nif __name__', 1)[0])

t0 = time.time()
MODE = "5avg"
Gp = gen(2000, 1000)                 # same raw futures as cands.py's ref pool, plus price noise for scoring
ref = {k: v for k, v in zip(("cap_m", "lead_a", "cost_m", "w", "dis", "noise"), Gp["raw"])}
ref["PX"] = np.log(Gp["bench"] / BENCH0[None, :, None])     # includes the noise; fine for median shifts
Gp.pop("raw")
Tp = Truth(Gp, MODE)
main_p = Tp.score(MAIN).total.values
print(f"pool ready {time.time() - t0:.0f}s; E[main] pool {main_p.mean():.2f}")
Fe = gen(500, 777); Fe.pop("raw")
Te = {m: Truth(Fe, m) for m in ("sum5", "5avg")}
names, Fs = stress()
Ts = {m: Truth(Fs, m) for m in ("sum5", "5avg")}
main_e = {m: Te[m].score(MAIN).total.values for m in Te}
main_s = {m: Ts[m].score(MAIN).total.values for m in Ts}


def report(tag, sub):
    out = [tag]
    for m in ("sum5", "5avg"):
        e = Te[m].score(sub).total.values; s = Ts[m].score(sub).total.values
        out.append(f"{m}: gen gain {np.maximum(main_e[m], e).mean() - main_e[m].mean():.2f} "
                   f"stress gain {np.maximum(main_s[m], s).mean() - main_s[m].mean():.2f}")
    print(" | ".join(out))


for start in ("calm", "kes", "pm0.5", "pm2.5"):
    B = pd.read_csv(f"cands/{start}.csv")
    for it in range(4):
        b = Tp.score(B).total.values
        win = b > main_p
        print(f"[{start} it{it}] pool: E[max] gain {np.maximum(main_p, b).mean() - main_p.mean():.2f}, B wins {win.mean():.2f}")
        report(f"   eval {start} it{it}", B)
        if win.sum() < 20:
            break
        idx = np.where(win)[0][:400]
        p = subset(ref, idx)
        Bn = assemble(plan_on(p), *shifts(p, ref))
        Bn.to_csv(f"cands_iter/{start}_it{it + 1}.csv", index=False)
        B = Bn
    print(f"   {time.time() - t0:.0f}s")
