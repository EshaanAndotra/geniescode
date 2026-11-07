"""Score every candidate with main fixed: E[B], E[max(main, B)], P(B > main) on generator futures (fresh seed,
price noise) and on 57's stress scenarios, under both starting-stock rules. Total points (30 P + 70 orders)."""
import os, glob, time, pickle
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e.py")).read().split("if __name__")[0])

t0 = time.time()
cands = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in sorted(glob.glob("cands/*.csv"))}
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 500
Fg = gen(N_EVAL, 777)
Fg.pop("raw")
names, Fs = stress()
mer_closed = Fg["lead"][:, [u for u in range(U) if W.sup.lane[u] == "MERROW"]].max(axis=(1, 2)) > \
    W.base_lead[[u for u in range(U) if W.sup.lane[u] == "MERROW"]].max()
res = {}
for setname, F in (("gen", Fg), ("stress", Fs)):
    for mode in ("sum5", "5avg"):
        T = Truth(F, mode)
        res[(setname, mode)] = {k: T.score(s) for k, s in cands.items()}
        print(setname, mode, f"{time.time() - t0:.0f}s")
pickle.dump(dict(res=res, names=names, mer_closed=mer_closed), open("evalmenu.pkl", "wb"))

rows = []
for (setname, mode), r in res.items():
    main = r["main"].total.values
    for k, df in r.items():
        b = df.total.values
        rows.append(dict(set=setname, stock=mode, cand=k, E_B=b.mean(), E_max=np.maximum(main, b).mean(),
                         gain=np.maximum(main, b).mean() - main.mean(), p_B_wins=(b > main + 1e-9).mean(),
                         E_P30=30 * df.P.mean(), E_orders70=df.orders70.mean()))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for setname in ("gen", "stress"):
    t = T[T.set == setname].pivot(index="cand", columns="stock", values=["gain", "E_B", "p_B_wins", "E_P30"]).round(2)
    print(f"\n== {setname}: gain = E[max(main,B)] - E[main]; E[main] sum5 / 5avg =",
          T[(T.set == setname) & (T.cand == "main")].E_B.round(2).tolist())
    print(t.sort_values(("gain", "5avg"), ascending=False).to_string())
# per stress scenario: max(main, B) - main, 5avg
st = res[("stress", "5avg")]
print("\n== stress per scenario (5avg): main total, then B - main for each candidate")
tab = pd.DataFrame({k: v.total.values - st["main"].total.values for k, v in st.items() if k != "main"}, index=names)
tab.insert(0, "main", st["main"].total.values)
print(tab.round(1).to_string())
# generator split by Merrow closure
g = res[("gen", "5avg")]
print("\n== gen (5avg) split: Merrow closure in", mer_closed.mean().round(2), "of futures")
out = {}
for k, v in g.items():
    out[k] = dict(closed_B_minus_main=(v.total.values - g["main"].total.values)[mer_closed].mean(),
                  open_B_minus_main=(v.total.values - g["main"].total.values)[~mer_closed].mean())
print(pd.DataFrame(out).T.round(2).to_string())
print(f"done {time.time() - t0:.0f}s")
