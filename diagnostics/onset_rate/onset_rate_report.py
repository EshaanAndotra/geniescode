"""Generator in-window onset rate vs realized, by source type, from onset_rate_fits.pkl.
P(source starts in the window) is computed exactly as Model 3 iter3/iter4 draw_path does it:
clip(q * prob_mult), renormalized if the sum exceeds 1, so P = min(sum q * prob_mult, 1); only calm sources.
Usage: python3 onset_rate_report.py [prob_mult]   (default 1.0 = main)"""
import os, sys, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
PM = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
LOW = 0.05
d = pickle.load(open(os.path.join(HERE, "onset_rate_fits.pkl"), "rb"))
res, meta = d["res"], d["meta"]
sup = meta["suppliers"]
rows = []
for r in res:
    o = r["origin"]
    for u in range(len(sup)):
        if r["sup_state"][u] == "":
            rows.append(("supplier", f"{sup.material[u]}/{sup.country[u]}", o, min(r["q_sup"][u].sum() * PM, 1), r["real_sup"][u]))
        if r["tar_state"][u] == "":
            rows.append(("tariff", f"{sup.material[u]}/{sup.country[u]}", o, min(13 * r["q_tariff"] * PM, 1), r["real_tar"][u]))
    for i, c in enumerate(meta["countries"]):
        if r["cty_state"][i] == "":
            rows.append(("country", c, o, min(r["q_cty"][i].sum() * PM, 1), r["real_cty"][i]))
    for i, l in enumerate(meta["lanes"]):
        if r["lane_state"][i] == "":
            rows.append(("lane", l, o, min(r["q_lane"][i].sum() * PM, 1), r["real_lane"][i]))
df = pd.DataFrame(rows, columns=["type", "key", "origin", "p", "h_real"])
df["hit"] = df.h_real > 0
df["onset_week"] = np.where(df.hit, df.origin + df.h_real, -1)
nO = df.origin.nunique()
pd.set_option("display.width", 200)
print(f"prob_mult {PM}; {nO} origins ({df.origin.min()}-{df.origin.max()}), window = origin+1..origin+13; "
      f"counts are source-windows, overlapping windows count one event many times\n")
g = df.groupby("type").agg(expected=("p", "sum"), realized=("hit", "sum"), n=("p", "size"))
g["exp_per_window"] = g.expected / nO
g["real_per_window"] = g.realized / nO
g["ratio_real/exp"] = g.realized / g.expected
g["distinct_events"] = [df[(df.type == t) & df.hit][["key", "onset_week"]].drop_duplicates().shape[0] for t in g.index]
print("1. EXPECTED vs REALIZED in-window onsets\n" + g.round(3).to_string() + "\n")
bins = [0, LOW, 0.2, 0.5, 0.9, 1.0001]
df["bin"] = pd.cut(df.p, bins, right=False)
rel = df.groupby(["type", "bin"], observed=True).agg(n=("p", "size"), mean_p=("p", "mean"), real_rate=("hit", "mean"),
                                                     expected=("p", "sum"), realized=("hit", "sum"))
print("2. RELIABILITY by P(onset in window)\n" + rel.round(3).to_string() + "\n")
lo = df[df.p < LOW]
blind = lo[lo.hit]
print(f"3. BLINDSIDES (realized onset, generator P < {LOW})")
for t in ["lane", "country", "supplier", "tariff"]:
    b, l = blind[blind.type == t], lo[lo.type == t]
    print(f"  {t:8s}: realized {len(b):3d} vs expected {l.p.sum():6.2f} in the P<{LOW} cells; per window {len(b)/nO:.3f} vs {l.p.sum()/nO:.3f}; "
          f"distinct events {b[['key','onset_week']].drop_duplicates().shape[0]}")
cap = df[df.type != "tariff"]
win = cap.groupby("origin").apply(lambda g: pd.Series({
    "blind": bool(((g.p < LOW) & g.hit).any()),
    "gen_p_blind": 1 - np.prod(1 - g.p[g.p < LOW]),       # generator's own chance of >=1 onset among its <5% sources
    "any_real": bool(g.hit.any()), "gen_p_any": 1 - np.prod(1 - g.p)}))
print(f"\n  capacity sources (lane, country, supplier): windows with >=1 blindside {win.blind.mean():.3f} "
      f"({int(win.blind.sum())}/{nO}); generator's own P(>=1 onset among its <{LOW} sources) averages {win.gen_p_blind.mean():.3f}")
print(f"  windows with >=1 realized capacity onset {win.any_real.mean():.3f}; generator P(>=1) averages {win.gen_p_any.mean():.3f}")
ev = blind.groupby(["type", "key", "onset_week"]).agg(n_windows=("origin", "size"), first_origin=("origin", "min"),
                                                       max_p=("p", "max")).reset_index()
# P at the origin one week before onset (last chance to react)
ev["p_at_onset-1"] = [df[(df.type == a) & (df.key == b) & (df.origin == w - 1)].p.round(3).tolist() for a, b, w in zip(ev.type, ev.key, ev.onset_week)]
print("\n  distinct blindside events\n" + ev.sort_values("onset_week").round(3).to_string(index=False))
allev = df[df.hit].groupby(["type", "key", "onset_week"]).agg(p_min=("p", "min"), p_max=("p", "max"),
        p_last=("p", "last")).reset_index()
print("\n4. EVERY REALIZED ONSET in origins' windows: P range across the windows that contain it, and P one week before\n"
      + allev.sort_values(["onset_week"]).round(3).to_string(index=False))
m = df[(df.type == "lane") & (df.key == "MERROW")]
print("\n5. MERROW: origins with P >= 0.2\n" + m[m.p >= 0.2][["origin", "p", "hit", "onset_week"]].round(3).to_string(index=False))
print(f"  Merrow overall: expected {m.p.sum():.2f} vs realized {int(m.hit.sum())} source-windows")
df.to_csv(os.path.join(HERE, f"onset_rate_cells_pm{PM:g}.csv"), index=False)
