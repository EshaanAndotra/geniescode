"""Scenario stress test of the private-window rows (222-234). Scores F, fill skill, diversification
(points out of 70: 40F + 21 fill + 9 div) under hand-built futures. Diagnostic only."""
import os, sys, copy, warnings; warnings.filterwarnings("ignore")
PL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PL); os.chdir(PL)
os.environ["MODEL1_DIR"] = os.path.join(PL, "Model 1", "iter4")
import numpy as np, pandas as pd
from run_pipeline import load_module
m3 = load_module("model3")
W0 = m3.World(m3.load_data("../../Data_new"))
lc = pd.read_csv("../../submissions/phase2/landed_costs_p2.csv"); lc = lc[lc.window == "222-234"]
S, E = 222, 234
sup = W0.sup; key = {(m, c): u for u, (m, c) in enumerate(zip(sup.material, sup.country))}
base_t = lc[lc.week == S].set_index(["material", "country"]).tariff_forecast

def base_world():
    w = copy.copy(W0)
    w.cap, w.lead, w.landed, w.disrupted = W0.cap.copy(), W0.lead.copy(), W0.landed.copy(), W0.disrupted.copy()
    for u in range(len(sup)):
        w.cap[u, S-1:E] = 0.984 * W0.normal_cap[u]; w.lead[u, S-1:E] = W0.base_lead[u]
    for r in lc.itertuples():
        u = key[(r.material, r.country)]
        w.landed[u, r.week-1] = r.landed_cost_calm + r.spot_calm * (r.tariff_forecast - base_t[(r.material, r.country)])
    w.disrupted[:, S-1:E] = 0
    for m in ("aluminum", "semiconductors"):
        w.disrupted[w.materials.index(m), 224-1:E] = 1
    w.last_week = E
    return w

def hit(w, units, a, b, cap=None, lead_add=0, price_mult=None, price_units=None):
    for u in units:
        if cap is not None: w.cap[u, a-1:b] = cap * W0.normal_cap[u]
        w.lead[u, a-1:b] += lead_add
    for u in (price_units if price_units is not None else units):
        if price_mult: w.landed[u, a-1:b] *= price_mult
    for u in units:
        w.disrupted[w.materials.index(sup.material[u]), a-1:b] = 1

U = lambda pred: [u for u in range(len(sup)) if pred(sup.loc[u])]
mat_units = lambda m: U(lambda r: r.material == m)
def merrow(w, a=223, b=230): hit(w, U(lambda r: r.lane == "MERROW"), a, b, cap=0.78, lead_add=3, price_mult=1.06)
def val_cobalt(w, a=223, b=233): hit(w, [key[("cobalt","VAL")]], a, b, cap=0.30, price_units=mat_units("cobalt"), price_mult=1.25)
def kes_re(w, a=223, b=230): hit(w, [key[("rare_earths","KES")]], a, b, cap=0.30, price_units=mat_units("rare_earths"), price_mult=1.25)
def kes_country(w, a=226, b=234): hit(w, U(lambda r: r.country == "KES"), a, b, cap=0.20, price_mult=1.10,
                                   price_units=[u for m in set(sup.material[sup.country=="KES"]) for u in mat_units(m)])
def kes_country_early(w): kes_country(w, 223, 234)
def merrow_big(w): hit(w, U(lambda r: r.lane == "MERROW"), 223, 234, cap=0.40, lead_add=4, price_mult=1.15)
def atlantic(w): hit(w, U(lambda r: r.lane == "ATLANTIC"), 225, 230, cap=0.78, lead_add=3, price_mult=1.06)
def south(w): hit(w, U(lambda r: r.lane == "SOUTH"), 225, 230, cap=0.78, lead_add=3, price_mult=1.06)
SCEN = {
    "calm (tariff only)": [],
    "Merrow closure 223-230": [merrow],
    "VAL cobalt ban 223-233": [val_cobalt],
    "KES rare_earths 223-230": [kes_re],
    "cobalt + rare_earths (model's view)": [val_cobalt, kes_re],
    "cobalt + RE + Merrow": [val_cobalt, kes_re, merrow],
    "KES country event 226-234": [kes_country],
    "cobalt + RE + KES country 226+": [val_cobalt, kes_re, kes_country],
    "KES country 223-234": [kes_country_early],
    "KES country 223+ + Merrow": [kes_country_early, merrow],
    "Merrow 40% cap 223-234 (bigger than training)": [merrow_big],
    "Atlantic closure 225-230": [atlantic],
    "South closure 225-230": [south],
}
files = {"main": "../../submissions/phase2/submission_moderate_p2.csv", "hedge": "../../submissions/phase2/submission_aggressive_p2.csv"}
plans = {}
for k, f in files.items():
    d = pd.read_csv(f); x = d.id.str.extract(r"^w(\d+)_(.+)$"); d["week"], d["material"] = x[0].astype(int), x[1]
    plans[k] = d[d.week.between(S, E)]
nv = m3.naive_plan(W0, S, E)
plans["naive"] = nv
td = nv.copy()   # naive, but aluminum/semis: move KES volume to the next-cheapest supplier from wk 224
for m, alt in (("aluminum", "HAL"), ("semiconductors", "SUN")):
    i = (td.material == m) & (td.week >= 224)
    td.loc[i, "share_KES"] = 0.0; td.loc[i, f"share_{alt}"] = 1.0
plans["naive+tariff dodge"] = td
rows = []
for name, fns in SCEN.items():
    w = base_world()
    for fn in fns: fn(w)
    for k, p in plans.items():
        r = m3.score_window(w, p, S, E, "sum5")
        fs = r["fill_skill"] if np.isfinite(r["fill_skill"]) else 0.0
        rows.append(dict(scenario=name, plan=k, pts70=40*r["F"] + 21*fs + 9*r["div"], F=r["F_raw"], fill_skill=r["fill_skill_raw"],
                         fill_ours=r["fill_ours"], fill_naive=r["fill_naive"], div=r["div"]))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.pivot(index="scenario", columns="plan", values="pts70").reindex(SCEN.keys()).round(1).to_string())
print(df[df.plan.isin(["main","hedge"])].round(3).to_string(index=False))
