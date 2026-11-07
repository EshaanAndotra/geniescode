# =====================================================================================
# Model 1: Disruption forecast  (DSDG Hackathon 2026 - Supply Chain Disruptor)
#
# Fills disruption_prob and writes the tables Model 3 needs. Two parts:
#   - Duration model (material already in a shock): Kaplan-Meier survival of past shock
#     lengths, blended with a geometric curve and given a longer tail.
#   - Onset model (material calm): logistic regression + gradient boosting on tariff,
#     market, shipping, macro, supply, history and news-warning features, blended and
#     Platt-calibrated on out-of-fold backtest predictions.
#   In a shock: p = P_cont + (1 - P_cont) * p_onset.  Clipped to [0.02, 0.98].
#
# Everything is computed "as of" an origin week T using only data up to T, so the same
# code runs the rolling backtest (T = 130, 143, ..., 208) and the real forecast (T = last
# released week). Runs as one Databricks cell or as a plain Python script.
#
# Outputs (CSV in OUTPUT_DIR, and Delta tables if running on Databricks):
#   m1_disruption     origin_week, material, week, h, p, p_onset, p_cont, in_shock, label
#   m1_supplier_risk  origin_week, material, country, week, p_supplier, risk_share
#   m1_metrics        Brier skill by version and slice (the ablation table)
#   m1_submission_col id, disruption_prob  (weeks 209-234, ready to merge)
# =====================================================================================
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
DATA_DIR = "/Workspace/Repos/DSDG Hackathon/DSDG Hackathon/Data"
OUTPUT_DIR = "/tmp/m1_outputs"
TABLE_PREFIX = "workspace.default."   # Unity Catalog schema for the shared Delta tables
WRITE_TABLES = True                   # only takes effect where `spark` exists (Databricks)
USE_NEWS = True                       # news warning-template features
USE_MLFLOW = True
MLFLOW_EXPERIMENT = None              # e.g. "/Users/you@example.com/model1_disruption"

BACKTEST_START, BACKTEST_STEP = 130, 13   # origins 130, 143, ..., up to the last full one
REPORT_FROM = 156                         # origins >= this go into the headline metrics
PUBLIC_ORIGIN = 208                       # forecasts weeks 209-221 (public window)
SUB_FIRST, SUB_LAST, PUBLIC_LAST = 209, 234, 221
WARMUP = 26                               # first origin week used for training rows
NEWS_LOOKAHEAD = 6                        # an article "came true" if an onset follows in 1-6 wks
NEWS_SMOOTH = 5                           # pseudo-articles pulling template rates to the base
CLIP = (0.02, 0.98)
LR_C = 0.1                                # logistic regression regularisation (smaller = stronger)
KM_PRIOR = 20                             # weight of the geometric curve vs raw KM counts
TAIL_STRETCH = 1.25                       # longer-than-history tail for the big test shock

# ------------------------------------------------------------------ load
def load(name):
    return pd.read_csv(f"{DATA_DIR}/{name}.csv")

news, prices, tariffs, shipping, macro = (load(n) for n in ["news", "prices", "tariffs", "shipping", "macro"])
sup, outc, hist = load("suppliers"), load("outcomes"), load("supply_history")

MATERIALS = sorted(sup["material"].unique())
LAST = int(outc["week"].max())            # last week with labels (208 in Phase 1, 221 in Phase 2)
FINAL_ORIGIN = LAST
MAX_H = max(13, SUB_LAST - FINAL_ORIGIN)  # 13 with Phase 2 data, 26 with Phase 1 only
print(f"labels through week {LAST} | final origin {FINAL_ORIGIN} | max horizon {MAX_H}")

D = outc.pivot(index="week", columns="material", values="disrupted").sort_index().fillna(0).astype(int)
onset_weeks = {}
for m in MATERIALS:
    s = D[m].values
    onset_weeks[m] = D.index.values[(s == 1) & (np.r_[0, s[:-1]] == 0)]
print(f"disruption onsets in labels: {sum(len(v) for v in onset_weeks.values())}")


def lag(df, keys, col, k=4):
    return df.groupby(keys)[col].shift(k)


# ------------------------------------------------------------------ history features
hist_rows = []
for m in MATERIALS:
    s, weeks = D[m].values, D.index.values
    age = np.zeros(len(s), int)
    since = np.zeros(len(s))
    c, last_dis = 0, None
    for i, v in enumerate(s):
        c = c + 1 if v == 1 else 0
        age[i] = c
        if v == 1:
            last_dis = weeks[i]
        since[i] = weeks[i] - last_dis if last_dis is not None else weeks[i]
    onset = ((s == 1) & (np.r_[0, s[:-1]] == 0)).astype(int)
    hist_rows.append(pd.DataFrame({
        "material": m, "week": weeks, "in_shock": s, "age": age,
        "weeks_since_dis": since,
        "shocks_52": pd.Series(onset).rolling(52, min_periods=1).sum().values,
        "base_so_far": np.cumsum(s) / np.arange(1, len(s) + 1),
    }))
hist_feat = pd.concat(hist_rows, ignore_index=True)

# ------------------------------------------------------------------ supplier-week panel
p = prices.merge(sup[["material", "country", "lane", "base_lead_weeks", "normal_capacity_units"]],
                 on=["material", "country"], how="left")
p = p.merge(tariffs[["week", "material", "country", "tariff_rate", "announced_rate", "announced_effective_week"]],
            on=["week", "material", "country"], how="left")
p = p.merge(hist[["week", "material", "country", "capacity_units", "lead_time_weeks"]],
            on=["week", "material", "country"], how="left")

ship = shipping.sort_values(["lane", "week"]).copy()
ship["cong"] = ship["congestion_index"]
ship["cong_chg4"] = ship["cong"] - lag(ship, "lane", "congestion_index")
ship["transit_ratio"] = ship["transit_days"] / ship.groupby("lane")["transit_days"].transform(
    lambda x: x.expanding().median())
ship["crate_chg4"] = ship["container_rate"] / lag(ship, "lane", "container_rate") - 1

mac = macro.sort_values(["country", "week"]).copy()
g = mac.groupby("country")["tension_index"]
mac["tension"] = mac["tension_index"]
mac["tension_chg4"] = mac["tension"] - lag(mac, "country", "tension_index")
mac["tension_z"] = (mac["tension"] - g.transform(lambda x: x.expanding(8).mean())) / \
    g.transform(lambda x: x.expanding(8).std()).replace(0, np.nan)
mac["pmi_chg4"] = mac["pmi"] - lag(mac, "country", "pmi")
mac["fx_chg4"] = mac["fx_rate"] / lag(mac, "country", "fx_rate") - 1

p = p.merge(ship[["week", "lane", "cong", "cong_chg4", "transit_ratio", "crate_chg4"]], on=["week", "lane"], how="left")
p = p.merge(mac[["week", "country", "tension", "tension_chg4", "tension_z", "pmi_chg4", "fx_chg4"]],
            on=["week", "country"], how="left")
p = p.sort_values(["material", "country", "week"]).reset_index(drop=True)
K = ["material", "country"]
p["fut_prem"] = p["futures_4w"] / p["spot_price"] - 1
p["fut_prem_chg4"] = p["fut_prem"] - lag(p, K, "fut_prem")
p["ret4"] = p["spot_price"] / lag(p, K, "spot_price") - 1
p["tariff_chg4"] = p["tariff_rate"] - lag(p, K, "tariff_rate")
upcoming = p["announced_effective_week"].notna() & (p["announced_effective_week"] > p["week"])
p["pending"] = np.where(upcoming, (p["announced_rate"] - p["tariff_rate"]).clip(lower=0), 0.0)
p["pending_eff"] = np.where(upcoming, p["announced_effective_week"], np.nan)
p["cap_ratio"] = p["capacity_units"] / p["normal_capacity_units"]
p["lead_excess"] = p["lead_time_weeks"] - p["base_lead_weeks"]

SUP_FEATS = ["fut_prem", "fut_prem_chg4", "ret4", "tariff_rate", "tariff_chg4", "pending",
             "cong", "cong_chg4", "transit_ratio", "crate_chg4",
             "tension", "tension_chg4", "tension_z", "pmi_chg4", "fx_chg4", "lead_excess"]
grp = p.groupby(["material", "week"])
F = grp[SUP_FEATS].agg(["max", "mean"])
F.columns = [f"{a}_{b}" for a, b in F.columns]
F["cap_ratio_min"] = grp["cap_ratio"].min()
F["cap_ratio_mean"] = grp["cap_ratio"].mean()
F = F.reset_index().merge(hist_feat, on=["material", "week"], how="left")

# ------------------------------------------------------------------ news warning features
COUNTRY_NAMES = {"Sunda Coast": "SUN", "Kestria": "KES", "Valdmar": "VAL",
                 "Orinoque": "ORI", "Halvard": "HAL", "Aurelia": "AUR"}
MATERIAL_PATTERNS = {
    "lithium": r"lithium", "cobalt": r"cobalt", "copper": r"copper", "aluminum": r"alumin",
    "steel": r"steel", "rare_earths": r"rare[ _-]?earth", "semiconductors": r"semiconductor|chip",
    "polymer_resin": r"polymer(?:[ _]resin)?|resin", "specialty_glass": r"(?:specialty[ _])?glass",
    "natural_gas": r"natural[ _]gas|\blng\b|\bgas\b",
}
LANE_PATTERNS = {"MERROW": r"merrow", "NORTH": r"northern sea|north sea", "SOUTH": r"southern route",
                 "ATLANTIC": r"atlantic", "RAIL": r"\brail"}
country_mats = sup.groupby("country")["material"].apply(set).to_dict()
lane_countries = sup.groupby("lane")["country"].apply(set).to_dict()


def template(h):
    t = str(h)
    for name in COUNTRY_NAMES:
        t = re.sub(name + r"\w*", "<C>", t)
    for pat in MATERIAL_PATTERNS.values():
        t = re.sub(r"(?i)\b(?:" + pat + r")\w*", "<M>", t)
    t = re.sub(r"(?i)merrow strait|northern sea(?: route)?|southern route|atlantic route", "<LANE>", t)
    return re.sub(r"\d+(?:\.\d+)?%?", "<N>", t)


def tag_materials(text):
    direct = {m for m, pat in MATERIAL_PATTERNS.items() if re.search(pat, text, re.I)}
    if direct:
        return sorted(direct)
    ctry = {c for n, c in COUNTRY_NAMES.items() if n in text}
    for lane, pat in LANE_PATTERNS.items():
        if re.search(pat, text, re.I):
            ctry |= lane_countries.get(lane, set())
    return sorted(set().union(*[country_mats.get(c, set()) for c in ctry])) if ctry else []


NEWS_FEATS = ["news_warn_1w", "news_warn_4w", "news_warn_8w", "news_max_4w", "news_sources_4w"]


def build_news_features():
    """Point-in-time warning score per article: how often its headline template was followed
    by a disruption onset, using only articles whose outcome was known by publication week
    (published at least NEWS_LOOKAHEAD weeks earlier). Then summed per material and week."""
    n = news.copy()
    n["template"] = n["headline"].map(template)
    n["material"] = (n["headline"].fillna("") + " " + n["body"].fillna("")).map(tag_materials)
    r = n.explode("material").dropna(subset=["material"])
    r = r[r["material"].isin(MATERIALS)].copy()
    dis = outc.set_index(["material", "week"])["disrupted"].to_dict()
    r["calm"] = [dis.get((m, w), 0) == 0 for m, w in zip(r["material"], r["week"])]

    def came_true(m, w):
        ws = onset_weeks.get(m, np.array([]))
        return bool(((ws > w) & (ws <= w + NEWS_LOOKAHEAD)).any())

    r["hit"] = [came_true(m, w) for m, w in zip(r["material"], r["week"])]
    W = int(max(r["week"].max(), LAST))
    weeks = np.arange(1, W + 1)
    lab = r[r["calm"]]
    tw = lab.groupby(["template", "week"])["hit"].agg(["size", "sum"]).reset_index()
    cum_n = tw.pivot(index="week", columns="template", values="size").reindex(weeks).fillna(0).cumsum()
    cum_h = tw.pivot(index="week", columns="template", values="sum").reindex(weeks).fillna(0).cumsum()
    cum_n = cum_n.shift(NEWS_LOOKAHEAD).fillna(0)   # row w = articles published by week w-6
    cum_h = cum_h.shift(NEWS_LOOKAHEAD).fillna(0)
    base = (cum_h.sum(axis=1) + 1) / (cum_n.sum(axis=1) + 20)
    rate = (cum_h + NEWS_SMOOTH * base.values[:, None]) / (cum_n + NEWS_SMOOTH)
    lift = rate.div(base, axis=0)
    lift.index.name, lift.columns.name = "week", "template"
    lift_long = lift.stack().rename("lift").reset_index()
    r = r.merge(lift_long, on=["week", "template"], how="left")
    r["warn"] = (r["lift"].fillna(1.0) - 1).clip(lower=0)
    r["warn_src"] = np.where(r["warn"] > 0.5, r["source"], None)
    mw = r.groupby(["material", "week"]).agg(warn_sum=("warn", "sum"), warn_max=("warn", "max"),
                                             src=("warn_src", "nunique")).reset_index()
    grid = pd.MultiIndex.from_product([MATERIALS, weeks], names=["material", "week"])
    mw = mw.set_index(["material", "week"]).reindex(grid).fillna(0).reset_index()
    g = mw.groupby("material")
    mw["news_warn_1w"] = mw["warn_sum"]
    mw["news_warn_4w"] = g["warn_sum"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    mw["news_warn_8w"] = g["warn_sum"].transform(lambda x: x.rolling(8, min_periods=1).sum())
    mw["news_max_4w"] = g["warn_max"].transform(lambda x: x.rolling(4, min_periods=1).max())
    mw["news_sources_4w"] = g["src"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    final_lift = lift.iloc[-1].sort_values(ascending=False)
    counts = cum_n.iloc[-1]
    top = pd.DataFrame({"lift": final_lift, "labelled_rows": counts.reindex(final_lift.index)})
    print("\nNews templates with the highest warning lift (as of the last week):")
    print(top[top["labelled_rows"] >= 3].head(15).round(2).to_string())
    return mw[["material", "week"] + NEWS_FEATS]


if USE_NEWS:
    F = F.merge(build_news_features(), on=["material", "week"], how="left")
else:
    NEWS_FEATS = []

# ------------------------------------------------------------------ (material, origin, horizon) rows
base_T = F[(F["week"] >= WARMUP) & (F["week"] <= LAST)].rename(columns={"week": "T"})
ALL = base_T.merge(pd.DataFrame({"h": np.arange(1, MAX_H + 1)}), how="cross")
ALL["week"] = ALL["T"] + ALL["h"]
ALL["log_h"] = np.log(ALL["h"])
lab_series = outc.set_index(["material", "week"])["disrupted"]
ALL["label"] = lab_series.reindex(pd.MultiIndex.from_arrays([ALL["material"], ALL["week"]])).values
ALL = ALL.reset_index(drop=True)
ALL["rid"] = np.arange(len(ALL))

# tariff increase announced to take effect on or before the target week
tp = p[["material", "week", "pending", "pending_eff"]].rename(columns={"week": "T"})
mm = ALL[["rid", "material", "T", "week"]].merge(tp, on=["material", "T"], how="left")
mm["ph"] = np.where(mm["pending_eff"] <= mm["week"], mm["pending"], 0.0)
ALL["pending_by_h"] = mm.groupby("rid")["ph"].max().reindex(ALL["rid"]).fillna(0).values

EXCLUDE = {"material", "T", "week", "label", "rid", "in_shock", "age"}
FEATURES = [c for c in ALL.columns if c not in EXCLUDE]
MONO_POS = {"pending_by_h", "pending_max", "tariff_chg4_max", "tension_z_max", "tension_chg4_max",
            "cong_chg4_max", "lead_excess_max", *NEWS_FEATS}
MONO_NEG = {"cap_ratio_min"}
MONO = [1 if f in MONO_POS else -1 if f in MONO_NEG else 0 for f in FEATURES]
print(f"\ntraining-row universe: {len(ALL):,} rows | {len(FEATURES)} features")

# ------------------------------------------------------------------ onset model
def fit_onset(train):
    X, y = train[FEATURES], train["label"].astype(int)
    if y.sum() < 5 or y.nunique() < 2:
        return None
    med = X.median().fillna(0)
    scaler = StandardScaler().fit(X.fillna(med))
    lr = LogisticRegression(C=LR_C, max_iter=3000).fit(scaler.transform(X.fillna(med)), y)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         monotonic_cst=MONO, random_state=0).fit(X, y)
    return {"med": med, "scaler": scaler, "lr": lr, "hgb": hgb}


def predict_onset(model, X):
    Xi = model["scaler"].transform(X[FEATURES].fillna(model["med"]))
    return model["lr"].predict_proba(Xi)[:, 1], model["hgb"].predict_proba(X[FEATURES])[:, 1]


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def fit_calibrator(oof):
    """Blend weight by Brier on out-of-fold rows, then Platt: logit(p) = a + b*logit(blend) + c*h."""
    if oof is None or oof["label"].sum() < 10:
        return 0.5, None
    y = oof["label"].astype(int).values
    ws = np.linspace(0, 1, 11)
    bs = [np.mean((w * oof["p_lr"] + (1 - w) * oof["p_hgb"] - y) ** 2) for w in ws]
    w = float(ws[int(np.argmin(bs))])
    pb = w * oof["p_lr"] + (1 - w) * oof["p_hgb"]
    cal = LogisticRegression(C=1e4, max_iter=2000).fit(np.c_[logit(pb), oof["h"]], y)
    return w, cal


def apply_calibrator(w, cal, p_lr, p_hgb, h):
    pb = w * p_lr + (1 - w) * p_hgb
    return pb if cal is None else cal.predict_proba(np.c_[logit(pb), h])[:, 1]


# ------------------------------------------------------------------ duration model
DMAX = 600


def episodes_upto(O):
    eps = []   # (length, completed?)
    for m in MATERIALS:
        c = 0
        for v in D.loc[:O, m].values:
            if v == 1:
                c += 1
            elif c > 0:
                eps.append((c, True))
                c = 0
        if c > 0:
            eps.append((c, False))   # still running at O: censored
    return eps


def survival_curve(eps):
    """S[d] = P(shock lasts more than d weeks). KM blended with a stretched geometric tail."""
    if not eps:
        return 0.85 ** np.arange(DMAX + 1)
    L = np.array([e[0] for e in eps])
    done = np.array([e[1] for e in eps])
    km = np.ones(DMAX + 1)
    for k in range(1, DMAX + 1):
        at_risk = (L >= k).sum()
        km[k] = km[k - 1] * (1 - ((L == k) & done).sum() / at_risk) if at_risk else km[k - 1]
    mean_len = L[done].mean() if done.any() else 6.0
    q = 1 - 1 / max(mean_len * TAIL_STRETCH, 1.5)
    geo = q ** np.arange(DMAX + 1)
    w = len(L) / (len(L) + KM_PRIOR)
    S = w * km + (1 - w) * geo
    top = int(L.max())
    S[top + 1:] = S[top] * q ** np.arange(1, DMAX - top + 1)
    return S


def p_continue(S, age, h):
    a, b = min(age - 1, DMAX), min(age + h - 1, DMAX)
    return float(S[b] / max(S[a], 1e-9))


# ------------------------------------------------------------------ one origin, end to end
def run_origin(O, oof_before):
    train = ALL[(ALL["week"] <= O) & (ALL["in_shock"] == 0) & ALL["label"].notna()]
    pred = ALL[ALL["T"] == O].copy()
    base = D.loc[:O].mean()
    pred["p_base"] = pred["material"].map(base).values
    model = fit_onset(train)
    if model is None:
        pred["p_lr"] = pred["p_hgb"] = pred["p_base"]
    else:
        pred["p_lr"], pred["p_hgb"] = predict_onset(model, pred)
    w, cal = fit_calibrator(oof_before)
    pred["p_onset"] = apply_calibrator(w, cal, pred["p_lr"], pred["p_hgb"], pred["h"])
    S = survival_curve(episodes_upto(O))
    pred["p_cont"] = [p_continue(S, a, h) if s else 0.0
                      for s, a, h in zip(pred["in_shock"], pred["age"], pred["h"])]

    def combine(p_new):
        return np.clip(np.where(pred["in_shock"] == 1, pred["p_cont"] + (1 - pred["p_cont"]) * p_new, p_new), *CLIP)

    pred["p_dur"] = combine(pred["p_base"])      # v1: duration model + base rate
    pred["p_lr_only"] = combine(pred["p_lr"])
    pred["p_hgb_only"] = combine(pred["p_hgb"])
    pred["p"] = combine(pred["p_onset"])         # full model
    pred["origin_week"] = O
    info = {"origin": O, "blend_w": w, "calibrated": cal is not None, "train_rows": len(train),
            "train_pos": int(train["label"].sum())}
    return pred, model, info


# ------------------------------------------------------------------ rolling backtest
origins = list(range(BACKTEST_START, LAST, BACKTEST_STEP))
preds, oof, infos, models = [], [], [], {}
for O in origins:
    pr, mdl, info = run_origin(O, pd.concat(oof) if oof else None)
    preds.append(pr)
    infos.append(info)
    models[O] = mdl
    oof.append(pr[(pr["in_shock"] == 0) & pr["label"].notna()][["p_lr", "p_hgb", "h", "label"]])
print("\nbacktest origins:")
print(pd.DataFrame(infos).to_string(index=False))

bt = pd.concat(preds, ignore_index=True)
scored = bt[bt["label"].notna() & (bt["origin_week"] >= REPORT_FROM)]
VERSIONS = {"v0_base_rate": "p_base", "v1_duration": "p_dur", "lr_only": "p_lr_only",
            "hgb_only": "p_hgb_only", "full_blend_cal": "p"}


def brier_skill(df, col):
    if len(df) == 0:
        return np.nan
    bs = np.mean((df[col] - df["label"]) ** 2)
    bb = np.mean((df["p_base"] - df["label"]) ** 2)
    return 1 - bs / bb if bb > 0 else np.nan


rows = []
for name, col in VERSIONS.items():
    rows.append({"version": name,
                 "skill_all": brier_skill(scored, col),
                 "skill_calm": brier_skill(scored[scored["in_shock"] == 0], col),
                 "skill_in_shock": brier_skill(scored[scored["in_shock"] == 1], col)})
metrics = pd.DataFrame(rows)
print(f"\nBRIER SKILL vs base rate (origins >= {REPORT_FROM}, {len(scored):,} scored rows):")
print(metrics.round(3).to_string(index=False))

per_origin = scored.groupby("origin_week").apply(lambda d: pd.Series(
    {"skill_full": brier_skill(d, "p"), "rows": len(d), "disrupted_rate": d["label"].mean()}))
print("\nper origin:")
print(per_origin.round(3).to_string())

scored = scored.assign(bucket=pd.cut(scored["p"], [0, .05, .1, .2, .3, .5, .7, 1.0]))
rel = scored.groupby("bucket", observed=True).agg(mean_pred=("p", "mean"), actual=("label", "mean"),
                                                  rows=("label", "size"))
print("\nreliability (mean prediction vs actual rate per bucket; close = calibrated):")
print(rel.round(3).to_string())

# ------------------------------------------------------------------ final forecasts
final = {O: bt[bt["origin_week"] == O] for O in {PUBLIC_ORIGIN} if O in origins}
for O in sorted({PUBLIC_ORIGIN, FINAL_ORIGIN}):
    if O not in final and O <= LAST:
        before = [o for o, d in zip(origins, oof) if o < O]
        pr, mdl, info = run_origin(O, pd.concat([d for o, d in zip(origins, oof) if o < O]) if before else None)
        final[O] = pr
        models[O] = mdl
        print(f"final origin {O}: blend weight {info['blend_w']}, calibrated {info['calibrated']}")

if models.get(FINAL_ORIGIN) is not None:
    coef = pd.Series(models[FINAL_ORIGIN]["lr"].coef_[0], index=FEATURES)
    print("\nlargest logistic-regression coefficients at the final origin (standardised features):")
    print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(15).round(3).to_string())

sub_rows = []
for wk in range(SUB_FIRST, SUB_LAST + 1):
    O = PUBLIC_ORIGIN if (wk <= PUBLIC_LAST and PUBLIC_ORIGIN in final) else FINAL_ORIGIN
    d = final[O]
    d = d[d["week"] == wk]
    for m in MATERIALS:
        val = d.loc[d["material"] == m, "p"]
        sub_rows.append({"id": f"w{wk}_{m}", "disruption_prob": float(val.iloc[0]) if len(val) else float(D[m].mean())})
submission_col = pd.DataFrame(sub_rows)
assert len(submission_col) == 10 * (SUB_LAST - SUB_FIRST + 1) and submission_col["disruption_prob"].between(0, 1).all()

# ------------------------------------------------------------------ supplier-level risk for Model 3
# Quick version: split each material's probability across its suppliers by a risk score,
# with p_i = 1 - (1 - p)^share_i so that 1 - prod(1 - p_i) = p exactly.
risk_rows = []
for O, d in final.items():
    s = p[p["week"] == O][["material", "country", "cap_ratio", "lead_excess", "tension_z", "cong_chg4", "pending"]].copy()
    hit = ((s["cap_ratio"] < 0.7) | (s["lead_excess"] >= 1)).astype(float)
    s["risk"] = (1 + 3 * hit + s["tension_z"].clip(0, 3).fillna(0)
                 + (s["cong_chg4"] / 10).clip(0, 2).fillna(0) + 20 * s["pending"].fillna(0))
    s["risk_share"] = s["risk"] / s.groupby("material")["risk"].transform("sum")
    j = d[["material", "week", "p"]].merge(s[["material", "country", "risk_share"]], on="material")
    j["p_supplier"] = 1 - (1 - j["p"]) ** j["risk_share"]
    j["origin_week"] = O
    risk_rows.append(j[["origin_week", "material", "country", "week", "p_supplier", "risk_share"]])
supplier_risk = pd.concat(risk_rows, ignore_index=True)

# ------------------------------------------------------------------ save
COLS = ["origin_week", "material", "week", "h", "p", "p_onset", "p_cont", "in_shock", "label"]
m1_disruption = pd.concat([bt] + [d for O, d in final.items() if O not in origins], ignore_index=True)[COLS]
metrics_out = metrics.assign(report_from=REPORT_FROM, last_week=LAST, use_news=USE_NEWS)


def save(df, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/{name}.csv", index=False)
    if WRITE_TABLES:
        try:
            spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true") \
                .saveAsTable(TABLE_PREFIX + name)  # noqa: F821  (spark exists on Databricks)
            print(f"wrote table {TABLE_PREFIX + name}")
        except NameError:
            pass
        except Exception as e:
            print(f"could not write table {name}: {e}")


for df, name in [(m1_disruption, "m1_disruption"), (supplier_risk, "m1_supplier_risk"),
                 (metrics_out, "m1_metrics"), (submission_col, "m1_submission_col")]:
    save(df, name)
print(f"\nsaved outputs to {OUTPUT_DIR}")

if USE_MLFLOW:
    try:
        import mlflow
        if MLFLOW_EXPERIMENT:
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"m1_disruption_last{LAST}_news{int(USE_NEWS)}"):
            mlflow.log_params({"last_week": LAST, "use_news": USE_NEWS, "lr_c": LR_C, "km_prior": KM_PRIOR,
                               "tail_stretch": TAIL_STRETCH, "clip": str(CLIP), "origins": str(origins),
                               "n_features": len(FEATURES)})
            for r in metrics.itertuples():
                for k in ["skill_all", "skill_calm", "skill_in_shock"]:
                    v = getattr(r, k)
                    if pd.notna(v):
                        mlflow.log_metric(f"{r.version}_{k}", float(v))
            mlflow.log_artifact(f"{OUTPUT_DIR}/m1_metrics.csv")
        print("logged MLflow run")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")