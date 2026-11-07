# =====================================================================================
# Model 1, part C: combined + calibrated model  (DSDG Hackathon 2026 - Supply Chain Disruptor)
#   Run text_model.py and numerical_model.py first. This file joins their feature tables
#   (m1_news_features + m1_numeric_features) and trains ONE model on all of them.
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
OUTPUT_DIR = "/tmp/m1_outputs"        # keep the same in all three files
TABLE_PREFIX = "workspace.default."   # keep the same in all three files
WRITE_TABLES = True                   # only takes effect where `spark` exists (Databricks)
USE_NEWS = True                       # use m1_news_features from text_model.py if present
USE_MLFLOW = True
MLFLOW_EXPERIMENT = None              # e.g. "/Users/you@example.com/model1_disruption"

BACKTEST_START, BACKTEST_STEP = 104, 4    # origins 104, 108, ..., up to the last labelled week
REPORT_FROM = 130                         # origins >= this go into the headline metrics
PUBLIC_ORIGIN = 208                       # forecasts weeks 209-221 (public window)
SUB_FIRST, SUB_LAST, PUBLIC_LAST = 209, 234, 221
WARMUP = 26                               # first origin week used for training rows
CLIP = (0.02, 0.98)
FEATURE_SET = "lean"                      # "lean" (~20 key features; few shocks) or "full" (everything)
MODEL_TYPE = "lr"                         # "lr" (logistic regression) or "blend" (+ gradient boosting)
LR_C = 0.1                                # logistic regression regularisation (smaller = stronger)
KM_PRIOR = 20                             # weight of the geometric curve vs raw KM counts
TAIL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]   # duration-tail stretches tried; backtest picks one
SHRINK_GRID = [0.0, 0.25, 0.5, 0.75, 1.0] # weight on the onset model vs base rate; backtest picks one
IN_SHOCK_NEW = "auto"                     # in-shock rows: "model" adds onset risk on top of the duration
                                          # curve, "base" adds only the base rate, "auto" picks by backtest

RUN_FEATURE_FILES = True                  # True: run text_model.py + numerical_model.py first (one click)
                                          # False: reuse their saved tables (faster when only tuning here)
FEATURE_DIR = None                        # folder holding the two feature files; None = this file's folder

# ------------------------------------------------------------------ run the feature files first
def _run_feature_file(name):
    """Run text_model / numerical_model in a fresh namespace. Works whether they are plain .py
    files or Databricks notebooks sitting next to this one."""
    folder = FEATURE_DIR or (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
    for path in (os.path.join(folder, name + ".py"), os.path.join(folder, name)):
        if os.path.isfile(path):
            print(f"\n========== running {path} ==========")
            ns = {"__name__": "__main__", "__file__": path}
            if "spark" in globals():
                ns["spark"] = globals()["spark"]
            with open(path) as fh:
                exec(compile(fh.read(), path, "exec"), ns)
            return
    try:  # Databricks notebook fallback
        print(f"\n========== running notebook ./{name} ==========")
        dbutils.notebook.run(f"./{name}", 3600)  # noqa: F821
    except NameError:
        print(f"WARNING: could not find {name}.py in {folder}.")
    except Exception as e:
        print(f"WARNING: could not run {name}: {str(e).splitlines()[0][:200]}")
    try:
        print(f"files in {folder}: {sorted(os.listdir(folder))}")
    except Exception:
        pass
    print(f"Continuing with the {name} tables saved by its last run. If they are missing or old, run {name} "
          "yourself, or set FEATURE_DIR (or RUN_FEATURE_FILES = False) at the top of this file.")


if RUN_FEATURE_FILES:
    for _name in ("text_model", "numerical_model"):
        _run_feature_file(_name)
    print("\n========== feature files done, training the combined model ==========\n")

# ------------------------------------------------------------------ load labels + feature tables
sup = pd.read_csv(f"{DATA_DIR}/suppliers.csv")
outc = pd.read_csv(f"{DATA_DIR}/outcomes.csv")
MATERIALS = sorted(sup["material"].unique())
LAST = int(outc["week"].max())            # last week with labels (208 in Phase 1, 221 in Phase 2)
FINAL_ORIGIN = LAST
MAX_H = max(13, SUB_LAST - FINAL_ORIGIN)  # 13 with Phase 2 data, 26 with Phase 1 only
print(f"labels through week {LAST} | final origin {FINAL_ORIGIN} | max horizon {MAX_H}")

D = outc.pivot(index="week", columns="material", values="disrupted").sort_index().fillna(0).astype(int)
n_onsets = sum(int(((D[m] == 1) & (D[m].shift(1, fill_value=0) == 0)).sum()) for m in MATERIALS)
print(f"disruption onsets in labels: {n_onsets}")


def load_table(name):
    """Read a feature table written by text_model.py / numerical_model.py (Delta first, then CSV)."""
    try:
        if spark.catalog.tableExists(TABLE_PREFIX + name):  # noqa: F821
            return spark.table(TABLE_PREFIX + name).toPandas()  # noqa: F821
        print(f"table {TABLE_PREFIX + name} does not exist yet")
    except NameError:
        pass
    except Exception as e:
        print(f"could not read table {name}: {str(e).splitlines()[0]}")
    path = f"{OUTPUT_DIR}/{name}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


F = load_table("m1_numeric_features")
p = load_table("m1_supplier_panel")
if F is None or p is None:
    raise RuntimeError("m1_numeric_features / m1_supplier_panel not found. Set RUN_FEATURE_FILES = True, or run "
                       "numerical_model.py first and check its output says 'wrote table ...'.")
if int(F["week"].max()) < LAST:
    print(f"WARNING: numeric features end at week {int(F['week'].max())} but labels run to {LAST} "
          "- rerun numerical_model.py on the new data.")

NEWS_FEATS = []
if USE_NEWS:
    nf = load_table("m1_news_features")
    if nf is None:
        print("\nWARNING: no m1_news_features found - run text_model.py first. Continuing without news.")
    else:
        NEWS_FEATS = [c for c in nf.columns if c.startswith("news_")]
        F = F.merge(nf[["material", "week"] + NEWS_FEATS], on=["material", "week"], how="left")
        if int(nf["week"].max()) < LAST:
            print(f"\nWARNING: news features end at week {int(nf['week'].max())} but labels run to {LAST} "
                  "- rerun text_model.py on the new data.")
        print(f"\nnews features loaded: {NEWS_FEATS}")
USE_SOURCE_RELIABILITY = "news_warn_rel_4w" in NEWS_FEATS

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

if "news_fixed_4w" in ALL.columns:
    ALL["warn_near"] = ALL["news_fixed_4w"] * (ALL["h"] <= 6)   # warnings matter most 1-6 weeks out

EXCLUDE = {"material", "T", "week", "label", "rid", "in_shock", "age"}
LEAN_FEATURES = ["pending_by_h", "pending_max", "tariff_chg4_max", "tension_z_max", "tension_chg4_max",
                 "cong_chg4_max", "transit_ratio_max", "fut_prem_max", "fut_prem_chg4_max", "cap_ratio_min",
                 "lead_excess_max", "weeks_since_dis", "shocks_52", "base_so_far",
                 "news_fixed_4w", "news_fixed_8w", "news_fixed_types_8w", "warn_near", "h", "log_h"]
if FEATURE_SET == "lean":
    FEATURES = [c for c in LEAN_FEATURES if c in ALL.columns]
else:
    FEATURES = [c for c in ALL.columns if c not in EXCLUDE]
MONO_POS = {"pending_by_h", "pending_max", "tariff_chg4_max", "tension_z_max", "tension_chg4_max",
            "cong_chg4_max", "lead_excess_max", "warn_near", *NEWS_FEATS}
MONO_NEG = {"cap_ratio_min"}
MONO = [1 if f in MONO_POS else -1 if f in MONO_NEG else 0 for f in FEATURES]
print(f"\ntraining-row universe: {len(ALL):,} rows | {len(FEATURES)} features ({FEATURE_SET}) | model: {MODEL_TYPE}")


# ------------------------------------------------------------------ onset model
def fit_onset(train):
    X, y = train[FEATURES], train["label"].astype(int)
    if y.sum() < 5 or y.nunique() < 2:
        return None
    med = X.median().fillna(0)
    scaler = StandardScaler().fit(X.fillna(med))
    lr = LogisticRegression(C=LR_C, max_iter=3000).fit(scaler.transform(X.fillna(med)), y)
    hgb = None
    if MODEL_TYPE == "blend":
        hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             monotonic_cst=MONO, random_state=0).fit(X, y)
    return {"med": med, "scaler": scaler, "lr": lr, "hgb": hgb}


def predict_onset(model, X):
    p_lr = model["lr"].predict_proba(model["scaler"].transform(X[FEATURES].fillna(model["med"])))[:, 1]
    p_hgb = model["hgb"].predict_proba(X[FEATURES])[:, 1] if model["hgb"] is not None else p_lr
    return p_lr, p_hgb


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def fit_calibrator(oof):
    """Blend weight by Brier on out-of-fold rows (LR only: weight 1), then Platt:
    logit(p) = a + b*logit(blend) + c*h, lightly regularised because shocks are few."""
    w_default = 1.0 if MODEL_TYPE == "lr" else 0.5
    if oof is None or oof["label"].sum() < 10:
        return w_default, None
    y = oof["label"].astype(int).values
    w = w_default
    if MODEL_TYPE == "blend":
        ws = np.linspace(0, 1, 11)
        bs = [np.mean((w_ * oof["p_lr"] + (1 - w_) * oof["p_hgb"] - y) ** 2) for w_ in ws]
        w = float(ws[int(np.argmin(bs))])
    pb = w * oof["p_lr"] + (1 - w) * oof["p_hgb"]
    cal = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[logit(pb), oof["h"] / 13], y)
    return w, cal


def apply_calibrator(w, cal, p_lr, p_hgb, h):
    pb = w * p_lr + (1 - w) * p_hgb
    return pb if cal is None else cal.predict_proba(np.c_[logit(pb), np.asarray(h) / 13])[:, 1]


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


def survival_curve(eps, stretch):
    """S[d] = P(shock lasts more than d weeks). KM blended with a geometric curve whose mean
    length is scaled by `stretch` (<1 = shocks end sooner, >1 = longer tail)."""
    if not eps:
        return 0.85 ** np.arange(DMAX + 1)
    L = np.array([e[0] for e in eps])
    done = np.array([e[1] for e in eps])
    km = np.ones(DMAX + 1)
    for k in range(1, DMAX + 1):
        at_risk = (L >= k).sum()
        km[k] = km[k - 1] * (1 - ((L == k) & done).sum() / at_risk) if at_risk else km[k - 1]
    mean_len = L[done].mean() if done.any() else 6.0
    q = 1 - 1 / max(mean_len * stretch, 1.5)
    geo = q ** np.arange(DMAX + 1)
    w = len(L) / (len(L) + KM_PRIOR)
    S = w * km + (1 - w) * geo
    top = int(L.max())
    S[top + 1:] = S[top] * q ** np.arange(1, DMAX - top + 1)
    return S


def p_continue(S, age, h):
    a, b = min(age - 1, DMAX), min(age + h - 1, DMAX)
    return float(S[b] / max(S[a], 1e-9))


eps_all = episodes_upto(LAST)
lens = sorted(e[0] for e in eps_all)
print(f"\nshock episodes so far: {len(eps_all)} ({sum(e[1] for e in eps_all)} finished) | "
      f"median length {np.median(lens):.0f} wks, mean {np.mean(lens):.1f} | lengths: {lens}")


# ------------------------------------------------------------------ one origin, end to end
def run_origin(O, oof_before):
    """Raw ingredients for origin O. The final probability is assembled by finalize() once the
    backtest has chosen the duration tail, the shrinkage and the in-shock rule."""
    train = ALL[(ALL["week"] <= O) & (ALL["in_shock"] == 0) & ALL["label"].notna()]
    pred = ALL[ALL["T"] == O].copy()
    pred["p_base"] = pred["material"].map(D.loc[:O].mean()).values
    model = fit_onset(train)
    if model is None:
        pred["p_lr"] = pred["p_hgb"] = pred["p_base"]
    else:
        pred["p_lr"], pred["p_hgb"] = predict_onset(model, pred)
    if oof_before is not None:
        oof_before = oof_before[oof_before["week"] <= O]      # only labels known at O
    w, cal = fit_calibrator(oof_before)
    pred["p_onset"] = apply_calibrator(w, cal, pred["p_lr"], pred["p_hgb"], pred["h"])
    eps = episodes_upto(O)
    for k, stretch in enumerate(TAIL_GRID):
        S = survival_curve(eps, stretch)
        pred[f"p_cont_{k}"] = [p_continue(S, a, h) if s else 0.0
                               for s, a, h in zip(pred["in_shock"], pred["age"], pred["h"])]
    pred["origin_week"] = O
    info = {"origin": O, "blend_w": w, "calibrated": cal is not None, "train_rows": len(train),
            "train_pos": int(train["label"].sum())}
    return pred, model, info


def assemble(df, k, alpha, shock_new):
    """p = shrunk onset for calm materials; duration curve + (shrunk onset or base) in a shock."""
    pc = df[f"p_cont_{k}"]
    p_new = alpha * df["p_onset"] + (1 - alpha) * df["p_base"]
    add = p_new if shock_new == "model" else df["p_base"]
    return np.clip(np.where(df["in_shock"] == 1, pc + (1 - pc) * add, p_new), *CLIP)


# ------------------------------------------------------------------ rolling backtest
origins = list(range(BACKTEST_START, LAST, BACKTEST_STEP))
preds, oof, infos, models = [], [], [], {}
for O in origins:
    pr, mdl, info = run_origin(O, pd.concat(oof) if oof else None)
    preds.append(pr)
    infos.append(info)
    models[O] = mdl
    oof.append(pr[(pr["in_shock"] == 0) & pr["label"].notna()][["p_lr", "p_hgb", "h", "label", "week"]])
info_df = pd.DataFrame(infos)
print(f"\nbacktest: {len(origins)} origins ({origins[0]}-{origins[-1]}, every {BACKTEST_STEP} weeks) | "
      f"calibrated at {int(info_df['calibrated'].sum())} of them")

bt = pd.concat(preds, ignore_index=True)
scored = bt[bt["label"].notna() & (bt["origin_week"] >= REPORT_FROM)].copy()


def brier_skill(df, col):
    if len(df) == 0:
        return np.nan
    bs = np.mean((df[col] - df["label"]) ** 2)
    bb = np.mean((df["p_base"] - df["label"]) ** 2)
    return 1 - bs / bb if bb > 0 else np.nan


calm_sc, shock_sc = scored[scored["in_shock"] == 0], scored[scored["in_shock"] == 1]
k_default = TAIL_GRID.index(1.0) if 1.0 in TAIL_GRID else 0

# 1) duration tail: best in-shock skill with the base rate as the "new shock" term
tail_res = {k: brier_skill(shock_sc.assign(_p=assemble(shock_sc, k, 0.0, "base")), "_p") for k in range(len(TAIL_GRID))}
k_best = max(tail_res, key=lambda k: -np.inf if pd.isna(tail_res[k]) else tail_res[k]) if len(shock_sc) else k_default
# 2) shrinkage toward the base rate for calm materials
alpha_res = {a: brier_skill(calm_sc.assign(_p=a * calm_sc["p_onset"] + (1 - a) * calm_sc["p_base"]), "_p")
             for a in SHRINK_GRID}
alpha_best = max(alpha_res, key=lambda a: -np.inf if pd.isna(alpha_res[a]) else alpha_res[a])
# 3) in-shock rows: add onset risk on top of the duration curve, or just the base rate
if IN_SHOCK_NEW == "auto" and len(shock_sc):
    s_model = brier_skill(shock_sc.assign(_p=assemble(shock_sc, k_best, alpha_best, "model")), "_p")
    s_base = brier_skill(shock_sc.assign(_p=assemble(shock_sc, k_best, alpha_best, "base")), "_p")
    shock_new = "model" if s_model > s_base else "base"
else:
    shock_new = IN_SHOCK_NEW if IN_SHOCK_NEW in ("model", "base") else "base"

print("\nSETTINGS CHOSEN BY THE BACKTEST")
print("  duration tail stretch -> in-shock skill: " +
      ", ".join(f"{TAIL_GRID[k]}: {v:.3f}" for k, v in tail_res.items() if pd.notna(v)) +
      f"   => {TAIL_GRID[k_best]}")
print("  shrink weight on the onset model -> calm skill: " +
      ", ".join(f"{a}: {v:.3f}" for a, v in alpha_res.items() if pd.notna(v)) + f"   => {alpha_best}")
print(f"  in-shock rows add: {'onset model' if shock_new == 'model' else 'base rate'}")
print("  (chosen on the same backtest it is scored on, so the final row is slightly optimistic)")


def finalize(df):
    df = df.copy()
    df["p"] = assemble(df, k_best, alpha_best, shock_new)
    df["p_cont"] = df[f"p_cont_{k_best}"]
    return df


bt = finalize(bt)
scored = finalize(scored)
scored["p_dur_default"] = assemble(scored, k_default, 0.0, "base")
scored["p_dur_tuned"] = assemble(scored, k_best, 0.0, "base")
scored["p_onset_raw"] = assemble(scored, k_best, 1.0, "model")
VERSIONS = {"v0_base_rate": "p_base", "v1_duration (stretch 1.0)": "p_dur_default",
            "v1_duration (tuned)": "p_dur_tuned", "onset model, no shrink": "p_onset_raw", "FINAL": "p"}
rows = []
for name, col in VERSIONS.items():
    rows.append({"version": name, "skill_all": brier_skill(scored, col),
                 "skill_calm": brier_skill(scored[scored["in_shock"] == 0], col),
                 "skill_in_shock": brier_skill(scored[scored["in_shock"] == 1], col)})
metrics = pd.DataFrame(rows)
print(f"\nBRIER SKILL vs base rate (origins >= {REPORT_FROM}, {len(scored):,} scored rows, "
      f"{int(scored['in_shock'].sum())} in-shock):")
print(metrics.round(3).to_string(index=False))

per_origin = scored.groupby("origin_week").apply(lambda d: pd.Series(
    {"skill_final": brier_skill(d, "p"), "rows": len(d), "disrupted_rate": d["label"].mean()}))
print("\nper origin:")
print(per_origin.round(3).to_string())

scored = scored.assign(bucket=pd.cut(scored["p"], [0, .05, .1, .2, .3, .5, .7, 1.0]))
rel = scored.groupby("bucket", observed=True).agg(mean_pred=("p", "mean"), actual=("label", "mean"),
                                                  rows=("label", "size"))
print("\nreliability (mean prediction vs actual rate per bucket; close = calibrated):")
print(rel.round(3).to_string())

# ------------------------------------------------------------------ final forecasts
final = {O: bt[bt["origin_week"] == O] for O in {PUBLIC_ORIGIN} if O in origins}
oof_all = pd.concat(oof) if oof else None
for O in sorted({PUBLIC_ORIGIN, FINAL_ORIGIN}):
    if O not in final and O <= LAST:
        pr, mdl, info = run_origin(O, oof_all)
        final[O] = finalize(pr)
        models[O] = mdl
        print(f"final origin {O}: calibrated {info['calibrated']}")

if models.get(FINAL_ORIGIN) is not None:
    coef = pd.Series(models[FINAL_ORIGIN]["lr"].coef_[0], index=FEATURES)
    print("\nlogistic-regression coefficients at the final origin (standardised features):")
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
metrics_out = metrics.assign(report_from=REPORT_FROM, last_week=LAST, use_news=USE_NEWS, use_source_reliability=USE_SOURCE_RELIABILITY)


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
            print(f"could not write table {name}: {str(e).splitlines()[0]}")


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
            mlflow.log_params({"last_week": LAST, "use_news": USE_NEWS, "use_source_reliability": USE_SOURCE_RELIABILITY, "lr_c": LR_C, "km_prior": KM_PRIOR,
                               "tail_stretch": TAIL_GRID[k_best], "shrink_alpha": alpha_best, "in_shock_add": shock_new, "feature_set": FEATURE_SET, "model_type": MODEL_TYPE, "clip": str(CLIP), "origins": str(origins),
                               "n_features": len(FEATURES)})
            for r in metrics.itertuples():
                for k in ["skill_all", "skill_calm", "skill_in_shock"]:
                    v = getattr(r, k)
                    if pd.notna(v):
                        mlflow.log_metric(re.sub(r"[^A-Za-z0-9_]+", "_", f"{r.version}_{k}"), float(v))
            mlflow.log_artifact(f"{OUTPUT_DIR}/m1_metrics.csv")
            final_model = models.get(FINAL_ORIGIN)
            if final_model is not None:
                import mlflow.sklearn
                mlflow.sklearn.log_model(final_model["lr"], name="onset_logistic")
                if final_model["hgb"] is not None:
                    mlflow.sklearn.log_model(final_model["hgb"], name="onset_boosting")
        print("logged MLflow run")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")