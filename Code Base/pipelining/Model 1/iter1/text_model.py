# =====================================================================================
# Model 1, part A: text model (news warning features)
#
# Turns ~21 articles a week into warning features per material and week, using only
# information available at the time (an article's outcome counts only once it is at least
# NEWS_LOOKAHEAD weeks old):
#   - headline templates and how often each was followed by a disruption onset (lift)
#   - source reliability on WARNINGS: (came true + 1) / (warnings + 2), relative to average
#   - certainty wording: confirming minus hedging words
#
# Output: m1_news_features (material, week, news_* columns), as a Delta table on Databricks
# and a CSV in OUTPUT_DIR. numerical_model.py reads it; run this file first.
# These are features, not a probability: calibration happens once, in numerical_model.py.
# =====================================================================================
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
DATA_DIR = "/Workspace/Repos/DSDG Hackathon/DSDG Hackathon/Data"
OUTPUT_DIR = "/tmp/m1_outputs"        # keep the same in both files
TABLE_PREFIX = "workspace.default."   # keep the same in both files
WRITE_TABLES = True                   # only takes effect where `spark` exists (Databricks)
USE_SOURCE_RELIABILITY = True         # weight warnings by each source's track record + certainty words
NEWS_LOOKAHEAD = 6                    # an article "came true" if an onset follows in 1-6 weeks
NEWS_SMOOTH = 5                       # pseudo-articles pulling template rates to the base rate

# ------------------------------------------------------------------ load
news = pd.read_csv(f"{DATA_DIR}/news.csv")
sup = pd.read_csv(f"{DATA_DIR}/suppliers.csv")
outc = pd.read_csv(f"{DATA_DIR}/outcomes.csv")

MATERIALS = sorted(sup["material"].unique())
LAST = int(outc["week"].max())
D = outc.pivot(index="week", columns="material", values="disrupted").sort_index().fillna(0).astype(int)
onset_weeks = {}
for m in MATERIALS:
    s = D[m].values
    onset_weeks[m] = D.index.values[(s == 1) & (np.r_[0, s[:-1]] == 0)]
print(f"news: {len(news):,} articles | labels through week {LAST} | "
      f"{sum(len(v) for v in onset_weeks.values())} disruption onsets")

# ------------------------------------------------------------------ templates, tagging, features
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
if USE_SOURCE_RELIABILITY:
    NEWS_FEATS += ["news_warn_rel_4w", "news_certainty_4w"]
CONFIRM_WORDS = r"\b(?:announced|announces|confirmed|confirms|effective|officially|signed|imposes|imposed|ordered|orders|enacted)\b"
HEDGE_WORDS = r"\b(?:reportedly|rumou?red|rumou?rs?|unconfirmed|denies|denied|speculation|could|may|might|considering|sources say)\b"


def build_news_features():
    """Point-in-time warning score per article: how often its headline template was followed
    by a disruption onset, using only articles whose outcome was known by publication week
    (published at least NEWS_LOOKAHEAD weeks earlier). Optionally weighted by the source's
    track record on warnings (same point-in-time rule) and scored for certainty wording.
    Then summed per material and week."""
    n = news.copy()
    text = (n["headline"].fillna("") + " " + n["body"].fillna("")).str.lower()
    n["certainty"] = text.str.count(CONFIRM_WORDS) - text.str.count(HEDGE_WORDS)
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

    # Source reliability on WARNINGS: (came true + 1) / (warnings + 2), counting only
    # warning-like articles (template lift > 1.5) published >= NEWS_LOOKAHEAD weeks earlier,
    # expressed relative to the average source (1.0 = average, 2.0 = twice as reliable).
    r["src_factor"] = 1.0
    if USE_SOURCE_RELIABILITY:
        art = (r[r["calm"]].groupby(["article_id", "source", "week"])
               .agg(hit=("hit", "max"), lift=("lift", "max")).reset_index())
        art = art[art["lift"].fillna(1.0) > 1.5]
        if len(art):
            sw = art.groupby(["source", "week"])["hit"].agg(["size", "sum"]).reset_index()
            cn = sw.pivot(index="week", columns="source", values="size").reindex(weeks).fillna(0).cumsum()
            ch = sw.pivot(index="week", columns="source", values="sum").reindex(weeks).fillna(0).cumsum()
            cn, ch = cn.shift(NEWS_LOOKAHEAD).fillna(0), ch.shift(NEWS_LOOKAHEAD).fillna(0)
            rel = (ch + 1) / (cn + 2)
            base_rel = (ch.sum(axis=1) + 1) / (cn.sum(axis=1) + 2)
            factor = rel.div(base_rel, axis=0).clip(0.25, 3.0)
            factor.index.name, factor.columns.name = "week", "source"
            r = r.drop(columns="src_factor").merge(factor.stack().rename("src_factor").reset_index(),
                                                   on=["week", "source"], how="left")
            r["src_factor"] = r["src_factor"].fillna(1.0)
            last = pd.DataFrame({"reliability_vs_avg": factor.iloc[-1], "warnings_scored": cn.iloc[-1]})
            print("\nSource reliability on warnings (as of the last week, 1.0 = average source):")
            print(last[last["warnings_scored"] > 0].sort_values("reliability_vs_avg", ascending=False)
                  .round(2).to_string())
    r["warn_rel"] = r["warn"] * r["src_factor"]
    r["cert_w"] = np.where(r["warn"] > 0, r["certainty"], 0.0)

    mw = r.groupby(["material", "week"]).agg(warn_sum=("warn", "sum"), warn_max=("warn", "max"),
                                             src=("warn_src", "nunique"), warn_rel=("warn_rel", "sum"),
                                             cert=("cert_w", "sum")).reset_index()
    grid = pd.MultiIndex.from_product([MATERIALS, weeks], names=["material", "week"])
    mw = mw.set_index(["material", "week"]).reindex(grid).fillna(0).reset_index()
    g = mw.groupby("material")
    mw["news_warn_1w"] = mw["warn_sum"]
    mw["news_warn_4w"] = g["warn_sum"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    mw["news_warn_8w"] = g["warn_sum"].transform(lambda x: x.rolling(8, min_periods=1).sum())
    mw["news_max_4w"] = g["warn_max"].transform(lambda x: x.rolling(4, min_periods=1).max())
    mw["news_sources_4w"] = g["src"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    mw["news_warn_rel_4w"] = g["warn_rel"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    mw["news_certainty_4w"] = g["cert"].transform(lambda x: x.rolling(4, min_periods=1).sum())
    final_lift = lift.iloc[-1].sort_values(ascending=False)
    counts = cum_n.iloc[-1]
    top = pd.DataFrame({"lift": final_lift, "labelled_rows": counts.reindex(final_lift.index)})
    print("\nNews templates with the highest warning lift (as of the last week):")
    print(top[top["labelled_rows"] >= 3].head(15).round(2).to_string())
    return mw[["material", "week"] + NEWS_FEATS]



# ------------------------------------------------------------------ save
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



news_features = build_news_features()
save(news_features, "m1_news_features")
print(f"\nm1_news_features: {len(news_features):,} rows | columns: {[c for c in news_features.columns if c.startswith('news_')]}")
