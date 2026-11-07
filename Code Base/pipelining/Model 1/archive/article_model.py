# News event study for Model 1: which headline templates actually warn of disruptions?
# Paste as a Databricks cell (or run as a script). Uses only news, outcomes and suppliers.
import re
import numpy as np
import pandas as pd

DATA_DIR = "/Workspace/Repos/DSDG Hackathon/DSDG Hackathon/Data"
LOOKAHEAD = 6  # an article "comes true" if a tagged material has a disruption onset within 1..6 weeks

news = pd.read_csv(f"{DATA_DIR}/news.csv")
out = pd.read_csv(f"{DATA_DIR}/outcomes.csv")
sup = pd.read_csv(f"{DATA_DIR}/suppliers.csv")

COUNTRIES = {"Sunda Coast": "SUN", "Kestria": "KES", "Valdmar": "VAL",
             "Orinoque": "ORI", "Halvard": "HAL", "Aurelia": "AUR"}
MATERIALS = {
    "lithium": r"lithium", "cobalt": r"cobalt", "copper": r"copper", "aluminum": r"alumin",
    "steel": r"steel", "rare_earths": r"rare[ -]?earth", "semiconductors": r"semiconductor|chip",
    "polymer_resin": r"polymer|resin", "specialty_glass": r"glass",
    "natural_gas": r"natural gas|\blng\b|\bgas\b",
}
LANES = {"MERROW": r"merrow", "NORTH": r"northern sea|north sea", "SOUTH": r"southern route",
         "ATLANTIC": r"atlantic", "RAIL": r"\brail"}


# 1. Headline templates: strip names and numbers so templated headlines collapse together
def template(h: str) -> str:
    t = h
    for name in COUNTRIES:  # also catches demonyms like "Kestrian"
        t = re.sub(name + r"\w*", "<C>", t)
    for pat in MATERIALS.values():
        t = re.sub(r"(?i)\b(?:" + pat + r")\w*", "<M>", t)
    t = re.sub(r"(?i)merrow strait|northern sea(?: route)?|southern route|atlantic route", "<LANE>", t)
    t = re.sub(r"\d+(?:\.\d+)?%?", "<N>", t)
    return t


news["template"] = news["headline"].map(template)
print(f"{len(news):,} articles collapse into {news['template'].nunique()} headline templates\n")
print(news["template"].value_counts().head(60).to_string())

# 2. Tag each article with the materials it can affect (direct mention, else via country or lane)
country_mats = sup.groupby("country")["material"].apply(set).to_dict()
lane_countries = sup.groupby("lane")["country"].apply(set).to_dict()


def tag_materials(text: str) -> list:
    direct = {m for m, p in MATERIALS.items() if re.search(p, text, re.I)}
    if direct:
        return sorted(direct)
    ctry = {c for n, c in COUNTRIES.items() if n in text}
    for lane, p in LANES.items():
        if re.search(p, text, re.I):
            ctry |= lane_countries.get(lane, set())
    return sorted(set().union(*[country_mats.get(c, set()) for c in ctry])) if ctry else []


news["materials"] = (news["headline"] + " " + news["body"]).map(tag_materials)
print(f"\narticles tagged to at least one material: {(news['materials'].str.len() > 0).mean():.1%}")

# 3. Disruption onsets per material (0 -> 1 transitions in outcomes)
out = out.sort_values(["material", "week"])
out["prev"] = out.groupby("material")["disrupted"].shift(1).fillna(0)
onset_weeks = (out[(out["disrupted"] == 1) & (out["prev"] == 0)]
               .groupby("material")["week"].apply(lambda s: np.sort(s.values)).to_dict())
disrupted = out.set_index(["material", "week"])["disrupted"].to_dict()
max_week = int(out["week"].max())
print(f"disruption onsets in labels: {sum(len(v) for v in onset_weeks.values())} "
      f"across {len(onset_weeks)} materials (weeks 1-{max_week})")


def lead_to_onset(material: str, week: int) -> float:
    ws = onset_weeks.get(material, np.array([]))
    nxt = ws[ws > week]
    return float(nxt[0] - week) if len(nxt) and nxt[0] - week <= LOOKAHEAD else np.nan


# Base rate: chance a calm material-week sees an onset within LOOKAHEAD weeks
calm = out[(out["disrupted"] == 0) & (out["week"] <= max_week - LOOKAHEAD)].copy()
calm["hit"] = [not np.isnan(lead_to_onset(m, w)) for m, w in zip(calm["material"], calm["week"])]
base_rate = calm.groupby("material")["hit"].mean().to_dict()

# 4. One row per (article, tagged material), calm at publication and fully labelled
rows = news.explode("materials").dropna(subset=["materials"]).rename(columns={"materials": "material"})
rows = rows[rows["week"] <= max_week - LOOKAHEAD].copy()
rows["active_now"] = [disrupted.get((m, w), 0) for m, w in zip(rows["material"], rows["week"])]
rows["lead"] = [lead_to_onset(m, w) for m, w in zip(rows["material"], rows["week"])]
rows["hit"] = rows["lead"].notna()
rows["expected"] = rows["material"].map(base_rate)
calm_rows = rows[rows["active_now"] == 0]

# 5. Template table: hit rate vs what chance would give (lift) and typical warning lead time
tmpl = calm_rows.groupby("template").agg(
    articles=("article_id", "nunique"),
    sources=("source", "nunique"),
    hit_rate=("hit", "mean"),
    expected=("expected", "mean"),
    median_lead=("lead", "median"),
)
tmpl["lift"] = tmpl["hit_rate"] / tmpl["expected"].clip(lower=1e-6)
tmpl = tmpl[tmpl["articles"] >= 3].sort_values(["lift", "articles"], ascending=False)
print("\nTEMPLATES MOST PREDICTIVE OF A DISRUPTION ONSET (calm materials only):")
display(tmpl.head(30).round(3))

# 6. Source reliability on WARNINGS (not macro claims): only templates with lift > 2
warning_templates = set(tmpl[tmpl["lift"] > 2].index)
w = calm_rows[calm_rows["template"].isin(warning_templates)]
art = w.groupby(["source", "article_id"])["hit"].max().reset_index()  # one verdict per article
src = art.groupby("source")["hit"].agg(warnings="count", came_true="sum")
src["reliability"] = (src["came_true"] + 1) / (src["warnings"] + 2)
src["median_lead"] = w[w["hit"]].groupby("source")["lead"].median()
print(f"\nSOURCE RELIABILITY ON WARNING ARTICLES ({len(warning_templates)} warning templates):")
display(src.sort_values("reliability", ascending=False).round(3))