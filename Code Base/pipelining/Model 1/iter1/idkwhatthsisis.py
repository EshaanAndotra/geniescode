# =====================================================================================
# Model 1, part B: numerical features
#
# Builds every non-text signal per material and week, using only data up to that week:
#   tariffs (current, 4-week change, announced increases), markets (futures premium, its
#   change, 4-week return), shipping (congestion, transit vs normal, container rate),
#   macro (tension level / change / z-score, PMI and FX moves), supply (capacity vs normal,
#   lead time above base) and shock history (in shock, shock age, weeks since, base rate).
# Supplier-level signals are combined per material (max and mean over its 3 suppliers).
#
# Outputs (Delta tables on Databricks + CSVs in OUTPUT_DIR), read by combined_model.py:
#   m1_numeric_features  material, week, feature columns
#   m1_supplier_panel    material, country, week + the supplier-level signals Model 3 uses
# =====================================================================================
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
DATA_DIR = "/Workspace/Repos/DSDG Hackathon/DSDG Hackathon/Data"
OUTPUT_DIR = "/tmp/m1_outputs"        # keep the same in all three files
TABLE_PREFIX = "workspace.default."   # keep the same in all three files
WRITE_TABLES = True                   # only takes effect where `spark` exists (Databricks)

# ------------------------------------------------------------------ load
def load(name):
    return pd.read_csv(f"{DATA_DIR}/{name}.csv")


prices, tariffs, shipping, macro = (load(n) for n in ["prices", "tariffs", "shipping", "macro"])
sup, outc, hist = load("suppliers"), load("outcomes"), load("supply_history")
MATERIALS = sorted(sup["material"].unique())
D = outc.pivot(index="week", columns="material", values="disrupted").sort_index().fillna(0).astype(int)
print(f"prices through week {int(prices['week'].max())} | labels through week {int(outc['week'].max())}")


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


PANEL_COLS = ["material", "country", "week", "cap_ratio", "lead_excess", "tension_z", "cong_chg4",
              "pending", "pending_eff", "tariff_rate", "spot_price", "futures_4w"]
save(F, "m1_numeric_features")
save(p[PANEL_COLS], "m1_supplier_panel")
print(f"\nm1_numeric_features: {len(F):,} rows x {F.shape[1] - 2} features | "
      f"m1_supplier_panel: {len(p):,} rows")
