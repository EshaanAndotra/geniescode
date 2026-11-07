"""
validate_submission.py - check submission files against every rule before they're handed over or uploaded.

Fails (exit code 1) on anything Kaggle rejects (docs/competition.md) or the team's rules forbid (docs/BRIEF.md, section 7):
  - columns: exactly the sample submission's columns, in its order
  - ids: exactly the sample's 260 ids (w{week}_{material}, weeks 209-234), no missing, extra or duplicate ids
  - no blank or non-numeric values
  - disruption_prob in [0, 1]; price_forecast > 0
  - order_units >= 0 and whole numbers
  - each share in [0, 1], shares summing to 1, and 0 for a country that doesn't sell the material
The share-sum tolerance matches validate() in Model 3/iter1/model3.py, so a file run_pipeline.py accepts
also passes here. Unlike that function, this reports every problem, not just the first.

Also prints, without failing: units ordered / demand per window, mean diversification, and rows that
are identical to the naive sample submission.

  python3 validate_submission.py ../../submissions/FILE.csv [MORE.csv ...] [--data ../../Data_new]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline import WINDOWS, find_data, find_sample  # noqa: E402

PROB, PRICE, ORDER = "disruption_prob", "price_forecast", "order_units"
SHARE_TOL = 1e-9     # same as model3.validate
WHOLE_TOL = 1e-9


def _ids(rows):
    return ", ".join(rows[:5]) + (f" (+{len(rows) - 5} more)" if len(rows) > 5 else "")


def check(sub, sample, suppliers):
    """Every rule broken by `sub`, as a list of messages; empty = valid."""
    problems = []
    if list(sub.columns) != list(sample.columns):
        missing = [c for c in sample.columns if c not in sub.columns]
        extra = [c for c in sub.columns if c not in sample.columns]
        problems.append(f"columns differ from the sample: missing {missing}, extra {extra}"
                        if missing or extra else "columns are in a different order from the sample")
        if missing:
            return problems
    share_cols = [c for c in sample.columns if c.startswith("share_")]

    if len(sub) != len(sample):
        problems.append(f"{len(sub)} rows, need {len(sample)}")
    dup = sub.id[sub.id.duplicated()].tolist()
    if dup:
        problems.append(f"duplicate ids: {_ids(dup)}")
    want, got = set(sample.id), set(sub.id)
    if want - got:
        problems.append(f"missing ids: {_ids(sorted(want - got))}")
    if got - want:
        problems.append(f"unexpected ids: {_ids(sorted(got - want))}")

    num = sub[[c for c in sample.columns if c != "id"]].apply(pd.to_numeric, errors="coerce")
    blank = sub.drop(columns="id").isna() | sub.id.isna().values[:, None]
    nonnum = num.isna() & ~blank
    if blank.any().any():
        problems.append(f"blank values in {sorted(blank.columns[blank.any()].tolist())}")
    if nonnum.any().any():
        problems.append(f"non-numeric values in {sorted(nonnum.columns[nonnum.any()].tolist())}")
    if not np.isfinite(num.fillna(0).values).all():
        problems.append("infinite values")

    def rule(mask, msg):
        bad = sub.id[mask.fillna(False)].tolist()
        if bad:
            problems.append(f"{msg}: {_ids(bad)}")

    rule(~num[PROB].between(0, 1) & num[PROB].notna(), f"{PROB} outside [0, 1]")
    rule(~(num[PRICE] > 0) & num[PRICE].notna(), f"{PRICE} must be > 0")
    rule(num[ORDER] < 0, f"negative {ORDER}")
    rule((num[ORDER] - num[ORDER].round()).abs() > WHOLE_TOL, f"{ORDER} not whole")
    shares = num[share_cols]
    rule(((shares < 0) | (shares > 1)).any(axis=1), "a share outside [0, 1]")
    rule((shares.sum(axis=1) - 1).abs() > SHARE_TOL, "shares don't sum to 1")
    sells = set(zip(suppliers.material, suppliers.country))
    mat = sub.id.str.split("_", n=1).str[1]
    for c in share_cols:
        no_sell = pd.Series([(m, c[len("share_"):]) not in sells for m in mat], index=sub.index)
        rule((shares[c] > 0) & no_sell, f"{c} > 0 for a country that doesn't sell the material")
    return problems


def summary(sub, sample, demand):
    """Informational lines: order volume vs demand per window, diversification, rows equal to naive."""
    wk = sub.id.str.extract(r"^w(\d+)_")[0].astype(int)
    mat = sub.id.str.split("_", n=1).str[1]
    d = demand.set_index(["week", "material"]).demand_units
    dem = pd.Series([d.get((w, m), np.nan) for w, m in zip(wk, mat)], index=sub.index)
    ratio = {f"{a}-{b}": round(sub[ORDER][wk.between(a, b)].sum() / dem[wk.between(a, b)].sum(), 3)
             for a, b in WINDOWS}
    share_cols = [c for c in sample.columns if c.startswith("share_")]
    # 1 - HHI rescaled so single sourcing = 0 and an even 3-way split = 1 (docs/competition.md)
    div = ((1 - (sub[share_cols] ** 2).sum(axis=1)) / (1 - 1 / 3)).clip(0, 1).mean()
    cols = [ORDER] + share_cols
    naive = sample.set_index("id").loc[sub.id, cols].values
    same = int(np.isclose(sub[cols].values, naive).all(axis=1).sum())
    return f"units ordered / demand {ratio}; diversification {div:.3f}; rows identical to naive {same}/{len(sub)}"


def main():
    ap = argparse.ArgumentParser(description="Validate submission files against the competition and team rules")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data", default=None, help="data folder with suppliers.csv and sample_submission.csv")
    args = ap.parse_args()
    data_dir = find_data(args.data)
    sample = pd.read_csv(find_sample(data_dir))
    suppliers = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    demand = pd.read_csv(os.path.join(data_dir, "demand.csv"))

    failed = 0
    for path in args.files:
        try:
            sub = pd.read_csv(path)
        except Exception as e:  # unreadable file counts as a failure, keep checking the rest
            print(f"FAIL  {path}\n      can't read: {e}")
            failed += 1
            continue
        problems = check(sub, sample, suppliers)
        if problems:
            failed += 1
            print(f"FAIL  {path}")
            for p in problems:
                print(f"      - {p}")
        else:
            print(f"PASS  {path}\n      {summary(sub, sample, demand)}")
    print(f"\n{len(args.files) - failed} of {len(args.files)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
