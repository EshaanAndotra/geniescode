# Submissions and scores

Every file uploaded to Kaggle (or built to upload), what produced it, and its public score.
The public leaderboard scores weeks 209 to 221 only. All files below were built on Phase 1 data
(weeks 1 to 208), so both windows are forecast from week 208.

Score = 100 x (0.30 P + 0.40 F + 0.30 R). With the sample's orders (files A and B), F and R are 0,
so the score comes only from the forecast columns.

| File | disruption_prob | price_forecast | order_units + shares | Public score |
|---|---|---|---|---|
| `A_model1_model2iter2.csv` | Model 1 iter3 | Model 2 iter2 (structural) | sample (naive) | not recorded |
| `B_model1_model2iter1.csv` | Model 1 iter3 | Model 2 iter1 (blend), before the teammate update of 2026-11-07 09:29 UTC | sample (naive) | not recorded |
| `C_full_moderate.csv` | Model 1 iter3 | Model 2 iter2 | Model 3 iter1, moderate | **62.33** |
| `D_full_aggressive.csv` | Model 1 iter3 | Model 2 iter2 | Model 3 iter1, aggressive (stress futures) | not recorded |
| `E_divtest_zero_rows_single.csv` | same as C | same as C | C, but zero-order rows sent 100% to the largest supplier (scorer test) | pending |
| `submission_moderate_wild.csv` | **Model 1 iter4 (news stories)** | Model 2 iter2 | Model 3 iter1, moderate | **65.29** |
| `submission_aggressive_wild.csv` | Model 1 iter4 (news stories) | Model 2 iter2 | Model 3 iter1, aggressive | pending |
| `landed_costs_wild.csv` | not a submission: Model 2's per-supplier landed cost forecasts | | | |

What the scores show so far:

- **News layer: +2.96.** `submission_moderate_wild.csv` (65.29) differs from `C_full_moderate.csv` (62.33) only through the Model 1 iter4 news story layer. See `WILD_CHANGES.md`, section 1.
- **E vs C** tests how Kaggle counts diversification. E and C have identical orders, so only the diversification score can differ.
  - About 59.4 means every row counts equally, including zero-order rows.
  - 62.33 means rows are weighted by order units.

## How to rebuild each file

From `Code Base/pipelining/`:

```
python run_pipeline.py                         # submission_moderate.csv = submission_moderate_wild.csv (verified identical)
                                               # submission_aggressive.csv = submission_aggressive_wild.csv
python run_pipeline.py --model1-iter iter3     # submission_moderate.csv = C_full_moderate.csv (verified identical)
                                               # submission_aggressive.csv = D_full_aggressive.csv
```

A, B and E came from one-off commands:

- **A:** `model1.fill_submission` on the sample, then `model2.fill_submission` from Model 2 iter2.
- **B:** the same, but with Model 2 iter1 as it stood before the teammate update.
- **E:** C with the shares on every `order_units == 0` row set to 1.0 for the material's largest-capacity supplier (`suppliers.csv`).

## Final selection (by Sunday Nov 8, 12 PM Central)

Rebuild both files with Phase 2 data, so the private window is planned from week 221. Upload them, and select them on Kaggle as the 2 final submissions. Public scores after Phase 2 do not predict the private leaderboard.
