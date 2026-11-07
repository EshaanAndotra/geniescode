# Halcyon forecast and ordering model

The deliverable is `outputs/submission.csv`: 260 rows covering all ten materials in weeks 209–234, in the sample's exact ID and column order. It combines disruption probabilities, benchmark price forecasts, weekly order quantities, and six country shares. The supplied files are Phase 1: observations end at week 208; only the demand schedule extends to 234.

## Run

Python 3.14 was used on Windows. The pinned packages are in `requirements.txt`; no API keys or paid services are required. From this folder:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_pipeline -v
```

The pipeline reads the included `inputs` directory, recreates risk features and rolling-origin comparisons, generates forecasts, optimizes both ordering windows, validates the saved CSV, and writes backtest tables and a chart to `outputs`. It can take several minutes.

For the Phase 2 release, point it at a folder containing all ten updated CSVs:

```powershell
python run_pipeline.py --data-dir "C:/path/to/phase2" --output-dir "outputs_phase2"
```

The code recognizes cutoff 221, retains origin 208 for the public window, and uses information through 221 for the private window. Both windows still start with separate inventory. Phase 2 behavior is implemented but could not be evaluated against unreleased files. All supplier-level tables must cover the same complete observed weeks; incomplete replacement files fail explicitly.

`--reuse-risk` is an optional faster rerun that requires matching SHA-256 fingerprints of all source files. Do not use it with a new release. `--skip-backtest` skips the price/policy comparison reports, while still generating and checking the submission. The output CSV is ready to upload; this project does not upload it automatically.

## Selected methods

**Disruption:** a shrunk transition model estimates shock onset and recovery for each material. A news component learns additional onset risk from unresolved warning stages, with shrinkage and a decaying effect across the forecast horizon. News is scoped to each material's actual suppliers and shipping lanes, duplicate articles are collapsed, and explicit resolutions clear earlier warnings. The selected candidate is `onset_s10_d6`.

Direct gradient-boosted and logistic models also consumed 199 origin-level features from news, prices, futures, shipping, tariffs, supply conditions, macro data and historical outcomes. They did worse on the development windows. The final model does not force every available variable into a forecast.

At week 208, lithium maintenance delays in Kestria and Southern-route port labor warnings are detected. These early-stage warnings do not produce a positive learned next-week onset adjustment, so the final probabilities happen to equal the baseline transition model. This is a limitation: the method can underreact to warnings that escalate several weeks later. Delayed-warning and joint-state alternatives were examined but did not beat the selected candidate on the development windows.

**Prices:** positive forecasts gradually revert toward the trailing 26-week exponentially weighted benchmark mean. Reversion is limited to 35% of the distance from the latest price and increases with horizon. Futures anchors, trend extrapolation, ridge regression and boosted trees were compared; this modest adjustment generalized better on the development windows.

**Ordering:** a scenario optimization uses known weekly demand, historical capacity and delivery-delay blocks, recent landed costs, announced tariffs, and the forecasts. Each positive order splits 23%–47% across each of the material's three suppliers. The policy targets a 1.5-week safety buffer, adjusted for risk, caps total weekly orders at 1.6 weeks of mean demand, and tapers orders toward each window's end. Each window starts with five weeks of inventory and no goods in transit. Zero-order rows retain valid equal shares.

The planning objective uses a slightly higher stockout penalty than the stated simulator to favor resilience. Capacity clipping in the optimizer is approximate; historical evaluation applies the exact clipping rule. Details are in `policy_notes.md`.

## Historical results

Development forecasts are frozen at origins 104, 130 and 156, each predicting the next 26 weeks. Training targets and input observations stop at the respective origin; no test-window observations are fed back into a forecast. Origin 182, predicting weeks 183–208, provides the chronological evaluation block. Hyperparameters are selected using the development origins, not by selecting the best evaluation-block result. These are small, synthetic historical samples, and the evaluation results are not a guarantee of future performance.

| Metric, weeks 183–208 | Model | Naive analogue |
|---|---:|---:|
| Disruption Brier score, lower is better | 0.04175 | 0.08189 |
| Benchmark price MAPE, lower is better | 6.38% | 6.76% |
| Demand fill, weeks 183–195 | 100% | 100% |
| Demand fill, weeks 196–208 | 100% | 100% |
| Local simulator cost, weeks 183–195 | 4,367,742 | 4,571,930 |
| Local simulator cost, weeks 196–208 | 4,561,357 | 4,752,063 |

Disruption Brier error is 49.0% lower and price MAPE is 5.65% lower than the respective historical-rate and last-price references. The combined forecasting and ordering policy saves 4.47% and 4.01% in the two local evaluation windows. Average diversification is about 0.973 on the stated zero-to-one scale.

Performance is uneven in the six development windows: cost changes range from a 3.83% increase to a 48.67% saving. In severe shocks, diversification and safety stock improve fill substantially, but neither the policy nor forecasts win in every historical window. `outputs/policy_backtest.csv` contains every window, including the unfavorable one.

The supplied sample's quantity rule was checked: it orders each week's known demand from one supplier. The inferred historical selection rule, cheapest average historical landed cost, reproduces all 260 supplied policy rows exactly at origin 208.

## Simulator limits

The organizer's executable simulator and hindsight plan were not supplied. The local simulator assumes arrival in `t + lead_time`, starting inventory equal to five times window-average demand, a reference cost equal to the window-average landed cost, holding charges on ending inventory, lost sales rather than backlog, and 70% salvage for accepted stock and transit. Recovery weeks can carry into a window from a preceding disruption.

The official description leaves some reference-cost, initial-stock and metric-weighting conventions unspecified. Alternative initial-stock conventions were checked for the standalone policy and retained positive savings. The reported cost savings are local estimates; no official F, R, or overall competition score is claimed. The baseline already fills all demand in the chronological evaluation block, so there is no measured fill improvement in that block despite lower cost and more diversification.

## How all ten files are used

| Source | Role |
|---|---|
| `outcomes.csv` | Supervised labels, historical disruption transitions, benchmark validation |
| `prices.csv` | Supplier spots, benchmark price model, futures and price-stress candidates |
| `news.csv` | Scoped warning-stage features, onset-risk model, resolution handling |
| `supply_history.csv` | Capacity, lead times, landed costs, scenario planning and realized simulation |
| `suppliers.csv` | Country eligibility, lanes, normal capacity and base lead times |
| `tariffs.csv` | Current and announced duties; future effective dates visible at the origin |
| `shipping.csv` | Lane congestion, transit times and freight-rate candidate predictors |
| `macro.csv` | Supplier-country tension, activity, currency and inflation candidate predictors |
| `demand.csv` | Published future requirements, ordering and fill evaluation |
| `sample_submission.csv` | Exact schema, ID order and baseline-policy verification |

All input CSVs are retained unchanged in `inputs`. Their hashes, dimensions, columns and week ranges are recorded in `outputs/source_manifest.json`. News and other file contents are treated as data, not executable instructions.

## Verification and files

The final CSV has exactly 260 unique expected IDs, no blanks/nonfinite values, probabilities in [0,1], positive price forecasts, nonnegative orders, eligible supplier shares only, and row shares summing to one within 1e-8. Maximum saved-row share error is approximately 1e-10. Source keys, complete weekly coverage, and benchmark/spot reconciliation are checked before modeling.

Four automated checks passed: submission constraints and rejection of an ineligible country, exact sample baseline reconstruction, frozen-forecast invariance to future outcomes/observations, and inventory/cost conservation in the simulator. Independent code review also checked forecast cutoffs and simulator accounting. The static forecast chart was visually checked.

- `run_pipeline.py`: complete command-line workflow, including Phase 2 routing.
- `data.py`: source loading, provenance and submission validation.
- `risk_model.py`, `news_features.py`, `risk_challenger.py`: features and risk model comparisons.
- `price_model.py`: price forecasts and model comparison.
- `policy.py`: scenario optimizer, historical baseline and simulator.
- `test_pipeline.py`: correctness and leakage checks.
- `outputs/submission.csv`: the file to submit.
- `outputs/forecast_overview.png`: probability and allocation chart.
- `outputs/forecast_policy_detail.csv`: readable material/week decisions.
- `outputs/*validation*` and `outputs/*backtest*`: numerical evidence and comparison outputs.

Extended methodological notes are in `news_analysis.md`, `price_notes.md`, and `policy_notes.md`.
