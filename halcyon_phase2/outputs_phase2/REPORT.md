<!-- phase2-upload-note -->
> **Upload update:** Use `submission_phase2_ready.csv` or `submission_phase2_challenger.csv` for a Phase2 upload. They use the already-released public outcomes and preserve the private forecasts. The 55–59 scores discussed below are frozen-origin historical replays, not post-release public-score estimates. See `PHASE2_UPLOAD.md`.

# Phase 2 audit and final-stage model

## Files to use

`submission_final.csv` is the recommended 260-row submission. `submission_challenger.csv` uses the more aggressive policy that won mean development performance. Both are eligible alternatives for the competition's two final submission slots. Weeks 209–221 are a revised replay frozen at origin 208; weeks 222–234 use the full release through 221. Public truth is never copied into forecasts or used to optimize those public orders. The new release is preserved separately in `inputs_phase2`, and the original submission remains in `outputs/submission.csv`.

The empirical risk and ordering changes were developed after seeing the revealed public outcomes. Therefore the revised public result is a diagnostic backcast, not fresh out-of-sample evidence. The private weeks 222–234 are still genuinely unknown.

## Why the original scored about 55

The local reconstruction is **55.43**, compared with the reported Kaggle score 55. Its estimated components are:

| Component | Points earned | Points available |
|---|---:|---:|
| Prediction | 1.26 | 30 |
| Financial | 24.42 | 40 |
| Resilience | 29.75 | 30 |

The simulator reconstruction closely matches the reported total, but these are estimated component values, not a component breakdown supplied by Kaggle. The main lost points were prediction and financial efficiency. Resilience was already nearly saturated.

The public period was **calmer overall but concentrated in a few shocks**:18 of 130 material-weeks were disrupted (13.8%), compared with 19.4% during weeks 1–208. A Southern-route strike disrupted lithium, cobalt, copper and natural gas in weeks 212–214. The separate lithium supply disruption continued through 220. Six materials had no active public-window disruption.

Our original probabilities averaged 9.2%. Lithium was disrupted in 9 of 13 weeks (69.2%), against an average forecast of 7.2%. Its warnings were recognized, but the selected next-week onset model learned no immediate lift from an early warning. It failed to translate that warning into risk several weeks later. Brier skill was only 6.69%.

The original price MAPE was 5.74%, almost identical to carrying the last price forward (5.77%). Its end-window price direction was correct for 70% of materials. From 208 to 221, natural gas rose 19.0%, polymer resin rose 14.3%, lithium rose 10.5%, and semiconductors fell 13.7%. A mild mean-reversion forecast did not capture several of these moves. Large price changes do not necessarily imply a labeled supply disruption.

The original orders met 100% of demand in the public replay, including shock weeks. Cost was 4,805,754, versus 5,687,743 for the sample policy and 4,243,016 for our perfect-information LP. The original policy captured 61% of that local attainable saving. Excess ending inventory, holding costs, and rigid country shares left additional financial room. More blanket safety stock would not have addressed the main weakness.

## What changed

1. **Forecast horizon:** direct 13-week forecasting, matching the final private window. The original comparison mixed horizons 1–26.
2. **Supplier-level signals:** capacity, lead times, futures premiums, tariffs, macroeconomic and shipping conditions, and supplier-scoped news stages/ages are modeled separately for 30 suppliers. The observed supplier shock definition reproduces every historical material disruption label exactly. It is used only as a historical training label, never as a future observation.
3. **Risk forecasts:** the selected candidate is `gated_announced`. Two regularized boosted classifiers and a transition fallback produce supplier probabilities. A material-level gate gives the direct models more weight when futures, tension, congestion or tariff announcements corroborate stress, and preserves the transition estimate otherwise. This reduces the tendency of a complex model to become overconfident in quiet conditions. A separate candidate treats future announced tariff starts explicitly, assigning 98% probability for a conservative duration estimated only from completed past tariff episodes. This differs from guessing the remaining duration of an already-active tariff; those broader persistence rules did worse. All candidates remain in the comparison table.
4. **Prices:** `price_mix` averages a futures anchor and a regularized model of future supplier price returns, then averages across each material's three suppliers. Its development MAPE was within one paired standard error of the best single model, so the fixed ensemble was preferred to dependence on one signal.
5. **Ordering:** supplier-specific predicted capacity and delay scenarios replace most of the unconditional history, with shared-lane shocks kept correlated. Known tariffs change landed costs from their announced effective week. Leaner, more flexible and blended policies were compared using the actual 40/30 financial/resilience weights. The recommended policy is `hedged50`, an explicit 50/50 blend of original and conditional country order quantities. The aggressive challenger `conditional_flexible` won mean development performance. The hedge is a judgment made after the retrospective stress tests, sacrificing some development mean to protect against missed shocks. It is not presented as the maximizer of that mean. Configuration details are in `model_selection.json` and `policy_v2.py`.

At 221 the inputs contain new Merrow-route tension, Valdmar cobalt export warnings, Kestria rare-earth export warnings, and announced Kestria tariffs on aluminum and semiconductors effective 224. The structured tariff table specifies 35%; it takes precedence over an approximate 25% aluminum-news headline. The lithium resolution at 221 clears its prior supplier-specific warning; it does not clear a separate Merrow-route warning. Early warnings remain uncertain, not guaranteed future events.

## Comparison results

| Period and metric | Original | Revised |
|---|---:|---:|
| Public 209–221 Brier | 0.13103 | 0.11693 |
| Public 209–221 price MAPE | 5.74% | 5.61% |
| Public 209–221 local score proxy | 55.43 | 59.16 |
| Public 209–221 demand fill | 100.00% | 100.00% |
| Development windows mean local score | 37.88 | 44.44 |
| Later evaluation windows mean local score | 29.97 | 29.58 |
| Later evaluation windows mean Brier | 0.13916 | 0.14325 |
| Later evaluation windows price MAPE | 4.85% | 4.73% |

The new model does not win everywhere. In particular, the unexpected crisis in 169–181 is difficult for signal-driven models, and disruption accuracy deteriorates in the later evaluation group even when other components improve. A more aggressive policy was explicitly compared with a blend that retains the original policy's protection. Every fold, including losses and cold-start performance, is in `historical_score_comparison.csv` and `policy_replay.csv`. The earlier forecast's strong performance on 183–208 was too narrow a basis for expecting the same performance in a new regime.

These results are retrospective chronological validation, not an untouched test set: we have now inspected these historical outcomes while developing the approach. No row-level feature or fitted target is allowed past its forecast origin. Numerical forecast selection and the best-mean policy comparison use the designated development folds. The recommended hedge and model architecture were informed by the broader stress tests, so the later evaluation cannot be called independent of model development. No promise of a private leaderboard score is justified by this small synthetic history.

## Replay weeks 0–207

There is **no week 0** in the files. The new files also leave every original row through 208 unchanged. The defensible replay uses weeks 1–12 as initialization and produces **1,950 forecast-and-order rows covering weeks 13–207**, in 15 independent 13-week windows. Origins are 12,25,38,...,194. Every window starts with five weeks of demand in stock and no transit, as in the competition. The first two origins use an explicitly labeled historical-rate fallback for the original risk model, which required 26 weeks.

Development origins are 64,77,90,103,116,129,142. Later chronological evaluation origins are 155,168,181,194. Earlier folds are reported as cold-start diagnostics. The final historical window 195–207 is fully scored without using observations 208–221. Each training target must end at or before its origin. Training on all 221 weeks and then predicting old weeks would leak their answers; this replay never does that. The published demand schedule is the permitted exception for future covariates.

From the project folder:

```powershell
python run_phase2.py
```

This runs the audit, historical forecasting comparison, policy comparison, final submission generation and tests. Cached supplier forecasts can be reused only with matching data fingerprints. To force refitting:

```powershell
python run_phase2.py --refresh
```

For forecasts alone use `python replay_v2.py --refresh`; `python evaluate_policy_v2.py` evaluates the policies. `historical_replay.csv` contains the complete selected replay with actual labels and orders. `model_comparison.csv` lists all forecast candidates; `policy_selection.csv` lists the policy comparison.

## Private-window forecast snapshot

| Material | Mean disruption probability,222–234 | Price 222 | Price 234 |
|---|---:|---:|---:|
| aluminum | 81.2% | 3.1701 | 3.1751 |
| cobalt | 29.8% | 65.0620 | 64.9624 |
| copper | 5.3% | 10.7612 | 10.9723 |
| lithium | 18.3% | 52.2648 | 52.7487 |
| natural_gas | 4.4% | 5.1688 | 5.1396 |
| polymer_resin | 30.7% | 2.0104 | 2.0061 |
| rare_earths | 44.0% | 108.5260 | 109.4951 |
| semiconductors | 78.0% | 522.4401 | 523.2629 |
| specialty_glass | 9.6% | 7.7192 | 7.7403 |
| steel | 37.8% | 1.2547 | 1.2532 |

## Limits and verification

The local simulator uses the documented arrival, cancellation, holding, shortage and terminal-salvage mechanics. Initial stock uses five times window-average demand and reference material cost uses window-average landed cost. The hindsight comparator is an exact perfect-information procurement LP under those local assumptions. It is used for evaluation only. A small gap from Kaggle can still reflect unspecified normalization conventions. When the sample already has 100% shock fill, the local fill-skill improvement is defined as zero; no-disruption windows earn only the diversification part of resilience.

Source completeness, unique keys, historical prefix equality, supplier eligibility, price positivity, probability bounds, nonnegative orders, and share sums are checked. The final saved submission has 260 unique rows and 10 columns. Prefix-invariance and future-target exclusion tests exercise the new forecasts. Inventory mass and cost reconciliation are checked separately. Exact verification outputs are stored beside the submission.
