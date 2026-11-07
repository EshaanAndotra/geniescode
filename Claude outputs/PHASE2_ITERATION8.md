# Phase 2 review and final picks (iteration 8)

Built from the Phase 2 data (weeks 1-221) for the private window, weeks 222-234. Only those 13 weeks count for the private score, so every number was tuned to the week-221 state and checked against the closest past situations. Values that rest on one or two past events were left alone.

**Final picks:** `submissions/submission_main_it8.csv` (main) and `submissions/submission_war_it8.csv` (war hedge). Upload both and select both as final submissions. Kaggle keeps the better one. If they aren't selected by hand, Kaggle picks the two best public scores instead.

```
cd "Code Base/pipelining"
python run_pipeline_p2.py          # about 45 s; finds Data_new/ on its own, writes both files to submissions/
```

This work sits next to the existing code; it adds files and changes none. The new model folders are numbered past everything on `main` and on `alan/b2-forecast`:

| Folder | What it is | Built on |
|---|---|---|
| `Model 1/iter6` | disruption forecast | Model 1 iter4 (news layer) |
| `Model 2/iter6` | price forecast and landed costs | Model 2 iter3 |
| `Model 3/iter5` | ordering plan | Model 3 iter3 (score-aware LP) |

`main`'s Model 2 iter4 and Model 3 iter4, and the branch's Model 1 and 2 iter5, are separate work and untouched.

---

## 1. Weeks 209-221: what happened and how the models did

**Calmer than average.** 13.8% of material-weeks were disrupted, against 19% in weeks 157-208 and 43% in weeks 105-156. There were two events:
- **Southern route strike, weeks 212-214.** It hit Orinoque, so cobalt, copper, lithium and natural gas.
- **Kestria lithium shortage, weeks 215-220.** Capacity fell to 30%.

Benchmark prices moved too: lithium peaked at +36%, natural gas +22%, polymer resin +14%, and semiconductors fell 14%.

**Where the points went.** The 68.40 public file scores 71.7 locally; the local scorer runs 1.3-3.3 points above Kaggle. Out of the 100 points:
- **Forecasts (P), about 21 lost.** Brier skill was 0.52. Price skill was -0.03, slightly worse than carrying the last price forward.
- **Cost (F), about 4 lost.** F was 0.89.
- **Fill and diversification (R), about 3 lost.** Fill skill was 1.0 and diversification 0.66.

**Model by model:**
- **Model 1.** It got the six calm materials right. It gave the Southern strike only 0.2, and those 12 material-weeks are 95% of its Brier error. It also called the lithium shortage 4 weeks early and let it fade too fast.
- **Model 2.** It priced lithium's jump at week 211 (the jump came at 215). Its mean reversion pulled steel down 5-8% while steel stayed flat.
- **Model 3.** No shortfall in shock weeks, and 89% of the possible savings.

## 2. The state at week 221

| Signal | Evidence | What it led to before |
|---|---|---|
| Tariff | Kestria aluminum and semiconductors go from 10% to 35% at week 224 | All 3 past tariffs took effect and lasted 10-18 weeks |
| Valdmar cobalt export ban | "strategic priority" and "slower license approvals" headlines; futures premium 7% -> 10% -> 12% | Past onsets came 2-5 weeks after a premium first passed 6% (weeks 222-224 here). When Model 1 said 90%+ for a calm material, the shock came 42 times out of 46 |
| Kestria rare earths | futures premium 4% -> 9% | Same rule: weeks 223-226 |
| Merrow Strait clash | "Tensions rise between Kestria and neighbors" (weeks 220-221); Kestria tension 10 -> 41 -> 54; Sunda tension 25 -> 37; congestion 27 -> 40 -> 44 | Same opening as week 171, which led to the 8-week closure at week 173 |
| Kestria export limits | "Lawmakers in Kestria debate limits on raw material exports" | The first step of the Orinoque copper (week 132) and Valdmar rare earth (week 136) bans |

The news matches a Merrow clash, not a sanctions-style Kestria shutdown:

| | Week 171 -> 8-week Merrow closure | Weeks 92-99 -> 35-week Kestria event | Now (220-221) |
|---|---|---|---|
| Headline | "Tensions rise between Kestria and neighbors" | talks break down, then "prepare for sanctions" | "Tensions rise", twice |
| Kestria tension | 24 -> 47 in a week | slow climb over 8 weeks | 10 -> 41 -> 54 |
| Sunda tension | 36 -> 54 | flat | 25 -> 37 |
| Kestria currency | flat | -8% | flat |
| Merrow congestion | 38 -> 55 | flat | 27 -> 40 -> 44 |

Still missing: "Insurers raise war risk premiums for the Merrow Strait", which came one week before the last closure.

## 3. What changed, model by model

All evidence comes from `backtest_phase2.py` (section 6). Each window is planned from its origin with data up to that week only. Origins run from 40 or 52 to 208, so weeks 53-221 are all test weeks. "Warning windows" are windows where nothing was disrupted at the origin but Model 1 already expected trouble: a mean forecast of 0.15 or more. Week 221 is one of these, at 0.425.

### Model 1 iter6

| Change | Evidence |
|---|---|
| **Sharpen only when the outlook is calm.** p is sharpened to sigmoid(1.5 logit p) only when the window's mean p is below 0.15 (`sharpen_max_alert`). | Warning windows: P 0.168 -> 0.199. All 169 windows: unchanged at 0.242. Before the big shocks (origins 92-98, 171) sharpening cost 0.06-0.09 P per window, because it pushed the 0.3-0.5 forecasts down right where the shocks landed. |
| **Tariff minimum length** (`tariff_min_floor`). No completed tariff has lasted under 10 weeks, so an active or announced tariff stays at full probability for that long. After that it follows the fitted curve, conditional on having reached 10 weeks. Surprise tariffs keep the old curve. | Overall P unchanged (0.2416 -> 0.2420); it fits the 18-week tariff and misses the one that ended at exactly 10. Full score unchanged (52.95 -> 52.97). Weeks 224-233, aluminum and semiconductors: 0.71-0.97 -> 0.96-0.97. |
| **Scenario layer** (`source_overrides`). `("lane:MERROW", 0.9, 1, 3, 6)` means: Merrow closes in weeks origin+1 to origin+3 with probability 0.9, then stays closed at least 6 weeks. It replaces the model's onset curve for that source. | Used only by the war hedge. With no override the output is unchanged. |
| Own duration shape for tariffs (`shape_sd`). | Off. Net zero (77 windows up, 37 down). |

### Model 2 iter6

| Change | Evidence (price skill against carrying the last price forward) |
|---|---|
| **Jumps at every horizon** (`jump_max_h` 4 -> 13). In iter3 a shock expected to last past week 4 of the window fell back to the calm price; at week 221 that dropped cobalt and rare earths back to normal at week 226. | Warning windows: 0.037 -> 0.080. Overall unchanged. |
| **Half-strength calm mean reversion** (`beta_scale` 0.5). | All windows: 0.024 -> 0.029. Windows into weeks 209-221: -0.007 -> 0.024. |
| Both together | All 0.029, warning windows 0.078, public window -0.034 -> -0.004 |
| Blend with futures prices (`use_futures`, `futures_weight`). | Off; worse at every weight. 0/25/50/100% futures: all windows 0.029 / 0.029 / 0.028 / 0.024; held-out latest windows (origins 183-208) 0.025 / 0.023 / 0.019 / 0.012. |

Past supplier shocks back up the jump size. The shock week averaged about +22% (range +20% to +39%; the largest came after the steepest premium runs), and prices held until the shock ended. Model 2 puts cobalt and rare earths at +19% to +20%.

### Model 3 iter5

| Change | Evidence |
|---|---|
| **Hazards x 1.5** in the main preset (`prob_mult` 1.5). | 53 windows, 2 seeds. Single pick: 52.1 -> 52.9. Best of two with the old hedge: 56.2 -> 56.7. Shock windows: 59.0 -> 61.5. Calm windows: 47.2 -> 46.7. Most of the gain comes from a few windows where a shock started mid-window (origins 91, 97, 130, 202, 205); on the 48 older windows alone it is a tie. It is kept because week 221 looks like a shock window. |
| **Plan on the smaller of the two starting-stock readings** (`start_stock="min"`). The scorer could start with the sum of the first 5 weeks' demand or with 5 x the window mean; for weeks 222-234 they differ by 2-4% per material. | Best of two, main + war hedge, in the three worlds of section 4. Scorer reading 5 x mean: 67.6 / 76.2 / 81.8 -> 68.3 / 76.5 / 82.1. Scorer reading first 5 weeks: 67.8 / 76.4 / 82.0 -> 67.5 / 76.4 / 82.1. `verify_scorer.py` leans toward 5 x mean. |
| **`m1_config`.** The futures follow a Model 1 scenario; a scenario source can be given a minimum length. | Used by the war hedge. |
| 400 sampled futures instead of 200. | Not used: 52.8 / 52.5 against 53.0 / 52.8. |

## 4. The two picks

**Main pick.** Model 1's own odds give the Merrow Strait about a 40% chance of closing. The pick includes the tariff, the cobalt ban and the rare-earth shock as Model 1 sees them.

**War hedge.** It assumes the Merrow Strait closes in weeks 222-224 with probability 0.9 and stays closed at least 6 weeks; Models 2 and 3 plan on the same assumption. Its weeks 209-221 rows are the main pick's. How it differs from the main pick:
- **Disruption probabilities.** Lithium, steel, polymer resin and specialty glass are about 0.74 in weeks 222-226 (main: about 0.4) and about 0.57 in weeks 227-234 (main: about 0.27). Cobalt and rare earths are higher later in the window. Copper and natural gas are unchanged.
- **Prices.** Identical: a strait closure delays shipments but doesn't move spot prices.
- **Orders.** Total volume is about the same. It moves away from Merrow where that's cheap: rare earths 37% -> 15%, aluminum 37% -> 21%, steel 45% -> 32%. Where the backup suppliers are too small, it buys earlier instead: weeks 222-224 orders are up about half for glass, 20% for steel and 11% for polymer resin.

**Evidence.** Full score (30 P + 40 F + 21 fill skill + 9 diversification) on 300 sampled futures per world, using `scenarios/eval_scenarios.py`. There are three worlds:
- **Status quo:** Model 1's own odds.
- **Merrow war:** the strait closes in weeks 222-224.
- **Big war:** the strait closes and Kestria shuts down (capacity 20%, at least 8 weeks).

Starting stock is the first-5-weeks reading; the scores are before the starting-stock change.

| Second pick, paired with main (best of two) | Status quo | Merrow war | Big war | 25% war | 50% war |
|---|---|---|---|---|---|
| Old aggressive hedge (Model 3 iter1) | 67.2 | 74.6 | 76.3 | 69.3 | 71.3 |
| **War hedge** | **67.8** | **76.4** | 82.0 | 70.7 | 73.5 |
| Merrow + Kestria shutdown file (main's prices) | 67.7 | 76.1 | 84.1 | 70.8 | 73.9 |

In the mixed columns the war odds are split evenly between the two war worlds. With an 80/20 split toward Merrow-only, the war hedge is best or within 0.1 at every level.

As single files:

| | Status quo | Merrow war | Big war |
|---|---|---|---|
| Main | 66.8 | 74.4 | 74.7 |
| War hedge | 62.0 | 74.9 | 82.0 |

The war hedge beats the old hedge in every world, including no war, because Model 1's own odds already close Merrow about half the time. The Kestria-shutdown file is better only if a shutdown is about as likely as a Merrow-only war, and the news doesn't support that.

**How much to trust these numbers.** The worlds are built from our own Models 1 and 3, so the margins are a guide, not a forecast. The backtests suggested about 60 on Kaggle for the private window, with a range of roughly 45-70.

## 5. Tested and not used

| Idea | Result |
|---|---|
| Teammate's ordering changes to the main pick: copper +600 units at week 228; specialty glass 20 points from Sunda Coast to Aurelia; polymer resin toward 35% Aurelia; semiconductors toward 40% Halvard | Worse in 11 of 13 of the hand-built scenarios in `stress_private.py` (by 0.01 to 1.4). Better only with Merrow at 40% capacity for the whole window (+2.8). Sampled futures: -0.14 to -0.37. As a second pick it loses to the hedge by 0.3-0.4. |
| `submission_phase2_ready.csv` (another teammate's file) | Ordering points 50.3 / 54.7 against the main pick's 56.4 / 59.9 (Model 1 odds / odds x 1.5). It misses the cobalt and rare-earth warnings in weeks 222-224, its prices follow futures, and it caps backup suppliers at 47.5% of capacity. Its weeks 209-221 rows are the released answers. 108 order rows aren't whole numbers. |
| Recalibrating Model 1's high calls | When the model said 90%+ for a calm material, the shock came 42 times out of 46. All 4 misses were the copper ban arriving 1-2 weeks late. |
| Moving the rare-earth onset a week later | History points to weeks 223-226 and the model is about a week early. Worth about 0.1 point, so not re-tuned on 6 events. |
| Raising Merrow odds by hand in the main pick | Tension of 40-60 was followed by a lane closure within 4 weeks about 25% of the time, and congestion of 40-45 about 9% of the time. Model 1's 0.41 peak is in line with that; the higher bet lives in the war hedge instead. |

## 6. How to run

```
cd "Code Base/pipelining"

# final files
python run_pipeline_p2.py                                    # submission_main_it8.csv, submission_war_it8.csv

# walk-forward backtest on the Phase 2 data
python backtest_phase2.py --mode P --origins 40:209:1        # Models 1 + 2 only, about 30 s per plan on 2 cores
python backtest_phase2.py --mode full                        # all three models, 53 windows, about 75 s per plan
python backtest_phase2.py --mode full --plans '[{"name":"new"},{"name":"old","m1":"iter4","m2":"iter3","m3":"iter3"}]'

# scenario files and the full-score comparison in the three worlds
python scenarios/build_scenarios.py war '{"ovr": [["lane:MERROW", 0.9, 1, 3, 6]]}'
python scenarios/eval_scenarios.py main=../../submissions/submission_main_it8.csv war=../../submissions/submission_war_it8.csv
STOCK=5avg python scenarios/eval_scenarios.py ...            # same, scored with the 5 x mean starting stock
```

**`backtest_phase2.py`:**
- Each plan is a JSON object with any of: `m1`, `m2`, `m3` (folder), `preset`, `sharpen`, `m1cfg`, `m2cfg`, `m3cfg`.
- Results are reported for each group: all windows, design-era (ending by week 208), windows reaching the new weeks, the public window (origin 208), alert, warning, shock and calm. With two or more plans in full mode it adds best of two for each pair.
- Groups are fixed by `backtests/p2_alert_ref.csv`, so every plan is compared on the same windows.
- Results go to `backtests/p2_<mode>_<name>.csv`.

The saved `backtests/p2_*.csv` files were made before the folders were renumbered:

| In the saved files | Folder now |
|---|---|
| Model 1 iter5 | Model 1 iter6 |
| Model 2 iter4 | Model 2 iter6 |
| Model 3 iter4 | Model 3 iter5 |

Plan names: `main_new` is the main pick before hazards x 1.5, `m3i3_pm15` is with it, and `main_tfloor` adds the tariff minimum length.

## 7. Files added

```
Code Base/pipelining/run_pipeline_p2.py           final files (main + war hedge)
Code Base/pipelining/backtest_phase2.py           walk-forward backtest on the Phase 2 data
Code Base/pipelining/scenarios/build_scenarios.py files planned on a Model 1 scenario
Code Base/pipelining/scenarios/eval_scenarios.py  full score in status quo / Merrow war / big war
Code Base/pipelining/Model 1/iter6/               model1.py, news_stories.py (news_stories.py unchanged from iter4)
Code Base/pipelining/Model 2/iter6/model2.py
Code Base/pipelining/Model 3/iter5/model3.py
Code Base/pipelining/backtests/p2_*.csv           backtest results behind the tables above
submissions/submission_main_it8.csv               main pick
submissions/submission_war_it8.csv                war hedge
docs/PHASE2_ITERATION8.md                         this file
```

## 8. Open items

- **Scorer details.** Starting stock is handled by planning on the smaller reading. The local scorer still reads 1.3-3.3 points above Kaggle on the known public scores.
- **One-off bets.** Hazards x 1.5 and the war hedge are bets on week 221's specific signals. On calm windows they cost a little: about 0.5 for the hazards, and the war hedge alone scores 62.0 against the main pick's 66.8 in the status quo.
- **No-warning shocks.** 3 of the 15 past events had no warning; nothing here covers those beyond the duration curves' long tail.
- **What would change the call.** An "Insurers raise war risk premiums for the Merrow Strait" headline would be the strongest confirmation of the war hedge. Sanctions-chain headlines ("prepare for sanctions", a falling Kestria currency) would point to the Kestria-shutdown file instead (`scenarios/build_scenarios.py` with `["cty:KES", 0.7, 2, 5, 8]` added).
