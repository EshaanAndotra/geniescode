# Event-tree check of the final pair (weeks 222-234, planned from week 221)

## Hypothesis

Model the private window as a calm baseline plus the named week-221 events, each with its own odds, start week, length and effects. Then check whether the current pair (`submission_main_it8` + `submission_war_it8`) is the right best-of-two for this specific window.

## Finding

The current pair has no file that wins if the Merrow Strait stays open. The main file hedges for a closure (about 0.45 on Merrow-only materials) and the war file bets on one (about 0.91). If Merrow doesn't close, the better of the two scores only about 58.5-59.5.

`submission_nowar_it8.csv` is the opposite of the war file. It's built with iteration 8's own pipeline; the only change is the Merrow closure set to probability 0:

```
cd "Code Base/pipelining"
python scenarios/build_scenarios.py nowar_m1 '{"ovr": [["lane:MERROW", 0.0, 1, 13, 0]]}'
```

It keeps the tariff, the Valdmar cobalt ban and the Kestria rare-earths ban. It drops the closure hedging: lithium, steel, polymer resin and specialty glass sit at about 0.12-0.15, with no stocking up early and cheaper Merrow buying.

## Results

Best of two, mean total out of 100. The "it8" worlds use iteration 8's own scenario generator; the "tree" worlds use `event_tree.py`. Starting stock is the first 5 weeks of demand (the 5 x mean reading gives the same ranking; see `eval_results_5avg.txt`).

| Pair | it8 Merrow war | it8 big war | it8 no closure | it8 late closure | tree calibrated | tree no closure | tree short closure |
|---|---|---|---|---|---|---|---|
| main + war (current) | **76.35** | 82.07 | 58.50 | **70.78** | 69.45 | 59.45 | **73.70** |
| **war + nowar_it8** | 74.89 | 82.07 | 65.83 | 67.89 | 71.91 | 67.14 | 68.38 |
| main + nowar_it8 | 74.36 | 75.51 | **66.40** | 70.08 | 71.69 | 67.76 | 73.56 |

Weighted over it8 worlds (closures split 70% on time, 15% late, 15% with a Kestria shutdown):

| P(Merrow stays open) | 10% | 25% | 45% | 60% |
|---|---|---|---|---|
| main + war (current) | **74.59** | 71.90 | 68.33 | 65.65 |
| war + nowar_it8 | 74.01 | **72.65** | **70.83** | **69.46** |
| main + nowar_it8 | 73.14 | 72.02 | 70.52 | 69.40 |

The break-even is about a 12-15% chance that Merrow stays open. Every estimate we have is well above that: Model 1 about 55-60%, the news-chain read about 45%, base rates about 75%. Only the war file's own 90%-closure assumption favors the current pair.

## Also tested, not recommended

- **Event-tree files as picks** (`submission_tree_*.csv`). They beat Model 1 files only in the tree's own worlds, which is circular. In it8 worlds the calibrated tree file loses to main by 2.5 (status quo) and 11 (big war).
- **Moving the war file's closure start to weeks 223-225.** It was worse than the current war file in most worlds (it8 Merrow war 71.75 vs 74.79).

## Forcing the three near-certain events (tariff, cobalt ban, rare-earths ban)

Not worth it: the models already make that assumption.

- At week 221 Model 1 already gives both bans a 100.0% chance of starting inside the window (cobalt 29/57/12/3% across weeks 222-225; rare earths 3/42/50/5% across 223-226). The tariff gets 95% (`p_announced`). Model 3's planner multiplies onset odds by 1.5 and caps them at 100%, so the orders already treat the bans as certain.
- The forced versions (`build_certain.py`; `m1patch.py` adds a negative `p_onset` that rescales the model's own onset curve to 100% without moving its timing) change only aluminum and semiconductors, by +0.006 to +0.04. Rebuilding main without overrides reproduces `submission_main_it8.csv` exactly.

Best of two (sum5; `eval_certain_sum5.txt`. The 5avg reading in `eval_certain_5avg.txt` shows the same pattern):

| Pair | it8 status quo | it8 no closure | it8 all three certain | it8 tariff fails | it8 cobalt ban fails | it8 RE ban fails | tree calibrated |
|---|---|---|---|---|---|---|---|
| war + nowar_it8 | 70.00 | 65.83 | 71.24 | 60.14 | 66.35 | 61.19 | 71.91 |
| war_cert + nowar_cert | 70.03 | 65.67 | 71.18 | 60.25 | 66.33 | 60.87 | 72.01 |
| war + nowar_cert5 (bans last 5+ weeks) | 69.48 | 64.33 | 70.64 | 59.93 | 65.58 | 60.14 | 71.59 |

- Even in the world where all three happen, forcing gains nothing (71.18 vs 71.24): the files already sit at 0.96-0.98 at the peak. The remaining doubt is when the events start and how long they last, and forcing occurrence doesn't touch that.
- All differences are within about ±0.3 with mixed signs. That's the level of the order planner re-solving on a changed scenario set, not a real effect.
- Assuming the bans last at least 5 weeks (the shortest past ban) lowered Brier in both world sets and cost 0.5-1.5 points in the pair. As a main replacement it gains 1.6 in a big war and loses 1.1 if the bans are short.
- The tariff is the one large exposure: if it isn't enacted, every file drops about 10 points (main 66.6 -> 56.3), because aluminum and semiconductors sit at 0.96-0.97 for 10 weeks. At 3-5% odds, a file to hedge that isn't worth one of our two picks.

## submission_war_merrow_kestria.csv (in submissions/final/)

Built outside this session and not documented. Its week 209-221 rows and every price forecast match main_it8.

- **Probabilities:** a Merrow closure starting at random over weeks 222-227 (specialty glass rises about 0.15 a week to 0.91 at week 227, then fades to 0.39), plus a Kestria-wide shutdown from week 223 that lasts to the end (Kestria-supplied materials still 0.78-0.82 at week 234).
- **Orders:** close to war_it8's. Kestria's share of aluminum is 0.20 (war 0.21), steel 0.24 (0.32), semiconductors 0.36 (0.32). The orders mostly hedge Merrow rather than planning around a Kestria shutdown.

Single file (sum5): it beats war_it8 only in a big war (83.60 vs 82.07) or a late closure (69.45 vs 67.89). It loses 2.7 in the status quo and 3.5 in a plain Merrow war, and its Brier skill in the status quo is -0.08, worse than predicting the base rate.

Weighted best of two (same weighting as above; `eval_war_merrow_kestria_*.txt`):

| P(Merrow stays open) | 10% | 25% | 45% | 60% |
|---|---|---|---|---|
| war + nowar_it8 | 74.01 | **72.65** | **70.83** | **69.46** |
| war_merrow_kestria + nowar_it8 | 72.25 | 71.16 | 69.70 | 68.61 |
| main + war | 74.59 | 71.90 | 68.33 | 65.65 |
| main + war_merrow_kestria | 74.59 | 71.91 | 68.35 | 65.68 |

Next to main, it ties with war_it8. Next to nowar_it8, it only beats war_it8 if a Kestria-wide shutdown is more than 54% of closure outcomes. The week-221 news has an export-limits debate in Kestria but no sanctions chain. Both starting-stock readings agree.

## final_main.csv and final_hailmary.csv (pasted in chat, Nov 8)

Neither matches any file in the project. Their week 209-221 rows also differ from main_it8. Both pass the format checks on weeks 222-234.

- **final_main:** the Merrow closure peaks at about 45% (week 224). Tariff and bans are as in iteration 8. Prices are within ±3% of main_it8, except cobalt and rare earths drop 6-8% from weeks 230-231.
- **final_hailmary:** final_main with 62 of 130 probabilities raised. It assumes:
  - a Merrow closure at 0.80 from weeks 222-225, held to week 227 and gone by about week 231;
  - the cobalt ban held at 0.80 through week 234, although its price drops 8% at week 230 as if the ban had ended;
  - the tariff still in force at week 234.

  It assumes the Kestria/Sunda Coast clash closes Merrow, but not a Kestria-wide shutdown.

Weighted best of two (sum5 / 5avg; `eval_final_main_hailmary_*.txt`):

| Pair | 25% open | 45% open | 60% open |
|---|---|---|---|
| war_it8 + nowar_it8 | **72.65 / 72.84** | **70.83 / 71.13** | **69.46 / 69.85** |
| hailmary + nowar_it8 | 72.01 / 71.99 | 70.33 / 70.47 | 69.07 / 69.32 |
| final_main + war_it8 | 72.72 / 71.99 | 69.71 / 68.71 | 67.44 / 66.25 |
| final_main + hailmary | 71.97 / 70.94 | 69.15 / 67.81 | 67.04 / 65.46 |

final_main + hailmary has the same gap as main + war: if Merrow stays open, the better of the two scores 61.4 (sum5). final_main's orders depend on the starting-stock reading. It beats main_it8 under sum5 (+0.9 status quo) and loses under 5avg (-1.6). main_it8 plans for the smaller of the two readings.

## Not checked

- No walk-forward backtest. Both comparisons use sampled futures from models we built, not past windows.
- The odds come from 3 clash chains and 5 lane closures.
- The tree's ban price jump (+25%) is untested against Model 2's +19-20%.

## Files

- `event_tree.py PIPELINE_DIR OUT_DIR`: builds the tree files. It swaps Model 3 iter5's `make_scenarios` for the event tree; the ordering optimizer, simulator and scorer are unchanged.
- `eval_tree.py PIPELINE_DIR OUT_DIR name=file.csv ...`: scores files in both sets of worlds.
- `eval_results_*.txt`: full output, including score parts per world.
