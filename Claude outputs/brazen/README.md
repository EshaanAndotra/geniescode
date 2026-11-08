# Brazen second file: chance of a final score above 85 (main_it8 kept as the safe file)

## Hypothesis

With `submission_main_it8.csv` fixed as one of the two picks, choose the second file to maximize P(better of the two > 85) rather than the average score.

## Finding

Across 11 sampled worlds (5 from Model 1 / Model 3, 6 from the event tree) and both starting-stock readings:

- Only one scenario produces scores above 85: a Merrow closure plus a Kestria-wide shutdown ("big war"). main_it8 never clears 85; its 90th percentile is about 78 in every world.
- The best file for that scenario is `submission_war_merrow_kestria.csv`. Sharpening its probabilities (logit doubled, clipped to [0.005, 0.995]; orders and prices unchanged) raises its big-war chance further:

| P(score > 85) in a big war | it8 world, sum5 | tree world, sum5 | it8 world, 5avg | tree world, 5avg |
|---|---|---|---|---|
| war_it8 | 0.02 | 0.01 | 0.02 | 0.02 |
| new big-war build | 0.22 | 0.15 | 0.22 | 0.16 |
| war_merrow_kestria | 0.29 | 0.23 | 0.30 | 0.28 |
| **war_merrow_kestria, sharpened** (`submission_brazen_mk_sharp.csv`) | **0.32** | **0.28** | **0.32** | **0.32** |

Paired with main_it8, the sharpened file costs 0.05-0.16 points of average score against unsharpened war_merrow_kestria.

Overall chance of a final above 85 with main_it8 + the sharpened file: about 2.5% at our base-case odds (45% Merrow stays open, 15% of closures become a Kestria-wide shutdown), about 5% at a 30% shutdown share, about 9% at 50%, and about 30% if a big war is certain.

The cost: against main_it8 + nowar_it8, the average score at base-case odds drops by about 2.2 points (70.5 -> 68.3). If Merrow stays open, the better of the two is about 58.5 instead of about 66.4.

## Also tested, not recommended

- **Big-war build with scenario prices** (`lane:MERROW` 1.0 weeks 1-3 min 6 + `cty:KES` 1.0 weeks 2-5 min 8). Its +18% price forecasts on Kestria materials overshoot: price skill falls to about 0 even in the big-war world. Swapping those prices into war_merrow_kestria hurt as well.
- **Longer big war, long Merrow-only closure, sharpened war_it8.** All below war_merrow_kestria in the big-war world.

## Point-path models (a new model built only for P > 85): not recommended

`point_path.py` commits to one exact future: tariff from week 224 for 12 weeks, cobalt ban weeks 222-227, rare-earths ban weeks 223-228, Merrow closure from week 223 for 5 weeks (variants: from week 224, 8 weeks long, plus a Kestria-wide shutdown from week 224, or no closure). Probabilities are 0/1, and Model 3's planner orders for that path alone.

| P(> 85), sum5 / 5avg | exact modal path happens | timing off by about 1 week | event-tree calibrated odds | event-tree big war |
|---|---|---|---|---|
| path_m223_l5 (modal path) | **0.69 / 0.69** (mean 83.5) | 0.01 / 0.01 (mean 64.6) | 0.00 / 0.00 | 0.00 / 0.00 |
| path_m224_l5 (onset 1 week later) | 0.00 / 0.00 (mean 68.4) | 0.01 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 |
| path_bigwar | 0.00 / 0.00 | 0.01 / 0.01 | 0.00 / 0.00 | 0.08 / 0.08 |
| brazen_mk_sharp | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | **0.31 / 0.36** |

The ceiling is real: when the exact path happens, the better of main_it8 and that path averages 84.8 (modal) or 87.4 (big war). But a one-week miss on the Merrow onset drops the modal path from 83.5 to 68.4. The week-221 data can't pin down onset weeks (Merrow from 223 has about a 35% chance; a 5-week length is 1 of 6 options; each ban has its own onset and length), so the chance of hitting the whole path is well under 1%. The spread in our usual probabilities is what keeps a file scoring when timing slips. That's why brazen_mk_sharp clears 85 in 31-36% of big-war futures and the point-path big-war file in only 8%.

## Best score reached with at least a 30% chance (70th percentile of the better of two)

From the saved per-future totals (`totals_*.npz`). Odds: Merrow stays open 45%; of the closures, 85% are an on-time Merrow war and 15% a Kestria-wide big war.

| main_it8 + ... | 70th percentile, sum5 / 5avg | range over odds (25-60% open, 15-30% big war) | mean |
|---|---|---|---|
| **war_it8** | **78.0 / 78.1** | 75.7-79.4 | 68.8 / 69.4 |
| war_merrow_kestria | 77.2 / 77.3 | 75.1-78.3 | 68.7 / 69.2 |
| brazen_mk_sharp | 76.9 / 76.9 | 74.7-78.2 | 68.6 / 69.1 |
| nowar_it8 | 75.9 / 75.9 | 75.2-76.5 | **70.9 / 71.2** |
| main_it8 alone | 75.2 / 75.3 | 73.8-76.3 | 67.2 / 67.7 |

The event-tree calibrated world agrees: war_it8 gives 77.4 / 77.5. In the base case, P(main_it8 + war_it8 >= 78) is 0.30 / 0.31. 71-73% of that comes from an on-time Merrow closure (46-48% of those futures reach 78), 25-26% from a Kestria-wide war (95%), and 2% from no closure. A long closure reaches 78 in 73-75% of futures, a short one (2-3 weeks) in 11-12% and a late one in 3%.

The trade-off: main_it8 + war_it8 gives the highest score with a 30% chance, while main_it8 + nowar_it8 gives the highest average. They are different objectives.

## Not checked

- The "big war" worlds come from our own generators. The it8 big-war world uses the same override spec as the new big-war build, but the event-tree big-war world (Merrow closure plus Kestria country cap 0.2 from weeks 223-226) is independent and agrees.
- Each P(> 85) estimate rests on 300-600 futures (about ±2.5 points). Comparisons between files use the same futures, so the ranking is firmer than the levels.
- There's no evidence for the odds of a Kestria-wide shutdown beyond the week-221 news: export-limits debate, no sanctions chain.

## Files

- `submission_brazen_mk_sharp.csv`: passes every check in the brief. Weeks 209-221 are identical to main_it8. In weeks 222-234 only `disruption_prob` differs from war_merrow_kestria.
- `eval_brazen_*.txt`: 10 files x 11 worlds (mean, P > 85, 90th percentile, best of two with main_it8).
- `eval_hybrids_*.txt`: war_merrow_kestria hybrids, with score parts.
- `eval_tail.py`, `eval_quick.py`: the evaluators (they need `event_tree.py` and `m1patch.py` from `../event_tree/`).
