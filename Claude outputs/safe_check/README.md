# Safe file: final_main vs submission_main_it8, scored the same way

## Hypothesis

The earlier comparisons may have favored main_it8 for two reasons. Half the test worlds come from the Model 1 / Model 3 generator that built main_it8 (final_main was built elsewhere). And the starting-stock rule, which decides the ranking, was treated as a coin flip. This check scores both files the same way: the same futures, the same partners, independent event-tree worlds reported separately, and a real-data test on weeks 209-221.

## Which starting-stock rule

`verify_scorer.py --data ../../Data_new` against the 4 known public scores: the best fit is 5 x mean demand (worst error across the four files 1.32). Sum of the first 5 weeks is close behind at 1.51, but only with unit-weighted diversification. With row-averaged diversification (what our evaluator uses), sum-of-5 misses the 68.40 file by 3.3. So 5 x mean is more likely but not proven.

## Real outcomes, weeks 209-221

| Convention | final_main | main_it8 |
|---|---|---|
| 5 x mean, row-averaged diversification (best fit) | 67.86 | **69.29** |
| sum of 5, row-averaged | **70.95** | 69.63 |
| sum of 5, unit-weighted (second-best fit) | 66.84 | **67.35** |
| 5 x mean, unit-weighted | 63.74 | **67.01** |

final_main's week 209-221 disruption probabilities score the same as the Phase 1 main pick's (Brier skill 0.520), so both files were built without seeing those weeks.

## Private window (weeks 222-234), simulated

Single file, mean score (5 x mean / sum of 5):

| | tree calibrated | tree no closure | tree Merrow closes | tree big war | it8 status quo |
|---|---|---|---|---|---|
| main_it8 | **68.43** / 68.24 | **59.50** / 59.10 | **75.54** / **75.44** | **76.30** / **76.58** | **67.27** / 66.56 |
| final_main | 66.73 / **68.72** | 59.22 / **61.97** | 73.48 / 74.54 | 72.31 / 73.34 | 65.66 / **67.42** |

Pairs, best of two (5 x mean / sum of 5). "30% level" = the score the pair reaches or beats with a 30% chance.

| Pair | tree calibrated mean | tree calibrated 30% level | it8 mix mean | it8 mix 30% level |
|---|---|---|---|---|
| main_it8 + nowar_it8 | **71.78 / 71.69** | 76.66 / 76.46 | **71.23 / 70.87** | 75.94 / 75.87 |
| final_main + nowar_it8 | 69.90 / 70.84 | 75.46 / 76.19 | 69.75 / 70.19 | 74.37 / 75.13 |
| main_it8 + war_it8 | 69.66 / 69.45 | 77.52 / 77.44 | 69.36 / 68.79 | **78.10** / 78.01 |
| final_main + war_it8 | 69.64 / 70.72 | **77.60 / 77.79** | 69.33 / 70.19 | 78.08 / **78.20** |

(it8 mix: Merrow stays open 45%; of the closures, 85% on time and 15% big war.)

## Finding

- **As a standalone safe file, main_it8 is better under the likely rule (5 x mean):** in every world, and on the real weeks 209-221 under 3 of 4 conventions. final_main wins only if the scorer uses sum-of-5 stock, and then only in calm or short-closure worlds.
- **Paired with nowar_it8 (best average), main_it8 is the better partner** under both rules (+0.9 to +1.9).
- **Paired with war_it8 (best upside), final_main is at least as good.** It ties under 5 x mean (within 0.1) and is ahead under sum-of-5 (+1.3 mean, +0.35 at the 30% level), because it covers the "Merrow stays open" side better (62.2 vs 59.3).

## Files

- `../../submissions/final/submission_main_it8.csv`, `../event_tree/submission_nowar_it8.csv`, `../../submissions/final/submission_war_it8.csv`
- final_main (pasted in chat): the full file was rebuilt from the paste and passes every check in the brief. It isn't saved in the project.
- `eval_safe_*.txt`: full output (11 worlds, both readings)
