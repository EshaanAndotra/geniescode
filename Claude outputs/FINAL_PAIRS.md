# Final pair options (private window, weeks 222-234, planned from week 221)

Kaggle keeps the better of the two selected files. The table shows the average score of the better file in each kind of future. Figures are simulated (event-tree worlds, 5 x mean starting stock, the more likely scorer rule). Details are in `event_tree/`, `brazen/` and `safe_check/`.

| Pair | Merrow stays open | Short closure (2-3 wk) | Normal closure (3-8 wk) | Late closure (wk 227+) | Long closure (10-13 wk) | Kestria-wide war | Overall average | Hit with 30% chance | Worst 10% |
|---|---|---|---|---|---|---|---|---|---|
| **A. main_it8 + nowar_it8** (recommended) | **67.7** | 73.9 | 75.5 | 70.1 | 74.8 | 76.3 | 71.8 | 76.7 | **61.3** |
| B. war_it8 + nowar_it8 | 66.8 | 68.8 | 76.2 | 67.9 | 79.5 | 81.6 | **72.0** | **77.9** | 58.8 |
| C. main_it8 + war_it8 | 59.7 | 74.0 | **77.3** | 70.8 | 79.5 | 81.6 | 69.7 | 77.5 | 56.6 |
| D. final_main + war_it8 | 60.2 | **74.2** | 77.2 | 68.8 | 79.5 | 81.6 | 69.6 | 77.6 | 52.3 |
| E. main_it8 + brazen_mk_sharp | 59.7 | 74.1 | 76.6 | **71.6** | **81.3** | **83.8** | 69.4 | 77.0 | 56.5 |

When each pair fits best:

- **A** if it's unclear whether the clash closes Merrow. It has the most robust floor and is the recommended pair.
- **B** if it's all or nothing: tensions fade, or a long war. It's weak if Merrow closes for only 2-3 weeks.
- **C / D** only if Merrow is at least 85% likely to close early. D beats C only if the scorer uses sum-of-first-5-weeks stock.
- **E** only if the clash escalates into a full Kestria shutdown. It's the only pair with any chance of a score above 85: about 3% overall, 31-36% in a Kestria-wide war.

## Files

| Name | Path |
|---|---|
| main_it8 | `submissions/final/submission_main_it8.csv` |
| war_it8 | `submissions/final/submission_war_it8.csv` |
| nowar_it8 | `Claude outputs/event_tree/submission_nowar_it8.csv` |
| brazen_mk_sharp | `Claude outputs/brazen/submission_brazen_mk_sharp.csv` |
| final_main | not in the repo (shared separately) |

All repo files above pass the brief's checks: 260 rows, ids match the sample, no blanks, probabilities in [0, 1], prices > 0, whole orders >= 0, shares sum to 1 with 0 for non-suppliers. Their week 209-221 rows are identical to main_it8's.
