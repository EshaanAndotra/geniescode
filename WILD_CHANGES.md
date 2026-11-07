# Wild changes log

What changed from the original plan (the three Model PDFs), why, the evidence, and how to undo each change. Everything here uses only competition data for weeks 1 to 208 (Phase 1). Every backtest forecasts from past weeks using only data available at that week.

Submission files are in `submissions/`. `submissions/SUBMISSIONS.md` lists every file, what produced it, and how to rebuild it. Public leaderboard scores so far:

| File | What it is | Public score |
|---|---|---|
| `C_full_moderate.csv` | Model 1 iter3 + Model 2 iter2 + Model 3 moderate | 62.33 |
| `E_divtest_zero_rows_single.csv` | C with single-supplier shares on zero-order rows (scorer test) | pending |
| `submission_moderate_wild.csv` | C + the Model 1 iter4 news layer (change 1) | **65.29** (+2.96 vs C) |

---

## 1. Model 1 iter4: news story layer (the drastic one)

**Files:** `Code Base/pipelining/Model 1/iter4/news_stories.py` (new) and `Model 1/iter4/model1.py` (iter3 plus a hook). iter3 is untouched. To undo, set `Config(use_news_stories=False)` or run the pipeline with `--model1-iter iter3`.

**Why.** Model 1 iter3 never read `news.csv`. The news turns out to follow fixed headline templates that escalate in a set order before each shock type. The full chronology of non-routine headlines against shock onsets in weeks 1 to 208 shows:

| Story type | Target | Warning headlines (examples) | Rumor ending | Became a shock | Weeks from first warning to onset |
|---|---|---|---|---|---|
| Shortage | one supplier | "Kestria steel producers flag maintenance delays", "Accident halts major ... site", "rations water to ... producers" | none seen | 3 of 3 (steel KES w37, cobalt KES w117, steel HAL w158) | 3 to 4 |
| Export ban | one supplier | "ministry calls copper a strategic priority", "drafts restrictions", "signals export curbs within weeks" | "officials rule out new restrictions" | 3 of 4 (copper ORI w132, rare earths VAL w136, natural gas VAL w153) | 2 to 4 |
| Strike | one lane | "Port labor talks drag on", "Union sets strike date", "Shippers rush cargo ahead of strike" | "Deal reached, ... disruption averted" | 2 of 6 (Atlantic w70, Merrow w109) | 2 to 3 |
| Clash | one lane | "Tensions rise between Valdmar and neighbors", "Insurers raise war risk premiums", "Warning shots fired" | "Calm returns near the ..." | 1 of 3 (Merrow w173) | 2 |
| Sanctions | one country | "Diplomatic talks ... break down", "Companies told to prepare for Kestria sanctions", "Sanctions vote scheduled" | none seen | 1 of 1 (Kestria w99) | 7 |

Policy reversals ("Halvard abruptly suspends aluminum export licenses") have no warning headlines, as the competition says.

**How it works.**

1. Headlines are matched to the templates above.
2. A story opens at its first warning. It closes when its target's shock starts, when a rumor-ending headline appears, or after 10 weeks with no new warning.
3. At each forecast week, only stories already closed by then are used. For each story type, the share that became shocks has a prior of 0.5 with weight 2. Lead times use a flat prior over 1 to 6 weeks.
4. Each open story gets a first-onset curve, adjusted for how long it has already run.
5. For a calm source, that curve is combined with Model 1's own onset curve. The combined curve then goes through Model 1's existing duration curves.

Model 2's structural prices and Model 3's scenarios read these source probabilities, so the news flows into all three columns.

**Evidence: backtests.**

| Test | Without news (iter3) | With news (iter4) |
|---|---|---|
| Model 1 Brier skill, forecasts from every week 104 to 207 | 0.420 | 0.420 |
| Model 3 points (40 F + 21 fill + 9 diversification), 34 windows | 40.1 | 40.9 |

The layer is neutral on history. Most past supplier shocks were also flagged by the futures premium, which Model 1 already uses. Lane stories were mostly rumors. Its value is in cases where the news is ahead of the market.

**What it changes right now (forecasting from week 208).**

- **Kestria lithium shortage story,** with warnings in weeks 207 and 208. Lithium's disruption probability for weeks 211 to 214 goes from about 0.1 to 0.2 up to about 0.76 to 0.80. The price forecast adds about 8% from week 211. Model 3 buys about 6,300 units in weeks 209 and 210 to build stock before the expected capacity cut.
- **Southern route strike story,** with warnings in weeks 207 and 208. Lithium, cobalt, copper and natural gas from Orinoque get about 0.26 extra probability of disruption.
- **The Northern Sea clash story** ended with "Calm returns" in week 207, so it adds nothing.

**Risk.** The lithium bet rests on 3 past shortage stories. If no Kestria lithium shock comes in weeks 209 to 221, the wild file loses Brier and price points on lithium and pays for a buffer it didn't need.

**Result.** Public score 65.29 against 62.33 for C, a gain of 2.96. The two files differ only through this layer, so iter4 is now the pipeline default.

**How to read the result.** The wild file differs from C only through this layer. A public score above 62.33 supports the news layer, and below 62.33 counts against it. After Phase 2 lands, check whether "Kestria lithium production falls amid shortage" appears around weeks 210 to 211.

## 2. Model 2 iter2: structural shock-state price forecast

**Files:** `Model 2/iter2/model2.py`. iter1 is untouched. To undo, use `Config(method="blend")`.

**Evidence from weeks 1 to 208.**

- Spot prices move only with supplier-level shocks. All three suppliers of the material jump together, by 15 to 31% for a single-supplier shock and about 8% for the Kestria country event. The jump fully reverses when the shock ends.
- Lane closures and tariffs do not move spot prices.
- Calm prices behave like a random walk.

**The method.** If a price shock is active, the forecast removes its jump, weighted by Model 1's probability that the shock has ended. If calm, it adds the typical jump, weighted by the probability that a shock has started.

**Backtest.** Price skill, the accuracy gain over carry-forward, was measured from every second week between 60 and 194:

| Situation | iter1 | iter2 |
|---|---|---|
| Overall | 0.014 | 0.051 |
| Shock active at forecast time | 0.03 | 0.15 |

## 3. Model 3 iter1: built from scratch (ordering plan)

**Files:** `Model 3/iter1/model3.py`.

**Pieces:**
- a simulator written from the published rules;
- a scorer for F and R against the naive policy and a perfect-hindsight plan;
- scenarios from Model 1's hazards, with shock effects measured on weeks 1 to 208;
- a two-stage stochastic linear program per material.

**Shock effects used in the scenarios:**

| Shock | Capacity | Lead time | Price |
|---|---|---|---|
| Single supplier | Cut to 5% to 61% of normal | Unchanged | Material +15% to +31% |
| Country event | 20% on all of that country's suppliers | Unchanged | About +8% |
| Lane closure | ×0.78 | +3 weeks | Unchanged |

**Backtest:** 34 past windows, F + R points out of 70.

| Plan | All windows | Naive never runs out | Naive runs out in shock weeks |
|---|---|---|---|
| Cheapest supplier, no scenarios | 24.7 | 36.4 | 9.8 |
| Moderate (default) | 39.7 to 40.1 | 26.2 to 26.7 | 57 |
| Aggressive, with stress futures | 36.7 | 17.1 | 61.4 |

Moderate means a minimum share of 0.1, the stockout penalty at the real 5× level, and Model 1's hazards scaled up by 1.5.

## 4. Tested and not adopted

- **Overriding lithium's probability from the futures premium alone.** Rejected. At the supplier level, a premium around 4% was followed by a shock only 2 times against about 8 false alarms. Most false alarms were one-week spikes. One rose over two weeks like lithium now: natural gas ORI, week 118. The earlier "never wrong" claim used the premium averaged across a material's three suppliers. That average is far less noisy, but it is below 3% for lithium today.
- **A maximum share per supplier instead of a minimum share.** Added as `Config(s_max=...)` and left off by default. Caps of 0.6 to 0.7 scored 40.3 to 40.5 points against 40.1 for the default. That's within the seed-to-seed noise of about half a point.

## 5. Open questions for the official scorer

After Phase 2, `score_local.py` scores public-window submissions locally against the released truth, to compare with Kaggle:

- **Diversification weighting.** Do rows count equally, or are they weighted by order units? The E file tests this. Expect about 59.4 if rows count equally and 62.33 if they are weighted.
- **Starting stock.** Is it the sum of the first 5 weeks' demand, or 5 times the average? The scorer prints both.
- **Fill-rate skill when naive meets all shock-week demand.** The local scorer counts it as 0.

## How to rebuild

```
cd "Code Base/pipelining"
python run_pipeline.py                                   # default: iter4 with the news layer
python run_pipeline.py --model1-iter iter3               # without news (reproduces C)
python score_local.py ../../submissions/<file>.csv       # after Phase 2: local score on weeks 209-221
```
