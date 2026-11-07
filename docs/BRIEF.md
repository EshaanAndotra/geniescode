# Brief for building agents

As of Sat Nov 7, 2026, after the Phase 2 data release. Deadline: **Sun Nov 8, 12 PM Central**.

Read this file and `docs/competition.md` (the official rules). That's the full context. `README.md` has run instructions only.

This brief gives facts, evidence and open problems. **It doesn't prescribe an architecture.** The current three-model pipeline is a resource, not a spec: any part of it, including the Model 1 → 2 → 3 split itself, can be replaced if the evidence says so.

---

## 1. The goal in one paragraph

Two CSV files decide the result. Only their rows for weeks 222-234 matter, planned with data up to week 221. Kaggle keeps the **better** of the 2 selected files. So the target is the expected value of max(score of file A, score of file B) on that one 13-week window. Score = 30 P + 40 F + 30 R (see `competition.md`); per row group that's 18 Brier, 12 price, 40 cost, 21 shock-week fill, 9 diversification. Demand is known. All the uncertainty is supply: capacity, lead time and landed cost per supplier per week.

## 2. Verified facts about the data

These were checked against weeks 1-208, and the Phase 2 weeks behave the same way. Treat them as ground truth; re-verify only if something contradicts them.

**Shock structure.** Four sources rebuild the `disrupted` column exactly (2,080 of 2,080 rows in weeks 1-208):

| Source | Detected by | Count, wk 1-208 | Length | Effect | Early warning |
|---|---|---|---|---|---|
| Single supplier | capacity < 70% of normal | 7 | 2-11 wk | capacity 5-61%; all 3 suppliers' spot +15-35% | that supplier's futures premium (futures_4w / spot - 1) climbs ahead; **every clean run above 0.06 (6 of 6, runs of 1-5 wk) was followed by an onset the week after the run ended** (time from first week above 0.06 to onset: 2, 2, 2, 3, 4, 5 wk). The older "13 of 13" counts supplier-weeks, not runs. 2 of 8 single-supplier onsets had no premium warning; news shortage/export-ban stories |
| Country | 2+ suppliers of one country hit together | 1 (Kestria, wk 99-133) | 35 wk | capacity 20%; spot +5-20% | news sanctions story about 7 wk ahead; tension, fx, PMI |
| Lane closure | transit >= 1.5x normal | 4 (Merrow 3x, Atlantic 1x) | 2-8 wk | capacity 75-80%; lead +3 to +4 wk; container rate 1.6-3x | congestion about 1 wk ahead; **above 50 it preceded a closure 19 of 19 times**; strike or clash stories |
| Tariff | tariff above base | 3 | 10-18 wk | landed cost only | announced 2-8 wk ahead in `tariffs.csv` |

- The 31 material-level episodes are only 15 real events. 3 of the 15 had no warning at all: policy reversals, such as "Halvard abruptly suspends aluminum export licenses" at week 195.
- **Calm weeks:** capacity = 0.984 x normal (sd 0.012, no autocorrelation; Phase 2 calm weeks also give 0.984 / 0.013). Lead time is exactly the base lead.
- **Prices:** calm prices are a random walk, and futures carry no information when the premium is under 3%. Benchmark price = exact mean of the 3 suppliers' spot prices.
- **Landed cost** = spot x (1 + tariff) + b x container_rate + c per supplier, with b = 0 on NORTH, SOUTH and RAIL. Average error 0.06%.
- **The cheapest supplier is usually the riskiest.** For 9 of 10 materials it ships through the Merrow Strait (Kestria for 7, Sunda Coast for 2).
- **The naive policy** (`sample_submission.csv`) orders each week's demand, 100% from the largest-capacity supplier: Kestria for 7 materials, Sunda Coast for specialty_glass, Orinoque for copper and natural_gas. Demand averages about 72% of the largest supplier's capacity.
- **In calm windows the naive-to-hindsight cost gap is small**, about 1.5 weeks of purchases per material, because naive already buys mostly from the cheapest supplier. Hedging in a calm window burns F points fast.

**News is templated.** Headlines follow fixed chains that escalate before each shock type:

| Chain | Typical headlines in order | Rumor ending | Outcome, wk 1-208 |
|---|---|---|---|
| Shortage (supplier) | "producers flag maintenance delays" → "Accident halts major ... site" → "production falls amid shortage" | none seen | 3 of 3 became shocks, 3-4 wk lead |
| Export ban (supplier) | "Lawmakers in X debate limits on raw material exports" / "ministry calls M a strategic priority" → "exporters report slower license approvals" → "signals M export curbs within weeks" | "officials rule out new restrictions" | 3 of 4 |
| Strike (lane) | "Port labor talks drag on" → "Union sets strike date" | "Deal reached, ... averted" | 2 of 6 |
| Clash (lane) | "Tensions rise between X and neighbors" → "Insurers raise war risk premiums" | "Calm returns near the ..." | 1 of 3 |
| Sanctions (country) | "Companies told to prepare for X sanctions" → "Sanctions vote on X scheduled" | none seen | 1 of 1, 7 wk lead |

In the export-ban chains, the futures premium separates real chains from fizzles. Valdmar natural gas at week 151 had a premium of 0.07, then 0.22 at week 152, and became an 11-week shock at week 153. Orinoque copper ran 0.08 → 0.24 over weeks 128-131 and became a shock at week 132. The two chains that fizzled (Valdmar natural gas, wk 48-50; Valdmar lithium, wk 209-210) stayed flat near 0.

## 3. What weeks 209-221 showed, and the state at week 221

**Shocks in Phase 2:**
- **Southern route strike, wk 212-214.** Union strike date announced at week 210. It hit Orinoque for lithium, cobalt, copper and natural_gas: capacity 0.80, lead +3 weeks, SOUTH congestion peaking at 79.3.
- **Kestria lithium shortage, wk 215-220.** Maintenance delays (wk 207-210) → accident (wk 212) → shortage (wk 215). Capacity 0.30, lithium price +25%. At week 221: capacity is back to normal, and the news says "Disruption in Kestria lithium supply eases".
- Nothing is disrupted at week 221.

**Signals at week 221 (the planning origin):**

| What | Evidence | Past comparison |
|---|---|---|
| Valdmar cobalt export-ban chain | "ministry calls cobalt a strategic priority" (wk 219, 221), "exporters report slower license approvals" (wk 220). VAL cobalt futures premium 0.073 → 0.103 → 0.116 over wk 219-221 | Every clean premium run above 0.06 became an onset (6 of 6; run length 2-5 wk); VAL cobalt is at week 3 of its run; real export-ban chains showed the premium spike, fizzles didn't |
| Kestria tension and export talk | KES tension 10.5 → 41.2 → 53.8 over wk 219-221 (+39 in 4 wk). "Tensions rise between Kestria and neighbors" (wk 220, 221). "Lawmakers in Kestria debate limits on raw material exports" (wk 220, 221): **no material named**. KES rare_earths premium 0.042 → 0.093; KES aluminum 0.030 | The only country event (Kestria, wk 99-133) lasted 35 wk at 20% capacity. Kestria is naive's supplier for 7 materials and the cheapest for most |
| Announced tariff | KES aluminum and KES semiconductors: 10% → **35% from week 224** (`tariffs.csv`, plus "Aurelia confirms new duties on Kestria aluminum") | Tariffs count as disrupted weeks; past tariffs lasted 10-18 wk |
| Merrow congestion rising | 28-30 for most of Phase 2, then 40.4 (wk 220) and 43.8 (wk 221) | Above 50 preceded a closure 19 of 19 times. Merrow carries Kestria and Sunda Coast |

**The team was told (early planning notes; the source isn't in the repo) that the test period holds a shock bigger than anything in training.** Nothing in the data confirms it yet. The Kestria signals are the obvious candidate. Treat this as a hypothesis to weigh, not a fact.

### 3b. How the current pipeline responds at week 221 (builder run, Nov 7 ~3 PM)

Fallback picks built from Phase 2 data and committed: `submissions/submission_moderate_p2.csv` (main) and `submission_aggressive_p2.csv` (hedge). Both validated.

| Signal | disruption_prob and orders | price_forecast |
|---|---|---|
| 35% KES tariff from week 224 | **Responds.** aluminum and semiconductors 0.16 / 0.32 / 1.00 at weeks 222-224, decaying to 0.73 by week 234 (past tariffs lasted 10-18 wk). Orders front-load 1.9-2.4x demand before week 224; KES share drops after | n/a (tariffs hit landed cost, not the benchmark) |
| Valdmar cobalt export ban | **Responds.** 0.44 / 0.97 / 0.99 at weeks 222-224. Main orders 2.2x demand at week 222 and keeps VAL's share at 0.11-0.33 | **Slips.** +18% only for weeks 223-225, then back to calm from week 226 while P(active) is still 0.84-0.97. Model 2 cuts onset jumps at h = 4. Past bans kept the benchmark up 21-40% for 7+ weeks |
| Kestria rare earths (premium 0.093) | **Responds.** Material p 0.99 from week 224 | **Slips** the same way (+20% for 2 weeks, then calm) |
| Kestria country risk (tension 54, +39 in 4 wk; export-limit talk, no material named) | **Slips.** The country-event probability is 0.004-0.018. The country model learned from one event, whose precursor was a sanctions chain, which is absent now | n/a |
| Merrow congestion 43.8 + clash headlines | Responds via the clash story: lane p 0.41 peak at week 224. Analog: wk 171-173 (tension 24 → 47 → 59, congestion 38 → 55 → 79, then an 8-wk closure). Fizzled clash chains had tension 35-38, congestion 40-45. Now: tension looks like a closure, congestion like a fizzle | n/a |

**Stress test of the private rows** (`Code Base/pipelining/stress_private.py`, about 2 s; hand-built futures for weeks 222-234; points = 40 F + 21 fill skill + 9 diversification, out of 70; P is identical for both files):

| Scenario | Main | Hedge |
|---|---|---|
| Calm (tariff only) | 26.7 | 15.2 |
| VAL cobalt ban only | 27.0 | 16.7 |
| Cobalt + KES rare earths (the model's view) | 57.5 | 51.4 |
| Merrow closure 223-230 | 66.7 | 64.3 |
| KES country event 226-234 / 223-234 | 65.5 / 63.9 | 62.7 / **65.7** |
| Merrow at 40% capacity 223-234 (bigger than training) | 55.3 | **61.4** |
| Atlantic closure 225-230 | 26.1 | 7.4 |
| Southern route closure 225-230 | 33.0 | **36.0** |

What it shows: **at this origin the hedge rarely adds anything.** The main pick wins 9 of 12 scenarios; the hedge wins three, by 2-6 points. Both files score worst (about 27 / 70) in the futures where naive doesn't get hurt (calm, cobalt-only, Atlantic), because naive is close to hindsight there. That makes the second file's slot the biggest open lever. In the cobalt-only and rare-earths-only scenarios the main pick's fill is 1.00: its shares on the about-to-be-banned supplier don't cause shortfalls, because total orders are inflated (cobalt 2.2x demand at week 222) and unfilled units are cancelled at no cost while still counting toward diversification. Its loss there is F (0.48 cobalt-only), not fill. Caveats: the futures carry no price noise and use the sum-of-5 starting stock.

**Starting-stock convention.** The main pick's 209-221 rows equal the 68.40 public file. Locally they score 71.71 with sum-of-first-5-weeks stock but 67.98 with 5 x mean demand. `verify_scorer.py` also ranks 5 x mean first, though no convention reproduces every public score exactly (best max error 1.32). It looks like a small detail, but F moves 0.894 → 0.801 between the two readings. Whether Model 3 *plans* with the right convention hasn't been checked.

## 4. What exists in the repo

**One-command pipeline:** `Code Base/pipelining/run_pipeline.py`. It runs Model 1 (iter4) → Model 2 (iter3) → Model 3 and writes two files: a main pick (Model 3 iter3, lean) and a hedge pick (Model 3 iter1 aggressive). Every script takes `--data`, so point it at the Phase 2 folder without touching `Data/`:

```
cd "Code Base/pipelining"
python3 run_pipeline.py --data ../../Data_new --tag _p2      # ~1-2 min, writes submissions/submission_{moderate,aggressive}_p2.csv
python3 backtest_full.py --data ../../Data_new --tag NAME --origins 52:209:3   # whole pipeline on past windows, ~4+ min on 8 cores
python3 score_local.py ../../submissions/FILE.csv --data ../../Data_new        # local score on weeks 209-221
python3 verify_scorer.py --data ../../Data_new     # which scorer conventions reproduce the known public scores
```

- `backtest_full.py` plans each past window from the week before it, using only data up to that week, and scores all five parts. With Phase 2 data the origins can run up to 208 (the window 209-221 is now fully observed). Results land in `Code Base/pipelining/backtests/bt_NAME.csv`.
- The simulator and the scenario LP live in `Model 3/iter1/model3.py`; iter3 imports and extends it.
- Model 1 also computes per-source hazards and duration curves, which Model 3 samples futures from. They're computed in memory on every run; old CSV exports from week 208 were removed from the repo.
- `submissions/probes/` holds files built to pin down unconfirmed scorer conventions from their public scores (`predictions_F.csv` lists the predicted score under each convention). Whether they've been uploaded isn't recorded. Ask before using upload slots on them.

**Current numbers.** Public leaderboard (Phase 1 files): **68.40 best** (the current pipeline's main pick), 2nd on the leaderboard before Phase 2. On 48 past windows (origins 52-193, every 3 weeks):

| Plan | All 48 | Calm (28) | Naive runs short (20) | Worst |
|---|---|---|---|---|
| Main: Model 3 iter3 + sharpened Model 1 | 52.6 | 47.6 | 59.7 | 4.5 |
| Hedge: Model 3 iter1 aggressive | 43.5 | 24.9 | 69.4 | 7.7 |
| **Better of the two, per window** | **57.2** | 47.6 | 70.7 | 27.3 |

The main pick is the better one in 39 of 48 windows; the hedge rescues windows where a big shock starts mid-window. The windows overlap, and every design choice was made after seeing all of weeks 1-208, so these numbers are optimistic.

## 5. Evidence ledger: what's been tried

This is evidence, not a list of bans. Retrying something is fine when you have a new reason; say what the reason is.

**Disruption probability (Model 1).** Brier skill, 26 origins, weeks 104-204, unless noted.

| Tried | Result |
|---|---|
| Duration curves (survival on 15 events, Weibull + 15% long tail) + calm base rate | 0.33, most of the skill |
| + scheduled announced tariffs | 0.36 |
| + onset signals per source (premium, tension, congestion; logistic, "within k weeks" labels) | **0.435** |
| "Exactly k weeks" labels instead of "within k" | 0.37 vs 0.71 on 4 test weeks |
| One material-level model, country events not split out | Tension from one-supplier shocks pushed whole countries up (lithium 0.98 with nothing behind it) |
| Platt calibration | 0.27, hurt |
| Gradient boosting | No gain; the limit is the number of events |
| News story layer (template chains, hit rates learned from closed stories) | Backtest neutral (0.420 vs 0.420); **+2.96 on the public leaderboard** |
| Sharpening sigmoid(1.5 x logit p) | Mean Brier skill 0.158 → 0.204 on the 48 windows (P is clipped at 0 per window) |
| Lithium override from futures premium alone at 4% | About 2 real warnings vs 8 false alarms |

Skill fades with lead time: about 0.7-0.9 at 1-4 weeks, about 0.15 at 9-13 weeks. Most remaining error is from **when shocks end**, and from no-warning shocks.

**Price forecast (Model 2).** MAPE skill vs carry-forward.

| Tried | Result |
|---|---|
| Futures-implied forecast | -0.007, worse than carry-forward |
| Calm mean reversion toward the 26-week average (pull grows with horizon) | +0.022 |
| In-shock: median of a two-scenario mix (elevated vs pre-shock price), weighted by P(still on) | +0.071, the biggest piece |
| Hard switch: add the jump only if P(shock) > 0.5 | 0.053 vs 0.096 for the median of the mix |
| Onset jumps beyond 4 weeks ahead | Added calm-week error |
| Current iter3 | **0.096** (iter2 0.075, iter1 0.015; both since removed from the repo). Lithium -0.11 and steel flat, both hurt by the 35-week Kestria event |
| **iter4 (adopted, Nov 7):** keep the onset jump past h = 4 while P(shock) > 0.5, but only when an onset is already likely within 4 weeks; otherwise identical to iter3 | 53 backtest origins: 0.0695 vs 0.0698 (neutral). Premium-chain origins: **0.1019 vs 0.0877**. Calm-week MAPE 4.806% vs 4.789%, all from one false alarm (specialty_glass, origin 103). 89 of 9,930 rows change. At origin 221: cobalt holds +18% → +11% through week 229, rare earths through 230 (9 private rows). Jumping at every horizon (h = 13 everywhere) adds 3x the calm error |
| Not yet tried: source-specific jump size | The median jump (0.179 log) is pulled down by 7 small country-event onsets at week 99. The 8 single-supplier onsets have a median of about 0.20; the 3 export bans average 0.27. Rests on 8 events |
| Current pipeline on the newly observed window 209-221 (origin 208, Phase 2 data) | **-0.034**, below carry-forward. The same run: Brier skill 0.52, F 0.89, fill skill 1.0, total 71.7 locally. One window, so weak evidence, but nobody has looked into it yet |

**Ordering (Model 3).** Average full score on the 48 windows.

| Tried | Result |
|---|---|
| Two-stage stochastic LP minimizing dollar cost, min share 0.1, stockout penalty 5x, hazards x1.5 (iter1 moderate) | 47.3 |
| + plan calm weeks at 0.975 x normal capacity | 48.2 |
| + no forced minimum share | 49.9 |
| iter3: minimize **points lost** (dollars x 40 / that future's naive-hindsight gap + shock shortfall x 21 / naive shortfall), per-material diversification chosen by exact scoring, orders capped at calm mean - 2.5 sd | 51.5 |
| + Model 1 sharpening | **52.6** |
| iter3 with all hazards x2.5 | 51.1 |
| iter3 with stress futures | 50.4 (the iter1 aggressive hedge pairs better) |
| Max share per supplier instead of min share | Within seed noise |
| **iter4 (adopted, Nov 7):** iter3 planned with starting stock = min(sum of first 5 weeks, 5 x mean demand). The scorer's convention is unresolved, and iter1/iter3 plan with sum5 | Mean over 13 week-221 stress scenarios, out of 70. Scored with sum5: plan_sum5 52.16, plan_5avg 51.01, **plan_min 52.11**. Scored with 5avg: plan_sum5 50.17, plan_5avg 52.20, **plan_min 52.22**. The min plan is never more than 0.05 behind the plan for the right rule. Planning for the wrong rule costs 1-2.6 per scenario (4.6 in the Atlantic case). `backtest_full.py` scores sum5 only, so it can't show the 5avg side |
| Heavily hedged plan (min share 0.33, penalty 8x, probabilities x1.5) | **44.15 on the public leaderboard**, about 18 below the moderate plans in a mostly calm window |
| No scenarios, cheapest supplier only | 24.7 (older Model 3-only scale, out of 70) |

**Second file for the private window** (Nov 7, ~4 PM). Main held fixed at `submission_moderate_p2v2.csv`. Gain = E[max(main, B)] - E[main] in total points out of 100, on 500 generator futures at origin 221 (fresh seed, calm price noise) and on the 13 stress scenarios, under both stock rules. The harness matches `score_local` to 1e-3.

| Candidate B | Generator gain | Stress gain |
|---|---|---|
| Current hedge (iter1 aggressive) | 0.9 | 0.5 |
| Merrow / KES-country / calm specialists, own P columns | 1.0 / 0.8 / 2.1 | 0.8 / 4.7 / 3.0 |
| Onset hazard x0.5 / x1.5 / x2.5 | 1.7 / 1.3 / 1.6 | 1.9 / 2.2 / 2.2 |
| **E[max]-directed B:** from the x0.5 start, keep the futures where B beats main, replan B on them (orders + own P), repeat twice | **3.7** | **5.1** |

- **Own P columns help.** A B whose disruption_prob matches the futures it bets on beats copying main's P in 9 of 12 comparisons, by up to +3.4.
- **The winning B is a lean "Merrow and Kestria stay open" bet:** Merrow-lane share 0.78 vs main's 0.46, and disruption_prob about 0.01 on the Merrow-only materials. It scores 39 vs main's 25 in calm and cobalt-only futures, and 59 vs 35 in a South closure. It collapses in Merrow and Kestria shocks, where main covers.
- **No sign of overfitting** to the planning futures: pool gain 3.62 vs eval gain 3.68.
- **Trade-off:** in the bigger-than-training Merrow scenario the pair's max falls from 61.4 (with the old hedge) to 55.9. Being worse off overall would take P(bigger-than-training shock) above about 1/3.
- **Pending:** a historical check (`backtest_pair.py`), and re-planning main on its own cluster with bigger-than-training futures added (alternating optimization).
- Old ledger note corrected: the "all hazards x2.5" result was effectively onset-only, since the generator's multiplier applies only to new onsets.

**Historical check of that B (`portfolio/backtest_pair.py`, 18 origins, each window planned from its origin, scored on real history):** it **fails**. Overall, max(main, B) = 49.1 vs max(main, current hedge) = 55.6. B gains +1.6 to +6.3 in the calm, premium-chain and recent categories, but collapses in the Merrow (166/169/172: 29 vs 68) and Kestria (94/97: 45 vs 67) windows. Worst window 5.6 vs 23.3. Why: B hedges main *as the generator sees the world*. In the windows that mattered historically the signals were weak at the origin, so main was lean and got blindsided; B was leaner still. The old hedge's value is insurance against what the generator misses (4 of 18 windows). Not promoted. Next: alternating optimization, with main specialized for shocks using generic blindside futures whose weight comes from calibration evidence, not tuning.

**Number of sampled futures (infra, Nov 7).** The final plan for weeks 222-234 is one LP solve. Stress mean over 13 scenarios, out of 70: at **n_scen = 200** (the default) seeds 0-4 give 47.97-53.45 (sd 2.04). At **1000**, 51.72-53.37 (sd 0.87, mean **+1.4**). The spread sits almost entirely in the Atlantic and South closure scenarios (rare lane events); without them the 1000-future spread is 0.13. **Final picks must use n_scen >= 1000** (one solve takes about 70 s; 2000 is about 3-5 min). Rare lane events may still need more futures or stratified sampling. Backtests at 200 carry this noise per window; it averages down over many windows.

**Biggest lesson so far:** the largest gain came from optimizing the score's actual structure (points per window, with per-window normalizers) instead of a proxy (dollars).

## 6. Open problems: where the points are

1. **This window, not the average window.** The private window is one specific 13-week window with live signals at its origin (section 3). The 48-window backtest is a proxy for "a typical window". How should the two files split the scenarios that are plausible *now* (Valdmar cobalt ban, Kestria country event, Merrow closure, the 35% tariff from week 224, calm)? The pair was tuned for best-of-two on average; it was never chosen for this origin.
2. **Mid-window shock onsets.** The lean main pick loses 25-50 points when a big shock starts mid-window (origins 91, 94, 166). The hedge covers that by being expensive everywhere: 24.9 in calm windows.
3. **Shock endings.** The biggest source of error in both Model 1 and Model 2, e.g. the 35-week Kestria event.
4. **A shock bigger than training.** Only the long-tail duration mix and the hedge's stress futures cover it, and neither has been tested against a real one.
5. **Calibration of the sampled futures,** not just of `disruption_prob`. Do the futures Model 3 plans against put the right weight on a new shock starting inside the window? Nobody has checked this directly.
6. **Unconfirmed scorer conventions.** How diversification is averaged (all rows vs weighted by units), what fill skill scores when naive meets all shock-week demand, what "average landed cost" means, and whether starting stock is the sum of the first 5 weeks' demand or 5x the mean. With Phase 2, local scores on weeks 209-221 can be checked against known public scores (`verify_scorer.py`).

**Ideas from similar past competitions** (prompts, not instructions):
- **VN2 inventory challenge (1st place, arXiv 2601.18919, read in full):** predict demand, then order up to a critical-ratio quantile with a safety term. The paper reports no tuned value and no ablation, so it doesn't show how much the safety factor itself mattered. Its 13.2% gain is against the organizers' moving-average benchmark, not other teams. VN2 had fixed lead times, unlimited supply and no end-of-horizon effect, and the paper lists simulator-tuned policies (what Model 3 iter3 already is) as future work. If borrowed, the roles flip: demand is known and supply is uncertain, so forecast delivered units. Rough critical ratio here: short ≈ 4 (5x penalty minus the purchase saved), over at the end ≈ 0.3 → q* ≈ 0.93, near 1 mid-window. Tune on points, not dollars. The only clear appeal is that it's built differently from the LP, so it may win different windows.
- **M5 Uncertainty (claims unverified; primary sources weren't reachable):** the reported lesson is that calibrated uncertainty beat better point forecasts. The check stands on its own either way: compare realized capacity and lead time against the sampled futures' 10/50/90 percentiles, and the rate of new onsets inside a window against how often they actually happened.
- **MITSUI commodity prediction (from talk slides and summaries, unverified):** simple linear models were hard to beat. Possible cheap gain: per-horizon shrinkage toward carry-forward, fit on past windows. Make sure every estimated quantity is refit on weeks 1-221.

**Research literature mapped to our decisions** (targeted search, Nov 7, ~3:30 PM; read from abstracts and summaries, not full texts). A search for a closer past competition (Kaggle, hackathons, simulator-scored ordering challenges) found none with published solutions.

| Idea | Source | What it would change here | Status |
|---|---|---|---|
| **Decision-focused scenario generation:** tune the generator of futures for the quality of the downstream decision, not the accuracy of the forecast | Hornewall, Delannoy-Pavy, Leclère, Homem-de-Mello, NeurIPS 2025, *Decision Focused Scenario Generation for Contextual Two-Stage Stochastic LP* (convex 2-stage problems only; ours is an LP, so it fits); Islip et al., arXiv 2502.05349 | The next step after iter3's lesson (optimize points, not dollars), applied one level up. Tune the scenario generator's knobs (onset-hazard multiplier and duration multiplier, per source type) to maximize **backtest points**, not Model 1's Brier. A cheap version is a small grid via `backtest_full.py`; no neural network needed | Untested. Closest evidence: all hazards x2.5 gave 51.1 vs 51.5. Onsets alone and per-type multipliers haven't been tried. Risk: 48 overlapping windows and few events, so overfitting is easy |
| **Ordering when a disruption is anticipated:** when the start is known (or forecast) but the length is random, the optimal policy builds stock before the onset, sized by the duration distribution; with advance supply information, the optimal base-stock level depends on the information state | *Optimal ordering policies when anticipating a disruption in supply or demand*, EJOR (1992); EPFL work on advance supply information (state-dependent (s,S) optimal); *EOQ with supply disruptions under different advance information regimes* (2023) | The stochastic LP already does this implicitly, **but only as well as its futures place the onset week.** The premium-run rule (onset the week after a run ends; 6 of 6 clean runs) is exactly the information state these papers say the order level should depend on | **Checked (BUILDER-2, T3):** the generator gets *whether* right (P(onset within 13 wk) 0.98-1.00 once a run starts) but is overconfident on *when*. Of 18 past origins, 5 fell outside the 80% band, all late (copper 132, lithium 215). At origin 221 it leaves out the late tail: P(onset at 225+) is 0.03 vs 0.23 by the rule for VAL cobalt, 0.05 vs 0.41 for KES RE, with no fizzle mass. An early onset is the cheap error for orders (5x stockout), so the plan's futures are kept as they are; a calibrated mix is being tested for the P columns only |
| **Mitigation vs inventory vs contingent rerouting:** rare-but-long disruptions favor sourcing mitigation (diversifying away) over inventory; frequent-short ones favor inventory; contingent rerouting needs a backup that can ramp up | Tomlin, *Management Science* 52(5), 2006 | Our bans, country events and the 35-week Kestria event are long relative to 1-6 wk lead times, so the remedy is a **sourcing shift**. Lane closures (2-8 wk, capacity still about 78%) favor a **buffer**. Contingent rerouting isn't available: orders are fixed up front and backup capacity can't ramp. A Kestria/Merrow specialist for the second file should mostly shift shares toward NORTH/ATLANTIC/RAIL suppliers, not stockpile | Design guidance for the second file's menu |
| News-precursor event models (Hawkes processes, LLMs on news) | Yamin et al., arXiv 2211.12091 (Hawkes change-point *detection* on order data, not forecasting); arXiv 2604.01298 (LLMs on news for probabilistic disruption forecasts) | Little to gain. Our news is templated, and the story-chain layer already works as a hand-built marked event model. An LLM adds cost and noise, not information | Not recommended |

Where innovation is most likely to pay, in my view: (1) choosing the second file for **this** origin by E[max] (open problem 1), and (2) decision-focused tuning of the scenario generator. Neither is a move a generic "look at past competitions" prompt would produce. Both also make a clear story for the video.

## 7. Ground rules

**Freedom.** Replace, rewrite or bypass any model, or the whole decomposition, if the evidence supports it. New approaches go in new files or folders; don't edit the current iterations in place.

**Hard limits:**
- **No leakage.** A plan for a window starting at week w uses only data up to week w-1. The private window is planned from week 221.
- **Measure, don't assert.** Judge every change with `backtest_full.py` (or an equivalent that plans each window from its own past). Report its effect on the **best-of-two** with the other final pick, plus all-window, calm and shock-window means and the worst window. The public leaderboard no longer predicts the private one.
- **Keep the safety net.** The current pipeline's Phase 2 outputs (`submissions/*_p2.csv`) are the fallback picks. Don't overwrite them or any file listed in the README scoreboard. Write new files with a distinct `--tag`.
- **No uploads, final selection or git commits.** The user does those.
- **Validate every file you hand over:** 260 rows, ids `w{week}_{material}`, no blanks, probabilities in [0, 1], price > 0, orders >= 0 and whole, shares summing to 1 with 0 for non-suppliers.
- **Time.** Everything must be decided well before Sun 12 PM CT, and the video needs time too. Prefer an approach that can be backtested in a few hours over one that can't be validated in time.

**When you report back, include:**
1. What you changed and why: the hypothesis, in one or two lines.
2. Backtest numbers in the format above, compared with the current pair (57.2 best-of-two; main 52.6, hedge 43.5).
3. What you did **not** check, and anything that surprised you.
4. Any file produced: path and validation result.
