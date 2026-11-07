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
| Single supplier | capacity < 70% of normal | 7 | 2-11 wk | capacity 5-61%; all 3 suppliers' spot +15-35% | that supplier's futures premium (futures_4w / spot - 1) climbs 1-4 wk ahead; **above 0.06 it preceded a shock 13 of 13 times**; news shortage/export-ban stories |
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
| Valdmar cobalt export-ban chain | "ministry calls cobalt a strategic priority" (wk 219, 221), "exporters report slower license approvals" (wk 220). VAL cobalt futures premium 0.073 → 0.103 → 0.116 over wk 219-221 | Premium above 0.06 preceded a shock 13 of 13 times; real export-ban chains showed the premium spike, fizzles didn't |
| Kestria tension and export talk | KES tension 10.5 → 41.2 → 53.8 over wk 219-221 (+39 in 4 wk). "Tensions rise between Kestria and neighbors" (wk 220, 221). "Lawmakers in Kestria debate limits on raw material exports" (wk 220, 221): **no material named**. KES rare_earths premium 0.042 → 0.093; KES aluminum 0.030 | The only country event (Kestria, wk 99-133) lasted 35 wk at 20% capacity. Kestria is naive's supplier for 7 materials and the cheapest for most |
| Announced tariff | KES aluminum and KES semiconductors: 10% → **35% from week 224** (`tariffs.csv`, plus "Aurelia confirms new duties on Kestria aluminum") | Tariffs count as disrupted weeks; past tariffs lasted 10-18 wk |
| Merrow congestion rising | 28-30 for most of Phase 2, then 40.4 (wk 220) and 43.8 (wk 221) | Above 50 preceded a closure 19 of 19 times. Merrow carries Kestria and Sunda Coast |

**The team was told (Model 1 plan notes) that the test period holds a shock bigger than anything in training.** Nothing in the data confirms it yet. The Kestria signals are the obvious candidate. Treat this as a hypothesis to weigh, not a fact.

## 4. What exists in the repo

**One-command pipeline:** `Code Base/pipelining/run_pipeline.py`. It runs Model 1 (iter4) → Model 2 (iter3) → Model 3 and writes two files: a main pick (Model 3 iter3, lean) and a hedge pick (Model 3 iter1 aggressive). Every script takes `--data`, so point it at the Phase 2 folder without touching `Data/`:

```
cd "Code Base/pipelining"
python run_pipeline.py --data ../../Data_new --tag _p2      # ~1-2 min, writes submissions/submission_{moderate,aggressive}_p2.csv
python backtest_full.py --data ../../Data_new --tag NAME --origins 52:209:3   # whole pipeline on past windows, ~4+ min on 8 cores
python score_local.py ../../submissions/FILE.csv --data ../../Data_new        # local score on weeks 209-221
python verify_scorer.py --data ../../Data_new     # which scorer conventions reproduce the known public scores
```

- `backtest_full.py` plans each past window from the week before it, using only data up to that week, and scores all five parts. With Phase 2 data the origins can run up to 208 (the window 209-221 is now fully observed). Results land in `Code Base/pipelining/backtests/bt_NAME.csv`.
- The simulator and the scenario LP live in `Model 3/iter1/model3.py`; iter3 imports and extends it.
- Model 1 also outputs per-source hazards and duration curves (`hazards.csv`, `hazards_sources.csv`, `duration_curves.csv`), which Model 3 samples futures from.
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
| Current iter3 | **0.096** (iter2 0.075, iter1 0.015). Lithium -0.11 and steel flat, both hurt by the 35-week Kestria event |

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
| Heavily hedged plan (min share 0.33, penalty 8x, probabilities x1.5) | **44.15 on the public leaderboard**, about 18 below the moderate plans in a mostly calm window |
| No scenarios, cheapest supplier only | 24.7 (older Model 3-only scale, out of 70) |

**Biggest lesson so far:** the largest gain came from optimizing the score's actual structure (points per window, with per-window normalizers) instead of a proxy (dollars).

## 6. Open problems: where the points are

1. **This window, not the average window.** The private window is one specific 13-week window with live signals at its origin (section 3). The 48-window backtest is a proxy for "a typical window". How should the two files split the scenarios that are plausible *now* (Valdmar cobalt ban, Kestria country event, Merrow closure, the 35% tariff from week 224, calm)? The pair was tuned for best-of-two on average; it was never chosen for this origin.
2. **Mid-window shock onsets.** The lean main pick loses 25-50 points when a big shock starts mid-window (origins 91, 94, 166). The hedge covers that by being expensive everywhere: 24.9 in calm windows.
3. **Shock endings.** The biggest source of error in both Model 1 and Model 2, e.g. the 35-week Kestria event.
4. **A shock bigger than training.** Only the long-tail duration mix and the hedge's stress futures cover it, and neither has been tested against a real one.
5. **Calibration of the sampled futures,** not just of `disruption_prob`. Do the futures Model 3 plans against put the right weight on a new shock starting inside the window? Nobody has checked this directly.
6. **Unconfirmed scorer conventions.** How diversification is averaged (all rows vs weighted by units), what fill skill scores when naive meets all shock-week demand, what "average landed cost" means, and whether starting stock is the sum of the first 5 weeks' demand or 5x the mean. With Phase 2, local scores on weeks 209-221 can be checked against known public scores (`verify_scorer.py`).

**Ideas from similar past competitions** (prompts, not instructions):
- **VN2 inventory challenge (1st place):** predict, then a simple order-up-to rule with one tuned safety factor. Here the roles flip: demand is known and supply is uncertain, so the thing to forecast is delivered units. Rough critical ratio: short ≈ 4 (5x penalty minus the purchase saved), over at the end ≈ 0.3 → q* ≈ 0.93, near 1 mid-window. Tune on points, not dollars. Appeal: fast, and built differently from the LP, so it may win different windows.
- **M5 Uncertainty:** calibrated uncertainty beat better point forecasts. Check coverage of the sampled futures: realized capacity and lead time against their 10/50/90 percentiles, and the rate of new onsets inside a window against how often they actually happened.
- **MITSUI commodity prediction:** simple linear models were hard to beat. Possible cheap gain: per-horizon shrinkage toward carry-forward, fit on past windows. Make sure every estimated quantity is refit on weeks 1-221.

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
