# DSDG Hackathon 2026: Supply Chain Disruptor

This is the team's one-page guide: how to build the final submissions, what each model does, what every Kaggle submission scored, and the evidence behind each choice.

**Deadline:** Sunday Nov 8, 12 PM Central. Pick the 2 final submissions on Kaggle by then.

---

## 1. Build the final submissions

When the Phase 2 data is out (weeks 1 to 221), copy the new CSVs into `Data/`, keeping the same file names. Then run:

```
cd "Code Base/pipelining"
python run_pipeline.py
```

This runs Model 1, then Model 2, then Model 3, and writes two files:

- `submissions/submission_moderate.csv` is the **main pick**: Model 3 iter3, a lean plan that wins calm windows.
- `submissions/submission_aggressive.csv` is the **hedge pick**: Model 3 iter1 aggressive, a heavily hedged plan that wins when a big shock starts mid-window.

Both files share the same `disruption_prob` (Model 1 iter4, sharpened) and `price_forecast` (Model 2 iter3). Kaggle ranks a team by the **better** of its 2 selected submissions, so the two plans are chosen to win in different futures (section 3a).

The private window, weeks 222 to 234, is planned from week 221. Both files pass every format rule before they're written. The run takes 1 to 2 minutes. Use `--tag _x` to add a suffix to the file names. Before uploading, read the checks it prints:

- which data weeks it found;
- which shocks are active at the last week;
- which news stories are open;
- which suppliers' futures premium is above 3%;
- each plan's order volume.

Upload both files, then **select both as your final submissions**. If you don't select them, Kaggle picks your 2 best public scores. After Phase 2, public scores reflect weeks that are already in the data, so they don't predict the private leaderboard.

**Notebook route (Databricks).** Run `Model 1/iter3/model1_run.ipynb`, then `Model 2/iter3/model2_run.ipynb`, then `Model 3/iter2/model3_run.ipynb`. Each notebook finds the data and the other models on its own. This route uses Model 1 iter3, without the news layer, and Eshaan's Model 3.

**Checking a submission locally.** After Phase 2, this scores a file on weeks 209 to 221 and prints every score part:

```
python score_local.py ../../submissions/<file>.csv
```

Compare the result with the file's Kaggle public score. A match confirms the simulator. `verify_scorer.py` does this for every file with a known public score at once: it tries all combinations of the open scorer details (section 6) and ranks them by how well they reproduce the Kaggle scores.

**Backtesting the whole pipeline.** `python backtest_full.py --tag name` plans 48 past windows the way the private window will be planned (each from the week before it, data up to that week only), scores all five parts, and writes `backtests/bt_name.csv`. It takes about 4 minutes on 8 cores.

---

## 2. The competition in brief

Halcyon Industries buys 10 materials, each from 3 of 6 countries. Every row of the submission is one material in one week, from week 209 to week 234, so there are 260 rows. Each row holds a forecast and an order.

| Score part | Weight | Column | Compared against |
|---|---|---|---|
| Disruption forecast (Brier skill) | 18% | `disruption_prob` | each material's historical disruption rate |
| Price forecast (MAPE skill) | 12% | `price_forecast` | carrying the last benchmark price forward |
| Financial F | 40% | `order_units`, `share_*` | naive policy, and a perfect-hindsight plan |
| Fill-rate skill in shock weeks | 21% | same | the naive policy's fill rate |
| Diversification | 9% | `share_*` | 1 - HHI, where an even 3-way split scores 1 |

**Simulator rules.**
- Each window starts with 5 weeks of demand in stock.
- Each country fills its share of an order up to its capacity that week. Unfilled units are cancelled and not charged.
- Filled units are paid at that week's landed cost and arrive after that week's lead time.
- Holding costs 2% of the average landed cost per unit-week. Each unit of unmet demand costs 5 times the average landed cost.
- Stock and units in transit at the end sell back at 70%.

**Leaderboards.** The public leaderboard covers weeks 209 to 221. The private one, weeks 222 to 234, decides 80% of the final placement. The 3-minute video decides the other 20%.

**Limits.** 5 submissions per team per day, resetting at midnight UTC, which is 6 PM Central.

The naive policy is `Data/sample_submission.csv`, which scores 0. It orders each week's demand, 100% from the largest-capacity supplier: Kestria for 7 materials, Sunda Coast for glass, and Orinoque for copper and natural gas.

---

## 3. Scoreboard: every submission

All of these were built on Phase 1 data (weeks 1 to 208) and scored on public weeks 209 to 221.

| Submission file | Model 1 | Model 2 | Model 3 | Public score |
|---|---|---|---|---|
| `submissions/submission_moderate_iter5.csv` | **iter4 (news), sharpened** | **iter3** | **iter3** (main pick) | **68.40** (best) |
| `submissions/submission_aggressive_iter5.csv` | iter4 (news), sharpened | iter3 | iter1 aggressive (hedge pick) | not uploaded |
| `submissions/submission_moderate_wild.csv` | iter4 (news) | iter2 | iter1 moderate | 65.29 |
| `submissions/C_full_moderate.csv` | iter3 | iter2 | iter1 moderate | 62.33 |
| `Code Base/pipelining/Model 3/iter2/outputs/submission_B.csv`, as committed in `ebacbf8` | iter3 | iter3 | iter2 plan B (aggressive) | 44.15 |
| `submissions/E_divtest_zero_rows_single.csv` | same as C | same as C | C, but zero-order rows sent 100% to the largest supplier (scorer test) | not recorded |
| `submissions/submission_aggressive_wild.csv` | iter4 (news) | iter2 | iter1 aggressive | not recorded |
| `submissions/D_full_aggressive.csv` | iter3 | iter2 | iter1 aggressive | not recorded |
| `submissions/A_model1_model2iter2.csv` | iter3 | iter2 | sample orders | not recorded |
| `submissions/B_model1_model2iter1.csv` | iter3 | iter1 | sample orders | not recorded |

What the scores show:

- **Iteration 5 adds 3.11 on top of that, for a best of 68.40.** The 68.40 and 65.29 files differ in all three columns: sharpened Model 1, Model 2 iter3 instead of iter2, and Model 3 iter3 instead of iter1 moderate. So the public score alone can't split the gain between them. On the 48-window backtest the same change was +5.3, mostly from Model 3 (section 3a). It is the first out-of-sample check of the new ordering model, on weeks no design choice saw.
- **The news layer adds 2.96.** The 65.29 and 62.33 files differ only in Model 1.
- **The heavily hedged plan B scored about 18 points below the moderate plans** on the same weeks. Those weeks rewarded buying cheaply over heavy insurance.
- **E against C tests how Kaggle counts diversification.** Their orders are identical, so only diversification can differ. A score of about 59.4 means every row counts equally, including zero-order rows. A score of 62.33 means rows are weighted by units.

**The current pipeline is iteration 5** (section 3a): Model 2 iter3 for `price_forecast` (it beat iter2 head to head, section 5), sharpened Model 1 probabilities, Model 3 iter3 for the main pick, and Model 3 iter1 aggressive for the hedge pick.

**Rebuilding older files.** `python run_pipeline.py --model1-iter iter3` with Model 2 iter2 and Model 3 iter1 reproduces C and D exactly. The same with Model 1 iter4 reproduces the two "wild" files. Both were checked byte for byte before iteration 5.

### 3a. Iteration 5: what changed and the evidence

The earlier backtests scored Model 3 alone. `backtest_full.py` scores the **whole submission** (P, F and R) on 48 past windows (origins 52 to 193, every 3 weeks). Each window is planned from the week before it with data up to that week only, which is how the private window is planned from week 221. Results are in `Code Base/pipelining/backtests/`.

| Plan | All 48 | Calm (28) | Naive runs short (20) | Worst |
|---|---|---|---|---|
| Old main: Model 3 iter1 moderate | 47.3 | 34.2 | 65.7 | 9.4 |
| Old hedge: Model 3 iter1 aggressive | 42.4 | 22.9 | 69.8 | 7.7 |
| **New main: Model 3 iter3 + sharpened Model 1** | **52.6** | **47.6** | 59.7 | 4.5 |
| New hedge: Model 3 iter1 aggressive + sharpened Model 1 | 43.5 | 24.9 | 69.4 | 7.7 |
| Old pair, better of the two per window | 49.4 | 34.2 | 70.7 | 20.0 |
| **New pair, better of the two per window** | **57.2** | **47.6** | 70.7 | **27.3** |

"Calm" means naive never runs short in a shock week, so fill skill is undefined (counted as 0). The new pair beats the old pair in 37 of the 48 windows.

**Why two different plans.** Kaggle keeps the better of a team's 2 selected submissions. The lean plan wins calm windows, and the hedged plan wins windows where a big shock starts mid-window: the Kestria ban at week 99, or the 8-week Merrow closure at week 173. The lean plan is the better one in 39 of the 48 windows, and the hedge rescues most of the rest.

**The changes, one at a time** (average over the 48 windows, unsharpened Model 1 unless noted):

| Step | All 48 |
|---|---|
| Model 3 iter1 moderate | 47.3 |
| + plan calm weeks at 0.975 × normal capacity | 48.2 |
| + no forced minimum share | 49.9 |
| Model 3 iter3 (all changes below) | 51.5 |
| + Model 1 sharpening (k = 1.5) | 52.6 |
| Model 3 iter3, hazards × 2.5 | 51.1 (not used) |
| Model 3 iter3 aggressive (stress futures) | 50.4 (not used; the iter1 hedge pairs better) |

What iter3 fixes (details in section 5, Model 3):

1. **Calm capacity is 0.984 × normal** (sd 0.012, no autocorrelation), and calm lead time is exactly the base lead. iter1 planned calm weeks at 1.0 × normal, so orders sized to capacity came up short, and each missing unit cost 5×. In calm windows that was 1 to 4 weeks of demand cost lost per window.
2. **In calm windows the naive-to-hindsight gap is tiny.** Naive already buys from the cheapest supplier, so its only waste is leftover stock, about 1.5 weeks of purchases per material. iter1 gave back over half of it on hedges: buying from pricier suppliers (4 to 6 weeks of demand cost across materials), extra holding (about 3), and stockouts (1 to 4). iter3 therefore optimizes **points, not dollars**. In each sampled future, dollars are weighted by 40 / (that future's naive-minus-hindsight gap), and shock-week shortfall by 21 / (naive's shortfall).
3. **Diversification is chosen per material** by scoring candidate share floors exactly. Cheap diversification is taken, for example lithium Kestria/Orinoque, which are 0.5% apart; expensive diversification is skipped, for example aluminum Aurelia at +26%.

**Model 1 sharpening.** P is clipped at 0 per window, and Brier skill is a per-window ratio to the base rate. Sharpening the probabilities to sigmoid(1.5 × logit p) raised mean Brier skill from 0.158 to 0.204 and mean clipped P from 0.213 to 0.247 on the same windows. `Config(sharpen_k=1)` turns it off.

**Caveats.** The 48 windows overlap (13 weeks long, every 3 weeks), so they aren't independent. Both changes were designed on the same 208 weeks they are tested on. The sharpening k was picked from a small grid on these windows.

---

## 4. Repository layout

```
README.md                      this page
Data/                          competition CSVs (Phase 1: weeks 1-208; demand to 234)
docs/                          design docs: Model 1/2/3 plan PDFs, Eshaan's model write-up (PDF),
                               competition overview (Supplemental.docx); the cheat sheet stays local (private links)
submissions/                   every file uploaded or built for upload; the pipeline writes here
Code Base/
  pipelining/
    run_pipeline.py            one command: Models 1 -> 2 -> 3, both final files, checks
    score_local.py             local score of a submission on released weeks (+ alternative scorer conventions)
    backtest_full.py           whole pipeline on 48 past windows, every score part (writes backtests/)
    verify_scorer.py           after Phase 2: which scorer conventions reproduce the known public scores
    backtests/                 backtest_full.py results, one CSV per configuration (evidence for section 3a)
    Model 1/
      iter4/                   CURRENT: iter3 + news story layer (model1.py, news_stories.py);
                               the pipeline turns on sharpening (Config.sharpen_k = 1.5)
      iter3/                   supplier/country/lane/tariff hazard model + run notebook + outputs
      iter1/, iter2/, archive/ earlier text/numerical/combined versions (superseded)
    Model 2/
      iter3/                   CURRENT (Eshaan): calm mean reversion + shock on/off mix, landed costs
      iter2/                   structural shock-state forecast (Alan), superseded by iter3
      iter1/                   futures/mean-reversion blend (superseded)
    Model 3/
      iter3/                   CURRENT main pick: score-aware stochastic LP (lean)
      iter1/                   CURRENT hedge pick (AGGRESSIVE preset): simulator, scenarios, stochastic LP
      iter2/                   Eshaan's version + run notebook + plan A/B outputs
    dq_rules/, sql/, 04_pipeline.py   Databricks data-quality pipeline (rules review/approve)
  lanies nb.ipynb              exploration notebook
```

---

## 5. Models

### Data findings behind all three models

Four shock sources rebuild the `disrupted` column exactly, matching all 2,080 rows.

| Source | Detected by | Weeks 1-208 | Length | Capacity | Lead time | Spot price | Early warning |
|---|---|---|---|---|---|---|---|
| Single supplier | capacity < 70% of normal | 7 | 2-11 wk | 5-61% | unchanged | all 3 suppliers +15-31% | futures premium of that supplier climbs to 7-25% in 2-3 weeks; news shortage or export-ban stories |
| Country | 2+ suppliers of one country hit together | 1 (Kestria, wk 99-133) | 35 wk | 20% | unchanged | about +8% | news sanctions story, 7 weeks ahead |
| Lane closure | transit >= 1.5x normal | 4 | 2-8 wk | about 78% | +3 to 4 weeks | unchanged | congestion about 1 week ahead; news strike or clash stories |
| Tariff | tariff above base | 3 | 10-18 wk | unchanged | unchanged | unchanged (landed cost up) | announced 2-8 weeks ahead in `tariffs.csv` |

Other facts used throughout:

- **Calm prices behave like a random walk.** Futures carry no information in calm weeks.
- **The benchmark price** is exactly the mean of the 3 suppliers' spot prices.
- **Landed cost** = spot × (1 + tariff) + b × container rate + c.
- **Demand averages about 72% of the largest supplier's capacity,** so an even 3-way split is feasible in calm weeks.
- **The cheapest supplier is usually the riskiest.** For 9 of 10 materials it ships through the Merrow Strait.

**News story chains.** Headlines follow fixed templates that escalate before each shock type.

| Story | Warning headlines | Rumor ending | Became a shock | Lead |
|---|---|---|---|---|
| Shortage (supplier) | "producers flag maintenance delays", "Accident halts major ... site" | none seen | 3 of 3 | 3-4 wk |
| Export ban (supplier) | "ministry calls ... a strategic priority", "signals export curbs within weeks" | "officials rule out new restrictions" | 3 of 4 | 2-4 wk |
| Strike (lane) | "Port labor talks drag on", "Union sets strike date" | "Deal reached, ... averted" | 2 of 6 | 2-3 wk |
| Clash (lane) | "Tensions rise between ... and neighbors", "Insurers raise war risk premiums" | "Calm returns near the ..." | 1 of 3 | 2 wk |
| Sanctions (country) | "Companies told to prepare for ... sanctions", "Sanctions vote scheduled" | none seen | 1 of 1 | 7 wk |

Policy reversals, such as "Halvard abruptly suspends aluminum export licenses", come with no warning.

### Model 1: disruption probability (current: iter4)

**iter3** forecasts each source on its own. A source already active uses a duration curve: P(still active h weeks ahead) = S(age + h) / S(age), from discrete Weibull curves fit per type on the 15 real events, plus a long tail. A calm source uses an onset model: a logistic model of P(first onset within k weeks) on futures premium, tension, fx, PMI and congestion features. Announced tariffs are scheduled with probability 0.95. Material probability = 1 - the product of (1 - p) over its sources, clipped to [0.02, 0.98].

Backtest Brier skill, forecasting from 26 weeks between 104 and 204:

| Version | Skill |
|---|---|
| Duration curves only | 0.33 |
| + tariff schedule | 0.36 |
| + signals (iter3) | 0.435 |
| + Platt calibration | 0.27, not used |

Skill fades with lead time: 0.73 at 1-4 weeks ahead, 0.16 at 9-13 weeks.

**iter4 adds the news story layer** (`news_stories.py`).

1. Warning headlines open a story.
2. A story closes when its target is hit, when a rumor-ending headline appears, or after 10 quiet weeks.
3. Each story type's hit rate and lead time are estimated only from stories already closed at the forecast week. The rate prior is 0.5 with weight 2, and the lead prior is flat over 1 to 6 weeks.
4. For a calm source, the story's onset curve combines with the model's as F = 1 - (1 - F_model)(1 - F_news). The result then goes through the same duration curves.

Results:

- **Backtest:** neutral, 0.420 against 0.420 from every week 104 to 207. Most past shocks were also flagged by futures.
- **Model 3 points:** 40.9 against 40.1.
- **Public leaderboard:** +2.96.
- **At week 208,** Kestria lithium's shortage story raised lithium's disruption probability for weeks 211-214 from about 0.15 to about 0.8.

To turn the layer off, use `Config(use_news_stories=False)` or `--model1-iter iter3`.

### Model 2: price forecast and landed costs (current: iter3, Eshaan)

- **Calm material:** carry-forward plus drift, minus a pull toward the 26-week calm average. The pull grows from 0.06 at 1 week to 0.52 at 13 weeks.
- **Material in a shock:** the median of a two-scenario mix, today's elevated price against the pre-shock price, weighted by Model 1's chance the shock is still on.
- **Calm with a shock coming:** the same mix with a typical +16% jump, up to 4 weeks ahead.
- **Landed costs** come in three versions: expected, calm and most likely.

**Head to head on identical rows.** Price skill over carry-forward:

| Forecast weeks | iter3 (Eshaan) | iter2 (Alan) | iter1 |
|---|---|---|---|
| Every 4 weeks from 104 | **0.096** | 0.075 | 0.015 |
| Every 2 weeks from 60 | **0.067** | 0.051 | 0.014 |

iter2 forecasts the same shock jumps with a probability-weighted mean instead of the mix median. It loses mostly at 9-13 weeks ahead, where iter3's calm mean reversion helps. Futures alone score below carry-forward, at -0.007.

### Model 3: ordering plan (main: iter3, hedge: iter1 aggressive)

**iter3** (`Model 3/iter3/model3.py`) keeps iter1's two-stage stochastic LP and simulator and changes four things.

- **Futures** carry calm capacity noise and lane closures of +3 or +4 lead weeks. Each future records which materials are disrupted, including recovery weeks carried over from before the origin.
- **Normalizers.** For every future: naive cost, hindsight cost (one block LP over all futures) and naive's shock-week shortfall.
- **Objective.** Expected points lost: each future's dollars × 40 / gap, plus its shock-week shortfall units × 21 / naive shortfall. The shortfall weight is capped by flooring naive's shortfall at 5% of shock-week demand.
- **Shares and orders.** Six share-floor options per material, each scored exactly with fill = min(order, capacity) on the noisy futures, plus 9 × row diversification; the best is kept. Orders per supplier stay at or below (calm mean − 2.5 sd) × normal capacity, so a calm future never fills short. That also lets the LP drop fill variables in calm cells, making it about 10× faster.

The table below is the earlier Model 3-only comparison of iter1 against Eshaan's iter2, kept for the record. Section 3a has the full-score comparison against iter3.

A two-stage stochastic linear program per material and window. Stage 1 is the submitted orders, shared by all futures. Stage 2 is fills, stock and shortfall in each sampled future. The objective is the weighted cost: purchases, holding, the stockout penalty, minus the sell-back.

- **Futures:** about 200 per window, drawn from Model 1's source hazards and duration curves, with the shock effects in the table above. The same draws apply to every material, so a lane closure hits every supplier on that lane.
- **Constraints:** fills are capped by orders and capacity, and each supplier gets at least a minimum share of each week's order.
- **Presets:**
  - *moderate:* minimum share 0.1, stockout penalty 5×, hazards ×1.5, no stress futures.
  - *aggressive:* adds stress futures at weight 0.15: Merrow closed 8 weeks, cheapest supplier banned, capacity halved, and a Kestria-wide event.

**Head to head on 41 past windows,** each planned from the week before it, with the same Model 1 (iter4). Every plan was scored the same way. Points = 40 F + 21 fill skill + 9 diversification, out of 70.

| Plan | Avg | Worst | Naive never runs out | Naive runs out |
|---|---|---|---|---|
| **iter1 moderate** (main pick) | **40.6** | 9.4 | **28.2** | 55.0 |
| iter2 plan A | 39.4 | **13.8** | 24.0 | 57.2 |
| **iter1 aggressive** (hedge pick) | 37.0 | 7.7 | 16.3 | **60.9** |
| iter2 plan B | 32.5 | 9.0 | 9.0 | 59.6 |

iter1 moderate beat plan A in 32 of the 41 windows. For reference, a plan without scenarios, buying from the cheapest supplier only, scores 24.7.

**Different scorers.** Eshaan's PDF reports plan A at 0.517 of 0.70. That scorer credits the raw fill rate, usually 0.95 or more. The competition text, and the table above, use fill-rate *skill* against naive.

---

## 6. Open questions and weak spots

**Scorer details still unconfirmed:**
- How diversification is averaged. The E submission tests this.
- Whether starting stock is the sum of the first 5 weeks' demand or 5 times the mean. Eshaan found this changes naive's cost by only 0.04%.
- What fill-rate skill scores when naive meets all shock-week demand. Our local scorer counts it as 0.
- What "average landed cost" means. We assume the mean over the material's suppliers and the window's weeks.

**Weak spots:**
- **The lean main pick is exposed to a big shock starting mid-window.** Origins 91, 94 and 166 cost it 25 to 50 points against iter1. The hedge pick covers that case, so both must be selected.
- **The backtests are optimistic.** Every design choice was made after seeing all 208 weeks.
- **Shock endings drive most remaining error.** That includes the 35-week Kestria event, which the duration curves expected to end sooner.
- **3 of the 15 past shocks had no warning,** whether from news, futures or anything else.
- **A shock bigger than anything in training** is covered only by the long-tail duration mix and the aggressive plan's stress futures. Neither has been tested against a real one.
- **The news rates rest on few stories:** 3 shortages, 4 export bans, 6 strikes, 3 clashes and 1 sanctions sequence.

**Tested and not adopted:**
- **Overriding lithium from its futures premium alone.** At the supplier level, a 4% premium was usually a false alarm: 2 real warnings against about 8 false ones.
- **A maximum share per supplier instead of a minimum share.** It scored 40.3 to 40.5 against 40.1, which is within seed noise. It's available as `Config(s_max=...)`.
- **Platt calibration for Model 1,** which hurt, and **boosted trees for Model 2,** which weren't needed.
