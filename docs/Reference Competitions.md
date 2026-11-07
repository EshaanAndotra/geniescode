# Reference Competitions for New Models

As of Nov 7, 2026. This is a starting point for a second modelling approach. It covers three past competitions that are close to parts of this hackathon, what their top solutions did, and how each idea maps onto our simulator and code.

No past Kaggle competition matches this one as a whole: disruption forecasting, price forecasting and supplier ordering scored against one simulator. Each of our three parts has a close relative, though.

| Our part | Score weight | Closest competition | What to borrow |
|---|---|---|---|
| Ordering, Model 3 (F + fill-rate skill) | 40% + 21% | **VN2 Inventory Planning Challenge** (2025) | Predict, then a simple cost-ratio order rule |
| Futures that Model 3 plans against (Model 1) | feeds all of the above | **M5 Uncertainty** (Kaggle, 2020) | Calibrated uncertainty beats a better point forecast |
| Price forecast, Model 2 (MAPE skill) | 12% | **MITSUI&CO. Commodity Prediction** (Kaggle, 2025-26) | Simple linear models and weekly refits held up |

**How well sourced each section is.** The VN2 section comes from the winner's report, read directly through its Section 6.2; the later sections came from a summary and are marked *(summary)*. The M5 and MITSUI sections rest on search results and talk summaries, not the winners' own write-ups. Read those before building on the details.

---

## 1. VN2 Inventory Planning Challenge: the template for a new Model 3

### Setup
- Weekly ordering for 599 store-product pairs over 6 rounds, with lost sales and no backorders.
- Lead time is 2 weeks: an order placed at the end of week t is sellable from week t+3.
- Cost = 1.0 per lost sale + 0.2 per unit on hand at the end of each week. No holding cost in transit.
- Supply was unconstrained and lead times were fixed. **All the uncertainty was in demand.**
- Scored by the organisers' simulator on total cost, with the same starting state for everyone.

### Winning architecture (1st place, Bartosz Szabłowski)
Two stages, *predict then optimize*:

1. **Forecast.** One global CatBoost model across all series, with a separate model per horizon h = 1, 2, 3.
    - Stockout-aware target: weeks that were out of stock are masked (y = NaN), so lost sales don't drag the forecast down.
    - Scaling: each series is divided by about its trailing 52-week mean, so one model fits all sizes. Recent years get higher weight (1.0, 0.5, 0.25). *(summary)*
    - Features: lags, seasonal lags, rolling mean/median/std/IQR, EWMs, momentum, slope.
    - Tuned with Optuna on a time-ordered validation split, then refit on all history.
2. **Order rule.** A newsvendor-style target stock level. *(summary)*

```
Project stock to the delivery week with the point forecast:
  Ẽ(t+1) = max(E(t) + R(t+1) − D̂(t+1), 0)
  Ĩ(t+3) = max(Ẽ(t+1) + R(t+2) − D̂(t+2), 0)

Critical ratio:   q* = c_s / (c_s + c_h) = 1.0 / 1.2 ≈ 0.833,   z = Φ⁻¹(q*) ≈ 0.967
Uncertainty:      σ(t+3) = φ · sqrt(D̂(t+3))     (φ tuned on validation to minimise total cost)
Target:           B(t+3) = D̂(t+3) + z · σ(t+3)
Order:            Q(t)   = max(B(t+3) − Ĩ(t+3), 0)
```

- **Result:** first place. The report gives a total cost of 3.763 against a benchmark of 4.334, 13.2% lower. *(summary)*
- **Author's lessons:** the forecasting model is only one part of the system. Modelling the lead time and the asymmetric costs explicitly mattered more than extra model complexity. A single tuned safety factor (φ) did most of the work. *(summary)*

### How it maps onto our problem
The key difference: **in VN2 demand is uncertain and supply is fixed; here it's the reverse.** Our demand is given out to week 234. The uncertainty is in capacity, lead time and landed cost per supplier. So the VN2 rule carries over with the roles swapped: forecast *delivered units*, not demand.

| VN2 | Our simulator ([Model 3/iter1/model3.py](../Code%20Base/pipelining/Model%203/iter1/model3.py), `simulate`) |
|---|---|
| Demand forecast D̂ | Demand is known |
| Fixed 2-week lead time | Lead time per supplier per week, +3 to 4 weeks in a lane closure |
| Unlimited supply | Fill = min(order, capacity); unfilled units are cancelled and **not charged** |
| Lost sale costs 1.0, holding 0.2 per week | Unmet demand costs 5 × average landed cost (lost, not backlogged), holding 0.02 × average cost per week |
| No end-of-horizon effect | Stock + in-transit at week 13 sells back at 0.7 × average cost |
| σ from √demand | Spread of delivered units across Model 1's sampled futures |

**Critical ratio for our costs** (rough, in units of average landed cost; ignores price differences between suppliers):
- A unit short costs the 5× penalty but saves its ~1.0 purchase, so about **4.0** net.
- A unit over in mid-window just carries to next week: **0.02 per week** held.
- A unit over at the end loses 1.0 − 0.7 = **0.3**, plus holding.
- So q* ≈ 4 / (4 + 0.3) ≈ **0.93** for orders that arrive near the end, and close to 1 for mid-window arrivals. Our rule should hold much more safety stock than VN2's did, except in the last few weeks.

**Caveat:** we're scored in points, not dollars. F is normalised by each window's naive-to-hindsight gap, and fill skill is relative to naive. In calm windows that gap is only about 1.5 weeks of purchases per material, so over-hedging is expensive in points. Model 3 iter3 already optimises points; a VN2-style rule optimises dollars unless φ is tuned on points.

### A sketch of a VN2-style Model 3 ("base-stock" variant)
Per material, per order week t, per supplier c:

1. **Supplier split first.** Shares go cheapest-first, with diversification only where it's cheap. iter3 already chooses share floors per material, so reuse that.
2. **Delivered-units forecast.** From Model 1's futures, take each supplier's capacity ratio and lead time per week. Compute the distribution of units that would arrive by week a for a given order.
3. **Projection.** Project stock to each arrival week using known demand and the median arrivals of orders already placed.
4. **Target.** Order up to the q*-quantile of what's needed to cover demand through the next arrival, with q* from the costs above. It falls in the last 2-3 weeks of the window because of the 70% sell-back.
5. **Tune one knob.** Tune a single safety multiplier (like VN2's φ) on the 48 backtest windows in [backtest_full.py](../Code%20Base/pipelining/backtest_full.py), maximising **points**.

**Why it's worth building:** it's simple, fast, and built differently from the LP. Kaggle keeps the better of the 2 selected files, so a different plan that wins different windows is worth more than a slightly better copy of the main pick. Judge it on best-of-two with the main pick across the 48 windows, not on its average alone.

---

## 2. M5 Uncertainty: calibrating the futures

### Setup
Kaggle, 2020, Walmart sales across 12 aggregation levels. Teams submitted the median and eight other quantiles for each series, scored on scaled pinball loss.

### What the winners did
- **1st:** David Lander (Northquay) and Russ Wolfinger (SAS). **2nd:** Nikolay Mamonov (GoodsForecast). **3rd:** Ioannis Nassios (Nodalpoint).
- The organisers singled out how accurate the winner's uncertainty estimates were.
- Other documented approaches:
    - 6th: a state-space model with negative binomial errors, 1st at the finest level of detail.
    - 13th: a LightGBM + DeepAR hybrid, with the method varying by aggregation level.
- *I haven't read the winning write-up. Its quantile method needs checking before we borrow specifics.*

### How it maps onto our problem
Model 3 optimises against futures sampled from Model 1's hazards and duration curves. If those futures put too little weight on a shock starting mid-window, the lean plan loses big. That's exactly what happened at origins 91, 94 and 166, costing 25-50 points. The M5 lesson: **check calibration directly, not just average accuracy.**

Checks we can run with what's already in the repo:
1. **Reliability by horizon.** On the 48 backtest windows, bin `disruption_prob` (e.g. deciles) and compare with the observed disruption rate, split by weeks ahead (1-4, 5-8, 9-13).
2. **Onset coverage.** For each origin, what fraction of sampled futures has a new shock starting inside the window? Compare with how often one actually did across origins.
3. **Delivered-units coverage.** For each supplier-week, check the realised capacity ratio and lead time against the futures' 10th/50th/90th percentiles. Count how often the outcome falls outside the 80% band; well calibrated is about 20%.
4. **Fix.** If onsets are under-covered, scale onset hazards only, not shock durations. Re-score both picks on the 48 windows. An earlier ×2.5 scaling of all hazards gave 51.1 against 51.5, which is a reason to try onsets alone rather than everything.

---

## 3. MITSUI&CO. Commodity Prediction: the price forecast

### Setup
Kaggle, July 2025 to January 2026, $100k in prizes. Teams forecast 424 commodity and market targets 1-4 days ahead. New labels arrived during the scoring period.

### What top solutions did
- Two write-ups from the same team describe an ensemble of ridge regressions. They disagree on its placement (top vs 14th); check the leaderboard.
- **8th:** one token-based Transformer predicting all 424 targets at once, with lag-4 features and a masked loss for missing labels.
- **15th:** standard financial time-series practice, adding a new model every 7 days as labels arrived and ensembling them.
- Across write-ups, simple linear models and baselines were hard to beat.
- *These come from talk slides and write-up summaries, not the 1st-place post.*

### How it maps onto our problem
- **It confirms our finding.** Calm prices behave like a random walk, futures carry no information in calm weeks, and futures alone score −0.007 against carry-forward. The gains are in shock handling and mean reversion, which Model 2 iter3 already does (skill 0.096 against 0.075 for iter2).
- **Refit on Phase 2.** The private window is planned from week 221, so every estimated quantity (the drift, the 26-week calm average, the shock jump size) should use weeks 1-221. Confirm that Model 2 iter3 recomputes these from the data and has no constants fitted on weeks 1-208.
- **Possible cheap gain:** shrink each horizon's forecast toward carry-forward with a weight fit on past windows, MITSUI-style ridge. Only adopt it if it beats iter3 head to head on the same rows.

---

## 4. Suggested order of work before Sunday 12 PM CT

1. **M5-style calibration checks (section 2)**, about 1 hour. They use the existing backtests and tell us whether the hedge is protecting against the right thing.
2. **VN2-style base-stock Model 3 (section 1)**, a few hours. Score it on the 48 windows with `backtest_full.py` and compare best-of-two against the current pair (57.2). Adopt it as the hedge only if the pair improves.
3. **Model 2 Phase 2 check (section 3)**, about 30 minutes, once the data lands.

Whatever changes, the final step stays the same: run `run_pipeline.py`, upload both files, and **select both as finals**.

---

## 5. Considered and not used

| Competition | Why not |
|---|---|
| Two Sigma: Using News to Predict Stock Movements (Kaggle, 2018-19) | Main lesson is that news features added little, which our backtest already shows (news layer neutral at 0.420). Our template story chains suit synthetic headlines better than general text features. |
| Can You Predict Product Backorders? (Kaggle dataset) | Rare-event classification on static rows; our disruptions are time-based and rule-generated. |
| Corporación Favorita, M5 Accuracy | Pure demand forecasting; our demand is given. |

---

## Sources

Read directly:
- VN2 winner report: Szabłowski, *One Global Model, Many Behaviors* ([arXiv 2601.18919](https://arxiv.org/abs/2601.18919), [HTML](https://arxiv.org/html/2601.18919v1)). Read through Section 6.2; the order-policy formulas and results came from a summary of the later sections.

From search results and summaries (not yet verified against the original posts):
- VN2 partial standings: [LinkedIn post, A. Bernal Herrero](https://es.linkedin.com/in/abh9a)
- M5: [winners list (UNIC)](https://www.unic.ac.cy/iff/research/forecasting/m5-competition/) · [6th place, state-space approach](https://ar5iv.labs.arxiv.org/html/2111.14721) · [LightGBM/DeepAR hybrid](https://arxiv.org/pdf/2009.07701.pdf)
- MITSUI: [8th and 15th place talk (Speakerdeck)](https://speakerdeck.com/gamella/lineyahu-detasaiensu-meetup-san-jing-wu-chan-komodeiteiyu-ce-tiyarenzi-nowu-tai-li-alpacatechpato) · [ridge write-up (14th)](https://storage.googleapis.com/kaggle-forum-message-attachments/3396797/35904/Technical%20Documentation%20(1).docx) · [model summary](https://storage.googleapis.com/kaggle-forum-message-attachments/3396797/35905/KAGGLE%20WINING%20MODEL%20SUMMARY.docx) · [tree-model solution](https://www.besthub.dev/articles/open-source-kaggle-solution-predicting-multi-market-commodity-prices-with-tree-models-866c5b9e1b90) · [competition diary](https://dev.to/nk_maker/kaggle-diary-mitsuico-commodity-prediction-challenge-day-1-3kl4)
- Two Sigma: [feature engineering write-up](https://medium.com/predict/feature-engineering-for-stock-price-movement-54669d4a7efc) · [top-10% repo](https://github.com/Sibozhu/Predicting-Stock-Movements-With-News)
- Product backorders: [Sensors 2021 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659943)
- Favorita: [2nd place (arXiv)](https://arxiv.org/pdf/1803.04037.pdf)
