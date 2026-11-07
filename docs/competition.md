# Competition rules (official)

The organizers' text from the Kaggle overview and the participant cheat sheet, condensed. Nothing here is the team's interpretation; `BRIEF.md` holds that. Kaggle: https://www.kaggle.com/competitions/dsdg-hackathon-2026-supply-chain-disruptor

## Task

Halcyon Industries, a manufacturer in the fictional country of Aurelia, buys 10 raw materials from 6 fictional countries. Every material has exactly 3 supplier countries. Read the signals in news, prices, tariffs, shipping and macro data, forecast which materials will be disrupted and what they will cost, and turn that into a weekly ordering policy: how much to buy and from which countries. A simulator plays the orders out against what actually happened.

Everything is synthetic. Outside data and real-world knowledge won't help; the signals are all in the files.

## Schedule (Central time)

| When | What |
|---|---|
| Fri Nov 6, 6:30 PM | Phase 1 data (weeks 1-208) |
| Sat Nov 7, 12 PM | Phase 2 data (weeks 1-221) |
| **Sun Nov 8, 12 PM** | **Deadline: select 2 final submissions and upload the 3-minute video** |
| Sun Nov 8, 6:30 PM | Private leaderboard revealed |

- Public leaderboard = weeks 209-221. Private leaderboard = weeks 222-234. After Phase 2 the public weeks are in the data, so the public leaderboard is a warm-up only.
- Final placement: 80% private leaderboard, 20% video (judged).
- 5 submissions per team per day (resets at midnight UTC = 6 PM Central).
- Select up to 2 final submissions by the deadline, **or Kaggle picks your 2 best public scores**. Kaggle ranks a team by the better of its 2 selected submissions.
- AI coding assistants are allowed. Sharing code with another team is not.

## Score

```
Score = 100 x (0.30 P + 0.40 F + 0.30 R)
```

Each part is clipped to [0, 1] and scaled so the naive sample submission scores 0.

- **P, prediction (30%)**
  - 60%: Brier skill on `disruption_prob` = 1 - Brier / Brier of always predicting each material's historical disruption rate.
  - 40%: price skill on `price_forecast` = 1 - MAPE / MAPE of carrying the last known benchmark price forward.
- **F, financial (40%)** = (naive cost - your cost) / (naive cost - hindsight cost). Naive = the sample submission's policy. Hindsight = a plan that knew the future.
- **R, resilience (30%)**
  - 70%: fill-rate skill, i.e. how much more demand you met than naive during shock weeks. Shock weeks = weeks a material is disrupted, plus 4 recovery weeks.
  - 30%: diversification = 1 - Herfindahl index of your shares, rescaled so single sourcing is 0 and an even 3-way split is 1.

Cheat-sheet hint from the organizers: *"Buying a little early and from more than one supplier before a shock is usually what moves the score, since a stockout costs 5 times what the material does."*

## Simulator

- Each window (public and private) is simulated on its own. It starts with 5 weeks of demand in stock and nothing in transit.
- An order placed in week t is split by your shares. Each country's slice is filled only up to that country's capacity that week. Filled units are paid at that week's landed cost (spot price + tariff + shipping) and arrive after that week's lead time. Units a supplier can't fill are cancelled and not charged.
- Each week, arrivals go into stock, then Halcyon uses up to that week's demand.
- Holding: 2% of the material's average landed cost per unit per week.
- Unmet demand: 5x the average landed cost per unit.
- End of window: stock and units still in transit are sold back at 70% of the average landed cost.

## Submission file

A CSV with a header and exactly 260 rows: one per material per week for weeks 209-234.

| Column | Meaning |
|---|---|
| `id` | `w{week}_{material}`, e.g. `w209_lithium` |
| `disruption_prob` | Probability (0-1) that any supplier of this material is hit by an active shock that week |
| `price_forecast` | That week's benchmark price (mean spot price across the material's 3 suppliers). Must be > 0 |
| `order_units` | Units to order that week, >= 0 |
| `share_KES` ... `share_AUR` | Split of the order across countries. Each 0-1, summing to 1, and 0 for any country that doesn't sell the material (`suppliers.csv`) |

A file is rejected for missing or duplicate ids, blank values, probabilities outside [0, 1], shares that don't sum to 1, or a share for a country that doesn't sell the material.

## World and data

| Code | Country | Shipping lane |
|---|---|---|
| KES | Kestria | MERROW (Merrow Strait) |
| VAL | Valdmar | NORTH (Northern Sea route) |
| SUN | Sunda Coast | MERROW (Merrow Strait) |
| ORI | Orinoque | SOUTH (Southern route) |
| HAL | Halvard | ATLANTIC |
| AUR | Aurelia (domestic) | RAIL |

Materials: lithium, cobalt, copper, aluminum, steel, rare_earths, semiconductors, polymer_resin, specialty_glass, natural_gas. Week 1 = Mon Jan 3, 2022; 234 weeks in total.

| File | Contents |
|---|---|
| `news.csv` | About 21 articles a week: article_id, week, date, source, headline, body. Some are early warnings of real shocks, most are routine, some are rumors that never happen |
| `prices.csv` | Per week, material and country: `spot_price`, `futures_4w` (market price for delivery in 4 weeks) |
| `tariffs.csv` | Per week, material and country: `tariff_rate` in force; `announced_rate` and `announced_effective_week` when a future change is announced |
| `shipping.csv` | Per week and lane: `container_rate`, `transit_days`, `congestion_index` (0-100) |
| `macro.csv` | Per week and country: `pmi`, `fx_rate`, `cpi_yoy`, `policy_rate`, `tension_index` (0-100, noisy political tension with Aurelia) |
| `suppliers.csv` | Static: material, country, lane, `base_lead_weeks`, `normal_capacity_units`, `base_tariff` |
| `demand.csv` | Halcyon's production plan, all 234 weeks including the test windows |
| `outcomes.csv` | Released weeks only: `benchmark_price`, `disrupted` (1 if at least one of the material's suppliers was hit by an active shock) |
| `supply_history.csv` | Released weeks only: `capacity_units`, `lead_time_weeks`, `landed_cost` per supplier |
| `sample_submission.csv` | Valid file; the naive baseline that scores 0 |

Shock kinds named by the organizers: tariff hikes, export bans, strait closures, resource shortages, sanctions, port strikes, sudden policy reversals. Most give some warning in news and market data. Policy reversals give none. Closures and strikes hit every country on a lane at once.

## Video (20% of placement)

About 3 minutes, uploaded to the team's folder by Sunday noon. Cover the signals you found, how forecasts turn into orders, how the plan survives shocks, and what the approach would miss.
