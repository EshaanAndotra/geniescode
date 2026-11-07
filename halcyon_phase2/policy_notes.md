# Purchasing policy and local simulator

The policy treats weeks 209–221 and 222–234 as independent 13-week windows, both forecast using information through week 208. Each starts with five weeks of inventory and no orders in transit. The provided demand schedule is known through week 234.

## Method

For each material, a stochastic linear program chooses weekly quantities from its three eligible suppliers. Historical contiguous blocks supply scenarios for correlated capacity losses and delivery delays. Twelve historical blocks, a normal scenario, and a scenario that gradually relaxes the latest observed constraints are blended. All scenario observations are at or before the forecast origin. Supplier prices start from recent landed costs, scale with the benchmark price forecast, and incorporate future tariff changes already announced by the origin.

Orders are diversified: each eligible supplier receives 23%–47% of each positive weekly order. Zero-order rows carry equal shares. The optimizer caps total orders at 1.6 weeks of average window demand, which reduces large batches exposed to a single week's capacity cancellation. A normal-state inventory buffer starts at 1.5 weeks, can increase with forecast disruption probability, and decreases near the end of the window. Orders taper when their arrival would be too late to cover demand.

The objective includes procurement, 2% weekly holding, terminal salvage at 70%, and a shortage penalty of 6 times reference cost. The extra penalty above the simulator's stated 5 times cost is a deliberate planning preference for resilience. Share constraints approximate the diversification incentive; the code does not claim to optimize the unknown exact competition score.

Capacity cancellation is approximated in the LP by three fixed-point updates of scenario fill fractions. This approximation can differ from the exact nonlinear cancellation rule. Reserve bounds can also create planning states with both inventory and unmet demand. The evaluation simulator applies exact minimum-of-order-and-capacity acceptance, always consumes available inventory, and exposes actual performance of the resulting implementable orders.

## Validation and model selection

Policy development uses forecast origins 104, 130, and 156, with both 1–13-week and 14–26-week horizons. These are six independent 13-week simulations; they are used to select policy settings. The four candidate combinations were buffer/order-cap `(0.55, 10)`, `(0.55, 1.6)`, `(1.0, 1.6)`, and `(1.5, 1.6)` weeks. Selection favored the last setting on the joint cost and fill results. Development mean cost savings were 27.0%; one window cost 3.2% more than its naive analogue. Mean material-weighted metrics can differ from pooled unit-weighted metrics.

Origin 182 is the chronological evaluation holdout. The default policy, without the forecast model's optional price/probability inputs, saved 4.65% over the naive analogue for weeks 183–195 and 4.29% for weeks 196–208, with 100% demand fill in both. These are local simulation results, not official competition scores or evidence about weeks 209–234. The final pipeline separately evaluates the combined forecasts and policy.

The naive analogue orders each week's published demand from the supplier with lowest mean historical landed cost as of the origin. This rule reproduces the supplied sample's order quantities and supplier selections for all 260 rows at origin 208. The competition does not supply its baseline-generation source code, so the historical counterpart is still an inferred analogue.

## Explicit simulator conventions

- An order in week `t` with lead `L` arrives in week `t + L`.
- Starting inventory is five times mean demand in the evaluation window.
- Reference material cost is mean landed cost across the window's weeks and three suppliers.
- Holding is charged on inventory remaining after that week's demand.
- Unmet demand is lost, not backlogged.
- Terminal salvage includes both on-hand and accepted in-transit quantities.
- A shock week includes an active disruption plus its following four recovery weeks. Recovery can carry into a window from a disruption immediately before it.
- Both pooled unit-weighted and material-averaged fill metrics are returned. Diversification is reported per row and weighted by order size.

The written rules do not completely specify starting-inventory demand normalization, reference-cost averaging, holding timing, or aggregation weights. Alternative initial stock equal to five times historical mean demand or first-week demand was tested on the origin-182 holdout without changing the selected policy. Savings stayed positive (1.4%–2.8%) and fill stayed at least 99.67%. No exact `F`, `R`, or overall competition score is reported, because official hindsight costs, normalization, and simulator code were not supplied.

## Interfaces

`plan_policy(data, origin, start, end, risk_forecast=None)` returns one row per material/week with `order_units` and all six country-share columns. `data` is a mapping from source file stem to DataFrame. Optional forecasts contain `week`, `material`, `disruption_prob`, and `price_forecast`.

`simulate_policy(data, policy, start, end, origin=None)` evaluates only windows with observed supply history. It returns total cost components, fill metrics, material details, and weekly inventory traces. `naive_policy(...)` builds the baseline analogue. `tune_policy(input_directory, output_path)` reproduces the fixed development grid.
