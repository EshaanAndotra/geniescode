# Benchmark price forecasting

The selected price forecast is a small, horizon-dependent adjustment of the last observed benchmark toward its trailing 26-week exponentially weighted mean. It won the prespecified historical-origin comparison; more complex models were retained only as validation candidates.

For material `m`, cutoff `o`, and horizon `h`:

```
P_last = mean supplier spot price at o
P_ema  = exponentially weighted mean of historical benchmark prices,
         span 26, adjust=False, through o inclusive
weight = 0.35 * (1 - exp(-h / 8))
forecast = exp((1 - weight) * log(P_last) + weight * log(P_ema))
```

This maintains positive forecasts and limits reversion to 35% of the distance to the recent mean. It does not extrapolate an observed trend indefinitely. The benchmark computed from prices matches outcomes to the expected rounding tolerance: maximum discrepancy 0.00003334.

## Selection and validation

Forecasts are frozen at each origin for all 26 subsequent weeks. There is no within-window update using observations from the forecast window. Selection uses origins 104, 130 and 156. The subsequent origin 182, covering weeks 183–208, is a final holdout and does not select the model or strength. Every training target used by a fitted candidate satisfies `target_week <= origin`.

| Period | Selected MAPE | Last-price MAPE | Relative error reduction |
|---|---:|---:|---:|
| Three tuning origins, all horizons | 6.5423% | 6.6477% | 1.59% |
| Holdout origin 182, horizons 1–26 | 6.3778% | 6.7596% | 5.65% |
| Holdout origin 182, horizons 1–13 | 4.1345% | 4.4555% | 7.20% |
| Holdout origin 182, horizons 14–26 | 8.6211% | 9.0637% | 4.88% |

Performance varies across origins. The selected method slightly worsens full-window MAPE at origins 104 and 130 and improves origin 156. The evidence supports a conservative adjustment, not a claim that future benchmark prices are highly predictable.

Compared candidates include last-price carry, four-week futures anchors, mild trend extrapolation, several reversion strengths, direct ridge regressions, and a small gradient-boosted model. Fitted candidates use supplier price dispersion and futures premiums, capacity and lead time, tariffs and announcements, supplier-linked macroeconomic and shipping data, historical disruption state, seasonality, and the supplied future demand. Their validation error was higher. The selected price component consequently uses only the price history; the larger pipeline uses the remaining signals for disruption and ordering decisions.

The integrity check changes all future non-demand observations after week 156 to zero and confirms predictions from week 156 remain identical. Demand is a permitted future covariate because its entire schedule was supplied, although it is not used by the selected price component.

## Interface and reproduction

`predict_prices(data, origin, weeks)` returns `week`, `material`, and `price_forecast`. The data dictionary contains DataFrames keyed by source basename. The final Phase 1 forecasts use origin 208 for weeks 209–234. If Phase 2 files arrive, refit with the common observed cutoff and use origin 221 for private weeks 222–234; actual public outcomes should be identified separately from forecasts in any evaluation.

From the project directory:

```
python price_model.py --validate --data-dir "C:/Users/eshaa/Downloads"
python price_check.py
```

`price_validation.csv` records all candidate scores by origin and horizon range. `price_validation.json` contains the chosen model, validation summary and integrity check. These are forecasting diagnostics, not the organizer's composite simulator score.
