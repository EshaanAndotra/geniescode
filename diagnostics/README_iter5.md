# iter5 (BUILDER-2): calibrated P columns and source-specific price jump

- **What's in it:** `Code Base/pipelining/Model 1/iter5` adds 3 flags. All are off by default, and with them off the output equals iter4 exactly:
  - `premium_mix`: onset timing for a supplier in an open futures-premium run.
  - `lane_floor_m`: each lane's own calm closure rate.
  - `tariff_floor`: ae's rule, from `t1_tariff/p_overrides_ae.py`.

  `Model 2/iter5` adds `jump_by_source` and `cal_onset`. All of these change `disruption_prob` and `price_forecast` only. Model 3's futures (`q_sup`, `q_lane`) stay iter4's.
- **Build (adopted by the manager; lane floor off):** `model1.Config(sharpen_k=1.5, premium_mix=0.5, tariff_floor=0.95)` and `model2.Config(jump_by_source=True)`. `iter5_eval/build_check.py` checks that flags off equal iter4 and that the result validates.
- **Measured** (`iter5_eval/eval.log`: 18 premium-chain + 3 tariff + 7 calm origins; `eval_all157.log`: origins 52-208; Brier on sharpened P):
  - The mix lowers Brier on the changed premium rows from 0.187 to 0.173, but the gain comes from the lithium and steel KES runs.
  - The tariff floor lowers Brier on the changed rows from 0.024 to 0.0025.
  - The lane floor lowers Brier over all 157 origins from 0.1350 to 0.1338, but raises it on premium origins.
  - The jump: leave-one-out error 0.051 vs 0.069, and price skill goes from +0.060 to +0.061.
- **Evidence:**
  - `t3_onset_timing/`: the generator is too sure of the onset week; the premium-run counts are reconciled there.
  - `onset_rate/`: generator vs realized onsets per window. All capacity shocks together: 0.95 vs 0.97. Calm Merrow: 0.06 vs 0.20. Unwarned: 4 events, about 0.27 chance per window.
  - `t2_jump/`: the jump leave-one-out check.
- **Left:**
  - ae still has to wire iter5 into its build; `run_pipeline.py` can't load iter5.
  - The orders-side decision test of the mix: plan with `premium_mix_plan=True` at origin 221, then score on `stress_private`.
  - Dropped: the -0.034 price skill on weeks 209-221 (item 4), and the Kestria and Merrow probability evidence (T4).
