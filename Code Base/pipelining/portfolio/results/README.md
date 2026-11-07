# portfolio/results

Run from a scratch folder with the scripts in `..`; every CSV below came from those scripts.

- `bt_pair_lean.csv`: `backtest_pair.py --procedure lean`, 18 origins. Main v2, current hedge, and the lean E[max]-directed B, scored on real history (sum5 and 5avg). Lean B **failed**: max(main,B) 49.1 vs max(main,hedge) 55.6, with Merrow/Kestria windows collapsing.
- `lean_B_pm05_it2_origin221.csv`: that lean B at origin 221 (not promoted). At 221 it scores +3.7 on generator E[max] and +5.1 on stress.
- `bt_3b_smoke_o169_uniformlanes.csv`: 3b smoke test at origin 169 (weight 0.15, uniform lane mix, superseded). main3b 44.6 vs main 5.6 vs hedge 62.5.
- `bt_3b_w027.csv` / `bt_3b_w015.csv` (added when finished): `backtest_pair.py --procedure 3b --omega 0.27|0.15`, with blindside lane mix by realized closure frequency. 0.27 = BUILDER-2's measured P(at least one unwarned shock per window). Adoption test: beat max(main,hedge) overall, no category below it beyond noise, worst window >= 22.

**3b result (sum5, the confirmed scorer rule): not adopted.** At w=0.27, max(main3b, B) = 54.05 vs max(main, hedge) = 55.57. Kestria 49.8 vs 66.5, Merrow 57.6 vs 67.7; worst window 27.9 vs 23.3. At w=0.15: 51.42 (origin 166 collapses to 5.2). 3b gains in the calm, premium-chain and recent categories but cannot match the old hedge's blindside insurance. The final pair stays main v2 + iter1 aggressive hedge.
