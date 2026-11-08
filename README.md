# DSDG Hackathon 2026: Supply Chain Disruptor

Team repo for the Kaggle competition. **Deadline: Sun Nov 8, 12 PM Central.** Select the 2 final submissions on Kaggle by then (Kaggle keeps the better one; if none are selected, it picks the 2 best public scores) and upload the 3-minute video.

**Start here:**
- [docs/BRIEF.md](docs/BRIEF.md): verified data facts, the signals at week 221, everything tried so far with results, open problems, and ground rules for building.
- [docs/competition.md](docs/competition.md): the official rules, scoring and simulator.
- [docs/PHASE2_ITERATION8.md](docs/PHASE2_ITERATION8.md): Phase 2 review and the final picks, `submissions/final/submission_main_it8.csv` and `submissions/final/submission_war_it8.csv` (built with `python run_pipeline_p2.py` from `Code Base/pipelining`).
- [halcyon_phase2/README.md](halcyon_phase2/README.md): a separate, self-contained model package with its own report and two alternative Phase 2 submissions.

## Data

`Data/` is Phase 1 (weeks 1-208). `Data_new/` is Phase 2 (weeks 1-221), which is what the private window (weeks 222-234) is planned from. Every script takes `--data`.

## Build the current baseline

```
cd "Code Base/pipelining"
python3 run_pipeline.py --data ../../Data_new --tag _p2
```

This runs Model 1 (iter4, disruption) → Model 2 (iter3, price) → Model 3 and writes two plans to `submissions/phase2/`. It takes 1-2 minutes, validates both files, and prints checks worth reading before upload: data weeks, shocks active at the last week, open news stories, futures premiums above 3%, and order volumes.

| File | Model 3 | Character |
|---|---|---|
| `submission_moderate_p2.csv` | iter3 | Lean; optimizes points directly; wins calm windows |
| `submission_aggressive_p2.csv` | iter1, aggressive preset | Heavily hedged; wins windows where a big shock starts mid-window |

Both share the same `disruption_prob` and `price_forecast`. Without `--tag`, the files are `submission_moderate.csv` and `submission_aggressive.csv`.

## Evaluate

| Command (from `Code Base/pipelining`) | What it does |
|---|---|
| `python3 backtest_full.py --data ../../Data_new --tag NAME [--origins 52:209:3]` | Plans past windows the way the private window is planned (from the week before, earlier data only) and scores all five parts. Writes `backtests/phase1/bt_NAME.csv`. About 4 minutes on 8 cores |
| `python3 score_local.py ../../submissions/final/FILE.csv --data ../../Data_new` | Scores a file on weeks 209-221, part by part |
| `python3 verify_scorer.py --data ../../Data_new` | Tries the unconfirmed scorer conventions and ranks them by how well they reproduce known public scores |
| `python3 validate_submission.py ../../submissions/final/FILE.csv [...] --data ../../Data_new` | Checks files against every rejection rule and team rule (ids, blanks, ranges, whole orders, shares). Lists every problem, and exits 1 if any file fails. Run it on every file before handing it over or uploading |

**Databricks:** the team is in the Databricks track. The final pipeline gets ported to the team's Databricks workspace once it's validated here; `dq_rules/`, `sql/` and `04_pipeline.py` are the existing data-quality pieces.

## Public scores (Phase 1 files, weeks 209-221)

| File | Public score |
|---|---|
| `submissions/phase1/submission_moderate_iter5.csv` | **68.40** |
| `submissions/phase1/submission_moderate_wild.csv` | 65.29 |
| `submissions/phase1/C_full_moderate.csv` | 62.33 |
| `submissions/phase1/submission_B_iter2.csv` (Eshaan's Model 3 iter2 plan B) | 44.15 |

Since Phase 2, the public weeks are in the data, so public scores no longer predict the private leaderboard. Their remaining use is pinning down scorer conventions (`verify_scorer.py`, `submissions/probes/`).

## Layout

```
docs/                    BRIEF.md, competition.md, PHASE2_ITERATION8.md
Data/, Data_new/         competition CSVs (Phase 1, Phase 2)
halcyon_phase2/          self-contained alternative model (own inputs/ copies of Data, Data_new; run_phase2.py)
submissions/             final/ = the two final picks, phase2/ = Phase 2 fallback files,
                         phase1/ = Phase 1 builds and uploads, probes/ = scorer-convention probes
Code Base/pipelining/
  run_pipeline.py        one command: Models 1 -> 2 -> 3, two final files, checks
  run_pipeline_p2.py     iteration 8: Model 1 iter6 -> Model 2 iter6 -> Model 3 iter5, final main + war files
  backtest_phase2.py     iteration 8 backtests (backtests/phase2/)
  scenarios/             iteration 8 scenario builder and evaluator
  backtest_full.py       whole pipeline on past windows, every score part
  score_local.py         local scorer (with alternative conventions)
  verify_scorer.py       which conventions reproduce the public scores
  validate_submission.py every rejection and team rule, checked before handover or upload
  stress_private.py      hand-built futures for weeks 222-234, scored for both picks
  backtests/             one CSV per configuration: phase1/ (backtest_full.py), phase2/ (backtest_phase2.py)
  Model 1/iter6/         iteration 8 disruption model (built on iter4)
  Model 1/iter4/         disruption model (iter3 + news story layer)
  Model 1/iter3/         the same model without news (Model 3's fallback when MODEL1_DIR isn't set)
  Model 2/iter6/         iteration 8 price model (built on iter3)
  Model 2/iter3/         price model and landed costs (Eshaan)
  Model 3/iter5/         iteration 8 ordering plan (built on iter3)
  Model 3/iter3/         lean ordering plan
  Model 3/iter1/         simulator, scenario futures, LP, hedged preset (hedge pick)
  dq_rules/, sql/, 04_pipeline.py   Databricks data-quality pipeline
```

Superseded iterations, notebooks and Phase 1 output files are no longer in the repo; they're in git history.
