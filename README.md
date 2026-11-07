# DSDG Hackathon 2026: Supply Chain Disruptor

Team repo for the Kaggle competition. **Deadline: Sun Nov 8, 12 PM Central.** Select the 2 final submissions on Kaggle by then (Kaggle keeps the better one; if none are selected, it picks the 2 best public scores) and upload the 3-minute video.

**Start here:**
- [docs/BRIEF.md](docs/BRIEF.md): verified data facts, the signals at week 221, everything tried so far with results, open problems, and ground rules for building.
- [docs/competition.md](docs/competition.md): the official rules, scoring and simulator.

## Data

`Data/` is Phase 1 (weeks 1-208). `Data_new/` is Phase 2 (weeks 1-221), which is what the private window (weeks 222-234) is planned from. Every script takes `--data`.

## Build the current baseline

```
cd "Code Base/pipelining"
python3 run_pipeline.py --data ../../Data_new --tag _p2
```

This runs Model 1 (iter4, disruption) → Model 2 (iter3, price) → Model 3 and writes two plans to `submissions/`. It takes 1-2 minutes, validates both files, and prints checks worth reading before upload: data weeks, shocks active at the last week, open news stories, futures premiums above 3%, and order volumes.

| File | Model 3 | Character |
|---|---|---|
| `submission_moderate_p2.csv` | iter3 | Lean; optimizes points directly; wins calm windows |
| `submission_aggressive_p2.csv` | iter1, aggressive preset | Heavily hedged; wins windows where a big shock starts mid-window |

Both share the same `disruption_prob` and `price_forecast`. Without `--tag`, the files are `submission_moderate.csv` and `submission_aggressive.csv`.

## Evaluate

| Command (from `Code Base/pipelining`) | What it does |
|---|---|
| `python3 backtest_full.py --data ../../Data_new --tag NAME [--origins 52:209:3]` | Plans past windows the way the private window is planned (from the week before, earlier data only) and scores all five parts. Writes `backtests/bt_NAME.csv`. About 4 minutes on 8 cores |
| `python3 score_local.py ../../submissions/FILE.csv --data ../../Data_new` | Scores a file on weeks 209-221, part by part |
| `python3 verify_scorer.py --data ../../Data_new` | Tries the unconfirmed scorer conventions and ranks them by how well they reproduce known public scores |
| `python3 validate_submission.py ../../submissions/FILE.csv [...] --data ../../Data_new` | Checks files against every rejection rule and team rule (ids, blanks, ranges, whole orders, shares). Lists every problem, and exits 1 if any file fails. Run it on every file before handing it over or uploading |

**Notebook route (Databricks):** `Model 1/iter3/model1_run.ipynb` → `Model 2/iter3/model2_run.ipynb` → `Model 3/iter2/model3_run.ipynb`. This route uses Model 1 iter3 (no news layer) and Eshaan's Model 3.

## Public scores (Phase 1 files, weeks 209-221)

| File | Public score |
|---|---|
| `submissions/submission_moderate_iter5.csv` | **68.40** |
| `submissions/submission_moderate_wild.csv` | 65.29 |
| `submissions/C_full_moderate.csv` | 62.33 |
| `Code Base/pipelining/Model 3/iter2/outputs/submission_B.csv` (as in commit `ebacbf8`) | 44.15 |

Since Phase 2, the public weeks are in the data, so public scores no longer predict the private leaderboard. Their remaining use is pinning down scorer conventions (`verify_scorer.py`, `submissions/probes/`).

## Layout

```
docs/                    BRIEF.md, competition.md
Data/, Data_new/         competition CSVs (Phase 1, Phase 2)
submissions/             every file built or uploaded; probes/ = scorer-convention probes
Code Base/pipelining/
  run_pipeline.py        one command: Models 1 -> 2 -> 3, two final files, checks
  backtest_full.py       whole pipeline on past windows, every score part
  score_local.py         local scorer (with alternative conventions)
  verify_scorer.py       which conventions reproduce the public scores
  backtests/             backtest results, one CSV per configuration
  Model 1/iter4/         current disruption model (iter3 + news story layer); iter3 = no news
  Model 2/iter3/         current price model and landed costs (Eshaan)
  Model 3/iter3/         current lean ordering plan; iter1/ = simulator, scenarios, hedged preset;
                         iter2/ = Eshaan's version with notebook
  dq_rules/, sql/, 04_pipeline.py   Databricks data-quality pipeline
```

Older iterations (Model 1 iter1, iter2 and archive; Model 2 iter1 and iter2) are superseded and kept for the record.
