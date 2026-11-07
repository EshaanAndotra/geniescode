# Halcyon final-stage model

This package preserves the original model and adds a Phase 2 audit, expanded historical replay, supplier-level forecasts, and two final submission policies.

Start with `outputs_phase2/REPORT.md`. It explains the original score, calm versus severe periods, forecast errors, changes, and limitations. `outputs_phase2/summary.json` and the CSV tables contain the numerical evidence.

## Submission files

- `outputs_phase2/submission_phase2_ready.csv`: balanced private policy with the already-known public outcomes and optimized public orders.
- `outputs_phase2/submission_phase2_challenger.csv`: aggressive private policy with the same known public portion.

Both have 260 data rows plus the header, covering weeks 209–234. The Phase2 public rows use revealed data through221, and private rows use the forecasts through221. Their local public score estimate is about95.7, which is an in-sample result and does not measure private forecasting quality. The competition allows two final selections, so these are two alternatives, not files to concatenate. The older `submission_final.csv` and `submission_challenger.csv` retain the frozen-origin public replay for validation. See `outputs_phase2/PHASE2_UPLOAD.md` for this distinction.

The recommendation is a robustness judgment after retrospective stress tests. There is no claim that it dominates every historical window or guarantees a private score. The report includes failures as well as improvements. Original Phase 1 outputs are retained under `outputs/`.

## Reproduce everything

Python 3.14 was used. Dependencies are pinned in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python run_phase2.py
```

To force all forecast models to refit:

```powershell
python run_phase2.py --refresh
```

The end-to-end command audits the submitted original, runs the forecast replay, compares policies, forms the policy portfolio, exports and validates both submissions, writes charts and reports, and runs four correctness tests. It can take several minutes. Input CSVs are local and no API keys are required.

## Test the historical period

There is no week 0. Weeks 1–12 initialize the models; 15 independent 13-week windows cover weeks 13–207 exactly. Every fold uses only information available through its origin. The full published demand schedule and already-announced tariff effective dates are the permitted forward inputs. A model trained through 221 and scored directly on its training weeks would not be an honest test.

```powershell
python replay_v2.py --refresh
python evaluate_policy_v2.py
python select_policy_portfolio.py
python finalize_v2.py
python -m unittest test_phase2 -v
```

The release's historical rows through 208 are identical to the originals. `historical_replay.csv` contains 1,950 selected forecast-and-order rows with actual labels. `historical_score_comparison.csv` compares original and recommended policies. `policy_replay.csv` includes the aggressive and other development candidates. `model_comparison.csv` contains forecast metrics by data split.

## Validation boundaries

Development origins are 64, 77, 90, 103, 116, 129 and 142. Later evaluation origins are 155, 168, 181 and 194. Earlier windows are cold-start diagnostics. Numerical forecast selection and the mean-score policy comparison use development folds. The broader history informed architecture and the recommended policy hedge, so the later results are retrospective evaluation, not a pristine untouched holdout.

The original score is reconstructed using a local simulator and a perfect-information LP for the hindsight reference. Assumptions are explicit in the report. The near-match to the reported score supports the diagnosis but does not make these official Kaggle component scores.

Tests cover both final schemas, exact historical coverage, forecast prefix invariance, and policy independence from future supply observations. The simulator's accounting and original baseline also have tests in `test_pipeline.py`.

## Files and workflow

- `audit_phase2.py`: public-period score and forecast audit; hindsight reference used only in evaluation.
- `model_v2.py`: supplier signals, direct 13-week forecasting, signal corroboration and candidate aggregation.
- `replay_v2.py`: fixed-origin historical replay and forecast comparisons.
- `policy_v2.py`: conditional supply scenarios and alternative ordering policies.
- `evaluate_policy_v2.py`: development policy comparison and chronological replay.
- `select_policy_portfolio.py`: explicit recommended hedge and aggressive challenger.
- `finalize_v2.py`: submission generation, validation, report and charts.
- `build_phase2_upload.py`: update the known public portion while verifying that the private rows remain unchanged.
- `run_phase2.py`: end-to-end command.
- `inputs/`: original ten CSVs.
- `inputs_phase2/`: new ten CSVs, renamed from the supplied `(1)` filenames.

Cached forecast files are checked against source fingerprints. Cached pickle files are generated locally by this project; do not replace them with untrusted pickle files. Use `--refresh` after changing feature or model code, even if the data is unchanged.
