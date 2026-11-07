# Phase2 upload versus historical validation

Use `submission_phase2_ready.csv` as the balanced upload, or
`submission_phase2_challenger.csv` for the aggressive private policy.

The Phase2 release reveals all public weeks209–221. These uploads populate
their public prediction columns with the released outcomes/prices and use
perfect-information public orders. Their private rows222–234 are identical
to the corresponding existing forecast files.

The local public score estimate is about95.7. This is an in-sample result,
not an improvement in unseen forecasting. The official score may differ
because the organizer's exact simulator was not supplied. High public scores
after the release cannot establish private prediction quality.

The files `submission_final.csv` and `submission_challenger.csv` retain the
frozen-origin public replay for analysis. Do not compare that historical
replay score directly with post-release public scores using known outcomes.
The prior public score estimates in REPORT.md refer to that replay.

Rebuild these upload variants with `python build_phase2_upload.py` after
running `python run_phase2.py`. The builder validates all260 rows and confirms
that private rows are unchanged. It does not submit anything to Kaggle.
