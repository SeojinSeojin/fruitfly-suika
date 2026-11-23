# Rollout state estimation and physics uncertainty

The experiment starts at `712f08160696ef807e2e1746536617e77737d35c`. The deployed checkpoint, robust reference, six-action shortlist, 20-drop continuation policy, and three tails per action are shared with the rollout-transfer experiment. [Protocol](protocol.json) records the seeds, preselected joint replay, and website selection rule. [Frozen settings](frozen.json) records the development choices.

The browser uses the latest detection set and up to two preceding settling frames. Matches require the same tier and mutual proximity, with ambiguous assignments rejected. A median blend adjusts matched centers. The development velocity estimate fits centers against monotonic timestamps in pixels and converts to pixels per 60 Hz simulator tick. Unreliable fits fall back to zero. Local evaluation uses three pre-drop observations at the end of the previous drop, with independent detector noise for earlier frames.

On joint development boards, matched center RMSE was 7.20 pixels for the single frame, 6.03 for center tracking, and 6.13 with velocity estimation. Velocity RMSE was 1.47 pixels per tick with zero velocity and 1.48 with the velocity fit. Tracking therefore uses zero velocity. About 72% of matched joint-profile centers moved at most two pixels across the final 22 local ticks. [Tracking calibration](tracking-calibration.json) and [diagnostics](diagnostics.json) include the full clean and joint breakdowns.

Physics sampling pairs three fixed parameter sets with three independently generated fruit tails via a permutation from a separate RNG stream. Every candidate shares the same scenario assignments, for 18 branches per choice. The two alternatives and nominal physics are recorded in the frozen config. The archive provides one observation per executed drop and its score change. The nominal model agreed on whether points increased in 114 of 188 fit drops and 99 of 174 validation drops. The two alternatives reached 116 and 118 on fit, and 102 and 100 on validation. This checks event agreement on actions already taken. The images do not resolve velocity or identify individual physical coefficients. [Physics calibration](physics-calibration.json) records the candidate ranges and captures.

On 26 joint audit boards, mean regret against exact-state forecasts was 41.23 points with current rollout, 34.10 with tracking, 42.21 with physics sampling, and 45.06 with both. The audit uses matched futures and the same shortlist. Full-game evaluation measures the gameplay effect separately.

Run from the repository root with the existing CPU environment and frozen checkpoint.

```bash
.venv/bin/python -m flymerge.state_calibration
.venv/bin/python -m flymerge.state_diagnostics
.venv/bin/python -m flymerge.state_experiment --workers 12
.venv/bin/python -m flymerge.state_video
.venv/bin/python -m flymerge.state_video --verify
.venv/bin/python -m unittest discover -s tests -v
```

When the held-out joint extended mean exceeds robust, `bash flymerge/run-state-live.sh` launches three headed browser attempts, each capped at 1,500 seconds.
