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

## Held-out results

The 12-worker run took 16 minutes 48 seconds on the 16-core Intel Haswell host. Games use 24 joint and 12 clean fruit seeds, with six exact-state games per profile. The extended endpoint is natural overflow or 450 drops. Games that overflow before 300 carry their terminal score into the common-horizon comparison. Latency covers decision computation. Win, tie, and loss counts compare scores on matching seeds.

###  Joint

| Method | Mean at 300 | Extended mean | Δ vs robust | W/T/L | >3,000 | Alive at cap | Median/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| robust | 2,953.4 | 3,224.6 | +0.0 | 0/24/0 | 18/24 | 0 | 11.0/28.2 |
| current | 2,737.1 | 2,872.2 | -352.5 | 9/0/15 | 13/24 | 0 | 180.5/396.0 |
| tracking | 2,850.8 | 3,021.8 | -202.8 | 8/0/16 | 14/24 | 0 | 188.4/410.9 |
| physics | 2,828.8 | 3,042.2 | -182.4 | 9/0/15 | 13/24 | 0 | 180.9/410.5 |
| both | 2,704.1 | 2,869.5 | -355.2 | 7/0/17 | 10/24 | 0 | 185.6/415.9 |
| exact | 3,318.5 | 4,428.5 | +1,225.2 | 5/1/0 | 6/6 | 1 | 411.8/618.5 |

###  Clean

| Method | Mean at 300 | Extended mean | Δ vs robust | W/T/L | >3,000 | Alive at cap | Median/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| robust | 3,087.2 | 3,460.2 | +0.0 | 0/12/0 | 10/12 | 0 | 10.0/25.2 |
| current | 3,104.1 | 3,528.0 | +67.8 | 7/0/5 | 11/12 | 0 | 186.7/422.9 |
| tracking | 3,224.2 | 3,901.2 | +441.1 | 7/0/5 | 12/12 | 1 | 173.8/387.9 |
| physics | 3,072.8 | 3,353.2 | -106.9 | 5/0/7 | 9/12 | 0 | 179.4/399.0 |
| both | 3,107.2 | 3,512.6 | +52.4 | 5/0/7 | 9/12 | 0 | 175.9/358.4 |
| exact | 3,297.7 | 4,549.0 | +1,188.8 | 6/0/0 | 6/6 | 1 | 175.2/363.2 |

The best joint rollout variant used physics sampling. Its 3,042-point mean was 182 points below robust, so the website selection gate stayed closed and this experiment has no live scores. Center tracking raised the current rollout's extended mean by 150 points on joint boards. Combining the changes gave up those gains. On the six paired joint diagnostic seeds, tracking led the reconstructed runs at 3,034 points. Robust scored 3,203 and exact-state rollout scored 4,429. The exact-state method has access to local simulator information unavailable to the browser.

[Per-episode results](results.json) include paired statistics at both horizons, cap seeds, and the decisions from the preselected replay.
