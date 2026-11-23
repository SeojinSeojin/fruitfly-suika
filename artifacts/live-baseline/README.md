# Live robust baseline and physics diagnostic

Branch `experiment/live-robust-baseline` starts at `b2d689ae72652f15304e1b1ac96f6bf0d910e246`, the tip of `experiment/rollout-state-estimation`. The planner and weights are frozen: strategy `robust` with checkpoint `artifacts/transfer/checkpoints/flywire-seed11-final.pt` (SHA-256 `b2316d67…e1e7f`). Browser adapter, screenshot gating, aim verification and simulator are unchanged. The only code change to gameplay is an opt-in `--record` flag that saves frames.

[Interactive replay of all three attempts](index.html) · [Attempt 2 final 10 drops (MP4)](attempt-2-final-drops.mp4) · [Verification](verification.json) · [Per-release diagnostic ledger](physics-diagnostic.json)

## Website attempts

Three sequential headed Chromium sessions on https://playsuikagame.com/ko used a 30-minute cap, natural game over, and action seeds 49001–49003. The attempt count was fixed in advance.

| Attempt | Final score | Drops | Stop reason | Highest visible tier | > 3,000 | Duration |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 1 | 1,813 | 200 | game over | 10 (melon) | no | 20 min 58 s |
| 2 | **2,040** | 222 | game over | 10 (melon) | no | 23 min 46 s |
| 3 | 1,953 | 182 | detector stop: "fruit centers or ready cue remained uncertain" | 11 (watermelon) | no | 19 min 22 s |

The best live score is 2,040, and no attempt exceeded 3,000. The tier is taken from the terminal frames. In attempt 3, a watermelon formed after drop 182 and the detector never recognized it: the highest detected tier in any observation was 10. Unexplained foreground then kept the 70-second readiness recovery from confirming a stable board ([pause frame](attempt-3/paused.png)). Terminal frames: [attempt 1](attempt-1/end.png), [attempt 2](attempt-2/end.png).

A separate 2-drop recorder check ran before the batch (score 0). It is not one of the three attempts.

## Recording and cadence

For each release, the capture stores:
- the pre-drop observation and its FlyWire trace in `website.js`
- the executed aim, both requested `target_x` and the hand position actually verified (`hand_x`)
- monotonic timestamps for the decision, release and readiness
- the visible score before and after
- every screenshot the readiness loop already takes, which includes the settled board used for the next decision

The fixed 750 ms post-release wait is filled with timestamped screenshots instead of idling, and JPEG encoding runs off the control thread. Bulk frames (6,348 in total) stay in gitignored `runs/live-baseline/attempt-N/record/`. The per-drop annotated screens are committed at half resolution.

| Capture | Median / p95 release interval | Median post-release wait |
| --- | ---: | ---: |
| Attempt 1 | 4.89 / 11.69 s | 0.752 s |
| Attempt 2 | 4.98 / 17.31 s | 0.752 s |
| Attempt 3 | 4.75 / 11.12 s | 0.751 s |
| Archived robust run (no recording) | 4.81 / 13.88 s | 0.750 s fixed |

The longest recorded wait was 0.786 s. Readiness checks are unchanged code. Replaying the saved observations with the frozen checkpoint and each attempt's action seed reproduces all 604 executed actions.

## Simulator replay of executed actions

For each of the 601 releases with a following settled observation, `flymerge.live_diagnostic` rebuilds the local transfer board from the pre-drop detections. It drops the held fruit at the executed aim, runs the planner's 80-tick physics, and compares three things with the website:
- **Merge points:** predicted points vs the visible score change.
- **Merge events:** solved from tier-count changes, so they are unique when they exist.
- **Layout:** same-tier optimal assignment of predicted vs next detected fruit, within 40 local px.

A drop counts as *perception-consistent* when the detected tier counts before and after explain the visible score change exactly.

| Outcome on 601 releases | Count | Perception-consistent | Perception-inconsistent |
| --- | ---: | ---: | ---: |
| Points agree exactly | 140 | 120 | 20 |
| **Simulator merges, site scores nothing** | **267 (44%)** | 148 | 119 |
| Both merge, different points | 194 | 88 | 106 |
| Site merges, simulator predicts nothing | 0 | 0 | 0 |

The simulator predicted scoring on 543 drops, while the site scored on 276. Merge-event multisets agreed on 51 of the 356 perception-consistent drops. The median matched-center error of the settled layout was 14.2 local px, with 7.1 predicted and 11.4 detected fruit unmatched per board. Running 300 ticks instead of 80 changed the predicted points on 127 drops. The detector's tier transition failed to explain the score on 245 of 601 drops (41%), so perception noise is substantial. Even so, the dominant mismatch also appears on drops where perception is consistent.

**Most common observable mismatch: the simulator predicts a merge that the website never makes (267 of 601 releases, 44%).** The same board simulated *without any drop* already merges fruit within 80 ticks on 424 of 601 releases (71%). The detected resting board is not a resting configuration for the local physics.

Representative panels show the pre-release detections, the 80-tick replay and the website's settled board. In [attempt 1 drop 20](frames/attempt-1-drop020-simulator-merge-consistent.jpg), the orange released at the right wall shoves the whole bottom row left in the simulator. Two apples meet and score 21, while the site row does not move. In [attempt 1 drop 42](frames/attempt-1-drop042-simulator-merge-inconsistent.jpg), the apple resting on the peach rolls off in the simulator and merges with another apple, but stays put on the site. The detector also missed a cherry on that board, but that is not what produced the predicted merge. The [frames folder](frames/) holds two consistent and two inconsistent examples per attempt and category.

These cases are clear physics and state-model errors rather than missed fruit. The site's public game script (read as text, not executed) creates Matter.js circle bodies with **friction 1, bounce 0**, gravity 1.5, and sprite-derived radii at scale 0.75. Converted to local units, those radii are 2–13% larger than `TRANSFER_RADII`: 13.8, 17.7, 24.8, 27.0, 34.3, 43.9, 51.2, 62.8, 70.7, 87.0, 101.3. Measured spacing between detected neighbors (median 1.02–1.11× the local radius sum) agrees with that.

### Targeted calibration with a holdout

The rule was fixed before viewing the holdout: among nominal, site radii, and site radii with zero bounce, choose the variant with the most score-increase agreements on attempts 1–2, then report all of attempt 3 untouched. Site radii was chosen.

| Attempt 3 (holdout) | Exact points | Increase agreement | Sim-only merges | No-drop merges | Mean abs. points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 61 / 181 | 109 | 72 | 101 | 43.1 |
| Site radii (selected) | 76 / 181 | 118 | 62 | 89 | 27.1 |

Three exploratory sensitivities were run afterwards and are not holdout-selected:
- **Horizontal damping as a friction proxy:** [friction-sensitivity.json](friction-sensitivity.json). Small gain only.
- **Relaxing the board with merging disabled before the drop:** [presettle-sensitivity.json](presettle-sensitivity.json). No gain, because compaction creates same-tier contacts.
- **Subtracting the same board's no-drop simulation from each prediction:** [baseline-subtraction-sensitivity.json](baseline-subtraction-sensitivity.json). With site radii, attempt 3 reached 112 / 181 exact, 133 increase agreements and a mean absolute error of 9.3 points. Attempts 1–2 reached 120 / 199 and 126 / 221 exact.

The policy was not changed at any point in this batch.

## Next change supported by the footage

Make the planner's one-drop simulator count only merges *caused by the drop*. Use the site's sprite-derived radii, and score each candidate by its outcome relative to the same board simulated without the drop. In the footage, the simulator invents merges from boards that sit still on the website. This test does not check whether the change raises live scores. It should be tested on paired local games before any website run.

## Commands

```bash
bash flymerge/run-live-baseline.sh          # three sequential attempts, frames under runs/live-baseline
.venv/bin/python -m flymerge.live_package    # compact copies, replay page
.venv/bin/python -m flymerge.check_live_baseline
.venv/bin/python -m flymerge.live_diagnostic
.venv/bin/python -c "from flymerge.live_package import final_drops_video; final_drops_video()"
```

The site now refuses Python's default `urllib` user agent for sprite downloads (HTTP 403). The 11 public sprites were fetched once with curl into `data/site/`.
