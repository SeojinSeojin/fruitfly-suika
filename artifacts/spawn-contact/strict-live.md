# Strict gate + mask memory: live attempts (seeds 50004–50006)

Configuration: the robust planner, extended detector, and spawn-contact check, plus `--contact-strict --mask-memory` (commit `16e0dee`). Same checkpoint, a 30-minute cap and natural game over. Raw captures are in `runs/strict-live/attempt-N/` (manifest, website.js, screens, record, log).

| Attempt | Seed | Score | Drops | Stop reason | Strict/memory overrides | Final aim label |
|---|---|---|---|---|---|---|
| 1 | 50004 | 1,201 | 124 | **paused**: fruit centers or ready cue stayed uncertain (board low, game still running) | 0 | clear |
| 2 | 50005 | 1,342 | 178 | game over, board full | 7 (drops 134–178; 5 uncertain, 2 fatal) | clear |
| 3 | 50006 | 1,361 | 167 | game over, board full | 0 | clear |

For comparison:
- spawn-contact check (seeds 50001–50003): 1,739 / 2,031 / 2,375
- frozen robust planner: 1,813 / 2,040 / 1,953

## What this shows

- **No measurable gain; this batch scored lower.** With 3 runs and an uncontrolled fruit sequence, the drop is within noise, but nothing here supports the change either.
- **The new rules almost never acted.**
  - Attempt 3 made zero overrides, so it played exactly as the earlier configuration would have; its low score is the sequence and the planner, not the gate.
  - Attempt 2's overrides all came in the last 45 drops, once the board was already full.
  - Mask memory carried no fruit in any attempt.
- **Both game overs were boards packed with unmerged small and mid fruit.** The final aims were labeled clear. The spawn-contact layer only decides the last few drops; by then the game is already lost. The real limiter is earlier: the planner fills the board. That is consistent with the simulator over-predicting merges (about 44% of releases, Part A), which is still uncorrected.
- **Attempt 1 was cut short by the harness, not the game.** It paused at 1,201 with a low board and a melon in play: a robustness bug in the ready/coverage retry logic. It should retry instead of stopping.

## Recommended next steps

1. Fix the merge mismatch in the planner's simulator: site radii plus the no-drop board correction (exploratory holdout exact 112/181). This acts on every drop, not just the last few.
2. Make the ready-cue/coverage pause retry a few seconds longer before stopping.
3. Keep `--contact-strict --mask-memory` as an end-game safety layer. It is cheap (7 of 646 aims in the label replay, `strict-replay.json`), but don't expect score from it.
