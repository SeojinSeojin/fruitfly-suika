# Spawn-contact live validation (seeds 50001–50003)

These are three fresh visible attempts on https://playsuikagame.com/ko. Each run used the frozen robust planner with the extended band/corner detector and the spawn-contact check (`--extended-vision --contact-check`), a 30-minute cap and natural game over. Nothing else changed: no retraining, no merge-score changes, and the no-drop merge correction stays out.

The deployed code is unchanged from reviewed commit `056f40c` (`site.py`, `spawn_contact.py`, `robust_planner.py`). The checkpoint is `artifacts/transfer/checkpoints/flywire-seed11-final.pt`, SHA-256 `b2316d67…0e1e7f`.

## Results

| Attempt | Action seed | Score | Drops | Time | Highest visible tier | Highest detected tier | Stop reason |
|---|---|---|---|---|---|---|---|
| 1 | 50001 | **1,739** | 195 | 17.9 min | 10 melon | 10 | game over |
| 2 | 50002 | **2,031** | 218 | 23.7 min | 10 melon | 10 | game over |
| 3 | 50003 | **2,375** | 233 | 23.1 min | 11 watermelon | 11 | game over |

- **Best score: 2,375.** None of the 3 attempts went above 3,000.
- For context, the earlier live scores were 1,813, 2,040 and 1,953 (mean 1,935). This batch's mean is 2,048.
- The site's fruit sequence is not controlled, so three runs cannot separate the fix from luck. The best run's extra points come largely from the watermelon merge.

## Verification

[`verification.json`](verification.json) re-checks every decision in all three attempts (646 total) against the deployed checkpoint:

- It re-derives the robust (legacy) choice from the recorded observation and the action-seed random stream.
- It recomputes all 15 contact labels from the observation plus the recorded band fruit.
- It confirms the gate reproduces the executed aim.

Every decision matched. Each manifest records checkpoint, `strategy: robust`, `extended_vision: true`, its action seed, and `time_cap_seconds: 1800`. The band background was built from 3 shots with 0 uncovered columns in every run.

## Overrides

[`override-ledger.json`](override-ledger.json) is empty: **the contact check blocked 0 of 646 moves.**

- The robust choice was never labeled fatal. Fatal labels appeared on only 3 boards in total:
  - attempt 2, drops 217–218: 4 and 8 fatal aims; the chosen aim was clear;
  - attempt 3, drop 231: 1 fatal aim; the chosen aim was uncertain.
- Executed aims labeled uncertain:
  - attempt 1: drops 192 and 195;
  - attempt 2: drops 182 and 199;
  - attempt 3: drops 231 and 232.
- No override happened, so no overridden aim has an unknown outcome to report. If one ever does, its outcome stays unknown, because the original aim is not executed.

## Watermelon (attempt 3)

- **The detector recognized it.** It first appears in the observation at drop 175, after the merge, and stays detected on all 59 remaining boards.
- **Play continued for 58 more drops.** The game ended with the watermelon still on the board (see the end frame).
- The archived run's failure was a dropped watermelon detection (coverage 0.39). It did not recur.

## Terminal frames and recordings

Each attempt folder holds:

- `final-predrop.jpg`: the raw aim frame captured just before the final release;
- `final-release-strip.jpg`: the aim frame plus every recorded frame after the release (bursts at about 0.3 s and 0.6 s, then settle);
- `end.png`: the game-over overlay;
- `screens/`: half-size per-drop screens;
- `manifest.json` and `website.js`: the compact capture.

The final-drop recordings are `attempt-N-final-drops.mp4` in this folder.

| Attempt | Pre-drop | Release strip | End | Final drops |
|---|---|---|---|---|
| 1 | [frame](attempt-1/final-predrop.jpg) | [strip](attempt-1/final-release-strip.jpg) | [end](attempt-1/end.png) | [mp4](attempt-1-final-drops.mp4) |
| 2 | [frame](attempt-2/final-predrop.jpg) | [strip](attempt-2/final-release-strip.jpg) | [end](attempt-2/end.png) | [mp4](attempt-2-final-drops.mp4) |
| 3 | [frame](attempt-3/final-predrop.jpg) | [strip](attempt-3/final-release-strip.jpg) | [end](attempt-3/end.png) | [mp4](attempt-3-final-drops.mp4) |

The replay viewer is [`index.html`](index.html); it has one capture per attempt. Bulk recordings (every burst and settle frame) stay under `runs/spawn-live/attempt-N/record/` and are not committed.

## What still limited the score

1. **The final drops were not knowingly fatal.**
   - **Attempt 1:** the final aim (15) was *uncertain* with 2.7 local units of slack, against a grape at the top-right rim. The gate only blocks *fatal* moves, by design, so it passed even though 13 aims were clear. The board was packed to the rim at the time.
   - **Attempts 2 and 3:** the final aims were labeled *clear*.
2. **Rim and corner detection is still incomplete.** Both "clear" endings trace back to fruit at the rim that the detector misread:
   - **Attempt 3:** a grape wedged against the right wall at the rim was detected at drop 232 (local 386.6, 12.5) but missing at drop 233. The held orange's mask column covered that spot (likely cause), and the orange was then aimed straight onto the grape.
   - **Attempt 2:** an apple partly behind the front rim was read as two cherries (tier 1, y 15.6). The predicted contact was with a phantom cherry, so the contact height was wrong.
   - The extended detector recovered the archived misses, but its masked/rim matching still drops or mis-tiers fruit in this zone.
3. **The simulator still mismatches the site's merges.** The local model merges where the site does not in about 44% of releases (Part A). This correction was deliberately left out, so the planner's value estimates for crowded boards remain optimistic. All three boards filled with mid-tier fruit (oranges, apples, pears) that were not merged in time.
4. **Uncertain moves were never refused.** About one executed aim per run was uncertain. Two of the three endings came at or right after an uncertain or rim-adjacent move. Refusing uncertain aims when a clear one exists is a policy change for a later experiment.
5. **The fruit sequence is uncontrolled.** Scores of 1,739 to 2,375 sit within the spread of the earlier runs (1,813 to 2,040).

Related: [archived-board replay ledger](../replay-ledger.json) and [before/after detection frames](../frames/) (see [`frames.json`](../frames.json)).
