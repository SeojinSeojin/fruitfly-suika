# Game-over audit before retraining

Same branch as the [live baseline](../README.md), as a separate commit. Planner and checkpoint are frozen. No penalty change was tested, because the decision gate pointed to perception and physics (see B4).

[Per-action decision ledger (final 10 drops of each run)](decision-ledger.json) · [Spawn-rule vs local-rule comparison](rule-comparison.json) · [Knowingly-fatal check over all robust decisions](knowingly-fatal-check.json) · [Training-style episode endings](training-endings.json)

## Runs audited

Four archived website runs ended at the site's own game-over screen. Attempt 3 of the baseline ended by detector stop and is skipped. The other earlier captures ended by time cap, aim stop, settling stop or ad occlusion.

| Run | Planner | Score | Drops | Terminal frames |
| --- | --- | ---: | ---: | --- |
| Attempt 1 | robust | 1,813 | 200 | [pre-drop](frames/attempt-1-drop200-predrop.jpg) · [burst](frames/attempt-1-drop200-burst.jpg) · [end](../attempt-1/end.png) |
| Attempt 2 | robust | 2,040 | 222 | [pre-drop](frames/attempt-2-drop222-predrop.jpg) · [burst](frames/attempt-2-drop222-burst.jpg) · [end](../attempt-2/end.png) |
| Archived robust | robust | 1,673 | 174 | [pre-drop](../../planner/website/screens/174.jpg) · [end](../../planner/website/end.png) |
| Archived hybrid | hybrid | 1,725 | 188 | [pre-drop](../../transfer/demo-game-over/screens/188.jpg) · [end](../../transfer/demo-game-over/end.png) |

## B1 ledger

Each decision is recomputed from its saved observation with the frozen checkpoint and the run's action seed, and every executed action reproduces. The ledger has one row per drop for the final 10 drops of each run. Each row records:
- the planned action and current/next tier
- the four highest detected fruit tops in local units, where the box top is 0 and the local overflow line is 11
- the nominal 80-tick post-drop top, clearance to the line, and points
- the actions whose nominal forecast survives
- for robust, every shortlisted candidate's physics term (mean simulated rank value), risk term (−2 × mean overflow risk), PPO term (0.20 × log probability), combined score, and the number of the 7 scenarios that overflow

For hybrid, overflow enters the physics term as the −8 rank penalty.

Across all 40 audited drops, the chosen action's nominal forecast survived. 14–15 of the 15 actions were forecast to survive on every terminal board. The forecast post-drop top is usually far *below* the detected stack: attempt 1 drop 200 goes from 18.6 detected to 143.9 forecast. This is the board collapse documented in the [physics diagnostic](../README.md#simulator-replay-of-executed-actions).

## B2 terminal drops

**The site's rule.** The public game script was read as text and not executed. Game over fires inside the Matter.js collision handler when two fruits of *different* tiers touch and either body has `position.y − centerOffset.y/2 < 70` game px. For a circle body the offset equals the radius, so the test is center − r/2 < 70. The held fruit's center sits at game y = 80, about 37 screen px above the box top.

A freshly released fruit of tier 2 or above therefore triggers game over if it touches a different fruit within a few pixels of leaving the hand. Converted to a straight-down drop, the fatal height is column-local and depends on the tier. A drop is fatal if a different-tier fruit under it has its top above about −15.9 local units for a cherry, −10.0 for a strawberry, 0.5 for a grape, 3.9 for tier 4, or 15.0 for an orange. The local check instead tests whether any fruit top is above 11, for the whole board, once after 80 ticks (1.33 s).

**Spawn-contact rule applied to detections**, using site-derived radii and the targeted screen x:
- It flags the terminal drop of both archived runs as fatal: robust contact −24.3 vs limit −19.4, hybrid −23.4 vs −19.4.
- It flags none of the 962 non-terminal releases across the five runs.

The local 80-tick check predicted overflow on 0 of 966 website releases, terminal drops included.

| Run | (a) Detector missed a fruit near the line | (b) Planner predicted overflow but chose it | (c) Local simulation predicted survival | When game over triggered |
| --- | --- | --- | --- | --- |
| Attempt 1 | **Yes.** The grape wedged in the top-right corner (screen ≈ 865, 232) has no detection. The strawberry was released at aim 15 directly onto it. With that grape added, the spawn rule fires (−30.0 vs −27.7). | No (risk 0 in all 7 scenarios) | Yes | The strawberry never falls: it is still on the grape at +0.16 s and +0.46 s, and the overlay is up at +0.94 s. The site shows the overlay after an 800 ms camera shake, so the trigger came ≤ 0.14 s after release (≤ ~9 ticks). |
| Attempt 2 | **Yes.** The top-right grape (screen ≈ 868, 250) is undetected. The orange at aim 14 slides against it. With it added, the rule is borderline (−18.8 vs −19.4). | No (mean risk 0.66, no scenario overflowing) | Yes | The orange is at spawn at +0.18 s and sliding against the grape at +0.51 s. There is no overlay at +0.94 s; the restart button is visible at the next check. The trigger falls between about +0.14 and +0.4 s. |
| Archived robust | No. The melon under the drop was detected (top 14.2). | No (risk 0) | Yes | **Unresolved.** No burst frames, only the pre-drop and end screens. The pre-drop geometry puts the orange's spawn in contact with the melon. |
| Archived hybrid | No. The orange column at the right wall was detected (top 12.4). | No (overflow not forecast) | Yes | **Unresolved**, for the same reason. The geometry puts the orange's spawn in contact with the column. |

The planner never knowingly selected a predicted-fatal move. Over all 778 robust website decisions, the chosen candidate never had all 7 scenarios overflowing, and its mean risk never exceeded 0.66.

The detector crops the board at the box top and masks its first 12 px. A fruit pushed into the band between the box top and the spawn height, or squeezed into a corner against the wall, is partly or wholly invisible. That band is exactly where the site's game over happens.

## B3 training episodes: overflow versus drop cap

No archived training log records how episodes ended. The episode log stores score, highest tier and length only, so an episode ending at length 300 is ambiguous. These counts are **new measurements** from an instrumented collection with the training worker environment: missing current/next 12%, transfer rules, cap 300. Actions are sampled from the frozen deployed checkpoint with no updates, 4 × 6,144 drops per profile.

| Profile | Completed episodes | Overflow before cap | Ended at cap | Overflow on the cap drop | Mean length | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed physics (deployed checkpoint's training profile) | 89 | 50 | 39 | 0 | 271.1 | 2,762 |
| Joint physics + vision noise | 86 | 41 | 45 | 1 | 279.2 | 2,848 |

When overflow and the cap occur on the same drop, `Orchard.step` sets `done |= truncated` and returns `terminal = done and not truncated`. The overflow is therefore reported as `terminal=False, truncated=True`, so episode accounting counts it as a cap ending. In `train.py`, both flags become `done`, and GAE uses `alive = 1 − done`. Overflow and cap endings get the same zero bootstrap, and there is no overflow penalty in the reward (reward = merge points). The label error does not change the PPO update. However, the 39–45% of episodes cut at the cap are also treated as terminal, even though the observation carries no drop count.

## B4 decision

The main terminal error was perception and physics, not penalty selection:
- In 2 of 4 runs the detector missed the corner fruit that caused the fatal contact.
- In all 4 runs the local simulator forecast survival.
- The local overflow test differs from the site's in both place and timing. It uses a fixed line inside the box, checked at 80 ticks. The site's rule is a tier-dependent contact at spawn height, decided within roughly 0.4 s of release.
- No candidate set ever contained the "every scenario overflows, another survives" pattern on the website.

The paired penalty test was therefore not run, and no planner or weight change was made.

Supported next steps, before any retraining or penalty change:
1. Extend the detector crop above the box top and into the wall corners so fruit in the spawn band are observed.
2. Replace the local overflow check with the site's spawn-contact test for the released fruit, and stop collapsing the static board, as covered by the physics-diagnostic recommendation.

The spawn-contact rule used here is a diagnostic only: it flagged both archived terminal drops from all 962 earlier releases, but that is a small sample.

## Commands

```bash
.venv/bin/python -m flymerge.gameover_audit   # ledger, rule comparison, knowingly-fatal check
.venv/bin/python -m flymerge.episode_audit
```
