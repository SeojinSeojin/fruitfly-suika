# Counterfactual drop ranking

The separate worktree for `experiment/counterfactual-drop-ranking` started at `bc06ddac9f3d16d7144f510cae3e9dae2f6bf6cb`, the fetched tip of `experiment/afterstate-value`. The CPU host exposed 16 logical cores. The deployed PPO checkpoint and FlyWire recurrent graph stayed fixed.

[Paired MP4](comparison.mp4) · [Interactive replay](index.html) · [Held-out episode results](evaluation.json) · [Ranking accuracy](ranking-accuracy.json)

## Teacher test

Each board shortlist contains six unique aims. It starts with the robust planner's aim and both previous afterstate scorers' aims, then fills missing slots with the highest nominal physics ranks. Ranking ties favor the smallest action index. Each first action is executed from the actual local board with its velocities and dynamics. A branch uses its immediate score plus subsequent points minus 150 for overflow. A fixed first-contact continuation policy favors same-tier contacts and low landing positions. It acts using the board and current preview at that step. The current fruit and the already visible preview are preserved. Three independently sampled tails use the existing uniform 0–4 fruit generator. The same tail, physics and per-step vision RNG seed are used across action branches where applicable. The search RNG is separate from the played episode's fruit generator. All branches stop at overflow or after 20 subsequent drops.

The audit sampled 184 boards from robust and physics-only games in clean and joint profiles. It included 26 boards with at least 20 fruits and three with a fruit center within 80 local pixels of the top. Under independently sampled future triples, the winner repeated on 43 of 184 boards. All 15 placements were tested on 95 boards and the shortlist contained the winning move on 48. The average estimated advantage of selecting the teacher's chosen action over the robust action shrank from 33.2 to 12.0 utility points under independent resampling. The audit took 29 seconds. [Teacher audit summary](teacher-diagnostics.json)

At 300 drops the teacher averaged 3,153.3 against 3,068.3 for robust, a paired gain of 84.9 points with 8 wins and 4 losses across 12 clean development seeds. This fell below the original +100-point gate. At the 450-drop horizon it averaged 4,391.4 versus 3,488.7, a +902.8 point gain with 10 wins and 2 losses. Five teacher episodes were alive at 450, and all robust episodes had overflowed. The extended result was the reason to proceed with supervised training after the 300-drop gate missed. That decision was recorded before generating the larger data. Teacher decisions took 185.6 ms median and 406.5 ms p95, compared with 9.1 and 27.0 ms for robust during the same worker run. The full teacher test took 139 seconds. [Teacher results](teacher-results.json) · [Stage decision and seeds](decision.json)

## Ranker training

Twelve CPU workers generated 1,520 board groups in 99.8 seconds, covering 9,120 candidate actions and 27,360 branch continuations. The split contains 1,155 training groups, 139 development groups and 226 independent test groups. Every candidate from a board stays in its group. Fruit seeds are disjoint across splits. Complete rollout arrays live in gitignored `runs/`, and the small checkpoints are committed.

Both rankers initialize from the corresponding previous afterstate scorer, using its compact layout encoder or frozen FlyWire reservoir. Training inputs are reconstructed candidate boards, the next fruit visible before the root move, and two simulator-derived features for the candidate's immediate score and immediate overflow. The learned scalar ranks the total risk-adjusted target, including immediate points. The model combines a confidence-weighted logistic loss over action pairs with auxiliary total-score regression and overflow prediction. Confidence uses the mean and standard error of *paired* rollout differences so noisy comparisons receive less weight. Old scorer and robust choices nominate the shared shortlist. The rankers choose among those candidates using their learned outputs. Each model's saved epoch minimizes mean candidate regret on development boards.

On the 111 held-out joint boards, compact correctly ordered 52.0% of non-tied reconstructed candidate pairs and 50.4% with exact local afterstates. FlyWire reached 48.1% and 48.0% respectively. Feeding exact boards reduced mean best-candidate regret by roughly 1.4 points for each scorer. On average, both rankers still chose lower-utility actions than the robust planner under the independent joint rollout labels. [Accuracy and regret by input](ranking-accuracy.json)

## Held-out gameplay and live selection

Each row contains 24 fresh paired games. The 300-drop column stops scoring at overflow or drop 300. The extended column continues that same episode through overflow or the 450-drop cap. Rightmost frequency and latency refer to the common horizon. The JSON includes paired deltas, win/tie/loss counts, highest tier, and the seeds still alive at each cap. All 192 extended gameplay episodes eventually overflowed. Local evaluation took 173 seconds with 12 workers. Compact training took 8.6 seconds, followed by 4.7 seconds for FlyWire training.

| Profile | Method | Mean at 300 | Median at 300 | Above 3,000 at 300 | Mean at extended horizon | Rightmost | Median / p95 decision time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clean | Robust | 3,153.5 | 3,179.5 | 22/24 | 3,673.2 | 77.0% | 9.2 / 21.7 ms |
| Clean | Old compact afterstate | 2,989.2 | 3,022.5 | 13/24 | 3,357.5 | 16.5% | 5.2 / 13.8 ms |
| Clean | Compact ranker | 2,928.6 | 3,085.0 | 15/24 | 3,264.3 | 11.1% | 34.0 / 99.7 ms |
| Clean | FlyWire ranker | 2,808.6 | 2,905.5 | 10/24 | 2,978.7 | 9.3% | 45.1 / 105.4 ms |
| Joint | Robust | 2,986.6 | 3,070.0 | 15/24 | 3,363.0 | 68.6% | 8.7 / 24.4 ms |
| Joint | Old compact afterstate | 2,684.7 | 2,707.5 | 8/24 | 2,793.3 | 15.0% | 5.9 / 14.3 ms |
| Joint | Compact ranker | 2,478.6 | 2,318.0 | 4/24 | 2,570.5 | 11.8% | 37.7 / 105.0 ms |
| Joint | FlyWire ranker | 2,731.3 | 2,903.0 | 7/24 | 2,857.9 | 10.2% | 43.3 / 101.1 ms |

Joint paired changes at 300 drops were −508.0 points for compact, with 6 wins and 18 losses, and −255.3 points for FlyWire, with 6 wins, one tie and 17 losses. The lower right-edge frequency coincided with lower scores. Inference latency includes generating the shared shortlist through robust and both old heads. The robust planner remains the best tested gameplay method. **Zero new website attempts qualified, and the 3,000-point website target remains unverified.**

The archived website diagnostic covers executed actions only. A visible score increase followed 93 of 185 predicted score increases at an edge, compared with 75 of 131 away from edges. On boards with at least 15 detected fruits the count was 77 of 158, versus 91 of 158 on less crowded boards. The independent local ranking results show weak ordering even with exact afterstates. More reliable relative labels are the next bottleneck. An adaptive audit can widen the shortlist and resample uncertain comparisons before spending on a larger training set. [Archived diagnostics](website-diagnostics.json)

## Reproduction

Generated rollouts and raw arrays remain under `runs/`. The local commands use the existing CPU environment.

```bash
.venv/bin/python -m flymerge.rollout_teacher --workers 12
.venv/bin/python -m flymerge.ranking_data --workers 12
.venv/bin/python -m flymerge.ranker compact
.venv/bin/python -m flymerge.ranker flywire
.venv/bin/python -m flymerge.ranker inspect
.venv/bin/python -m flymerge.ranking_eval --workers 12
.venv/bin/python -m flymerge.ranking_eval --diagnose
.venv/bin/python -m flymerge.ranking_video
.venv/bin/python -m flymerge.check_ranking_replay
.venv/bin/python -m unittest discover -s tests -v
```

## Showcase

The predeclared fresh seed was 20271201 under joint perturbations. Robust reached 3,208 in 330 drops and the best learned method, FlyWire, reached 2,919 in 293. The interactive view jumps to drop 6. FlyWire aimed at slot 9 for zero immediate points. A one-drop fork of that same local board awarded 10 points for the robust aim at slot 15. The played FlyWire trajectory then earned 93 points over the following 20 drops. The viewer synchronizes the FlyWire circuit to the selected simulated afterstate for each FlyWire decision.

FFmpeg decoded the 4.0 MB MP4. The verifier reproduced all 623 recorded decisions, immediate and cumulative scores, and circuit traces against the committed checkpoint hashes and per-episode results. Chromium verified playback, the disagreement and 20-drop seek controls, the terminal score, and the recurrent-step selector. All 21 unit tests passed.
