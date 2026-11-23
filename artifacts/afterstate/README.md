# Afterstate value experiment

Worktree branch `experiment/afterstate-value` starts at `bbe9cb85235eb42215a021be20aa195f6bbb6b27`, then the fetched tip of `experiment/robust-one-drop`. The fetched tip of main was `5174d72`. The machine had 16 logical CPUs and 125 GiB usable memory. The deployed PPO checkpoint and FlyWire graph were frozen throughout.

[Paired gameplay MP4](comparison.mp4) · [Replay and FlyWire inspection](index.html) · [Per-episode evaluation](evaluation.json) · [Seed and dataset manifest](dataset.json)

## Dataset and scores

The local collector sampled 624 episodes from both the deployed hybrid and the physics-only planner, under both clean and joint dynamics and vision. Each collector sampled uniformly from its top four positions on a seeded 20% of moves. It saved the executed action, immediate score, previously visible next fruit, actual settled board, and reconstructed candidate board. These make 191,515 transitions and two input views per transition. The training, validation, development, and held-out sets use mutually exclusive fruit seeds.

Future labels cover up to 20 subsequent releases and exclude the reward on the creating release. A true end of game makes the remaining label known through overflow. A 500-drop cap censors unfinished targets. Overflow on the creating drop is treated separately from future overflow, and its row is masked out of both objectives. By these rules 190,891 transitions have known future score, and 12,480 contain future overflow. The post-drop fruit revealed by the simulator is excluded from inputs. The observation slot for the current fruit carries the previously visible next fruit because that fruit is now due to drop. The following-fruit slot is masked.

The compact model uses a 12,698-parameter shared fruit encoder, masked mean and max pooling and five horizontal bins. The FlyWire model uses the frozen 512-node circuit with a new 32,962-parameter nonlinear readout. Each learned head sees the same 363,678 training examples and 19,352 validation examples across reconstructed and clean boards. Training uses masked Huber loss on future score divided by 100 plus half the masked binary cross-entropy on future overflow. The selected validation mean absolute errors were 59.1 points for compact and 63.6 for FlyWire. [Compact training](compact-training.json) · [FlyWire training](flywire-training.json)

All 15 candidate boards come from the existing transfer simulator. The model scores them in one batch. Ranking uses immediate simulated points plus 0.5 times predicted future points minus 70 times predicted overflow probability. Both heads selected those weights on eight joint development seeds from four preset combinations. An immediate simulated overflow excludes that action whenever a surviving candidate exists. If all candidates overflow, the largest immediate score wins. Exact ties go to the smallest action index. The earlier PPO actor is used by baseline A and the baseline collector.

## Fresh evaluation

Each entry summarizes 24 paired fruit sequences. A is deployed FlyWire PPO hybrid, B is physics-only, R is the existing perturbation-aware hybrid on these same fresh seeds, C is the compact afterstate head, and D is the FlyWire afterstate head. Environments reuse separate seeded physics and vision streams. The 300-drop and 450-drop scores come from the same episode, stopping earlier on overflow. Every row alive at either cap is identified by seed in the results. The highest tier field uses the engine's zero-based encoding. Latency covers selecting one action, measured during 12-worker evaluation.

| Profile | Method | Mean at 300 | Median at 300 | Above 3,000 at 300 | Overflows by 300 | Mean at 450 | Median / p95 latency at 300 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clean | A | 3,100.8 | 3,110.0 | 18 | 6 | 3,421.0 | 3.28 / 7.17 ms |
| Clean | B | 3,031.3 | 3,100.5 | 15 | 4 | 3,408.3 | 4.94 / 14.34 ms |
| Clean | R | 3,093.0 | 3,107.0 | 17 | 7 | 3,450.2 | 8.17 / 20.38 ms |
| Clean | C | 2,982.9 | 3,053.5 | 13 | 6 | 3,323.3 | 5.22 / 13.70 ms |
| Clean | D | 2,939.0 | 3,079.5 | 15 | 9 | 3,262.4 | 6.68 / 15.38 ms |
| Joint | A | 2,869.1 | 3,057.0 | 15 | 11 | 3,036.8 | 3.67 / 9.67 ms |
| Joint | B | 2,798.8 | 2,982.5 | 11 | 9 | 3,030.9 | 5.00 / 12.84 ms |
| Joint | R | 3,061.0 | 3,091.5 | 17 | 8 | 3,344.6 | 9.31 / 23.70 ms |
| Joint | C | 2,861.6 | 2,942.0 | 10 | 8 | 3,062.5 | 5.87 / 15.41 ms |
| Joint | D | 2,639.8 | 2,892.5 | 9 | 12 | 2,801.5 | 6.77 / 17.24 ms |

At joint 300, compact trailed A by 7.5 points and FlyWire trailed by 229.3. Both won 11 pairs and lost 13. At joint 450, compact led A by 25.7 points, split 12–12, and FlyWire trailed A by 235.2. All 240 extended trials overflowed before or at the cap. The robust planner led joint evaluation at both horizons. The prior robust-planner pilot found a +50.0 point gain versus A under joint noise at 300, compared with +191.9 on this seed set. [Earlier robust results](../planner/results.json) · [Development weight comparison](development.json)

Neither learned head met the selection rule of beating all baselines on joint 300-drop mean. There were **zero new live attempts**. The 3,000-point website target remains unverified.

## Transfer diagnostic

Across 362 executed actions from the archived deployed-hybrid and robust-planner runs, the local simulator predicted 316 score increases. The website showed a score increase on 168 of those drops. For predicted scoring releases, agreement was 93/185 at an edge versus 75/131 away from the edges. With a detected fruit center above 35% of board height, agreement was 21/46 versus 147/270 on other boards. The diagnostic combines perception, physics, and score observation; it uses the website outcome of the executed action only.

A concrete next test is to compare candidate rankings against multiple executed local continuations from the same board, using known physics and reconstructed vision as separate inputs. That would measure action ranking directly, while the current labels measure value regression for a single action collected under each behavior policy. The clean evaluation losses show that closing the website simulator gap alone leaves a learning problem. The archived discrepancies also make immediate simulated merges a poor sole target for transfer calibration. [Website error breakdown and per-drop data](website-diagnostics.json)

## Reproduce

Use the repository's existing Python environment. The generated data stays in gitignored `runs/afterstate.npz`. The committed checkpoint files are `compact.pt` and `flywire.pt`.

```bash
.venv/bin/python -m flymerge.afterstate_data
.venv/bin/python -m flymerge.afterstate_model compact
.venv/bin/python -m flymerge.afterstate_model flywire
.venv/bin/python -m flymerge.afterstate_eval diagnose
.venv/bin/python -m flymerge.afterstate_eval develop --workers 8
.venv/bin/python -m flymerge.afterstate_eval evaluate --workers 12
.venv/bin/python -m flymerge.afterstate_video
.venv/bin/python -m flymerge.check_afterstate
.venv/bin/python -m unittest discover -s tests -v
```

The predeclared video seed is 20270701 under joint perturbations. The FlyWire inspection can seek its corresponding afterstate trace by drop and recurrent step, using the same fruit sequence. The archived media manifest records scores, lengths and checkpoint identity.

On the predeclared showcase, the deployed hybrid scored 2,588 and overflowed at drop 293. Compact scored 2,848 at the 300-drop cap, and the independent FlyWire inspection trace reached 3,027 at the cap. The replay highlights drop 8, where compact scored zero and an alternate move on the same board would have scored 3 immediately. Drop 11 scored 16 total, including a 10-point merge. Individual merges were recovered from conservation of fruit-tier counts between drops. This illustration follows one observed local continuation.

FFmpeg decoded the 3.6 MB MP4. The verifier matched all 893 recorded decisions, scores, and available circuit traces to their deterministic replay inputs. Chromium playback, both illustration seek buttons, the drop slider, and the FlyWire recurrent-step control passed.
