# One-drop planner experiment

Branch `experiment/robust-one-drop` starts from `5174d72`. The fetched tip of main matched that commit at launch. All conditions use the deployed FlyWire checkpoint with SHA-256 `b2316d67a4538d55f2270633497dfe521a1dfd29dfc3ba0406ab8733380e1e7f`. The machine had 16 logical CPUs. No PPO updates were performed.

[Paired comparison MP4](comparison.mp4) · [Interactive circuit replay](index.html) · [Replay manifest](comparison.json) · [Held-out results](results.json) · [Per-change simulations](results-decisions.json.gz)

## Right-edge concentration

All 188 saved pre-drop observations were available. The board was mirrored by reflecting x, preserving tiers and current/next fruit, and sorting the resulting board with the shared observation encoder. Mirrored actions were compared with action index `14 - action`. All exact ties use the smallest action index, including shortlist ties. A near tie means the runner-up is within 0.001 of the winner.

| PPO weight | Rightmost | Logged actions reproduced | Exact ties | Near ties excluding exact | Mirrored rightmost / leftmost | Mirrored action agrees after unmirroring |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31 | 59 | 19 | 62 | 8 / 39 | 139 |
| 0.05 | 116 | 178 | 0 | 0 | 78 / 25 | 41 |
| 0.20 | 121 | 188 | 0 | 0 | 92 / 20 | 33 |

At weight zero, nine of the 31 rightmost wins were near ties. At the deployed weight, none of the 121 rightmost wins were near ties. Across all 188 deployed decisions the median winning margin was 0.284. The fixed checkpoint's policy term accounts for most of the right-edge concentration on these observations and often keeps pulling toward the screen's right after reflection. The archived game contains website outcomes only for executed moves. [All action scores and margins](edge-diagnostics.json)

## Planner and test protocol

A uses the legacy one-drop hybrid. B uses the same nominal physics scoring at PPO weight zero. C shortlists the top four under A, then scores the nominal case and three independent antithetic pairs, shared across all four candidates. It maximizes mean legacy simulated score minus twice mean overflow risk, plus 0.20 times the frozen PPO log probability. Overflow risk rises linearly as the simulated highest fruit top approaches the overflow boundary from 20 local pixels below it, and saturates at or above that boundary or at capacity. The legacy simulator's own overflow penalty is included in the simulated score.

Offsets are uniform over ±4 x and ±5 y local pixels per detected fruit, ±6 local pixels of aim, ±0.025 gravity, and ±0.045 collision impulse. The prior site's aiming residual was 0.34 screenshot pixels median and 0.876 p95. Those measurements describe the visible hand before release. The wider aim and center perturbations here are sensitivity settings for placement and vision error. Nominal center coordinates and release physics match the legacy implementation.

The development set used 20270101 through 20270103 in all profiles. The settings were fixed in code before that check and retained after it. All-action search agreed with the shortlist on all 26 sampled development boards and on 208 of 213 held-out audit boards. The replay fruit seed was set to 20270201 before scoring. Each method used the same 24 fresh fruit seeds from 20270201 through 20270224 in each profile. Dynamics, vision, and planner draws use separate RNG streams. The stress profile uses the larger episode-level physics variation from the prior experiment with 1.5× synthetic vision noise. This includes tier errors, missed detections, and false positives beyond the planner's perturbations. The common 300-drop scores are snapshots of the same trajectories that continue to 450 drops or overflow.

## Paired results

Each row lists mean score. C win/tie/loss compares paired scores against A. Counts in the last column are actual overflow events for A/B/C, out of 24 per method. Remaining episodes were alive at the cap. Highest tier and all paired episode records are in the results JSON and use zero-based tier numbering.

| Profile and horizon | A | B | C | C − A | C wins/ties/losses | Overflow A/B/C |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Clean · 300 | 3,125.5 | 2,952.2 | 3,110.0 | −15.5 | 11/0/13 | 9/10/5 |
| Joint · 300 | 2,912.8 | 2,672.8 | 2,962.7 | +50.0 | 16/0/8 | 12/15/10 |
| Stress · 300 | 2,889.1 | 2,983.5 | 2,891.1 | +2.0 | 12/0/12 | 12/10/10 |
| Clean · 450 | 3,420.6 | 3,294.5 | 3,623.2 | +202.5 | 16/0/8 | 24/24/23 |
| Joint · 450 | 3,098.0 | 2,781.5 | 3,115.5 | +17.5 | 13/0/11 | 24/24/24 |
| Stress · 450 | 3,059.7 | 3,212.5 | 3,110.2 | +50.5 | 11/0/13 | 24/24/24 |

C was selected for the website on the joint profile. The lower joint overflow count at 300 disappears by 450. The paired video intentionally uses the preselected seed, where A scored 3,252 and C scored 3,166 at 300.

At the same board state, C changed A's nominal choice 2,338 times. Replaying the two actions for one drop in copies of the actual local environment found 522 immediate improvements, 268 reductions, and 1,548 ties. Six C choices overflowed on that drop where A's choice would survive. In joint seed 20270213 on drop 236, C moved from action index 14 to 2, scoring zero where A's action would have scored 116. Median standard deviation of the selected candidate's simulated outcome on changed decisions was 0.031 rank units. Immediate reward comparisons describe one-step differences on C's trajectory.

On the same 188 archived boards with a single inference process, A took 3.12 ms median and 7.86 ms p95, B took 3.03 and 7.61 ms, and C took 7.72 and 21.10 ms. The measured increases for C versus A were 4.61 ms median and 13.24 ms p95. This includes policy inference and action selection, with compilation, screenshot processing, and trace rendering outside the timer. [Latency by board](latency.json)

## Commands run

```bash
.venv/bin/python -m flymerge.planner_experiment diagnose
.venv/bin/python -m flymerge.planner_experiment develop --workers 9
.venv/bin/python -m flymerge.planner_experiment evaluate --workers 12
.venv/bin/python -m flymerge.planner_video
.venv/bin/python -m flymerge.planner_experiment benchmark
.venv/bin/python -m unittest discover -s tests -v
setsid xvfb-run -a -s '-screen 0 1440x1024x24' .venv/bin/python -m flymerge.site \
  --checkpoint artifacts/transfer/checkpoints/flywire-seed11-final.pt \
  --strategy robust --until-game-over --out artifacts/planner/website \
  --replay-key plannerWebsite > runs/planner-website.log 2>&1 < /dev/null &
```
