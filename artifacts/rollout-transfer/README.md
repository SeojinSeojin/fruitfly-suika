# Direct rollout transfer

Worktree branch `experiment/rollout-transfer` starts at `ffd2d82029008f5ad98be4282d9489aba7321244`, the fetched tip of `experiment/counterfactual-drop-ranking`. The host exposed 16 logical CPUs. The existing deployed PPO and both previous afterstate checkpoints were frozen. [Protocol and frozen seed sets](protocol.json)

[Paired video](comparison.mp4) · [Interactive replay](index.html) · [Per-episode held-out results](results.json)

## Planner

The reconstructed planner converts normalized detected centers into transfer-board coordinates and copies fruit tiers. All starting velocities are zero. It rolls out nominal gravity, damping, bounce, and impact with the existing transfer simulator and frozen first-contact continuation policy. The held fruit is required. If its tier is unknown, control falls back to the robust planner. The known preview stays fixed. An unknown preview is sampled once per hypothetical tail from the site's local stand-in fruit distribution (tiers 0–4).

Each candidate in a decision gets the same preview and unseen fruit tail for each hypothetical rollout. The tail RNG is keyed by the episode seed, drop, and search stream, independent of the played episode's fruit and detector RNGs. The continuation sees only its own current and next fruits as they become available. Utility is the actual simulated points on the first move plus the next 20 drops' points, minus 150 for any overflow. A rollout ends immediately at overflow. Ties use the smallest action index.

The original six candidates reuse the robust pick, both previous afterstate picks, and nominal physics ranks. Fixed search uses three shared tails, at most 18 branches. The adaptive variant adds up to two candidates if the two best nominal scores are within 0.15 rank units. It adds three shared tails if the leading utility margin is at most 25 points or within one paired standard error. It caps the total at 48 branches. Both variants use identical models and legal moves.

## Development selection

Six fresh development fruit seeds were played under both clean and joint profiles. On the joint extended horizon, robust averaged 3,394.2, fixed 3,259.8, and adaptive 2,867.2. On the clean extended horizon the means were 3,205.0, 4,026.7 and 3,545.8, respectively. Fixed was therefore frozen for held-out evaluation. Across the 67 development boards with two independent search streams, fixed repeated the winning aim 17 times and adaptive 11 times. This small audit found no repeatability gain from the adaptive setting. Adaptive used 43.7 branches per decision on average. Median / p95 joint extended latency was 10.1 / 27.8 ms for robust, 194.8 / 401.4 ms for fixed, and 383.4 / 880.4 ms for adaptive. Development ran for 297 seconds with 12 workers. [Development results](development.json)

## Matched diagnostic forecasts

On 45 boards from the first six held-out fruit seeds under each profile, the diagnostic holds candidates and hypothetical fruit sequences fixed and changes only the simulated starting state and physics. In clean games, using the true board and velocities changed 15 of 23 winners. In joint games it changed 19 of 22. Switching that exact board to the episode's actual physics changed 18 of 22 joint winners again. In joint games mean utility regret against the exact-board, actual-physics forecast shrank from 31.4 to 26.8 points with the true starting board alone. These measurements use local counterfactuals. [Matched state and physics diagnostic](diagnostics.json)

## Held-out paired evaluation

The full comparison uses 24 fresh fruit sequences per profile, paired across robust and the frozen fixed reconstructed planner. The rows at 300 are snapshots of the same episodes that run through overflow or 450 drops. The rightmost column counts games alive at that horizon. Latency includes candidate selection and the rollout calculation, after warming each method before timing. The 12-worker held-out run took 455 seconds.

| Profile and horizon | Robust | Reconstructed | Paired change | Rollout wins / ties / losses | Above 3,000, robust / rollout | Alive at cap, robust / rollout |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Clean, 300 | 3,087.5 | 3,099.8 | +12.3 | 11 / 0 / 13 | 19/24 / 20/24 | 19 / 22 |
| Clean, 450 | 3,626.5 | 3,723.8 | +97.3 | 13 / 0 / 11 | 21/24 / 22/24 | 1 / 2 |
| Joint, 300 | 2,883.1 | 2,688.9 | −194.2 | 9 / 0 / 15 | 12/24 / 8/24 | 14 / 12 |
| Joint, 450 | 3,047.9 | 2,928.3 | −119.7 | 11 / 0 / 13 | 15/24 / 12/24 | 0 / 0 |

On the primary joint extended metric, robust took **9.8 ms median / 26.0 ms p95** and reconstructed search took **190.5 ms / 422.4 ms**. On the clean extended metric the times were 9.7 / 25.9 ms and 182.7 / 388.9 ms. Reconstructed search used exactly 18 branches per timed decision. [Episode data and cap seeds](results.json)

### Exact-state matching subset

The original exact-state teacher ran on the first six held-out seeds per profile. The table compares all three methods on those same seeds. The exact-state teacher has access to the true starting positions, velocities, dynamics and visible-preview state and uses the previous teacher's own joint vision rollout path. The matched forecast audit above isolates changes to starting state and physics with identical hypothetical tails.

| Profile and horizon | Robust | Reconstructed | Exact |
| --- | ---: | ---: | ---: |
| Clean, 300 | 2,973.3 | 3,159.8 | 3,202.7 |
| Clean, 450 | 3,344.2 | 3,527.5 | 4,657.7 |
| Joint, 300 | 2,663.5 | 2,723.3 | 3,184.7 |
| Joint, 450 | 2,743.7 | 3,001.8 | 4,639.5 |

The joint exact-state teacher beat robust on all six extended pairs and three exact-state episodes reached the 450-drop cap alive. Exact-state access therefore preserved most of the teacher's usable advantage under perturbations. The six-seed reconstructed subset was positive, while the full 24-seed paired effect was negative. The matched forecasts found both starting-board mismatch and dynamics mismatch. A practical next test is to measure transfer with motion tracked over multiple screenshots and physics sampled during search, checking each change against the same exact-state reference.

The live gate required a positive mean change on the **full held-out joint extended evaluation**. Reconstructed search fell short, so **zero website attempts were run**. The 3,000-point website threshold remains unverified.

## Reproduce

```bash
.venv/bin/python -m flymerge.rollout_eval develop --workers 12
.venv/bin/python -m flymerge.rollout_eval evaluate --workers 12
.venv/bin/python -m flymerge.rollout_diagnostics
.venv/bin/python -m flymerge.rollout_video
.venv/bin/python -m flymerge.rollout_video --verify
.venv/bin/python -m unittest discover -s tests -v
```

The browser adapter exposes `--rollout-mode fixed` and `adaptive`. The gated live runner allows natural game over within 1,500 seconds per attempt and uses the existing screenshot and aim checks. The predeclared showcase seed is 20271401 with the joint profile.
