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
