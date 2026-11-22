# Orchard Drop and FlyWire PPO

Orchard Drop is an original seeded fruit-merging game. This experiment compares a fixed reservoir derived from the adult female Drosophila FlyWire FAFB v783 connectome, a degree-preserving rewired control, and a small MLP. Each of nine CPU trials collected 100,352 fruit drops. Each trial used a separate learner and four game processes.

## Watch

Open [the interactive replay](artifacts/replays/index.html) in a modern browser after cloning. It plays the committed MP4s and synchronizes each fruit drop to six simulated recurrent steps. Choose a milestone, use the timeline or one-drop buttons, and click a neuron to inspect its direct input, recurrent input, activation, and zero-input reference.

| Milestone | Actual training drops | Completed PPO updates | Showcase score | MP4 |
| --- | ---: | ---: | ---: | --- |
| 5 drops | 8 | 0 | 1,051 | [5-drop checkpoint](artifacts/videos/drop5.mp4) |
| 1,000 drops | 1,000 | 0 | 1,051 | [1,000-drop checkpoint](artifacts/videos/drop1000.mp4) |
| Early · 5% | 5,020 | 4 | 929 | [Early](artifacts/videos/early.mp4) |
| Middle · 50% | 50,176 | 49 | 891 | [Middle](artifacts/videos/middle.mp4) |
| Final · 100% | 100,352 | 98 | 969 | [Final](artifacts/videos/final.mp4) |

The four-worker collector reached the requested 5 drops in the batch ending at drop 8. The 5% target is 5,017 integer drops, first reached at 5,020. The first two recordings show the same pre-update gameplay with different checkpoint metadata. The first optimizer update follows drop 1,024. Seed 11 and fruit sequence 20261122 were fixed before evaluation. [Video manifest](artifacts/videos/manifest.json) · [Evaluation protocol](artifacts/protocol.json)

### Website transfer

One headed Chromium episode ran on an Xvfb display on the training machine. Playwright used visible screenshots, public fruit sprites as image templates, and normal mouse input. The site showed the current fruit on the aiming line and the next fruit in its own preview. Match confidence and frame-to-frame board stability gated releases. The 1280 × 900 viewport gave a measured board of (370, 216) to (896, 781). The first aim matched the visible current fruit within 0.8 pixels and the drop landed. Before the second decision, a sticky ad banner covered the bottom of the board and the first fruit. Foreground coverage fell to zero and the detector paused. The attempt ended after one drop at observed score 0. Screenshot replay is available without contacting the site. [Transfer replay](artifacts/replays/index.html#website) · [Annotated decision](artifacts/demo/screens/001.jpg) · [Paused view](artifacts/demo/paused.png) · [Capture manifest](artifacts/demo/manifest.json)

## Reproduce

Linux and Python 3.12 were used. Start from the repository root. CPU PyTorch is installed from its CPU wheel index. FFmpeg is provided by imageio-ffmpeg. Chromium is only required for the transfer demo and browser playback checks.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m unittest discover -s tests -v
```

The checked-in graph can be used immediately. To rebuild it from the source tables, download both files once and run the importer. Its inputs can also point to local copies.

```bash
mkdir -p data
source_url=https://storage.googleapis.com/flywire-data/codex/data/fafb/783
curl -fL "$source_url/connections_princeton.csv.gz" -o data/connections.csv.gz
curl -fL "$source_url/coordinates.csv.gz" -o data/coordinates.csv.gz
.venv/bin/python -m flymerge.graph \
  --connections data/connections.csv.gz --coordinates data/coordinates.csv.gz \
  --out artifacts/graph.json

.venv/bin/python -m flymerge.train --kind flywire --seed 99 \
  --workers 2 --budget 1024 --out runs/smoke
bash flymerge/run-benchmark.sh
bash flymerge/run-suite.sh
.venv/bin/python -m flymerge.replay evaluate
.venv/bin/python -m flymerge.plot
.venv/bin/python -m flymerge.replay render
.venv/bin/python -m flymerge.verify
```

The render command reuses matching entries in the existing video manifest. For a fresh encoding, run it from a fresh checkout after moving `artifacts/videos/manifest.json` out of the way. To use the recorded checkpoint weights in a clean clone, extract the archive before verification. The training and evaluation directories are `runs/<agent>-<seed>/`. The full run produces initialization, 5-drop, 1,000-drop, early, middle, and final checkpoints for each trial. The compact archive contains all 54 policy checkpoints. [Checkpoint archive](artifacts/checkpoints/ppo-checkpoints.tar.gz) · [Final transfer policy](artifacts/checkpoints/flywire-seed11-final.pt)

```bash
tar -xzf artifacts/checkpoints/ppo-checkpoints.tar.gz
.venv/bin/python -m flymerge.verify
```

The visible transfer command starts one headed browser and stops after at most 100 drops or ten minutes, or earlier on game over or uncertain screenshots. Use an existing desktop session or prefix the command with `xvfb-run -a -s '-screen 0 1440x1024x24'` on a machine with Xvfb.

```bash
.venv/bin/playwright install chromium
.venv/bin/python -m flymerge.site \
  --checkpoint artifacts/checkpoints/flywire-seed11-final.pt \
  --out runs/new-transfer
```

A synthetic fixture verifies the graph and training path if a source table is unavailable. Its IDs are explicitly labeled and its graph should be stored under a separate filename.

```bash
.venv/bin/python -m flymerge.graph --synthetic --out runs/synthetic-graph.json
.venv/bin/python -m flymerge.train --kind flywire --seed 99 --workers 2 \
  --budget 1024 --graph runs/synthetic-graph.json --out runs/synthetic-smoke
```

## Method

- Each decision releases one fruit at one of 15 evenly spaced legal positions. Circle dynamics advance for 80 ticks, with equal tiers merging into the next tier and earning the score increment. A fruit protruding above the overflow line after those ticks ends the episode. The environment also has a 300-drop time limit.
- The shared observation contains 80 entries of normalized top-origin center x/y and tier divided by ten, sorted by y, x, then tier, followed by current and next tiers. Missing features and unused slots carry −1. In training each of current and next is independently masked with probability 0.12. The held-out local evaluation provides both.
- The graph importer aggregates directed pairs from the filtered v783 Princeton connection table, omits autapses, and visits the strongest outgoing connections breadth first, starting at the highest total outgoing synapse count. Ties use increasing FlyWire root ID. It retains the first 512 visited neurons and their induced directed edges. The graph records source SHA-256 hashes, exact root IDs, 1,631 directed edges, 28,285 retained synapses, representative 3D coordinates for all 512 neurons, and neuropil counts for every retained synapse. Coordinates are the first matching public v783 coordinates for each root ID. [Graph](artifacts/graph.json) · [FlyWire Codex v783](https://codex.flywire.ai/fafb) · [Dorkenwald et al., Nature 2024](https://doi.org/10.1038/s41586-024-07558-y)
- Each feature drives four of the first 128 nodes through a fixed seeded signed projection, with each input column normalized by its L2 norm. Before every action, activation resets to zero. Six steps use `h = tanh(direct + h @ W)`. Edge magnitudes start at the square root of retained synapse count, then normalize by incoming magnitude sum with gain 0.92. All 512 activations feed separate linear actor and critic readouts, for 8,208 trainable parameters. The model uses unsigned connection counts and generic activations to simulate a reservoir.
- The control swaps paired directed targets, rejecting duplicate edges and self-loops. In-degree and out-degree match exactly. Both graphs reach every readout within one directed hop from the set of input nodes. Rewiring replaces 312 of 1,631 edges (19.13%). Synapse counts move with their source edge, so the raw weight multiset is preserved. Incoming renormalization changes 893 of 1,319 retained shared weights by more than 0.000001. The MLP uses one 32-unit tanh layer and has 8,304 trainable parameters.
- PPO uses 1,024 drops per update, four optimization epochs over randomized 256-drop minibatches, Adam at 0.0003, discount 0.99, GAE 0.95, ratio clip 0.2, value coefficient 0.5, entropy coefficient 0.01 and gradient norm clip 0.5. Reservoir features are frozen and cached across optimization epochs. All trials had 98 PPO updates. Evaluation uses the frozen policy with categorical action sampling and the same 12 fruit seeds and paired action RNG seeds for every checkpoint and architecture.
- The replay compares simulated activation against a zero-game-input pass with the same initial state. Diamonds identify nonzero direct drive. Other changes come through the recurrent network. The neuropil proxy weights each presynaptic neuron's absolute activation change by its outgoing retained synapse counts in that region and divides by the annotated count in that region. Coordinates are projected into 2D and can be rotated. The replay uses a fixed −1 to +1 color scale.

## Results and runtime

Mean held-out score over 12 paired episodes per cell. Each value in the two right columns is a paired mean difference at the final checkpoint.

| Training seed | FlyWire init → final | Rewired init → final | MLP init → final | FlyWire − rewired | FlyWire − MLP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 917.3 → 977.2 | 917.3 → 1,158.9 | 915.2 → 971.2 | −181.8 | +5.9 |
| 22 | 975.8 → 1,263.2 | 983.0 → 1,073.2 | 932.4 → 1,057.4 | +190.0 | +205.8 |
| 33 | 984.6 → 1,030.3 | 1,029.9 → 1,044.3 | 969.2 → 998.8 | −14.0 | +31.6 |

The FlyWire reservoir finished above the MLP on these three training seeds. Its paired difference against the degree-preserving control changed sign across seeds, with a mean of −1.9 points. The comparison covers one selected subgraph and three training seeds. [Learning curves](artifacts/results/learning-curves.png) · [Evaluation episodes](artifacts/results/evaluation.json) · [Paired differences](artifacts/results/paired.json) · [Training episode and update logs](artifacts/results/runs)

The machine reported 16 Intel Haswell-class logical CPUs and 125 GiB of usable RAM. PyTorch, OpenMP and BLAS each used one thread per process. The two-worker smoke run reached 793 drops/s. Eight-worker benchmarks reached 1,988 drops/s dense and 2,207 drops/s sparse. Three concurrent four-worker dense runs reached about 4,039 drops/s in aggregate. The full suite trained in architecture waves of three learners and 12 workers, using sparse recurrence for the reservoirs and advancing 903,168 drops in total. Full-suite throughput was 1,286–1,305 drops/s per reservoir trial and 1,569–1,591 drops/s per MLP trial. Those rates include game physics, batch inference, interprocess communication, and PPO updates. Each architecture wave took about 80.3 s, 80.7 s, and 66.3 s respectively including initialization. Graph preprocessing took 20.4 s, evaluation took 96.5 s, and rendering plus decoding took about 149 s across five videos. Some independent stages ran concurrently.

For the showcased FlyWire trial, the learning loop took 76.9 s, with 29.6 s for batched inference, 38.8 s in environment dispatch/receive, and 7.3 s in PPO. The environment phase included an estimated 9.2 s of communication and scheduling wait beyond the slowest worker's measured duration per batch. The maximum process RSS was 443 MiB for a learner and 200 MiB for a worker. These per-process peaks include shared pages. [Benchmarks](artifacts/results/benchmarks) · [Per-trial runtime and memory](artifacts/results/training.json)

FFmpeg decoded all five H.264 MP4s. The verifier checks video lengths against the recorded physics frame counts, reproduces every local and website replay action and circuit trace with the corresponding frozen policy, and checks local scores and termination. Chromium loaded all five videos and the website capture and exercised playback, drop navigation, and recurrent-step controls.
