#!/usr/bin/env bash
# Three sequential headed attempts: frozen robust planner, extended vision, spawn-contact gate.
set -uo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
mkdir -p runs/spawn-live
for attempt in 1 2 3; do
  out="runs/spawn-live/attempt-$attempt"
  xvfb-run -a -s '-screen 0 1440x1024x24' "$py" -m flymerge.site \
    --checkpoint artifacts/transfer/checkpoints/flywire-seed11-final.pt \
    --strategy robust --extended-vision --contact-check --until-game-over --wall-seconds 1800 \
    --action-seed "$((50000+attempt))" --replay-key "spawnLive$attempt" \
    --out "$out" --record "$out/record" > "runs/spawn-live/attempt-$attempt.log" 2>&1
  echo "attempt $attempt exit $?"
done
