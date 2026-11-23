#!/usr/bin/env bash
# Three sequential headed attempts with the frozen robust planner and passive release recording.
set -uo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
mkdir -p runs/live-baseline
for attempt in 1 2 3; do
  out="runs/live-baseline/attempt-$attempt"
  xvfb-run -a -s '-screen 0 1440x1024x24' "$py" -m flymerge.site \
    --checkpoint artifacts/transfer/checkpoints/flywire-seed11-final.pt \
    --strategy robust --until-game-over --wall-seconds 1800 \
    --action-seed "$((49000+attempt))" --replay-key "liveBaseline$attempt" \
    --out "$out" --record "$out/record" > "runs/live-baseline/attempt-$attempt.log" 2>&1
  echo "attempt $attempt exit $?"
done
