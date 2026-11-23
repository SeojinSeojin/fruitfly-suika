#!/usr/bin/env bash
set -euo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
mode=$("$py" -c 'import json; d=json.load(open("artifacts/rollout-transfer/results.json")); assert d["live_eligible"], "Joint held-out gate failed"; print(d["selected"])')
mkdir -p runs
pids=()
for attempt in 1 2 3; do
  xvfb-run -a -s '-screen 0 1440x1024x24' "$py" -m flymerge.site \
    --checkpoint artifacts/transfer/checkpoints/flywire-seed11-final.pt \
    --rollout-mode "$mode" --vision-workers 4 --until-game-over --wall-seconds 1500 \
    --action-seed "$((47100+attempt))" --replay-key "rolloutAttempt$attempt" \
    --out "artifacts/rollout-transfer/live-$attempt" > "runs/rollout-live-$attempt.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
