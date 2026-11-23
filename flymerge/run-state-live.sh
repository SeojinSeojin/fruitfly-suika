#!/usr/bin/env bash
set -euo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
mode=$("$py" -c 'import json; d=json.load(open("artifacts/state-estimation/results.json")); assert d["live_eligible"], "Joint held-out gate failed"; print(d["best"])')
mkdir -p runs
pids=()
for attempt in 1 2 3; do
  xvfb-run -a -s '-screen 0 1440x1024x24' "$py" -m flymerge.site \
    --checkpoint artifacts/transfer/checkpoints/flywire-seed11-final.pt \
    --state-mode "$mode" --vision-workers 4 --until-game-over --wall-seconds 1500 \
    --action-seed "$((48000+attempt))" --replay-key "stateAttempt$attempt" \
    --out "artifacts/state-estimation/live-$attempt" > "runs/state-live-$attempt.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
