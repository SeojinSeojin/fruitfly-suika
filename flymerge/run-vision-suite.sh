#!/usr/bin/env bash
set -euo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
for seed in 51 52 53; do
  pids=()
  for condition in fixed physics both; do
    options=()
    if [[ $condition != fixed ]]; then options+=(--randomize-physics); fi
    if [[ $condition == both ]]; then options+=(--vision-noise 1); fi
    "$py" -m flymerge.train --kind flywire --rules transfer --seed "$seed" \
      --workers 4 --budget 49152 "${options[@]}" \
      --initialize artifacts/transfer/checkpoints/flywire-seed11-final.pt \
      --out "runs/vision-suite/$condition-$seed" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then exit 1; fi
done
