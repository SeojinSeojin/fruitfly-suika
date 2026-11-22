#!/usr/bin/env bash
set -euo pipefail
python_bin=${PYTHON_BIN:-.venv/bin/python}
budget=${BUDGET:-100352}
for kind in flywire rewired mlp; do
  pids=()
  for seed in 11 22 33; do
    "$python_bin" -m flymerge.train --kind "$kind" --seed "$seed" --workers 4 \
      --budget "$budget" --sparse --out "runs/${kind}-${seed}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then exit 1; fi
done
