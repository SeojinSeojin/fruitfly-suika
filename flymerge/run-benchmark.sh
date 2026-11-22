#!/usr/bin/env bash
set -euo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
$py -m flymerge.train --kind flywire --seed 99 --workers 8 --budget 8192 --out runs/bench-dense
$py -m flymerge.train --kind flywire --seed 99 --workers 8 --budget 8192 --out runs/bench-sparse --sparse
pids=()
for seed in 98 99 100; do
  $py -m flymerge.train --kind flywire --seed "$seed" --workers 4 --budget 8192 --out "runs/bench-concurrent-$seed" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
