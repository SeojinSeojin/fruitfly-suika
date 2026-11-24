#!/usr/bin/env bash
# Sequential headed attempts with the strict contact gate and mask memory (seeds 50004..).
# Each finished attempt is committed and pushed immediately so results survive a shutdown.
set -uo pipefail
py=${PYTHON_BIN:-.venv/bin/python}
mkdir -p runs/strict-live
for attempt in ${ATTEMPTS:-1 2 3}; do
  out="runs/strict-live/attempt-$attempt"
  xvfb-run -a -s '-screen 0 1440x1024x24' "$py" -m flymerge.site \
    --checkpoint artifacts/transfer/checkpoints/flywire-seed11-final.pt \
    --strategy robust --extended-vision --contact-check --contact-strict --mask-memory \
    --until-game-over --wall-seconds 1800 \
    --action-seed "$((50003+attempt))" --replay-key "strictLive$attempt" \
    --out "$out" --record "$out/record" > "runs/strict-live/attempt-$attempt.log" 2>&1
  status=$?
  echo "attempt $attempt exit $status"
  score=$("$py" -c "import json;m=json.load(open('$out/manifest.json'));print(m.get('final_score'),m.get('outcome'))" 2>/dev/null)
  git add -f "runs/strict-live/attempt-$attempt.log" "$out"
  git commit -q -m "Strict-gate live attempt $attempt (seed $((50003+attempt))): $score

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main
  echo "attempt $attempt committed: $score"
done
