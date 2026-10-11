#!/usr/bin/env bash
# Start or resume every remaining experiment in the background, safe to run any number of times.
# Finished images are skipped, so after a shutdown this simply continues where it stopped.
#
#   ./resume.sh            # launch (returns immediately; logs in logs/)
#   ./resume.sh status     # show progress
set -uo pipefail
builtin cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore
mkdir -p logs

# patterns include "bin/python" so they match the real jobs, never this script's own command line
running() { pgrep -f "$1" >/dev/null; }
DEV="bin/python src/run_all.py --phase dev"; TEST="bin/python src/run_all.py --phase test"
SDXL="bin/python src/run_all.py --phase sdxl"; EVAL="bin/python src/eval_watch.py"

if [ "${1:-}" = "status" ]; then
  for p in "$DEV" "$TEST" "$SDXL" "$EVAL" "resume_chain"; do
    running "$p" && echo "running : $p" || echo "stopped : $p"
  done
  echo "--- evaluated runs"
  ls results/*/*/summary.json 2>/dev/null | sed 's|results/||; s|/summary.json||' | tr '\n' ' '; echo
  [ -f results/main/sdxl/meta.json ] && $PY -c "import json;print('SDXL images:', len(json.load(open('results/main/sdxl/meta.json'))['times']), '/ 80')"
  [ -f results/.all_done ] && echo "ALL DONE"
  exit 0
fi

rm -f results/.all_done

running "$EVAL" || (nohup $PY src/eval_watch.py >> logs/eval.txt 2>&1 &)

nohup bash -c '
  # resume_chain
  PY=.venv/bin/python
  wait_for() { while pgrep -f "bin/python src/$1" >/dev/null; do sleep 30; done; }
  (
    wait_for "run_all.py --phase dev";  $PY src/run_all.py --phase dev  >> logs/run_dev.txt 2>&1
    wait_for "run_all.py --phase test"; $PY src/run_all.py --phase test >> logs/run_test.txt 2>&1
  ) &
  (
    wait_for "run_all.py --phase sdxl"; $PY src/run_all.py --phase sdxl >> logs/run_sdxl.txt 2>&1
  ) &
  wait
  touch results/.all_done
  wait_for "eval_watch.py"
  $PY src/analysis.py >> logs/analysis.txt 2>&1
  echo "ALL DONE $(date)" >> logs/run_test.txt
' > logs/resume_chain.txt 2>&1 &

# keep the Mac awake for the whole chain, not just this shell
sleep 1
caffeinate -i -s -w "$(pgrep -f resume_chain | head -1)" >/dev/null 2>&1 &
echo "launched; check with: ./resume.sh status"
