#!/usr/bin/env bash
# Print session.txt of a running SITL instance (exit 0), or exit 3 when it is not running.
# Env: SITL_RUN_DIR (required), SITL_INSTANCE (0).
set -euo pipefail
: "${SITL_RUN_DIR:?status_sitl: SITL_RUN_DIR is not set}"
run_dir=${SITL_RUN_DIR/#\~/$HOME}
run="$run_dir/i${SITL_INSTANCE:-0}"
if [ -f "$run/supervisor.pid" ] && kill -0 "$(cat "$run/supervisor.pid")" 2>/dev/null; then
    echo "pid=$(cat "$run/supervisor.pid")"
    echo "run_dir=$run"
    cat "$run/session.txt" 2>/dev/null || true
    exit 0
fi
exit 3
