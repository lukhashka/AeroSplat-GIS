#!/usr/bin/env bash
# Stop a SITL instance started by run_sitl.sh (its supervisor kills arducopter and MAVProxy).
# Env: SITL_RUN_DIR (required), SITL_INSTANCE (0). Exit 0 when nothing was running.
set -euo pipefail
: "${SITL_RUN_DIR:?stop_sitl: SITL_RUN_DIR is not set}"
run_dir=${SITL_RUN_DIR/#\~/$HOME}
run="$run_dir/i${SITL_INSTANCE:-0}"
pidfile="$run/supervisor.pid"

if [ ! -f "$pidfile" ]; then echo "stop_sitl: not running"; exit 0; fi
pid=$(cat "$pidfile")
if ! kill -0 "$pid" 2>/dev/null; then
    echo "stop_sitl: stale pid file ($pid), removed"; rm -f "$pidfile"; exit 0
fi
kill -TERM "$pid"
for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || { echo "stop_sitl: stopped ($pid)"; exit 0; }
    sleep 0.1
done
# Supervisor did not exit: kill it and anything it started.
pkill -KILL -P "$pid" 2>/dev/null || true
kill -KILL "$pid" 2>/dev/null || true
rm -f "$pidfile"
echo "stop_sitl: killed ($pid)"
