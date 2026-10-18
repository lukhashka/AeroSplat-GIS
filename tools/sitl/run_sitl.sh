#!/usr/bin/env bash
# Run ArduCopter SITL (built-in physics) + MAVProxy in the foreground until stopped.
#
# Runs inside Linux / WSL2. Normally started by `python -m aerosplat.sim.sitl start`, which keeps
# the wsl.exe process that runs this script alive in the background. Stop it with stop_sitl.sh
# (or Ctrl+C when run by hand): children are killed on exit.
#
# Required env: AP_WORKTREE AP_COMMIT AP_VENV SITL_HOME (lat,lon,alt,heading) SITL_RUN_DIR
# Optional env: SITL_INSTANCE (0) SITL_MODEL (+) SITL_SPEEDUP (1) SITL_WIPE (1)
#               SITL_DEFAULTS (comma list, relative to the worktree) SITL_TEST_PORT (14560)
#               SITL_GCS_PORT (14550) SITL_GCS_HOST (auto: Windows host as seen from WSL)
#
# Endpoints (instance I adds 10*I to every port, like sim_vehicle.py):
#   tcp:127.0.0.1:5760          SITL SERIAL0, MAVProxy master
#   udpin:0.0.0.0:$TEST_PORT    pytest / pymavlink; the client must send first (heartbeat)
#   udp:$GCS_HOST:$GCS_PORT     Mission Planner on the Windows host
set -euo pipefail

for v in AP_WORKTREE AP_COMMIT AP_VENV SITL_HOME SITL_RUN_DIR; do
    if [ -z "${!v:-}" ]; then echo "run_sitl: $v is not set" >&2; exit 2; fi
done
expand() { local p="$1"; echo "${p/#\~/$HOME}"; }
die() { echo "run_sitl: $*" >&2; exit 1; }

wt=$(expand "$AP_WORKTREE")
venv=$(expand "$AP_VENV")
inst=${SITL_INSTANCE:-0}
model=${SITL_MODEL:-+}
speedup=${SITL_SPEEDUP:-1}
wipe=${SITL_WIPE:-1}
defaults=${SITL_DEFAULTS:-Tools/autotest/default_params/copter.parm}
test_port=$(( ${SITL_TEST_PORT:-14560} + 10 * inst ))
gcs_port=$(( ${SITL_GCS_PORT:-14550} + 10 * inst ))
master_port=$(( 5760 + 10 * inst ))
rc_port=$(( 5501 + 10 * inst ))
run=$(expand "$SITL_RUN_DIR")/i$inst
bin="$wt/build/sitl/bin/arducopter"

# Pinned build only: the worktree HEAD and the build stamp must both be the pinned commit.
[ -x "$bin" ] || die "$bin missing; run: python -m aerosplat.sim.sitl setup"
head=$(git -C "$wt" rev-parse HEAD)
[ "$head" = "$AP_COMMIT" ] || die "$wt is at $head, pinned $AP_COMMIT"
grep -q "\"commit\": \"$AP_COMMIT\"" "$wt/build/sitl/aerosplat-pin.json" 2>/dev/null \
    || die "build stamp does not match $AP_COMMIT; rerun setup"

abs_defaults=""
IFS=',' read -ra parts <<< "$defaults"
for p in "${parts[@]}"; do
    case "$p" in /*) f="$p" ;; *) f="$wt/$p" ;; esac
    [ -f "$f" ] || die "defaults file $f not found"
    abs_defaults="${abs_defaults:+$abs_defaults,}$f"
done

gcs_host=${SITL_GCS_HOST:-auto}
if [ "$gcs_host" = auto ]; then
    mode=$(wslinfo --networking-mode 2>/dev/null || true)
    if grep -qi microsoft /proc/version && [ "$mode" != mirrored ]; then
        # WSL2 NAT: the Windows host is the default gateway (same rule as sim_vehicle.py).
        gcs_host=$(ip route show default | awk '{print $3; exit}')
    else
        gcs_host=127.0.0.1
    fi
fi

mkdir -p "$run"
if [ -f "$run/supervisor.pid" ] && kill -0 "$(cat "$run/supervisor.pid")" 2>/dev/null; then
    die "instance $inst already running (pid $(cat "$run/supervisor.pid")); stop it first"
fi
for port in "$master_port" "$test_port"; do
    if ss -Hlntu "sport = :$port" | grep -q .; then die "port $port is in use"; fi
done
echo $$ > "$run/supervisor.pid"

pids=()
cleanup() {
    trap - EXIT TERM INT HUP
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 0.5
    for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    rm -f "$run/supervisor.pid"
}
trap cleanup EXIT
trap 'exit 0' TERM INT HUP

{
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit=$AP_COMMIT"
    echo "home=$SITL_HOME"
    echo "instance=$inst"
    echo "model=$model"
    echo "speedup=$speedup"
    echo "wipe=$wipe"
    echo "defaults=$abs_defaults"
    echo "master=tcp:127.0.0.1:$master_port"
    echo "test=udpin:0.0.0.0:$test_port"
    echo "gcs=udp:$gcs_host:$gcs_port"
    echo "networking=$(wslinfo --networking-mode 2>/dev/null || echo n/a)"
    echo "mavproxy=$("$venv/bin/python" -c 'import importlib.metadata as m; print(m.version("MAVProxy"))')"
} > "$run/session.txt"

cd "$run"   # SITL keeps eeprom.bin and DataFlash logs/ in its working directory
wipe_flag=()
[ "$wipe" = 1 ] && wipe_flag=(-w)
# The binary must keep a live parent (this script): SITL exits when it is re-parented to init.
"$bin" -I"$inst" --model "$model" --speedup "$speedup" --home "$SITL_HOME" \
    --defaults "$abs_defaults" "${wipe_flag[@]}" --sim-address=127.0.0.1 \
    > "$run/arducopter.log" 2>&1 < /dev/null &
pids+=($!)

for _ in $(seq 1 100); do
    ss -Hlnt "sport = :$master_port" | grep -q . && break
    kill -0 "${pids[0]}" 2>/dev/null || die "arducopter exited: $(tail -5 "$run/arducopter.log")"
    sleep 0.1
done

"$venv/bin/python" "$venv/bin/mavproxy.py" \
    --master "tcp:127.0.0.1:$master_port" --sitl "127.0.0.1:$rc_port" \
    --out "udpin:0.0.0.0:$test_port" --out "udp:$gcs_host:$gcs_port" \
    --state-basedir "$run" --aircraft mavproxy --retries 10 --daemon --non-interactive \
    > "$run/mavproxy.log" 2>&1 < /dev/null &
pids+=($!)

echo "run_sitl: instance $inst up; test udpin:0.0.0.0:$test_port, gcs udp:$gcs_host:$gcs_port, logs $run"
# Exit (and stop the other process) as soon as either one dies.
set +e
wait -n "${pids[@]}"
status=$?
echo "run_sitl: a child exited with status $status" >&2
exit "$status"
