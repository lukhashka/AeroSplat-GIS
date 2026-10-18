#!/usr/bin/env bash
# Print the parameter metadata of the pinned ArduCopter source as JSON (apm.pdef.json).
#
# Runs ArduPilot's own Tools/autotest/param_metadata/param_parse.py from the pinned worktree in a
# scratch directory, so the worktree stays clean (setup_ardupilot.sh refuses a dirty worktree).
# Used by `python -m aerosplat.sim.params snapshot`.
#
# Required env: AP_WORKTREE AP_COMMIT AP_VENV
set -euo pipefail

for v in AP_WORKTREE AP_COMMIT AP_VENV; do
    if [ -z "${!v:-}" ]; then echo "param_metadata: $v is not set" >&2; exit 2; fi
done
expand() { local p="$1"; echo "${p/#\~/$HOME}"; }
wt=$(expand "$AP_WORKTREE")
venv=$(expand "$AP_VENV")
head=$(git -C "$wt" rev-parse HEAD)
[ "$head" = "$AP_COMMIT" ] || { echo "param_metadata: $wt is at $head, pinned $AP_COMMIT" >&2; exit 1; }

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
(cd "$out" && "$venv/bin/python" "$wt/Tools/autotest/param_metadata/param_parse.py" \
    --vehicle ArduCopter --format json > param_parse.log 2>&1) \
    || { tail -20 "$out/param_parse.log" >&2; exit 1; }
cat "$out/apm.pdef.json"
