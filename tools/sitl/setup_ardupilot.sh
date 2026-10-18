#!/usr/bin/env bash
# Create the pinned ArduPilot worktree used by AeroSplat-GIS and build ArduCopter SITL in it.
#
# Runs inside Linux / WSL2. Normally called through `python -m aerosplat.sim.sitl setup`, which
# passes the pin from configs/sitl.yaml as environment variables:
#   AP_TAG, AP_COMMIT     pinned release tag and the commit it must resolve to
#   AP_REMOTE             upstream URL to fetch the tag from when the clone lacks it
#   AP_SOURCE_CLONE       existing ArduPilot clone; only its object store and tags are used
#   AP_WORKTREE           where the pinned, detached worktree lives
#   AP_VENV               Python venv with waf's dependencies (empy 3.x, pexpect, future)
#
# The source clone's checkout, branches and build directory are never touched: the pinned tree is
# a separate `git worktree` in detached HEAD state with its own build/ directory.
# Idempotent: re-running verifies the pin and rebuilds incrementally.
set -euo pipefail

for v in AP_TAG AP_COMMIT AP_REMOTE AP_SOURCE_CLONE AP_WORKTREE AP_VENV; do
    if [ -z "${!v:-}" ]; then echo "setup_ardupilot: $v is not set" >&2; exit 2; fi
done
expand() { local p="$1"; echo "${p/#\~/$HOME}"; }
src=$(expand "$AP_SOURCE_CLONE")
wt=$(expand "$AP_WORKTREE")
venv=$(expand "$AP_VENV")

die() { echo "setup_ardupilot: $*" >&2; exit 1; }

[ -d "$src/.git" ] || die "source clone $src not found"
[ -x "$venv/bin/python" ] || die "venv $venv not found"

# 1. The tag must exist locally and resolve to the pinned commit (a moved tag is a hard error).
if ! git -C "$src" rev-parse -q --verify "refs/tags/$AP_TAG^{commit}" >/dev/null; then
    echo "fetching tag $AP_TAG from $AP_REMOTE"
    git -C "$src" fetch --no-tags "$AP_REMOTE" "refs/tags/$AP_TAG:refs/tags/$AP_TAG"
fi
tag_commit=$(git -C "$src" rev-parse "refs/tags/$AP_TAG^{commit}")
[ "$tag_commit" = "$AP_COMMIT" ] || die "tag $AP_TAG -> $tag_commit, pinned $AP_COMMIT"

# 2. Detached worktree at the pinned commit.
if [ ! -e "$wt" ]; then
    git -C "$src" worktree add --detach "$wt" "$AP_COMMIT"
fi
head=$(git -C "$wt" rev-parse HEAD)
[ "$head" = "$AP_COMMIT" ] || die "$wt is at $head, pinned $AP_COMMIT (remove it or check out the pin)"
if [ -n "$(git -C "$wt" status --porcelain --untracked-files=no --ignore-submodules=all)" ]; then
    die "$wt has local modifications"
fi

# 3. Submodules. Borrow objects from the source clone's module stores when they exist
#    (--dissociate copies them, so the pinned tree does not depend on that clone later).
git -C "$wt" config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
while read -r key path; do
    name=${key#submodule.}; name=${name%.path}
    if [ -e "$wt/$path/.git" ]; then continue; fi
    ref="$src/.git/modules/$name"
    if [ -d "$ref" ]; then
        git -C "$wt" submodule update --init --reference "$ref" --dissociate -- "$path"
    else
        git -C "$wt" submodule update --init -- "$path"
    fi
done
git -C "$wt" submodule update --init --recursive

# 4. Build. waf runs with the venv's interpreter; nothing from the source clone is on PATH.
cd "$wt"
export PATH="$venv/bin:/usr/lib/ccache:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
"$venv/bin/python" ./waf configure --board sitl
"$venv/bin/python" ./waf copter
bin="$wt/build/sitl/bin/arducopter"
[ -x "$bin" ] || die "build finished but $bin is missing"

printf '{"tag": "%s", "commit": "%s", "built": "%s", "binary": "%s"}\n' \
    "$AP_TAG" "$AP_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$bin" > "$wt/build/sitl/aerosplat-pin.json"
echo "OK $AP_TAG ($AP_COMMIT) -> $bin"
