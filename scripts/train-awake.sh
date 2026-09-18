#!/usr/bin/env bash
# Train without letting the Mac doze off.
#
# A long run on battery, or with the lid shut, will sleep between wake-ups and
# take twice as long for no reason. This starts the run, then pins the machine
# awake until that process exits.
set -euo pipefail
cd "$(dirname "$0")/.."

dragon "$@" train &
pid=$!
caffeinate -ims -w "$pid" &
trap 'kill %2 2>/dev/null || true' EXIT
wait "$pid"
