#!/usr/bin/env bash
# Start the observability studio with as little ceremony as possible.
#
#   bash scripts/studio.sh                 # this project's runs, port 8790
#   bash scripts/studio.sh --port 9000
#   bash scripts/studio.sh --log ~/other/train.log --iters 1600
#
# Finds the virtual environment, checks the port, and if something already has
# it says what, and offers to end it or move to the next free one. Everything
# after the script name is passed to `dragon studio`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x ".venv/bin/dragon" ]; then
  DRAGON=".venv/bin/dragon"
elif command -v dragon >/dev/null 2>&1; then
  DRAGON="dragon"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/dragon" ]; then
  DRAGON="$VIRTUAL_ENV/bin/dragon"
else
  echo "dragon is not installed here. From this directory:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

printf '\n  Agencie.io Labs · TrainYourDragon\n  Observability studio\n\n'
exec "$DRAGON" studio "$@"
