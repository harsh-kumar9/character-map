#!/bin/bash
# Trait-file generation on ada (CPU and network only, no GPU). Usage: bash scripts/gen_traits.sh [cm.gen_traits args]
# Reads the OpenRouter key from the demand-worlds .env at run time; the key is never printed or copied.
set -euo pipefail
KEYFILE="${CM_KEYFILE:-/ada1/u/harsh/demand-worlds-corpus/.env}"
OPENROUTER_API_KEY="$(grep -m1 '^OPENROUTER_API_KEY=' "$KEYFILE" | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
export OPENROUTER_API_KEY
[ -n "$OPENROUTER_API_KEY" ] || { echo "FATAL: no OPENROUTER_API_KEY in $KEYFILE"; exit 1; }
cd /ada1/u/harsh/character-map
exec /ada1/u/harsh/miniconda3/envs/sote/bin/python -u -m cm.gen_traits "$@"
