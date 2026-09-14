#!/usr/bin/env bash
# Alternate benchsparse runs between control binary and variant binary.
set -euo pipefail
CTRL="${1:-./qpow_hip}"
VAR="${2:-./qpow_hip_opt}"
SECS="${3:-10}"
ROUNDS="${4:-3}"

echo "control=$CTRL variant=$VAR secs=$SECS rounds=$ROUNDS"
for i in $(seq 1 $ROUNDS); do
    echo "--- round $i ---"
    c=$($CTRL benchsparse $SECS | grep -oP 'benchsparse: \K[0-9.]+')
    v=$($VAR benchsparse $SECS | grep -oP 'benchsparse: \K[0-9.]+')
    echo "control: ${c} MH/s  variant: ${v} MH/s"
done
